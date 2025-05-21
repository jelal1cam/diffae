import os
import json
import copy
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *

from config import *
from dataset import *
from model.classifier_architecture import build_classifier, get_classifier_info

from tqdm import tqdm

def compute_discrete_time_from_target_snr(autoenc_conf, target_snr):
    desired_alpha = target_snr / (1 + target_snr)
    latent_diffusion = autoenc_conf.make_latent_eval_diffusion_conf().make_sampler()
    alphas_cumprod = latent_diffusion.alphas_cumprod  # assumed to be a numpy array
    diffs = np.abs(alphas_cumprod - desired_alpha)
    best_t = int(np.argmin(diffs))
    computed_snr = alphas_cumprod[best_t] / (1 - alphas_cumprod[best_t])
    print(f"Closest discrete diffusion time index found: {best_t}")
    print(f"alpha_cumprod at index {best_t}: {alphas_cumprod[best_t]:.4f}")
    print(f"Computed SNR at this index: {computed_snr:.4f}")
    return best_t

class LatentDataset(Dataset):
    def __init__(self, latents, labels):
        self.latents = latents
        self.labels = labels

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):  # if index is a tensor, convert to int
            idx = idx.item()
        return {
            "cond": self.latents[idx],
            "labels": self.labels[idx],
        }

# EMA helper function
def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                  source_dict[key].data * (1 - decay))

class ClsModel(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.save_hyperparameters(conf.as_dict_jsonable())

        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.model.requires_grad_(False)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        if conf.pretrain is not None:
            print(f'Loading pretrain from {conf.pretrain.name}')
            state = torch.load(conf.pretrain.path, map_location="cpu")
            print("Loaded step:", state["global_step"])
            self.load_state_dict(state["state_dict"], strict=False)

        if conf.manipulate_znormalize:
            print("Loading latent stats...")
            state = torch.load(conf.latent_infer_path)
            self.conds = state["conds"]
            self.register_buffer("conds_mean", state["conds_mean"][None, :])
            self.register_buffer("conds_std", state["conds_std"][None, :])
        else:
            self.conds_mean = None
            self.conds_std = None

        self.diffusion_time_dependent = getattr(conf, "diffusion_time_dependent_classifier", False)
        if self.diffusion_time_dependent:
            self._setup_diffusion_params(conf)

        if conf.manipulate_mode == ManipulateMode.celebahq_all:
            num_cls = len(CelebAttrDataset.id_to_cls)
        elif conf.manipulate_mode.is_single_class():
            num_cls = 1
        else:
            raise NotImplementedError()

        input_dim = conf.style_ch
        self.classifier = build_classifier(conf, input_dim, num_cls)
        self.ema_classifier = copy.deepcopy(self.classifier)
        print("Classifier configuration:", get_classifier_info(conf))
        
        # Store the EMA decay parameter from the config
        self.ema_decay = getattr(conf, 'ema_decay', 0.9999)

    def _setup_diffusion_params(self, conf):
        self.time_embedding_dim = getattr(conf, 'time_embedding_dim', 64)
        self.autoenc_config = conf.autoenc_config
        latent_diffusion = self.autoenc_config.make_latent_eval_diffusion_conf().make_sampler()
        self.alpha_vals = torch.tensor(latent_diffusion.sqrt_alphas_cumprod, dtype=torch.float32)
        self.sigma_vals = torch.tensor(np.sqrt(1.0 - latent_diffusion.alphas_cumprod), dtype=torch.float32)
        self.max_diffusion_time = compute_discrete_time_from_target_snr(self.autoenc_config, conf.lower_trainable_snr) if conf.lower_trainable_snr is not None else 1000

    def setup(self, stage=None):
        latent_dir = os.path.join("datasets", f"celeba_{self.conf.img_size}_latents")
        os.makedirs(latent_dir, exist_ok=True)

        paths = {
            'train': os.path.join(latent_dir, 'train.pt'),
            'val': os.path.join(latent_dir, 'val.pt'),
            'test': os.path.join(latent_dir, 'test.pt'),
        }

        if all(os.path.exists(p) for p in paths.values()):
            print("[ClsModel] Loading precomputed latent datasets...")
            self.train_data = torch.load(paths['train'])
            self.val_data   = torch.load(paths['val'])
            self.test_data  = torch.load(paths['test'])
        else:
            print("[ClsModel] Precomputed latents not found. Generating...")
            full_dataset = CelebHQAttrDataset(data_paths['celebahq'], self.conf.img_size, data_paths['celebahq_anno'])
            loader = DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=4)

            self.ema_model.eval()
            self.ema_model.to(self.device)
            latents, labels = [], []
            with torch.no_grad():
                for batch in tqdm(loader, desc="Encoding CelebA-HQ"):
                    imgs = batch['img'].to(self.device)
                    z = self.ema_model.encoder(imgs).cpu()
                    y = batch['labels']
                    if self.conf.manipulate_znormalize:
                        z = (z - self.conds_mean.cpu()) / self.conds_std.cpu()
                    latents.append(z)
                    labels.append(y)

            latents = torch.cat(latents).cpu()
            labels = torch.cat(labels).cpu()
            dataset = LatentDataset(latents, labels)

            n = len(dataset)
            n_train = int(n * 0.94)
            n_val   = int(n * 0.05)
            n_test  = n - n_train - n_val
            train_data, val_data, test_data = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))

            torch.save(train_data, paths['train'])
            torch.save(val_data, paths['val'])
            torch.save(test_data, paths['test'])

            self.train_data = train_data
            self.val_data   = val_data
            self.test_data  = test_data

    def train_dataloader(self):
        return self.conf.make_loader(self.train_data, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self.conf.make_loader(self.val_data, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self.conf.make_loader(self.test_data, shuffle=False, drop_last=False)

    def training_step(self, batch, batch_idx):
        cond = batch["cond"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.diffusion_time_dependent:
            t_rand = torch.randint(0, self.max_diffusion_time + 1, (cond.size(0),), device=cond.device)
            alpha_vals = self.alpha_vals.to(cond.device).to(cond.dtype)
            sigma_vals = self.sigma_vals.to(cond.device).to(cond.dtype)
            alpha_t = alpha_vals[t_rand.long()].view(t_rand.size(0), 1)
            sigma_t = sigma_vals[t_rand.long()].view(t_rand.size(0), 1)
            noise = torch.randn_like(cond)
            cond_perturbed = alpha_t * cond + sigma_t * noise
            pred = self.classifier(cond_perturbed, t=t_rand)
            pred_ema = self.ema_classifier(cond_perturbed, t=t_rand)
        else:
            pred = self.classifier(cond)
            pred_ema = self.ema_classifier(cond)

        gt = torch.where(labels > 0, torch.ones_like(labels).float(), torch.zeros_like(labels).float())
        loss = F.binary_cross_entropy_with_logits(pred, gt)
        loss_ema = F.binary_cross_entropy_with_logits(pred_ema, gt)

        self.log("train_loss", loss)
        self.log("train_loss_ema", loss_ema)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Apply EMA update to the classifier weights
        ema(self.classifier, self.ema_classifier, self.ema_decay)

    def validation_step(self, batch, batch_idx):
        cond = batch["cond"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.diffusion_time_dependent:
            # Use random timesteps during validation as well, matching training behavior
            t_rand = torch.randint(0, self.max_diffusion_time + 1, (cond.size(0),), device=cond.device)
            alpha_vals = self.alpha_vals.to(cond.device).to(cond.dtype)
            sigma_vals = self.sigma_vals.to(cond.device).to(cond.dtype)
            alpha_t = alpha_vals[t_rand.long()].view(t_rand.size(0), 1)
            sigma_t = sigma_vals[t_rand.long()].view(t_rand.size(0), 1)
            noise = torch.randn_like(cond)
            cond_perturbed = alpha_t * cond + sigma_t * noise
            pred = self.classifier(cond_perturbed, t=t_rand)
            pred_ema = self.ema_classifier(cond_perturbed, t=t_rand)
        else:
            pred = self.classifier(cond)
            pred_ema = self.ema_classifier(cond)

        gt = torch.where(labels > 0, torch.ones_like(labels).float(), torch.zeros_like(labels).float())
        loss = F.binary_cross_entropy_with_logits(pred, gt)
        loss_ema = F.binary_cross_entropy_with_logits(pred_ema, gt)

        self.log('val_loss', loss, prog_bar=False, on_epoch=True, batch_size=cond.size(0))
        self.log('val_loss_ema', loss_ema, prog_bar=True, on_epoch=True, batch_size=cond.size(0))
        return {"val_loss": loss, "val_loss_ema": loss_ema}

    def configure_optimizers(self):
        if self.conf.optimizer == OptimizerType.adam:
            optimizer_class = torch.optim.Adam
        elif self.conf.optimizer == OptimizerType.adamw:
            optimizer_class = torch.optim.AdamW
        else:
            raise NotImplementedError(f"Optimizer {self.conf.optimizer} not supported.")

        return optimizer_class(
            self.classifier.parameters(),
            lr=self.conf.lr,
            weight_decay=self.conf.weight_decay
        )
    
    def normalize(self, cond):
        return (cond - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)

    def denormalize(self, cond):
        return cond * self.conds_std.to(self.device) + self.conds_mean.to(self.device)

    def forward(self, x, t=None):
        return self.classifier(x, t=t)