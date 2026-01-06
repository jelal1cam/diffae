#!/usr/bin/env python3
"""
Train an alignment network using contrastive learning to map custom latent space 
to a distance-preserving space.

The key insight: instead of trying to match ArcFace directly, we learn a 
distance-preserving transformation using contrastive learning on augmented pairs.
"""

import os
import argparse
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor, 
    EarlyStopping,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger

from torchvision import transforms
from torchvision.transforms import functional as TF
import random

from enum import Enum
from typing import Sequence, Tuple, Optional
import math

# Import your existing architecture components
from ro_optimization.evaluation.fast_arcface_similarity.align import (
    Activation, _make_linear, MLPSkipAlignNet, L2NormLayer
)

# Import your datasets
from dataset import CelebHQAttrDataset, CelebAlmdb, FFHQlmdb
from experiment import LitModel

# -----------------------------------------------------------------------------
# Face-preserving augmentations for contrastive learning
# -----------------------------------------------------------------------------

class FaceIdentityAugmentation:
    """
    Augmentations that preserve face identity while creating variation.
    Based on SimCLR and face recognition literature.
    """
    def __init__(self, img_size: int = 128, strength: float = 1.0):
        self.img_size = img_size
        self.strength = strength
        
        # Define individual transforms
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4 * strength,
            contrast=0.4 * strength,
            saturation=0.2 * strength,  # Lower saturation change
            hue=0.1 * strength  # Minimal hue change for faces
        )
        
        # Minimal rotation to preserve face structure
        self.max_rotation = 10 * strength  # degrees
        
        # Small translation
        self.max_translate = int(0.1 * img_size * strength)
        
        # Gaussian blur
        kernel_size = int(0.1 * img_size) // 2 * 2 + 1  # Ensure odd
        self.gaussian_blur = transforms.GaussianBlur(
            kernel_size=(kernel_size, kernel_size),
            sigma=(0.1, 2.0)
        )
        
    def __call__(self, img):
        """Apply random augmentations to image."""
        # Start with original image
        aug_img = img
        
        # Apply color jitter (80% chance)
        if random.random() < 0.8:
            aug_img = self.color_jitter(aug_img)
        
        # Apply slight rotation (50% chance)
        if random.random() < 0.5:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            aug_img = TF.rotate(aug_img, angle, fill=0)
        
        # Apply slight translation (50% chance)
        if random.random() < 0.5:
            tx = random.randint(-self.max_translate, self.max_translate)
            ty = random.randint(-self.max_translate, self.max_translate)
            aug_img = TF.affine(aug_img, angle=0, translate=(tx, ty), 
                               scale=1.0, shear=0, fill=0)
        
        # Apply Gaussian blur (50% chance)
        if random.random() < 0.5:
            aug_img = self.gaussian_blur(aug_img)
        
        # Horizontal flip (50% chance) - faces are roughly symmetric
        if random.random() < 0.5:
            aug_img = TF.hflip(aug_img)
            
        # Random grayscale (20% chance)
        if random.random() < 0.2:
            aug_img = TF.rgb_to_grayscale(aug_img, num_output_channels=3)
        
        return aug_img


# -----------------------------------------------------------------------------
# Contrastive Dataset
# -----------------------------------------------------------------------------

class ContrastiveLatentDataset(Dataset):
    """
    Dataset that provides augmented pairs for contrastive learning.
    Applies augmentations in image space, then encodes to latent space.
    """
    def __init__(
        self,
        base_dataset,
        encoder: nn.Module,
        z_mean: Optional[torch.Tensor] = None,
        z_std: Optional[torch.Tensor] = None,
        augmentation_strength: float = 1.0,
        device: str = 'cuda'
    ):
        self.base_dataset = base_dataset
        self.encoder = encoder
        self.z_mean = z_mean
        self.z_std = z_std
        self.device = device
        
        # Initialize augmentation
        self.augment = FaceIdentityAugmentation(
            img_size=base_dataset.image_size,
            strength=augmentation_strength
        )
        
        # Move encoder to device and set to eval
        self.encoder.to(device).eval()
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original image
        sample = self.base_dataset[idx]
        img = sample['img']  # Should be in [-1, 1]
        
        # Convert to [0, 1] for augmentation
        img_01 = (img + 1) / 2
        
        # Create augmented version
        aug_img_01 = self.augment(img_01)
        
        # Convert back to [-1, 1]
        aug_img = aug_img_01 * 2 - 1
        
        # Encode both images
        with torch.no_grad():
            # Move to device for encoding
            img_tensor = img.unsqueeze(0).to(self.device)
            aug_img_tensor = aug_img.unsqueeze(0).to(self.device)
            
            # Encode and flatten to 1D
            z_orig = self.encoder(img_tensor).flatten()
            z_aug = self.encoder(aug_img_tensor).flatten()
            
            # Apply z-normalization if available
            if self.z_mean is not None and self.z_std is not None:
                z_orig = (z_orig - self.z_mean) / self.z_std
                z_aug = (z_aug - self.z_mean) / self.z_std
            
            # Move back to CPU for dataloader
            z_orig = z_orig.cpu()
            z_aug = z_aug.cpu()
        
        return {
            'z_orig': z_orig,
            'z_aug': z_aug,
            'idx': idx
        }


# -----------------------------------------------------------------------------
# Contrastive Loss
# -----------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning.
    Treats augmented pairs as positives, all others as negatives.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: [N, D] normalized embeddings from original images
            z2: [N, D] normalized embeddings from augmented images
        
        Returns:
            InfoNCE loss
        """
        N = z1.shape[0]
        
        # Concatenate for efficient computation
        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]
        
        # Create positive pair mask
        pos_mask = torch.zeros((2*N, 2*N), dtype=torch.bool, device=z1.device)
        pos_mask[:N, N:] = torch.eye(N, dtype=torch.bool)
        pos_mask[N:, :N] = torch.eye(N, dtype=torch.bool)
        
        # Mask out self-similarities
        self_mask = torch.eye(2*N, dtype=torch.bool, device=z1.device)
        sim.masked_fill_(self_mask, -float('inf'))
        
        # For each sample, compute log prob of positive pair
        pos_sim = sim[pos_mask].view(2*N, 1)
        
        # Log-sum-exp for numerical stability
        neg_sim = sim.masked_fill(pos_mask, -float('inf'))
        
        # Compute loss
        loss = -pos_sim + torch.logsumexp(sim, dim=1, keepdim=True)
        
        return loss.mean()


# -----------------------------------------------------------------------------
# Lightning Module for Contrastive Learning
# -----------------------------------------------------------------------------

class ContrastiveAlignmentModel(pl.LightningModule):
    """Lightning module for training alignment network with contrastive learning."""

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        super().__init__()

        # Handle loading from checkpoint (config=None, hparams in kwargs)
        if config is None:
            from types import SimpleNamespace
            config = SimpleNamespace(**kwargs)

        self.config = config

        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        elif hasattr(config, '__dict__'):
            config_dict = vars(config)
        else:
            config_dict = config

        self.save_hyperparameters(config_dict)
        
        # Build alignment network
        self.alignment_net = self._build_network()
        
        # EMA network
        self.ema_alignment_net = copy.deepcopy(self.alignment_net)
        self.ema_alignment_net.requires_grad_(False)
        self.ema_decay = config.ema_decay
        
        # Loss function
        self.criterion = InfoNCELoss(temperature=config.temperature)
        
        # For logging
        self.val_embeddings = []
        self.val_labels = []
        
    def _build_network(self):
        """Build the alignment network based on config."""
        cfg = self.config
        
        if cfg.architecture == "mlp":
            core = MLPSkipAlignNet(
                in_dim=cfg.input_dim,
                out_dim=cfg.output_dim,
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.num_layers,
                skip_layers=tuple(cfg.skip_layers),
                activation=cfg.activation,
                use_norm=cfg.use_norm,
                dropout=cfg.dropout,
                last_act=cfg.last_act,
            )
        else:
            raise ValueError(f"Unknown architecture: {cfg.architecture}")
        
        # Add L2 normalization layer
        return nn.Sequential(core, L2NormLayer())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alignment_net(x)
    
    def _ensure_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is 2D [N, D] for contrastive loss."""
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        elif tensor.dim() == 3:
            # [N, 1, D] -> [N, D]
            return tensor.squeeze(1)
        return tensor

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        z_orig = self._ensure_2d(batch['z_orig'])
        z_aug = self._ensure_2d(batch['z_aug'])

        # Forward pass through alignment network
        h_orig = self._ensure_2d(self.alignment_net(z_orig))
        h_aug = self._ensure_2d(self.alignment_net(z_aug))

        # Compute contrastive loss
        loss = self.criterion(h_orig, h_aug)
        
        # Logging
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Compute average similarity between positive pairs
        with torch.no_grad():
            pos_sim = F.cosine_similarity(h_orig, h_aug, dim=1).mean()
            self.log("train_pos_sim", pos_sim, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        z_orig = self._ensure_2d(batch['z_orig'])
        z_aug = self._ensure_2d(batch['z_aug'])

        # Forward pass with both networks
        h_orig = self._ensure_2d(self.alignment_net(z_orig))
        h_aug = self._ensure_2d(self.alignment_net(z_aug))

        h_orig_ema = self._ensure_2d(self.ema_alignment_net(z_orig))
        h_aug_ema = self._ensure_2d(self.ema_alignment_net(z_aug))
        
        # Compute losses
        loss = self.criterion(h_orig, h_aug)
        loss_ema = self.criterion(h_orig_ema, h_aug_ema)
        
        # Logging
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_loss_ema", loss_ema, prog_bar=True, on_epoch=True)
        
        # Compute positive pair similarities
        pos_sim = F.cosine_similarity(h_orig, h_aug, dim=1).mean()
        pos_sim_ema = F.cosine_similarity(h_orig_ema, h_aug_ema, dim=1).mean()
        
        self.log("val_pos_sim", pos_sim, prog_bar=True, on_epoch=True)
        self.log("val_pos_sim_ema", pos_sim_ema, prog_bar=True, on_epoch=True)
        
        # Store embeddings for epoch-level metrics
        self.val_embeddings.append(h_orig_ema.detach())
        self.val_labels.append(batch['idx'])
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute epoch-level validation metrics."""
        if len(self.val_embeddings) > 0:
            # Concatenate all embeddings
            embeddings = torch.cat(self.val_embeddings, dim=0)
            labels = torch.cat(self.val_labels, dim=0)
            
            # Compute pairwise similarities
            sim_matrix = torch.mm(embeddings, embeddings.t())
            
            # Compute average intra-class similarity (same image)
            # This is a proxy for how well the network preserves identity
            n = len(labels)
            intra_sims = []
            
            for i in range(n):
                for j in range(i+1, n):
                    if labels[i] == labels[j]:
                        intra_sims.append(sim_matrix[i, j])
            
            if intra_sims:
                avg_intra_sim = torch.stack(intra_sims).mean()
                self.log("val_intra_sim", avg_intra_sim, prog_bar=True)
            
            # Clear stored embeddings
            self.val_embeddings = []
            self.val_labels = []
    
    def on_after_backward(self):
        """Update EMA network."""
        with torch.no_grad():
            for param, ema_param in zip(
                self.alignment_net.parameters(),
                self.ema_alignment_net.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay
                )
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Optimizer
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.alignment_net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.alignment_net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "lars":
            # LARS optimizer (good for contrastive learning)
            from torch.optim import SGD
            base_optimizer = SGD(
                self.alignment_net.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
            optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Scheduler
        if self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.min_lr
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.config.scheduler == "cosine_warmup":
            total_steps = self.config.max_epochs * self.config.steps_per_epoch
            warmup_steps = self.config.warmup_epochs * self.config.steps_per_epoch
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return self.config.scheduler_min_lr + 0.5 * (1 - self.config.scheduler_min_lr) * (1 + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
        else:
            return optimizer


# -----------------------------------------------------------------------------
# LARS optimizer (for contrastive learning)
# -----------------------------------------------------------------------------

class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling optimizer.
    Based on https://arxiv.org/abs/1708.03888
    """
    def __init__(self, optimizer, eps=1e-8, trust_coef=0.001):
        self.optimizer = optimizer
        self.eps = eps
        self.trust_coef = trust_coef
        
    def step(self, closure=None):
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    p_norm = torch.norm(p.data)
                    g_norm = torch.norm(p.grad.data)
                    
                    if p_norm != 0 and g_norm != 0:
                        adaptive_lr = self.trust_coef * p_norm / (g_norm + self.eps)
                        p.grad.data *= adaptive_lr
        
        return self.optimizer.step(closure)
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


# -----------------------------------------------------------------------------
# Data Module
# -----------------------------------------------------------------------------

class ContrastiveDataModule(pl.LightningDataModule):
    """Data module for contrastive learning."""
    
    def __init__(self, config: Dict[str, Any], encoder: nn.Module, 
                 z_mean: Optional[torch.Tensor] = None,
                 z_std: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.z_mean = z_mean
        self.z_std = z_std
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        # Dataset dispatch
        dataset_cls = {
            "celebahq": CelebHQAttrDataset,
            "celebalmdb": CelebAlmdb,
            "ffhqlmdb": FFHQlmdb,
        }
        
        # Create base datasets
        base_datasets = []
        for ds_name in self.config.dataset_names:
            ds = dataset_cls[ds_name](image_size=self.config.img_size)
            base_datasets.append(ds)
        
        # Combine datasets
        if len(base_datasets) > 1:
            from torch.utils.data import ConcatDataset
            full_dataset = ConcatDataset(base_datasets)
        else:
            full_dataset = base_datasets[0]
        
        # Create contrastive dataset
        full_contrastive = ContrastiveLatentDataset(
            full_dataset,
            self.encoder,
            self.z_mean,
            self.z_std,
            augmentation_strength=self.config.augmentation_strength,
            device=self.config.device
        )
        
        # Split into train/val
        n_total = len(full_contrastive)
        n_train = int(0.95 * n_total)
        n_val = n_total - n_train
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_contrastive,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        # Calculate steps per epoch for scheduler
        self.config.steps_per_epoch = len(self.train_dataset) // self.config.batch_size
        
        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Val dataset: {len(self.val_dataset)} samples")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,  # Important for contrastive learning
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.config.num_workers > 0 else False
        )


# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------

def train_contrastive_alignment(config: Dict[str, Any]):
    """Main training function for contrastive alignment."""
    
    # Set seed
    if config.seed is not None:
        seed_everything(config.seed)
    
    # Create directories
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Load encoder
    from templates_latent import ffhq128_autoenc_latent, ffhq256_autoenc_latent
    
    template_dispatch = {
        "ffhq128": ffhq128_autoenc_latent,
        "ffhq256": ffhq256_autoenc_latent,
    }
    
    conf = template_dispatch[config.encoder_config]()
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    
    # Load encoder via LitModel
    from ro_optimization.evaluation.fast_arcface_similarity.generate_paired_latents import load_encoder_via_lit
    encoder, z_mean, z_std = load_encoder_via_lit(conf, device)
    
    # Initialize data module
    data_module = ContrastiveDataModule(config, encoder, z_mean, z_std)
    
    # Initialize model
    model = ContrastiveAlignmentModel(config)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename="contrastive-{epoch:03d}-{val_loss:.4f}",
            monitor=config.monitor_metric,
            mode="min",
            save_top_k=config.save_top_k,
            save_last=True
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.patience,
            mode="min"
        ),
        RichProgressBar()
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name="contrastive_alignment"
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        deterministic=True
    )
    
    # Train
    trainer.fit(model, data_module)
    
    print(f"Training completed. Best checkpoint saved at: {callbacks[0].best_model_path}")
    
    return model


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train contrastive alignment network")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Import and load config
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()
    
    # Train
    train_contrastive_alignment(config)


if __name__ == "__main__":
    main()