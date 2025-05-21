import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from templates_latent import ffhq128_autoenc_latent
from templates_cls import (
    ffhq128_autoenc_flexibleclassifier_time_cls,
    ffhq128_autoenc_flexibleclassifier_time_cls_tuned,
    ffhq128_autoenc_cls,
)
from experiment import LitModel
from experiment_classifier import ClsModel
from experiment_classifier_new import ClsModel as RefactoredClsModel
from dataset import CelebAttrDataset
from ..config_loader import load_riemannian_config
from ..utils import flatten_tensor

def load_shared_resources(config_path, device=None):
    """
    Load models, dataset, and configuration.

    Returns:
      ae          – autoencoder model
      cls_nl      – nonlinear classifier (Riemannian)
      cls_lin     – linear classifier (baseline)
      dataset     – full manipulation dataset
      pos_dataset – list of positive examples (labels == 1)
      neg_indices – list of indices of negative examples (labels == -1)
      cfg         – configuration dict
      cid         – target attribute index
      device      – torch device
    """
    # Determine device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load config and target attribute
    cfg  = load_riemannian_config(config_path)
    attr = cfg["target_attr"]

    # --- Autoencoder ---
    ae_conf = ffhq128_autoenc_latent()
    ae_conf.T_eval = 1000
    ae_conf.latent_T_eval = 1000
    ae = LitModel(ae_conf).to(device)
    ckpt = torch.load(
        os.path.join("checkpoints", ae_conf.name, "last.ckpt"),
        map_location="cpu"
    )
    ae.load_state_dict(ckpt["state_dict"], strict=False)
    ae.ema_model.eval().to(device)

    # --- Nonlinear classifier (Riemannian) ---
    cls_nl_conf = ffhq128_autoenc_flexibleclassifier_time_cls() #ffhq128_autoenc_non_linear_cls()
    cls_nl = ClsModel(cls_nl_conf).to(device)
    checkpoint_path = getattr(cls_nl_conf, 'checkpoint_path', None) or os.path.join("checkpoints", cls_nl_conf.name, "best.ckpt")
    ckpt_nl = torch.load(checkpoint_path, map_location="cpu")
    cls_nl.load_state_dict(ckpt_nl["state_dict"], strict=False)
    cls_nl.eval()

    # --- Linear classifier (Baseline) ---
    cls_lin_conf = ffhq128_autoenc_cls()
    cls_lin = ClsModel(cls_lin_conf).to(device)
    ckpt_lin = torch.load(
        os.path.join("checkpoints", cls_lin_conf.name, "last.ckpt"),
        map_location="cpu"
    )
    cls_lin.load_state_dict(ckpt_lin["state_dict"], strict=False)
    cls_lin.eval()

    # --- Dataset and positive/negative split (single pass) ---
    dataset     = cls_lin.load_dataset()
    cid         = CelebAttrDataset.cls_to_id[attr]
    pos_dataset = []
    neg_indices = []
    for idx, item in enumerate(dataset):
        label = item['labels'][cid]
        if label == 1:
            pos_dataset.append(item)
        elif label == -1:
            neg_indices.append(idx)

    return ae, cls_nl, cls_lin, dataset, pos_dataset, neg_indices, cfg, cid, device

def compute_median_logit(ae, classifier, cls_id, pos_dataset, cfg, device):
    """
    Compute the median logit score for positive examples.
    """
    # How many positives to use and batch size
    num_samples = cfg.get("median_samples", 1000)
    batch_size  = cfg.get("median_batch_size", 16)

    # Subset of positives
    items = pos_dataset[:num_samples] if num_samples and num_samples < len(pos_dataset) else pos_dataset
    logits = []

    # Process in mini-batches
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        imgs = torch.stack([it['img'] for it in batch_items]).to(device)

        with torch.no_grad():
            conds     = ae.encode(imgs)
            flat_z    = flatten_tensor(conds)
            norm_z    = classifier.normalize(flat_z)
            batch_log = classifier.classifier(norm_z)[:, cls_id]
            logits.extend(batch_log.cpu().tolist())

    if not logits:
        raise ValueError("No positive samples for median logit.")

    return float(np.median(logits))

def save_side_by_side_comparison(original, linear, riemannian_list, save_path, total_losses=None):
    B = original.size(0)
    num_seeds = len(riemannian_list)
    fig, axes = plt.subplots(B, 2 + num_seeds,
                             figsize=(3 * (2 + num_seeds), 3 * B))

    if B == 1:
        axes = axes.reshape(1, -1)

    titles = ["Original", "Linear"] + [f"Riemannian {i+1}" for i in range(num_seeds)]

    # find best seed per row
    best_idx = None
    if total_losses is not None:
        tl = total_losses.detach().cpu()
        if tl.shape == (num_seeds, B):
            tl = tl.transpose(0, 1)
        assert tl.shape == (B, num_seeds)
        best_idx = torch.argmin(tl, dim=1)

    for i in range(B):
        for j in range(2 + num_seeds):
            ax = axes[i, j]

            # select image
            if j == 0:
                img = original[i]
            elif j == 1:
                img = linear[i]
            else:
                img = riemannian_list[j - 2][i]

            # convert to HWC numpy float in [0,1]
            if isinstance(img, np.ndarray):
                im = img
                # if it’s uint8 [0..255], normalize
                if im.dtype == np.uint8:
                    im = im.astype(np.float32) / 255.0
            else:
                # torch Tensor CHW float [0,1]
                im = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])

            # hide spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            # set title on first row
            if i == 0:
                ax.set_title(titles[j])

            # highlight best
            if best_idx is not None and j >= 2 and j == best_idx[i].item() + 2:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('red')
                    spine.set_linewidth(6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
