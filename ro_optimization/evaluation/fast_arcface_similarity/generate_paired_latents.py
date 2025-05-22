#!/usr/bin/env python3
"""
Generate **paired and fully‑normalised** latent embeddings (custom‑encoder ↔ ArcFace).

Revision – keep it simple 💡
---------------------------------
We now reuse the project’s **`LitModel`** helper to load the auto‑encoder and
its EMA weights instead of manually stripping prefixes.  This matches exactly
what `load_shared_resources()` does in downstream manipulation code.

Key points
~~~~~~~~~~
* `LitModel(conf)` loads the checkpoint and exposes `ema_model.encoder` – the
  same encoder used at inference time.
* z‑normalisation stats (`conds_mean`, `conds_std`) are fetched directly from
  the instantiated model if available.
* All other behaviour (ArcFace L2‑norm, 90/5/5 split, CLI) is unchanged.
"""

import os
import argparse
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Repo‑local imports (root must be on PYTHONPATH)
# -----------------------------------------------------------------------------

from templates_latent import ffhq128_autoenc_latent, ffhq256_autoenc_latent  # extend if needed
from experiment import LitModel
from dataset import (
    CelebHQAttrDataset,
    CelebAlmdb,
    FFHQlmdb,
)
from ro_optimization.evaluation.arcface_similarity import (
    init_face_models,
    get_embedding_faceanalysis,
    get_embedding_arcface,
)

# -----------------------------------------------------------------------------
# Registries
# -----------------------------------------------------------------------------

AVAILABLE_CFGS = {
    "ffhq128_autoenc_latent": ffhq128_autoenc_latent,
    "ffhq256_autoenc_latent": ffhq256_autoenc_latent,
}

DATASET_DISPATCH = {
    "celebahq": CelebHQAttrDataset,
    "celebalmdb": CelebAlmdb,
    "ffhqlmdb256": FFHQlmdb,
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Auto‑encoder loader via LitModel (EMA)
# -----------------------------------------------------------------------------

def load_encoder_via_lit(conf, device: torch.device):
    """Instantiate `LitModel`, load checkpoint, and return EMA encoder + stats."""

    lit = LitModel(conf).to(device)

    ckpt_cfg = conf.pretrain or conf.continue_from or conf
    ckpt_path = os.path.join("checkpoints", conf.name, "last.ckpt") if ckpt_cfg is conf else ckpt_cfg.path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    lit.load_state_dict(state["state_dict"], strict=False)
    lit.ema_model.eval().to(device)

    # z‑norm stats (may be None)
    mean = getattr(lit, "conds_mean", None)
    std  = getattr(lit, "conds_std", None)
    if mean is not None:
        mean, std = mean.to(device), std.to(device)
        print("[INFO] Loaded z‑normalisation stats (μ, σ) from LitModel")
    else:
        print("[INFO] z‑normalisation stats not available – proceeding without.")

    return lit.ema_model.encoder, mean, std


# -----------------------------------------------------------------------------
# Dataset factory
# -----------------------------------------------------------------------------

def build_datasets(names: List[str], img_size: int):
    return [DATASET_DISPATCH[n](image_size=img_size) for n in names]


# -----------------------------------------------------------------------------
# ArcFace embedding helper – returns unit vectors
# -----------------------------------------------------------------------------

def arcface_emb_batch(img_batch: torch.Tensor, models) -> torch.Tensor:
    embs = []
    for img in img_batch:  # InsightFace works per‑image
        try:
            e = get_embedding_faceanalysis(models["app"], img)
        except RuntimeError:
            e = get_embedding_arcface(models["zoo_model"], img)
        e = torch.from_numpy(e).float()
        embs.append(e / e.norm(p=2))
    return torch.stack(embs)


# -----------------------------------------------------------------------------
# Core routine
# -----------------------------------------------------------------------------

def generate_pairs(
    cfg_name: str,
    dataset_names: List[str],
    output_dir: str,
    batch_size: int,
    gpu: int,
    arcface_model: str,
):
    set_seed(0)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # 1) Config & EMA encoder via LitModel
    conf = AVAILABLE_CFGS[cfg_name]()
    encoder, z_mu, z_sigma = load_encoder_via_lit(conf, device)

    # 2) Datasets
    datasets = build_datasets(dataset_names, conf.img_size)
    print(f"[INFO] Total images: {sum(len(d) for d in datasets)} across {dataset_names}")

    # 3) ArcFace models
    arc_models = init_face_models(method="arcface", model_name=arcface_model)

    # 4) Iterate
    cust, arc = [], []
    for ds in datasets:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        for batch in tqdm(loader, desc=f"{ds.__class__.__name__}"):
            imgs = batch["img"].to(device)          # [-1,1]
            imgs01 = (imgs + 1) / 2                 # [0,1]

            with torch.no_grad():
                z = encoder(imgs)
                if z_mu is not None and z_sigma is not None:
                    z = (z - z_mu) / z_sigma
            cust.append(z.cpu())
            arc.append(arcface_emb_batch(imgs01.cpu(), arc_models))

    custom_latents  = torch.cat(cust)
    arcface_latents = torch.cat(arc)
    assert custom_latents.size(0) == arcface_latents.size(0)
    N = custom_latents.size(0)
    print(f"[INFO] Pairs ready: {N}")

    # 5) Save tensors
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(custom_latents,  f"{output_dir}/custom_latents.pt")
    torch.save(arcface_latents, f"{output_dir}/arcface_latents.pt")

    # 6) Train/val/test split
    split_sizes = [int(0.90*N), int(0.05*N), N - int(0.90*N) - int(0.05*N)]
    train_ds, val_ds, test_ds = random_split(
        TensorDataset(custom_latents, arcface_latents),
        split_sizes,
        generator=torch.Generator().manual_seed(0),
    )
    torch.save(train_ds, f"{output_dir}/train.pt")
    torch.save(val_ds,   f"{output_dir}/val.pt")
    torch.save(test_ds,  f"{output_dir}/test.pt")

    for name, ds in zip(["train","val","test"], [train_ds, val_ds, test_ds]):
        print(f"[INFO] {name:>5}: {len(ds):>6}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Generate paired (custom, ArcFace) latents")
    p.add_argument("--cfg", choices=AVAILABLE_CFGS, default="ffhq128_autoenc_latent")
    p.add_argument("--datasets", nargs="+", default=["celebahq"])
    p.add_argument("--output_dir", default="datasets/paired_ArcFace_latents")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--arcface_model", default="buffalo_l")
    args = p.parse_args()

    generate_pairs(
        cfg_name=args.cfg,
        dataset_names=args.datasets,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gpu=args.gpu,
        arcface_model=args.arcface_model,
    )


if __name__ == "__main__":
    main()
