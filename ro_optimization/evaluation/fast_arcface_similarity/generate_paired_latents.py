#!/usr/bin/env python3
"""
Generate paired (custom‑latent, ArcFace‑latent) embeddings that can be used to
train an alignment network.  Initially the script is configured to work with
CelebA‑HQ, but it is written so that additional datasets (e.g. FFHQ) can be
added by simply passing extra dataset names on the command line.

The output directory will contain five files:

    custom_latents.pt   – concatenated tensor of all custom latents    (N, C)
    arcface_latents.pt  – concatenated tensor of all ArcFace latents   (N, 512)
    train.pt / val.pt / test.pt – PyTorch `Subset` objects with paired tensors

where N = total number of images over all chosen datasets.

Example usage
-------------
$ python generate_paired_latents.py \
        --output_dir datasets/celebahq128_paired_latents \
        --datasets celebahq \
        --batch_size 256 --gpu 0

To concatenate CelebA‑HQ and FFHQ 256×256:
$ python generate_paired_latents.py \
        --output_dir datasets/celebahq_ffhq256_paired_latents \
        --datasets celebahq ffhqlmdb256 \
        --cfg ffhq256_autoenc_latent
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
# Project‑level imports (assumes repository root is on PYTHONPATH)
# -----------------------------------------------------------------------------

from templates import ffhq128_autoenc_latent, ffhq256_autoenc_latent  # add more if needed
from dataset import data_paths  # global dict in dataset.py
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
# Helper utilities
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


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Autoencoder loader – returns only the encoder module (frozen, eval‑mode)
# -----------------------------------------------------------------------------

def load_encoder(conf, device: torch.device):
    """Instantiate the autoencoder, load weights, return the encoder."""
    print("[INFO] Loading autoencoder …")
    autoenc = conf.make_model_conf().make_model()

    # Locate checkpoint (either conf.pretrain or conf.continue_from)
    ckpt_cfg = conf.pretrain or conf.continue_from
    if ckpt_cfg is None:
        raise RuntimeError("The chosen config has no associated checkpoint; set conf.pretrain or conf.continue_from.")

    ckpt_path = ckpt_cfg.path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    autoenc.load_state_dict(state["state_dict"], strict=False)
    autoenc.eval().requires_grad_(False).to(device)

    # Latent normalisation statistics (optional)
    if conf.latent_znormalize and conf.latent_infer_path and os.path.exists(conf.latent_infer_path):
        stats = torch.load(conf.latent_infer_path, map_location="cpu")
        mean = stats["conds_mean"].to(device)  # (C,)
        std = stats["conds_std"].to(device)
        print("[INFO] Loaded latent z‑normalisation statistics.")
    else:
        mean, std = None, None
        if conf.latent_znormalize:
            print("[WARN] latent_znormalize=True but latent_infer_path missing; proceeding without normalisation.")

    return autoenc.encoder, mean, std


# -----------------------------------------------------------------------------
# Dataset factory – returns list of instantiated datasets (one per name)
# -----------------------------------------------------------------------------

def build_datasets(names: List[str], img_size: int):
    datasets = []
    for name in names:
        if name not in DATASET_DISPATCH:
            raise ValueError(f"Unknown dataset name '{name}'. Known: {list(DATASET_DISPATCH)}")
        cls = DATASET_DISPATCH[name]
        if name == "celebahq":
            ds = cls(image_size=img_size)  # CelebHQAttrDataset requires image_size
        elif name == "ffhqlmdb256":
            ds = cls(image_size=img_size)
        else:  # general fallback
            ds = cls(image_size=img_size)
        datasets.append(ds)
    return datasets


# -----------------------------------------------------------------------------
# ArcFace embedding helper (expects images in 0…1 range)
# -----------------------------------------------------------------------------

def extract_arcface_embeddings(img_batch, models):
    """Compute ArcFace embeddings for a batch of images."""
    embs = []
    B = img_batch.size(0)
    for i in range(B):
        img = img_batch[i]
        try:
            emb = get_embedding_faceanalysis(models["app"], img)
        except RuntimeError:
            # Fallback (no face detected by FaceAnalysis)
            emb = get_embedding_arcface(models["zoo_model"], img)
        embs.append(torch.from_numpy(emb))
    return torch.stack(embs)


# -----------------------------------------------------------------------------
# Main processing routine
# -----------------------------------------------------------------------------

def generate_pairs(cfg_name: str, dataset_names: List[str], output_dir: str, batch_size: int, gpu: int, arcface_method: str):
    set_seed(0)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # -------------------- 1. build config & load encoder -------------------- #
    conf_fn = AVAILABLE_CFGS[cfg_name]
    conf = conf_fn()
    encoder, z_mean, z_std = load_encoder(conf, device)

    # -------------------- 2. datasets -------------------------------------- #
    datasets = build_datasets(dataset_names, conf.img_size)
    total_samples = sum(len(ds) for ds in datasets)
    print(f"[INFO] Total images: {total_samples} across {dataset_names}")

    # -------------------- 3. ArcFace models -------------------------------- #
    arc_models = init_face_models(method=arcface_method)

    # -------------------- 4. iterate & encode ------------------------------ #
    all_custom = []
    all_arc = []

    for ds in datasets:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        for batch in tqdm(loader, desc=f"Encoding {ds.__class__.__name__}"):
            imgs = batch["img"].to(device)               # [-1,1]
            imgs_01 = (imgs + 1) / 2                     # [0,1] for ArcFace

            # ---- custom latent ----
            with torch.no_grad():
                z = encoder(imgs)
                if z_mean is not None and z_std is not None:
                    z = (z - z_mean) / z_std
            all_custom.append(z.cpu())

            # ---- arcface latent ----
            embs = extract_arcface_embeddings(imgs_01.cpu(), arc_models)
            all_arc.append(embs)

    custom_latents = torch.cat(all_custom)
    arcface_latents = torch.cat(all_arc)

    assert custom_latents.shape[0] == arcface_latents.shape[0]
    N = custom_latents.shape[0]
    print(f"[INFO] Finished. Paired samples: {N}")

    # -------------------- 5. save raw tensors ------------------------------ #
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(custom_latents, os.path.join(output_dir, "custom_latents.pt"))
    torch.save(arcface_latents, os.path.join(output_dir, "arcface_latents.pt"))

    # -------------------- 6. train/val/test splits ------------------------- #
    paired_ds = TensorDataset(custom_latents, arcface_latents)

    n_train = int(0.94 * N)
    n_val   = int(0.05 * N)
    n_test  = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(paired_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))

    torch.save(train_ds, os.path.join(output_dir, "train.pt"))
    torch.save(val_ds,   os.path.join(output_dir, "val.pt"))
    torch.save(test_ds,  os.path.join(output_dir, "test.pt"))

    print("[INFO] Saved:")
    for split, ds in zip(["train","val","test"], [train_ds, val_ds, test_ds]):
        print(f"    {split}: {len(ds)} samples → {output_dir}/{split}.pt")


# -----------------------------------------------------------------------------
# Command‑line interface
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate paired custom and ArcFace latents")
    parser.add_argument("--cfg", default="ffhq128_autoenc_latent", choices=list(AVAILABLE_CFGS), help="Autoencoder config to use")
    parser.add_argument("--datasets", nargs="+", default=["celebahq"], help="One or more dataset names (see DATASET_DISPATCH)")
    parser.add_argument("--output_dir", default="datasets/paired_latents")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arcface_method", choices=["arcface","facenet"], default="arcface")
    args = parser.parse_args()

    generate_pairs(
        cfg_name=args.cfg,
        dataset_names=args.datasets,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gpu=args.gpu,
        arcface_method=args.arcface_method,
    )


if __name__ == "__main__":
    main()
