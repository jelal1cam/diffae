#!/usr/bin/env python
import os
import argparse

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from ..config_loader import load_riemannian_config

# FaceNet import
from facenet_pytorch import InceptionResnetV1

# InsightFace imports
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model  # for fallback ONNX

def preprocess_for_faceanalysis(img: torch.Tensor) -> np.ndarray:
    img_u8 = (
        img.clamp(0, 1)
           .mul(255)
           .byte()
           .permute(1, 2, 0)
           .cpu()
           .numpy()
    )
    return img_u8[..., ::-1]  # RGB→BGR

def get_embedding_faceanalysis(app: FaceAnalysis, img: torch.Tensor) -> np.ndarray:
    bgr = preprocess_for_faceanalysis(img)
    faces = app.get(bgr)
    if not faces:
        raise RuntimeError("No face detected")
    return faces[0].embedding  # (512,)

def preprocess_for_arcface(img: torch.Tensor) -> np.ndarray:
    img_small = F.interpolate(
        img.unsqueeze(0),
        size=(112, 112),
        mode='bilinear',
        align_corners=False
    )
    img_small = img_small.clamp(0,1).mul(255).byte()
    img_np = img_small[0].permute(1,2,0).cpu().numpy()
    return img_np[..., ::-1]  # RGB→BGR

def get_embedding_arcface(model, img: torch.Tensor) -> np.ndarray:
    bgr = preprocess_for_arcface(img)
    blob = cv2.dnn.blobFromImage(
        bgr,
        scalefactor=1.0/127.5,
        size=(112,112),
        mean=(127.5,127.5,127.5),
        swapRB=False,
        crop=False
    )
    sess = model.session
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: blob})
    return out[0].reshape(-1)

def get_embedding_facenet(model, img: torch.Tensor, device: torch.device) -> np.ndarray:
    img_resized = F.interpolate(
        img.unsqueeze(0),
        size=(160,160),
        mode='bilinear', align_corners=False
    )
    img_norm = img_resized.clamp(0,1).mul(2).sub(1)
    x = img_norm.to(device)
    with torch.no_grad():
        emb = model(x)
    return emb.cpu().numpy().reshape(-1)

def save_grids(orig: torch.Tensor, fin: torch.Tensor, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    save_image(orig.clamp(0,1), os.path.join(save_dir, "orig_grid.png"), nrow=orig.size(0))
    save_image(fin.clamp(0,1), os.path.join(save_dir, "fin_grid.png"), nrow=fin.size(0))
    print(f"[INFO] Grids saved to {save_dir}")

def save_pairs(orig: torch.Tensor, fin: torch.Tensor, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2,2,figsize=(6,6))
    for i in range(2):
        axes[i,0].imshow(orig[i].permute(1,2,0).cpu()); axes[i,0].axis('off')
        axes[i,1].imshow(fin[i].permute(1,2,0).cpu()); axes[i,1].axis('off')
    plt.tight_layout()
    path = os.path.join(save_dir, "pairs.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Pairs saved to {path}")

def evaluate(image_dir: str,
             method: str,
             insightface_model_name: str,
             device: torch.device):
    # load tensors
    orig_path = os.path.join(image_dir, "original_imgs.pt")
    fin_path  = os.path.join(image_dir, "final_imgs.pt")
    if not (os.path.exists(orig_path) and os.path.exists(fin_path)):
        raise FileNotFoundError(f"Missing originals or finals in {image_dir}")
    orig = torch.load(orig_path)
    fin  = torch.load(fin_path)
    assert orig.shape == fin.shape, "Shape mismatch"

    print(f"[INFO] Loaded {orig.size(0)} pairs")
    save_grids(orig, fin, image_dir)
    save_pairs(orig, fin, image_dir)

    if method == "arcface":
        # 1) FaceAnalysis pipeline
        app = FaceAnalysis(
            name=insightface_model_name,
            providers=['CUDAExecutionProvider','CPUExecutionProvider']
        )
        app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
        # 2) Fallback ONNX model
        zoo_model = get_model(insightface_model_name)
        zoo_model.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    else:
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    sims = []
    for i in range(orig.size(0)):
        try:
            if method == "arcface":
                try:
                    e0 = get_embedding_faceanalysis(app, orig[i])
                    e1 = get_embedding_faceanalysis(app, fin[i])
                except RuntimeError:
                    # fallback if no face detected/aligned
                    e0 = get_embedding_arcface(zoo_model, orig[i])
                    e1 = get_embedding_arcface(zoo_model, fin[i])
            else:
                e0 = get_embedding_facenet(facenet, orig[i], device)
                e1 = get_embedding_facenet(facenet, fin[i], device)

            e0 /= np.linalg.norm(e0)
            e1 /= np.linalg.norm(e1)
            sim = cosine_similarity([e0], [e1])[0,0]

        except Exception as e:
            print(f"[WARN] img {i} failed: {e}")
            sim = np.nan

        sims.append(sim)
        print(f"[SIM] img {i}: cosine similarity = {sim:.4f}")

    sims = np.array(sims, dtype=np.float32)
    valid = sims[~np.isnan(sims)]
    if valid.size:
        print(f"[RESULT] Avg similarity: {valid.mean():.4f} over {valid.size}/{len(sims)}")
    else:
        print("[RESULT] No valid sims")
    out_np = os.path.join(image_dir, f"{method}_similarities.npy")
    np.save(out_np, sims)
    print(f"[INFO] Saved similarities to {out_np}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare faces with ArcFace or FaceNet")
    parser.add_argument("--ro-config", required=True)
    parser.add_argument("--method", choices=["arcface","facenet"], default="arcface")
    parser.add_argument("--insightface-model", default="buffalo_l",
                        help="e.g. buffalo_l, magface_64, antelopev2")
    args = parser.parse_args()

    cfg = load_riemannian_config(args.ro_config)
    img_dir = os.path.join(cfg.get("log_dir","logs"), cfg.get("target_attr",""))
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate(img_dir, args.method, args.insightface_model, device)
