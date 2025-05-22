#!/usr/bin/env python
import os
import argparse
import warnings

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from ..config_loader import load_riemannian_config

# Suppress InsightFace warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# FaceNet import
from facenet_pytorch import InceptionResnetV1

# InsightFace imports
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model  # for fallback ONNX

# ---- Utility functions ----

def setup_logging():
    """Configure less verbose logging for InsightFace"""
    import logging
    logging.basicConfig(level=logging.ERROR)

def preprocess_for_faceanalysis(img: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor image to the format needed for face analysis."""
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
    """Extract face embedding using the InsightFace FaceAnalysis pipeline."""
    bgr = preprocess_for_faceanalysis(img)
    faces = app.get(bgr)
    if not faces:
        raise RuntimeError("No face detected")
    return faces[0].embedding  # (512,)

def preprocess_for_arcface(img: torch.Tensor) -> np.ndarray:
    """Preprocess an image for direct input to ArcFace model."""
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
    """Extract face embedding using ArcFace ONNX model directly."""
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
    """Extract face embedding using FaceNet model."""
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

# ---- Core similarity computation functions for reuse ----

def init_face_models(method="arcface", model_name="buffalo_l", device=None, verbose=False):
    """Initialize face embedding models for reuse."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Suppress stdout/stderr during model loading
    if not verbose:
        import contextlib
        import io
        stdout_redirect = io.StringIO()
        stderr_redirect = io.StringIO()
    else:
        stdout_redirect = None
        stderr_redirect = None
    
    ctx_mgr = contextlib.redirect_stdout(stdout_redirect) if stdout_redirect else contextlib.nullcontext()
    ctx_mgr2 = contextlib.redirect_stderr(stderr_redirect) if stderr_redirect else contextlib.nullcontext()
    
    with ctx_mgr, ctx_mgr2:
        if method == "arcface":
            # 1) FaceAnalysis pipeline
            app = FaceAnalysis(
                name=model_name,
                providers=['CUDAExecutionProvider','CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']  # Only load what we need
            )
            app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
            # 2) Fallback ONNX model
            zoo_model = get_model(model_name)
            zoo_model.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
            facenet = None
        else:  # facenet
            app = None
            zoo_model = None
            facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    return {
        "app": app,
        "zoo_model": zoo_model,
        "facenet": facenet,
        "device": device
    }

def compute_face_similarity(orig_img, edited_img, models, method="arcface"):
    """
    Compute face similarity between original and edited images.
    
    Args:
        orig_img: Original image tensor (C, H, W)
        edited_img: Edited image tensor (C, H, W)
        models: Dictionary of models from init_face_models
        method: 'arcface' or 'facenet'
        
    Returns:
        similarity: Cosine similarity between face embeddings
    """
    try:
        if method == "arcface":
            try:
                e0 = get_embedding_faceanalysis(models["app"], orig_img)
                e1 = get_embedding_faceanalysis(models["app"], edited_img)
            except RuntimeError:
                # fallback if no face detected/aligned
                e0 = get_embedding_arcface(models["zoo_model"], orig_img)
                e1 = get_embedding_arcface(models["zoo_model"], edited_img)
        else:  # facenet
            e0 = get_embedding_facenet(models["facenet"], orig_img, models["device"])
            e1 = get_embedding_facenet(models["facenet"], edited_img, models["device"])

        e0 /= np.linalg.norm(e0)
        e1 /= np.linalg.norm(e1)
        sim = cosine_similarity([e0], [e1])[0,0]
        return sim
    except Exception as e:
        return np.nan

def compute_batch_similarities(orig_imgs, edited_imgs, method="arcface", model_name="buffalo_l", device=None, verbose=False):
    """
    Compute face similarity for a batch of images.
    
    Args:
        orig_imgs: Tensor of original images (B, C, H, W)
        edited_imgs: Tensor of edited images (B, C, H, W)
        method: 'arcface' or 'facenet'
        model_name: Model name for InsightFace
        device: Computation device
        verbose: Whether to print per-image similarities
        
    Returns:
        similarities: Array of similarities (B,)
    """
    models = init_face_models(method, model_name, device, verbose)
    
    sims = []
    for i in range(orig_imgs.size(0)):
        sim = compute_face_similarity(orig_imgs[i], edited_imgs[i], models, method)
        sims.append(sim)
        if verbose:
            print(f"[SIM] img {i}: {sim:.4f}")
    
    sims = np.array(sims, dtype=np.float32)
    valid = sims[~np.isnan(sims)]
    if valid.size > 0 and verbose:
        print(f"[RESULT] Avg similarity: {valid.mean():.4f}±{valid.std():.4f} over {valid.size}/{len(sims)}")
    
    return sims

# ---- Standalone script functionality ----
    
def evaluate_manipulation_dir(image_dir, method="arcface", model_name="buffalo_l", device=None, verbose=False):
    """
    Evaluate all saved images in a manipulation output directory.
    
    Args:
        image_dir: Directory containing saved manipulation images
        method: 'arcface' or 'facenet'
        model_name: Model name for InsightFace
        device: Computation device
        verbose: Whether to print detailed progress
    """
    setup_logging()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check for original images
    orig_path = os.path.join(image_dir, "original_imgs.pt")
    if not os.path.exists(orig_path):
        raise FileNotFoundError(f"Original images not found at {orig_path}")
    
    # Load original images
    orig = torch.load(orig_path)
    print(f"[INFO] Loaded original images: {orig.shape}")
    
    # Initialize models once
    models = init_face_models(method, model_name, device, verbose)
    
    results = {}
    
    # Process linear edits if they exist
    lin_path = os.path.join(image_dir, "linear_imgs.pt")
    if os.path.exists(lin_path):
        lin_imgs = torch.load(lin_path)
        print(f"[INFO] Processing linear edits...")
        
        # Compute similarities
        lin_sims = []
        for i in range(orig.size(0)):
            sim = compute_face_similarity(orig[i], lin_imgs[i], models, method)
            lin_sims.append(sim)
            if verbose:
                print(f"[LINEAR SIM] img {i}: {sim:.4f}")
        
        lin_sims = np.array(lin_sims, dtype=np.float32)
        valid_lin = lin_sims[~np.isnan(lin_sims)]
        if valid_lin.size > 0:
            results["linear"] = {
                "mean": valid_lin.mean(),
                "std": valid_lin.std(),
                "valid": valid_lin.size,
                "total": len(lin_sims)
            }
        
        # Save similarities
        np.save(os.path.join(image_dir, f"{method}_linear_similarities.npy"), lin_sims)
    
    # Process Riemannian best edits if they exist
    riem_best_path = os.path.join(image_dir, "riemannian_best_imgs.pt")
    if os.path.exists(riem_best_path):
        riem_best = torch.load(riem_best_path)
        print(f"[INFO] Processing best Riemannian edits...")
        
        # Compute similarities
        riem_best_sims = []
        for i in range(orig.size(0)):
            sim = compute_face_similarity(orig[i], riem_best[i], models, method)
            riem_best_sims.append(sim)
            if verbose:
                print(f"[RIEM BEST SIM] img {i}: {sim:.4f}")
        
        riem_best_sims = np.array(riem_best_sims, dtype=np.float32)
        valid_riem = riem_best_sims[~np.isnan(riem_best_sims)]
        if valid_riem.size > 0:
            results["riemannian_best"] = {
                "mean": valid_riem.mean(),
                "std": valid_riem.std(),
                "valid": valid_riem.size,
                "total": len(riem_best_sims)
            }
        
        # Save similarities
        np.save(os.path.join(image_dir, f"{method}_riemannian_best_similarities.npy"), riem_best_sims)
    
    # Process individual seeds if they exist
    riem_dir = os.path.join(image_dir, "images", "riemannian")
    if os.path.exists(riem_dir):
        seed_files = [f for f in os.listdir(riem_dir) if f.startswith("seed_") and f.endswith(".pt")]
        
        if seed_files:
            print(f"[INFO] Found {len(seed_files)} seed files")
            all_seed_sims = []
            results["seeds"] = {}
            
            for seed_file in sorted(seed_files):
                seed_num = int(seed_file.split("_")[1].split(".")[0])
                seed_imgs = torch.load(os.path.join(riem_dir, seed_file))
                
                print(f"[INFO] Processing seed {seed_num}...")
                
                # Compute similarities
                seed_sims = []
                for i in range(orig.size(0)):
                    sim = compute_face_similarity(orig[i], seed_imgs[i], models, method)
                    seed_sims.append(sim)
                    if verbose:
                        print(f"[SEED {seed_num} SIM] img {i}: {sim:.4f}")
                
                seed_sims = np.array(seed_sims, dtype=np.float32)
                valid_seed = seed_sims[~np.isnan(seed_sims)]
                if valid_seed.size > 0:
                    results["seeds"][seed_num] = {
                        "mean": valid_seed.mean(),
                        "std": valid_seed.std(),
                        "valid": valid_seed.size,
                        "total": len(seed_sims)
                    }
                
                # Save similarities
                np.save(os.path.join(riem_dir, f"{method}_seed_{seed_num}_similarities.npy"), seed_sims)
                all_seed_sims.append(seed_sims)
            
            if len(all_seed_sims) > 1:
                # Stack and analyze all seeds
                all_seeds_stack = np.stack(all_seed_sims, axis=1)  # (B, S)
                np.save(os.path.join(image_dir, f"{method}_all_seeds_similarities.npy"), all_seeds_stack)
                
                # Find best similarity per image
                best_sims = np.nanmax(all_seeds_stack, axis=1)
                best_seeds = np.nanargmax(all_seeds_stack, axis=1)
                
                valid_best = best_sims[~np.isnan(best_sims)]
                if valid_best.size > 0:
                    results["best_across_seeds"] = {
                        "mean": valid_best.mean(),
                        "std": valid_best.std(),
                        "valid": valid_best.size,
                        "total": len(best_sims)
                    }
    
    # Print organized results
    print("\n" + "="*50)
    print(f"FACE SIMILARITY RESULTS ({method.upper()})")
    print("="*50)
    
    if "linear" in results:
        print(f"\nLINEAR EDITS:")
        r = results["linear"]
        print(f"  Average similarity: {r['mean']:.4f} ± {r['std']:.4f}")
        print(f"  Valid samples: {r['valid']}/{r['total']}")
    
    if "riemannian_best" in results:
        print(f"\nBEST RIEMANNIAN EDITS:")
        r = results["riemannian_best"]
        print(f"  Average similarity: {r['mean']:.4f} ± {r['std']:.4f}")
        print(f"  Valid samples: {r['valid']}/{r['total']}")
    
    if "seeds" in results and len(results["seeds"]) > 0:
        print(f"\nINDIVIDUAL SEEDS:")
        for seed_num, r in sorted(results["seeds"].items()):
            print(f"  Seed {seed_num}: {r['mean']:.4f} ± {r['std']:.4f} ({r['valid']}/{r['total']} valid)")
    
    if "best_across_seeds" in results:
        print(f"\nBEST ACROSS ALL SEEDS:")
        r = results["best_across_seeds"]
        print(f"  Average similarity: {r['mean']:.4f} ± {r['std']:.4f}")
        print(f"  Valid samples: {r['valid']}/{r['total']}")
    
    # If we have both linear and either best_riemannian or best_across_seeds
    if "linear" in results and ("riemannian_best" in results or "best_across_seeds" in results):
        print("\n" + "-"*50)
        print("COMPARISON: LINEAR vs RIEMANNIAN")
        print("-"*50)
        
        lin_mean = results["linear"]["mean"]
        if "riemannian_best" in results:
            riem_mean = results["riemannian_best"]["mean"]
            riem_std = results["riemannian_best"]["std"]
            comparison_type = "best Riemannian"
        else:
            riem_mean = results["best_across_seeds"]["mean"]
            riem_std = results["best_across_seeds"]["std"]
            comparison_type = "best across seeds"
        
        diff = riem_mean - lin_mean
        rel_diff = (diff / lin_mean) * 100 if lin_mean != 0 else float('inf')
        
        print(f"  Linear similarity:       {lin_mean:.4f} ± {results['linear']['std']:.4f}")
        print(f"  Riemannian similarity:   {riem_mean:.4f} ± {riem_std:.4f}")
        print(f"  Absolute difference:     {diff:.4f}")
        print(f"  Relative improvement:    {rel_diff:.2f}%")
        
        # Significance indicator
        if abs(diff) > (results["linear"]["std"] + riem_std) / 2:
            significance = "Significant difference"
        else:
            significance = "Difference within standard deviation"
        print(f"  Assessment: {significance}")

    return results

# For use as a standalone script
def process_args():
    parser = argparse.ArgumentParser(description="Evaluate face similarity for image manipulations")
    parser.add_argument("--ro-config", required=True, help="Path to config YAML")
    parser.add_argument("--method", choices=["arcface", "facenet"], default="arcface",
                      help="Face similarity method to use")
    parser.add_argument("--insightface-model", default="buffalo_l",
                      help="InsightFace model name (e.g., buffalo_l, magface_64)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = process_args()
    
    # Load config and set up paths
    cfg = load_riemannian_config(args.ro_config)
    img_dir = os.path.join(cfg.get("log_dir", "logs"), cfg.get("target_attr", ""), 'images')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run evaluation
    evaluate_manipulation_dir(img_dir, args.method, args.insightface_model, device, args.verbose)