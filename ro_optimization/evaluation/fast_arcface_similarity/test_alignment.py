#!/usr/bin/env python3
"""
Test alignment network by comparing direct ArcFace similarities with aligned similarities.

Usage:
    python -m ro_optimization.evaluation.test_alignment \
        --config ro_optimization/evaluation/configs/alignment_config.py \
        --checkpoint path/to/checkpoint.ckpt \
        --n_pairs 1000
"""

import os
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Import from the project
from dataset import CelebHQAttrDataset
from experiment import LitModel
from ro_optimization.evaluation.arcface_similarity import (
    init_face_models,
    get_embedding_faceanalysis,
    get_embedding_arcface,
)
from ro_optimization.evaluation.align import AlignmentModel

# Import config loaders
from templates_latent import ffhq128_autoenc_latent, ffhq256_autoenc_latent

AVAILABLE_CFGS = {
    "ffhq128_autoenc_latent": ffhq128_autoenc_latent,
    "ffhq256_autoenc_latent": ffhq256_autoenc_latent,
}


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_arcface_embedding(img: torch.Tensor, models: Dict) -> torch.Tensor:
    """Extract ArcFace embedding from a single image."""
    try:
        emb = get_embedding_faceanalysis(models["app"], img)
    except RuntimeError:
        emb = get_embedding_arcface(models["zoo_model"], img)
    
    emb = torch.from_numpy(emb).float()
    return F.normalize(emb, p=2, dim=0)


def compute_direct_arcface_similarity(img1: torch.Tensor, img2: torch.Tensor, 
                                    arcface_models: Dict) -> float:
    """Compute ArcFace similarity directly between two images."""
    # Convert to [0, 1] range expected by ArcFace
    img1_01 = (img1 + 1) / 2
    img2_01 = (img2 + 1) / 2
    
    # Get embeddings
    emb1 = get_arcface_embedding(img1_01, arcface_models)
    emb2 = get_arcface_embedding(img2_01, arcface_models)
    
    # Compute cosine similarity
    sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    return sim


def compute_aligned_similarity(img1: torch.Tensor, img2: torch.Tensor,
                             encoder: torch.nn.Module,
                             alignment_net: torch.nn.Module,
                             z_mean: torch.Tensor = None,
                             z_std: torch.Tensor = None) -> float:
    """Compute similarity using custom encoder and alignment network."""
    with torch.no_grad():
        # Encode with custom encoder
        z1 = encoder(img1.unsqueeze(0))
        z2 = encoder(img2.unsqueeze(0))
        
        # Apply z-normalization if available
        if z_mean is not None and z_std is not None:
            z1 = (z1 - z_mean) / z_std
            z2 = (z2 - z_mean) / z_std
        
        # Map to ArcFace space
        arcface_z1 = alignment_net(z1)
        arcface_z2 = alignment_net(z2)
        
        # Compute cosine similarity
        sim = F.cosine_similarity(arcface_z1, arcface_z2).item()
    
    return sim


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------

def evaluate_alignment(
    config: Dict[str, Any],
    checkpoint_path: str,
    autoenc_config_name: str,
    n_pairs: int = 1000,
    arcface_model: str = "buffalo_l",
    device: str = "cuda"
):
    """Evaluate alignment by comparing direct and aligned similarities."""
    
    set_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    print(f"Loading models on {device}...")
    
    # Load autoencoder configuration
    autoenc_conf = AVAILABLE_CFGS[autoenc_config_name]()
    
    # Load encoder via LitModel
    lit = LitModel(autoenc_conf).to(device)
    autoenc_ckpt = os.path.join("checkpoints", autoenc_conf.name, "last.ckpt")
    state = torch.load(autoenc_ckpt, map_location="cpu")
    lit.load_state_dict(state["state_dict"], strict=False)
    lit.ema_model.eval()
    encoder = lit.ema_model.encoder
    
    # Get z-normalization stats if available
    z_mean = getattr(lit, "conds_mean", None)
    z_std = getattr(lit, "conds_std", None)
    if z_mean is not None:
        z_mean = z_mean.to(device)
        z_std = z_std.to(device)
        print("Using z-normalization statistics")
    
    # Load alignment model
    alignment_model = AlignmentModel.load_from_checkpoint(
        checkpoint_path,
        config=config
    ).to(device)
    alignment_model.eval()
    
    # Use EMA network for evaluation
    alignment_net = alignment_model.ema_alignment_net
    
    # Initialize ArcFace models
    arcface_models = init_face_models(method="arcface", model_name=arcface_model)
    
    # Load CelebA-HQ dataset
    dataset = CelebHQAttrDataset(image_size=autoenc_conf.img_size)
    
    # Generate random pairs
    print(f"\nGenerating {n_pairs} random image pairs...")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Create pairs (consecutive images)
    pair_indices = [(indices[i], indices[i+1]) for i in range(0, min(len(indices)-1, n_pairs*2), 2)]
    pair_indices = pair_indices[:n_pairs]
    
    # Compute similarities
    direct_sims = []
    aligned_sims = []
    
    print("\nComputing similarities...")
    for idx1, idx2 in tqdm(pair_indices):
        # Load images
        img1 = dataset[idx1]["img"].to(device)
        img2 = dataset[idx2]["img"].to(device)
        
        try:
            # Direct ArcFace similarity
            direct_sim = compute_direct_arcface_similarity(
                img1, img2, arcface_models
            )
            
            # Aligned similarity
            aligned_sim = compute_aligned_similarity(
                img1, img2, encoder, alignment_net, z_mean, z_std
            )
            
            direct_sims.append(direct_sim)
            aligned_sims.append(aligned_sim)
            
        except Exception as e:
            print(f"Error processing pair ({idx1}, {idx2}): {e}")
            continue
    
    # Convert to numpy arrays
    direct_sims = np.array(direct_sims)
    aligned_sims = np.array(aligned_sims)
    
    # Compute statistics
    print("\n" + "="*60)
    print("ALIGNMENT EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nNumber of successful pairs: {len(direct_sims)}/{n_pairs}")
    
    # Basic statistics
    print(f"\nDirect ArcFace similarities:")
    print(f"  Mean: {direct_sims.mean():.4f} ± {direct_sims.std():.4f}")
    print(f"  Min: {direct_sims.min():.4f}, Max: {direct_sims.max():.4f}")
    
    print(f"\nAligned similarities:")
    print(f"  Mean: {aligned_sims.mean():.4f} ± {aligned_sims.std():.4f}")
    print(f"  Min: {aligned_sims.min():.4f}, Max: {aligned_sims.max():.4f}")
    
    # Difference statistics
    differences = aligned_sims - direct_sims
    abs_differences = np.abs(differences)
    
    print(f"\nDifference statistics (aligned - direct):")
    print(f"  Mean difference: {differences.mean():.4f}")
    print(f"  Mean absolute difference: {abs_differences.mean():.4f}")
    print(f"  Std of differences: {differences.std():.4f}")
    print(f"  Max absolute difference: {abs_differences.max():.4f}")
    
    # Percentile analysis
    percentiles = [50, 90, 95, 99]
    print(f"\nAbsolute difference percentiles:")
    for p in percentiles:
        val = np.percentile(abs_differences, p)
        print(f"  {p}th percentile: {val:.4f}")
    
    # Correlation analysis
    pearson_r, pearson_p = stats.pearsonr(direct_sims, aligned_sims)
    spearman_r, spearman_p = stats.spearmanr(direct_sims, aligned_sims)
    
    print(f"\nCorrelation analysis:")
    print(f"  Pearson correlation: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})")
    
    # Success rate (e.g., difference < 0.05)
    thresholds = [0.01, 0.02, 0.05, 0.1]
    print(f"\nSuccess rates (|difference| < threshold):")
    for thresh in thresholds:
        success_rate = (abs_differences < thresh).mean() * 100
        print(f"  Threshold {thresh}: {success_rate:.1f}%")
    
    # Save detailed results
    results = {
        "direct_similarities": direct_sims,
        "aligned_similarities": aligned_sims,
        "differences": differences,
        "abs_differences": abs_differences,
        "statistics": {
            "n_pairs": len(direct_sims),
            "direct_mean": direct_sims.mean(),
            "direct_std": direct_sims.std(),
            "aligned_mean": aligned_sims.mean(),
            "aligned_std": aligned_sims.std(),
            "mean_difference": differences.mean(),
            "mean_abs_difference": abs_differences.mean(),
            "max_abs_difference": abs_differences.max(),
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "success_rates": {f"thresh_{t}": (abs_differences < t).mean() 
                            for t in thresholds}
        }
    }
    
    # Save results
    output_dir = Path(config.log_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / f"alignment_evaluation_n{n_pairs}.npz"
    np.savez(
        results_path,
        **{k: v for k, v in results.items() if isinstance(v, np.ndarray)}
    )
    
    # Save statistics as JSON
    import json
    stats_path = output_dir / f"alignment_evaluation_n{n_pairs}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(results["statistics"], f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  {results_path}")
    print(f"  {stats_path}")
    
    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scatter plot
        ax = axes[0, 0]
        ax.scatter(direct_sims, aligned_sims, alpha=0.5, s=10)
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect alignment')
        ax.set_xlabel('Direct ArcFace Similarity')
        ax.set_ylabel('Aligned Similarity')
        ax.set_title(f'Similarity Comparison (r={pearson_r:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Difference histogram
        ax = axes[0, 1]
        ax.hist(differences, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', label='Zero difference')
        ax.set_xlabel('Difference (Aligned - Direct)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Differences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Absolute difference histogram
        ax = axes[1, 0]
        ax.hist(abs_differences, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Absolute Difference')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of Absolute Differences\n(Mean: {abs_differences.mean():.4f})')
        ax.grid(True, alpha=0.3)
        
        # Cumulative distribution
        ax = axes[1, 1]
        sorted_abs_diff = np.sort(abs_differences)
        cumulative = np.arange(1, len(sorted_abs_diff) + 1) / len(sorted_abs_diff)
        ax.plot(sorted_abs_diff, cumulative)
        ax.set_xlabel('Absolute Difference')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution of Absolute Differences')
        ax.grid(True, alpha=0.3)
        
        # Add threshold lines
        for thresh in thresholds:
            success_rate = (abs_differences < thresh).mean()
            ax.axvline(thresh, color='gray', linestyle=':', alpha=0.5)
            ax.text(thresh, 0.5, f'{thresh}: {success_rate:.1%}', 
                   rotation=90, verticalalignment='center')
        
        plt.tight_layout()
        plot_path = output_dir / f"alignment_evaluation_n{n_pairs}_plots.pdf"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"\nPlots saved to: {plot_path}")
        
    except ImportError:
        print("\nMatplotlib not available, skipping plots.")
    
    print("\n" + "="*60)
    
    # Summary assessment
    if abs_differences.mean() < 0.02 and pearson_r > 0.95:
        print("✓ EXCELLENT: Alignment is very successful!")
    elif abs_differences.mean() < 0.05 and pearson_r > 0.9:
        print("✓ GOOD: Alignment is working well.")
    elif abs_differences.mean() < 0.1 and pearson_r > 0.8:
        print("○ MODERATE: Alignment shows reasonable performance.")
    else:
        print("✗ POOR: Alignment needs improvement.")
    
    print("="*60)
    
    return results


# -----------------------------------------------------------------------------
# Additional evaluation: Specific attribute preservation
# -----------------------------------------------------------------------------

def evaluate_attribute_preservation(
    config: Dict[str, Any],
    checkpoint_path: str,
    autoenc_config_name: str,
    n_samples: int = 100,
    device: str = "cuda"
):
    """Evaluate if alignment preserves identity while changing attributes."""
    
    print("\n" + "="*60)
    print("ATTRIBUTE PRESERVATION EVALUATION")
    print("="*60)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # This would require attribute manipulation code
    # For now, we'll skip this evaluation
    print("(Skipping attribute preservation evaluation - requires manipulation code)")
    

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test alignment network by comparing similarities"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to alignment configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to alignment model checkpoint"
    )
    parser.add_argument(
        "--autoenc-config",
        type=str,
        choices=list(AVAILABLE_CFGS.keys()),
        default="ffhq256_autoenc_latent",
        help="Autoencoder configuration name"
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=1000,
        help="Number of image pairs to evaluate"
    )
    parser.add_argument(
        "--arcface-model",
        type=str,
        default="buffalo_l",
        help="ArcFace model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--eval-attributes",
        action="store_true",
        help="Also evaluate attribute preservation"
    )
    
    args = parser.parse_args()
    
    # Import and load config
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()
    
    # Run evaluation
    results = evaluate_alignment(
        config=config,
        checkpoint_path=args.checkpoint,
        autoenc_config_name=args.autoenc_config,
        n_pairs=args.n_pairs,
        arcface_model=args.arcface_model,
        device=args.device
    )
    
    # Optional: evaluate attribute preservation
    if args.eval_attributes:
        evaluate_attribute_preservation(
            config=config,
            checkpoint_path=args.checkpoint,
            autoenc_config_name=args.autoenc_config,
            n_samples=100,
            device=args.device
        )


if __name__ == "__main__":
    main()