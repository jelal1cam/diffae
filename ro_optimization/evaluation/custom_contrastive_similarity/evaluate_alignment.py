#!/usr/bin/env python3
"""
Evaluation script for the trained contrastive alignment network.
Tests how well the learned space preserves face identity distances.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr

# Import the models and datasets
from ro_optimization.evaluation.contrastive_align import (
    ContrastiveAlignmentModel,
    ContrastiveLatentDataset
)
from ro_optimization.evaluation.arcface_similarity import (
    init_face_models,
    get_embedding_faceanalysis,
    get_embedding_arcface,
)


class AlignmentEvaluator:
    """Evaluates alignment network performance on various metrics."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = "cuda"
    ):
        self.device = torch.device(device)
        
        # Load config
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        self.config = config_module.get_config()
        
        # Load model
        self.model = ContrastiveAlignmentModel.load_from_checkpoint(
            checkpoint_path,
            config=self.config
        ).to(self.device)
        self.model.eval()
        
        # Use EMA model for evaluation
        self.alignment_net = self.model.ema_alignment_net
        
        # Initialize ArcFace for comparison
        self.arcface_models = init_face_models(method="arcface", model_name="buffalo_l")
        
    def compute_embeddings(
        self,
        dataloader: DataLoader,
        use_augmentation: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Compute embeddings for all samples in dataloader."""
        
        all_z_orig = []
        all_z_aug = []
        all_h_orig = []
        all_h_aug = []
        all_indices = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing embeddings"):
                z_orig = batch['z_orig'].to(self.device)
                z_aug = batch['z_aug'].to(self.device)
                idx = batch['idx']
                
                # Pass through alignment network
                h_orig = self.alignment_net(z_orig)
                h_aug = self.alignment_net(z_aug)
                
                all_z_orig.append(z_orig.cpu())
                all_z_aug.append(z_aug.cpu())
                all_h_orig.append(h_orig.cpu())
                all_h_aug.append(h_aug.cpu())
                all_indices.append(idx)
        
        return {
            'z_orig': torch.cat(all_z_orig),
            'z_aug': torch.cat(all_z_aug),
            'h_orig': torch.cat(all_h_orig),
            'h_aug': torch.cat(all_h_aug),
            'indices': torch.cat(all_indices)
        }
    
    def evaluate_identity_preservation(
        self,
        embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate how well the alignment preserves identity."""
        
        h_orig = embeddings['h_orig']
        h_aug = embeddings['h_aug']
        z_orig = embeddings['z_orig']
        z_aug = embeddings['z_aug']
        
        # Compute similarities in both spaces
        sim_h = F.cosine_similarity(h_orig, h_aug, dim=1)
        sim_z = F.cosine_similarity(z_orig, z_aug, dim=1)
        
        # Statistics
        results = {
            'mean_sim_aligned': sim_h.mean().item(),
            'std_sim_aligned': sim_h.std().item(),
            'mean_sim_original': sim_z.mean().item(),
            'std_sim_original': sim_z.std().item(),
            'sim_improvement': (sim_h.mean() - sim_z.mean()).item()
        }
        
        return results
    
    def evaluate_distance_preservation(
        self,
        embeddings: Dict[str, torch.Tensor],
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """Evaluate how well distances are preserved."""
        
        h_orig = embeddings['h_orig']
        z_orig = embeddings['z_orig']
        
        # Sample random pairs
        n = len(h_orig)
        idx1 = torch.randint(0, n, (n_samples,))
        idx2 = torch.randint(0, n, (n_samples,))
        
        # Compute distances in both spaces
        dist_h = 1 - F.cosine_similarity(h_orig[idx1], h_orig[idx2], dim=1)
        dist_z = 1 - F.cosine_similarity(z_orig[idx1], z_orig[idx2], dim=1)
        
        # Correlation analysis
        pearson_r, pearson_p = pearsonr(dist_z.numpy(), dist_h.numpy())
        spearman_r, spearman_p = spearmanr(dist_z.numpy(), dist_h.numpy())
        
        results = {
            'pearson_correlation': pearson_r,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_r,
            'spearman_p_value': spearman_p
        }
        
        return results
    
    def evaluate_clustering_quality(
        self,
        embeddings: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate clustering quality if labels are available."""
        
        if labels is None:
            return {}
        
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        h_orig = embeddings['h_orig'].numpy()
        z_orig = embeddings['z_orig'].numpy()
        
        # Compute clustering metrics
        results = {
            'silhouette_aligned': silhouette_score(h_orig, labels, metric='cosine'),
            'silhouette_original': silhouette_score(z_orig, labels, metric='cosine'),
            'calinski_harabasz_aligned': calinski_harabasz_score(h_orig, labels),
            'calinski_harabasz_original': calinski_harabasz_score(z_orig, labels),
        }
        
        return results
    
    def evaluate_arcface_alignment(
        self,
        dataset,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """Compare with ArcFace embeddings on a subset of data."""
        
        # Sample indices
        indices = torch.randperm(len(dataset))[:n_samples]
        
        arcface_embeddings = []
        aligned_embeddings = []
        original_embeddings = []
        
        with torch.no_grad():
            for idx in tqdm(indices, desc="Computing ArcFace embeddings"):
                sample = dataset[idx.item()]
                
                # Get image
                img = dataset.base_dataset[idx.item()]['img']
                img_01 = (img + 1) / 2  # Convert to [0, 1]
                
                # Get ArcFace embedding
                try:
                    arc_emb = get_embedding_faceanalysis(
                        self.arcface_models["app"], 
                        img_01
                    )
                except:
                    arc_emb = get_embedding_arcface(
                        self.arcface_models["zoo_model"], 
                        img_01
                    )
                arc_emb = torch.from_numpy(arc_emb).float()
                arc_emb = F.normalize(arc_emb, p=2, dim=0)
                
                # Get aligned and original embeddings
                z_orig = sample['z_orig'].unsqueeze(0).to(self.device)
                h_orig = self.alignment_net(z_orig).squeeze(0).cpu()
                
                arcface_embeddings.append(arc_emb)
                aligned_embeddings.append(h_orig)
                original_embeddings.append(z_orig.cpu().squeeze(0))
        
        # Stack embeddings
        arcface_embeddings = torch.stack(arcface_embeddings)
        aligned_embeddings = torch.stack(aligned_embeddings)
        original_embeddings = torch.stack(original_embeddings)
        
        # Compute similarity matrices
        sim_arc = torch.mm(arcface_embeddings, arcface_embeddings.t())
        sim_aligned = torch.mm(aligned_embeddings, aligned_embeddings.t())
        sim_original = torch.mm(original_embeddings, original_embeddings.t())
        
        # Compare similarity patterns
        # Flatten upper triangular parts
        mask = torch.triu(torch.ones_like(sim_arc), diagonal=1).bool()
        sim_arc_flat = sim_arc[mask]
        sim_aligned_flat = sim_aligned[mask]
        sim_original_flat = sim_original[mask]
        
        # Correlations with ArcFace
        pearson_aligned, _ = pearsonr(sim_arc_flat.numpy(), sim_aligned_flat.numpy())
        pearson_original, _ = pearsonr(sim_arc_flat.numpy(), sim_original_flat.numpy())
        
        results = {
            'arcface_correlation_aligned': pearson_aligned,
            'arcface_correlation_original': pearson_original,
            'arcface_alignment_improvement': pearson_aligned - pearson_original
        }
        
        return results
    
    def visualize_results(
        self,
        embeddings: Dict[str, torch.Tensor],
        save_path: str = "alignment_evaluation.png"
    ):
        """Create visualization of evaluation results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Similarity distribution
        ax = axes[0, 0]
        sim_h = F.cosine_similarity(embeddings['h_orig'], embeddings['h_aug'], dim=1)
        sim_z = F.cosine_similarity(embeddings['z_orig'], embeddings['z_aug'], dim=1)
        
        ax.hist(sim_z.numpy(), bins=50, alpha=0.5, label='Original space', density=True)
        ax.hist(sim_h.numpy(), bins=50, alpha=0.5, label='Aligned space', density=True)
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.set_title('Positive Pair Similarities')
        ax.legend()
        
        # 2. Distance correlation
        ax = axes[0, 1]
        n_samples = min(1000, len(embeddings['h_orig']))
        idx1 = torch.randint(0, len(embeddings['h_orig']), (n_samples,))
        idx2 = torch.randint(0, len(embeddings['h_orig']), (n_samples,))
        
        dist_h = 1 - F.cosine_similarity(embeddings['h_orig'][idx1], embeddings['h_orig'][idx2], dim=1)
        dist_z = 1 - F.cosine_similarity(embeddings['z_orig'][idx1], embeddings['z_orig'][idx2], dim=1)
        
        ax.scatter(dist_z.numpy(), dist_h.numpy(), alpha=0.3, s=10)
        ax.plot([0, 2], [0, 2], 'r--', label='y=x')
        ax.set_xlabel('Original Space Distance')
        ax.set_ylabel('Aligned Space Distance')
        ax.set_title('Distance Preservation')
        ax.legend()
        
        # 3. t-SNE visualization (subsample for speed)
        from sklearn.manifold import TSNE
        
        n_vis = min(500, len(embeddings['h_orig']))
        indices = torch.randperm(len(embeddings['h_orig']))[:n_vis]
        
        # Original space t-SNE
        ax = axes[1, 0]
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        z_tsne = tsne.fit_transform(embeddings['z_orig'][indices].numpy())
        ax.scatter(z_tsne[:, 0], z_tsne[:, 1], alpha=0.5, s=10)
        ax.set_title('Original Space (t-SNE)')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        # Aligned space t-SNE
        ax = axes[1, 1]
        h_tsne = tsne.fit_transform(embeddings['h_orig'][indices].numpy())
        ax.scatter(h_tsne[:, 0], h_tsne[:, 1], alpha=0.5, s=10)
        ax.set_title('Aligned Space (t-SNE)')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate contrastive alignment network")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--dataset", type=str, default="celebahq", help="Dataset to evaluate on")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AlignmentEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Load encoder and create dataset
    from templates_latent import ffhq128_autoenc_latent, ffhq256_autoenc_latent
    from ro_optimization.evaluation.align import load_encoder_via_lit
    from dataset import CelebHQAttrDataset, CelebAlmdb, FFHQlmdb
    
    # Get config and load encoder
    template_dispatch = {
        "ffhq128": ffhq128_autoenc_latent,
        "ffhq256": ffhq256_autoenc_latent,
    }
    
    conf = template_dispatch[evaluator.config.encoder_config]()
    encoder, z_mean, z_std = load_encoder_via_lit(conf, torch.device(args.device))
    
    # Create dataset
    dataset_cls = {
        "celebahq": CelebHQAttrDataset,
        "celebalmdb": CelebAlmdb,
        "ffhqlmdb": FFHQlmdb,
    }
    
    base_dataset = dataset_cls[args.dataset](image_size=evaluator.config.img_size)
    
    # Create contrastive dataset
    dataset = ContrastiveLatentDataset(
        base_dataset,
        encoder,
        z_mean,
        z_std,
        augmentation_strength=evaluator.config.augmentation_strength,
        device=args.device
    )
    
    # Subsample if needed
    if args.n_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:args.n_samples]
        dataset = Subset(dataset, indices)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=evaluator.config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Compute embeddings
    embeddings = evaluator.compute_embeddings(dataloader)
    
    # Run evaluations
    print("\n=== Identity Preservation ===")
    identity_results = evaluator.evaluate_identity_preservation(embeddings)
    for k, v in identity_results.items():
        print(f"{k}: {v:.4f}")
    
    print("\n=== Distance Preservation ===")
    distance_results = evaluator.evaluate_distance_preservation(embeddings)
    for k, v in distance_results.items():
        print(f"{k}: {v:.4f}")
    
    print("\n=== ArcFace Alignment ===")
    arcface_results = evaluator.evaluate_arcface_alignment(dataset, n_samples=min(100, len(dataset)))
    for k, v in arcface_results.items():
        print(f"{k}: {v:.4f}")
    
    # Create visualizations
    if args.visualize:
        evaluator.visualize_results(embeddings)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()