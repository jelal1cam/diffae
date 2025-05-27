#!/usr/bin/env python3
"""
Utility functions for using the trained alignment network.

This module provides convenient functions to load the alignment model
and use it to map custom latents to ArcFace space.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

from ro_optimization.evaluation.align import AlignmentModel


class AlignmentWrapper:
    """Wrapper class for easy use of the alignment network."""
    
    def __init__(
        self, 
        checkpoint_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        use_ema: bool = True
    ):
        """
        Initialize the alignment wrapper.
        
        Args:
            checkpoint_path: Path to the trained alignment model checkpoint
            config: Configuration dict (if None, will be loaded from checkpoint)
            device: Device to use ('cuda' or 'cpu')
            use_ema: Whether to use the EMA version of the network
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        if config is None:
            # Load config from checkpoint
            self.model = AlignmentModel.load_from_checkpoint(
                checkpoint_path,
                map_location=self.device
            )
            self.config = self.model.config
        else:
            self.model = AlignmentModel.load_from_checkpoint(
                checkpoint_path,
                config=config,
                map_location=self.device
            )
            self.config = config
        
        self.model.to(self.device)
        self.model.eval()
        
        # Select network (EMA or regular)
        if use_ema:
            self.alignment_net = self.model.ema_alignment_net
        else:
            self.alignment_net = self.model.alignment_net
        
        print(f"Loaded alignment model from {checkpoint_path}")
        print(f"Using {'EMA' if use_ema else 'regular'} network on {self.device}")
    
    def align(self, custom_latents: torch.Tensor) -> torch.Tensor:
        """
        Map custom latents to ArcFace space.
        
        Args:
            custom_latents: Tensor of shape (B, custom_dim) or (custom_dim,)
        
        Returns:
            Aligned ArcFace embeddings of shape (B, 512) or (512,)
        """
        single_input = custom_latents.dim() == 1
        if single_input:
            custom_latents = custom_latents.unsqueeze(0)
        
        # Move to device
        custom_latents = custom_latents.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            arcface_latents = self.alignment_net(custom_latents)
        
        # Remove batch dimension if input was single
        if single_input:
            arcface_latents = arcface_latents.squeeze(0)
        
        return arcface_latents
    
    def compute_similarity(
        self, 
        custom_latent1: torch.Tensor, 
        custom_latent2: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between two custom latents in ArcFace space.
        
        Args:
            custom_latent1: First custom latent (1D or 2D tensor)
            custom_latent2: Second custom latent (1D or 2D tensor)
        
        Returns:
            Cosine similarity score
        """
        # Ensure inputs are 2D
        if custom_latent1.dim() == 1:
            custom_latent1 = custom_latent1.unsqueeze(0)
        if custom_latent2.dim() == 1:
            custom_latent2 = custom_latent2.unsqueeze(0)
        
        # Align to ArcFace space
        arcface1 = self.align(custom_latent1)
        arcface2 = self.align(custom_latent2)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(arcface1, arcface2).item()
        
        return similarity
    
    def batch_similarities(
        self,
        custom_latents1: torch.Tensor,
        custom_latents2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise similarities for batches of latents.
        
        Args:
            custom_latents1: First batch of custom latents (B, custom_dim)
            custom_latents2: Second batch of custom latents (B, custom_dim)
        
        Returns:
            Tensor of similarities (B,)
        """
        # Align both batches
        arcface1 = self.align(custom_latents1)
        arcface2 = self.align(custom_latents2)
        
        # Compute similarities
        similarities = F.cosine_similarity(arcface1, arcface2, dim=1)
        
        return similarities


def load_alignment_model(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cuda",
    use_ema: bool = True
) -> AlignmentWrapper:
    """
    Convenience function to load an alignment model.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to config file (optional, will load from checkpoint if None)
        device: Device to use
        use_ema: Whether to use EMA network
    
    Returns:
        AlignmentWrapper instance
    """
    config = None
    if config_path is not None:
        # Load config from file
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.get_config()
    
    return AlignmentWrapper(checkpoint_path, config, device, use_ema)


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the best checkpoint in a directory based on validation loss.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Path to best checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for checkpoint files
    ckpt_files = list(checkpoint_dir.glob("alignment-*.ckpt"))
    
    if not ckpt_files:
        # Try last.ckpt
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            return str(last_ckpt)
        else:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by validation loss (assuming filename format: alignment-epoch-val_loss.ckpt)
    def get_val_loss(path):
        try:
            return float(path.stem.split("-")[-1])
        except:
            return float('inf')
    
    best_ckpt = min(ckpt_files, key=get_val_loss)
    return str(best_ckpt)


# Example usage
if __name__ == "__main__":
    # Example: Load model and compute similarities
    
    # Find best checkpoint
    ckpt_dir = "checkpoints/alignment"
    best_ckpt = find_best_checkpoint(ckpt_dir)
    print(f"Using checkpoint: {best_ckpt}")
    
    # Load model
    aligner = load_alignment_model(best_ckpt)
    
    # Example: Generate random latents and compute similarity
    custom_dim = aligner.config.custom_latent_dim
    latent1 = torch.randn(custom_dim)
    latent2 = torch.randn(custom_dim)
    
    similarity = aligner.compute_similarity(latent1, latent2)
    print(f"Similarity between random latents: {similarity:.4f}")
    
    # Example: Batch processing
    batch_size = 10
    batch1 = torch.randn(batch_size, custom_dim)
    batch2 = torch.randn(batch_size, custom_dim)
    
    similarities = aligner.batch_similarities(batch1, batch2)
    print(f"Batch similarities: {similarities}")
    print(f"Mean similarity: {similarities.mean():.4f}")