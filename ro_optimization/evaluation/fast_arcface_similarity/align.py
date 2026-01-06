#!/usr/bin/env python3
"""
Train an alignment network to map custom latent space to ArcFace space.

Usage:
    python -m ro_optimization.evaluation.align --config ro_optimization/evaluation/configs/alignment_config.py
"""

import os
import argparse
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor, 
    EarlyStopping,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger

from enum import Enum
from typing import Sequence, Tuple, Optional
import math

# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------

class Activation(Enum):
    relu = "relu"
    lrelu = "lrelu"
    silu = "silu"
    none = "none"

    def get(self):
        return {
            "relu": nn.ReLU(inplace=True),
            "lrelu": nn.LeakyReLU(0.2, inplace=True),
            "silu": nn.SiLU(inplace=True),
            "none": nn.Identity(),
        }[self.value]

def _make_linear(in_dim, out_dim, act: Activation, use_norm: bool, p_dropout: float):
    layers = [nn.Linear(in_dim, out_dim, bias=not use_norm)]
    if use_norm:
        layers.append(nn.LayerNorm(out_dim))
    layers.append(act.get())
    if p_dropout > 0:
        layers.append(nn.Dropout(p_dropout))
    return nn.Sequential(*layers)

# ----------------------------------------------------------------------
#  Two concrete nets
# ----------------------------------------------------------------------

class LinearAlignNet(nn.Module):
    """Single affine map 512→512 (identical to your baseline)."""
    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):                # (B,512) → (B,512)
        return self.proj(x)


class MLPSkipAlignNet(nn.Module):
    """
    MLP with optional skip injections: on every layer index i in `skip_layers`
    we concatenate the original input x to the running hidden state *before*
    it enters that layer.
    """

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 512,
        hidden_dim: int = 1024,
        n_layers: int = 6,
        skip_layers: Sequence[int] = (2, 4),
        activation: str = "silu",
        use_norm: bool = True,
        dropout: float = 0.1,
        last_act: str = "none",
    ):
        super().__init__()
        act_enum      = Activation(activation)
        last_act_enum = Activation(last_act)

        # ------------------------------------------------------------------
        # build (din, dout) tuples for every layer, now *accounting* for skip
        # ------------------------------------------------------------------
        dims: list[Tuple[int, int]] = []
        for i in range(n_layers):
            # ➊ what comes from previous layer
            if i == 0:
                din = in_dim
            else:
                din = hidden_dim

            # ➋ if this layer is a skip-layer we will concat x first
            if i in skip_layers:
                din += in_dim           # <- bug fix

            # ➌ output size
            dout = hidden_dim if i < n_layers - 1 else out_dim
            dims.append((din, dout))

        # ------------------------------------------------------------------
        # actual modules
        # ------------------------------------------------------------------
        self.layers = nn.ModuleList(
            [
                _make_linear(
                    din,
                    dout,
                    act_enum if k < n_layers - 1 else last_act_enum,
                    use_norm and k < n_layers - 1,
                    dropout if k < n_layers - 1 else 0.0,
                )
                for k, (din, dout) in enumerate(dims)
            ]
        )
        self.skip_layers = set(skip_layers)
        self.in_dim      = in_dim

    def forward(self, x):                       # (B,512) → (B,512)
        h = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_layers:           # inject input
                h = torch.cat([h, x], dim=1)
            h = layer(h)
        return h


# alignment_network.py  (add once)
class L2NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

# ----------------------------------------------------------------------
# Factory
def build_alignment_net(cfg):
    arch = cfg.architecture.lower()
    if arch == "linear":
        core = LinearAlignNet(dim=cfg.custom_latent_dim)
    elif arch == "mlp":
        hidden_dims = getattr(cfg, "hidden_dims", [1024])
        hidden = hidden_dims[0] if hidden_dims else 1024
        core = MLPSkipAlignNet(
            in_dim      = cfg.custom_latent_dim,
            out_dim     = cfg.arcface_latent_dim,
            hidden_dim  = hidden,
            n_layers    = getattr(cfg, "num_layers", 6),
            skip_layers = tuple(getattr(cfg, "skip_layers", (2, 4))),
            activation  = getattr(cfg, "activation", "silu"),
            use_norm    = getattr(cfg, "use_norm", True),
            dropout     = getattr(cfg, "dropout", 0.1),
            last_act    = getattr(cfg, "last_act", "none"),
        )
    else:
        raise ValueError(f"Unknown architecture {arch}")

    # optional post-layer
    if getattr(cfg, "output_norm", "none") == "l2":
        return nn.Sequential(core, L2NormLayer())
    return core



# -----------------------------------------------------------------------------
# Lightning Module
# -----------------------------------------------------------------------------

class AlignmentModel(pl.LightningModule):
    """Lightning module for training the alignment network."""

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
            
        # Initialize network
        self.alignment_net = build_alignment_net(config)
        
        # EMA network
        self.ema_alignment_net = copy.deepcopy(self.alignment_net)
        self.ema_alignment_net.requires_grad_(False)
        self.ema_decay = config.ema_decay
        
        # Loss function
        self.loss_type = config.loss_type
        self.mse_weight = getattr(config, "mse_weight", 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alignment_net(x)
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss."""
        if self.loss_type == "cosine":
            # Cosine similarity loss (1 - cosine_similarity)
            loss = 1 - F.cosine_similarity(pred, target, dim=1).mean()
        elif self.loss_type == "mse":
            # Mean squared error
            loss = F.mse_loss(pred, target)
        elif self.loss_type == "combined":
            # Combination of cosine and MSE
            cos_loss = 1 - F.cosine_similarity(pred, target, dim=1).mean()
            mse_loss = F.mse_loss(pred, target)
            loss = cos_loss + self.mse_weight * mse_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        custom_latents, arcface_latents = batch
        
        # Forward pass
        pred_arcface = self.alignment_net(custom_latents)
        loss = self.compute_loss(pred_arcface, arcface_latents)
        
        # Logging
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Compute cosine similarity & Euclidean distance for monitoring
        with torch.no_grad():
            cos_sim = F.cosine_similarity(pred_arcface, arcface_latents, dim=1).mean()
            l2_dist = F.pairwise_distance(pred_arcface, arcface_latents).mean()
            
            self.log("train_cos_sim", cos_sim, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_l2_dist", l2_dist, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        custom_latents, arcface_latents = batch
        
        # Forward pass with both networks
        pred_arcface = self.alignment_net(custom_latents)
        pred_arcface_ema = self.ema_alignment_net(custom_latents)
        
        # Compute losses
        loss = self.compute_loss(pred_arcface, arcface_latents)
        loss_ema = self.compute_loss(pred_arcface_ema, arcface_latents)
        
        # Logging
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_loss_ema", loss_ema, prog_bar=True, on_epoch=True)
        
        # Compute cosine similarities
        cos_sim = F.cosine_similarity(pred_arcface, arcface_latents, dim=1).mean()
        cos_sim_ema = F.cosine_similarity(pred_arcface_ema, arcface_latents, dim=1).mean()
        
        # Compute Euclidean distances
        l2_dist = F.pairwise_distance(pred_arcface, arcface_latents).mean()
        l2_dist_ema = F.pairwise_distance(pred_arcface_ema, arcface_latents).mean()
        
        self.log("val_cos_sim", cos_sim, prog_bar=True, on_epoch=True)
        self.log("val_cos_sim_ema", cos_sim_ema, prog_bar=True, on_epoch=True)
        
        self.log("val_l2_dist", l2_dist, prog_bar=True, on_epoch=True)
        self.log("val_l2_dist_ema", l2_dist_ema, prog_bar=True, on_epoch=True)
        
        return loss

    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        custom_latents, arcface_latents = batch
        
        # Use EMA network for testing
        pred_arcface = self.ema_alignment_net(custom_latents)
        
        # Compute metrics
        loss = self.compute_loss(pred_arcface, arcface_latents)
        cos_sim = F.cosine_similarity(pred_arcface, arcface_latents, dim=1)
        
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_cos_sim", cos_sim.mean(), on_epoch=True)
        
        return {
            "loss": loss,
            "cos_sim": cos_sim,
            "pred": pred_arcface,
            "target": arcface_latents
        }
    
    def on_after_backward(self) -> None:
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
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Scheduler
        if self.config.scheduler == "none":
            return optimizer
        elif self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_epochs,
                eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.config.scheduler == "cosine_warmup":
            from torch.optim.lr_scheduler import LambdaLR

            total_steps = self.config.max_epochs * len(self.trainer.datamodule.train_dataloader())

            def lr_lambda(step):
                if step < self.config.warmup_steps:              # linear warm-up
                    return step / self.config.warmup_steps
                # cosine decay
                progress = (step - self.config.warmup_steps) / max(1, total_steps - self.config.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))  # ← math.cos on float

            scheduler = LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
        elif self.config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
                min_lr=1e-6,
                threshold=1e-4
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif self.config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.config.scheduler == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[50, 100, 150],
                gamma=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.config.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.98
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")


# -----------------------------------------------------------------------------
# Data Module
# -----------------------------------------------------------------------------

class AlignmentDataModule(pl.LightningDataModule):
    """Data module for loading paired latents."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_dir = Path(config.paired_latents_dir)
        
    def setup(self, stage: Optional[str] = None):
        """Load the pre-computed paired latents."""
        if stage in (None, "fit"):
            # Load training data - now saved as tuple (custom_tensors, arcface_tensors)
            train_data = torch.load(self.data_dir / "train.pt", weights_only=False)
            train_custom, train_arcface = train_data
            self.train_dataset = TensorDataset(train_custom, train_arcface)
            
            # Load validation data
            val_data = torch.load(self.data_dir / "val.pt", weights_only=False)
            val_custom, val_arcface = val_data
            self.val_dataset = TensorDataset(val_custom, val_arcface)
            
            print(f"Train dataset: {len(self.train_dataset)} samples")
            print(f"Val dataset: {len(self.val_dataset)} samples")
        
        if stage in (None, "test"):
            # Load test data
            test_data = torch.load(self.data_dir / "test.pt", weights_only=False)
            test_custom, test_arcface = test_data
            self.test_dataset = TensorDataset(test_custom, test_arcface)
            
            print(f"Test dataset: {len(self.test_dataset)} samples")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )


# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------

def train_alignment(config: Dict[str, Any]):
    """Main training function."""
    
    # Set seed
    if config.seed is not None:
        seed_everything(config.seed)
    
    # Create directories
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    data_module = AlignmentDataModule(config)
    
    # Initialize model
    model = AlignmentModel(config)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename="alignment-{epoch:03d}-{val_loss:.4f}",
            monitor=config.monitor_metric,
            mode="min",
            save_top_k=config.save_top_k,
            save_last=True
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=config.monitor_metric,
            patience=12,
            mode="min"
        ),
        RichProgressBar()
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name="alignment"
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
    
    # Test
    trainer.test(model, data_module)
    
    print(f"Training completed. Best checkpoint saved at: {callbacks[0].best_model_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train alignment network")
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
    train_alignment(config)


if __name__ == "__main__":
    main()