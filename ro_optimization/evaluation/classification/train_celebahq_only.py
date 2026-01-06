# train_celebahq_only.py
"""
Train attribute classifier on CelebA-HQ only.
The CelebA LMDB has broken image-label alignment, so we train directly on CelebA-HQ.
"""
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateMonitor,
    RichProgressBar, RichModelSummary
)
from pytorch_lightning.loggers import TensorBoardLogger
from ro_optimization.evaluation.classification.lit_module import AttrClassifier
from ro_optimization.evaluation.classification.config import get_config
import torch


def setup_callbacks(cfg):
    """Setup callbacks for CelebA-HQ training."""
    callbacks = []

    # Checkpoint callback
    checkpoint_dir = os.path.join(cfg.logging.base_ckpt_dir, "celebahq_only")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='celebahq-{epoch:02d}-{val_loss:.4f}-{val_auroc:.4f}',
        monitor=cfg.logging.checkpoint.monitor,
        mode=cfg.logging.checkpoint.mode,
        save_top_k=cfg.logging.checkpoint.save_top_k,
        save_last=cfg.logging.checkpoint.save_last
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stop = EarlyStopping(
        monitor=cfg.training.early_stop.monitor,
        patience=cfg.training.early_stop.patience_fine_tune,  # Use fine-tune patience (20)
        mode=cfg.training.early_stop.mode,
        verbose=True
    )
    callbacks.append(early_stop)

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # Progress bar and model summary
    callbacks.append(RichProgressBar())
    callbacks.append(RichModelSummary(max_depth=2))

    return callbacks, checkpoint_callback


def train(cfg):
    """Train on CelebA-HQ dataset (30,000 images)."""
    print("=" * 60)
    print("Training Attribute Classifier on CelebA-HQ")
    print("=" * 60)
    print("Note: CelebA LMDB has broken alignment, using CelebA-HQ only")

    # Create model - train directly on CelebA-HQ
    model = AttrClassifier(
        # Data paths (CelebA paths not used but required for init)
        celeba_lmdb=cfg.data.celeba.lmdb_file,
        celeba_attr=cfg.data.celeba.attr_file,
        celeba_partition=cfg.data.celeba.partition_file,
        celebahq_lmdb=cfg.data.celebahq.lmdb_file,
        celebahq_attr=cfg.data.celebahq.attr_file,
        # Model params
        img_size=cfg.model.img_size,
        backbone=cfg.model.backbone,
        dropout=cfg.model.dropout,
        # Training params - use slightly higher LR since training from scratch
        lr=cfg.training.learning_rate.first_stage,  # 5e-4
        weight_decay=cfg.training.weight_decay,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.dataloader.num_workers,
        lr_scheduler='cosine',
        # Use CelebA-HQ
        stage='celebahq',
        use_augmentation=cfg.training.use_augmentation,
        cfg=cfg
    )

    # Freeze backbone - only train classifier head
    print("Freezing backbone - training classifier head only...")
    for param in model.model.net.parameters():
        param.requires_grad = False
    model.model.net.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} (classifier head only)")
    print("Backbone in eval mode (BatchNorm uses pretrained running stats)")

    # Setup callbacks
    callbacks, checkpoint_callback = setup_callbacks(cfg)

    # Logger
    log_dir = os.path.join(cfg.logging.base_log_dir, "celebahq_only_training")
    logger = TensorBoardLogger(log_dir, name='', version='')

    # Trainer - more epochs since smaller dataset
    trainer = pl.Trainer(
        max_epochs=100,  # More epochs for smaller dataset
        accelerator='gpu',
        devices=1,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=cfg.training.deterministic,
        val_check_interval=1.0  # Validate once per epoch
    )

    # Train
    trainer.fit(model)

    return checkpoint_callback.best_model_path


def main():
    # Load config
    cfg = get_config()

    # Set random seed
    pl.seed_everything(cfg.training.seed)

    # Create directories
    os.makedirs(cfg.logging.base_log_dir, exist_ok=True)
    os.makedirs(cfg.logging.base_ckpt_dir, exist_ok=True)

    # Train on CelebA-HQ
    best_checkpoint = train(cfg)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best model saved at: {best_checkpoint}")
    print("=" * 60)


if __name__ == '__main__':
    main()
