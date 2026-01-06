# train.py
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


def setup_callbacks(cfg, stage="first_stage"):
    """Setup callbacks based on config and stage."""
    callbacks = []
    
    # Get stage-specific config
    if stage == "first_stage":
        stage_cfg = cfg.logging.first_stage
        patience = cfg.training.early_stop.patience_first_stage
    else:
        stage_cfg = cfg.logging.fine_tune
        patience = cfg.training.early_stop.patience_fine_tune
    
    # Checkpoint callback
    checkpoint_dir = os.path.join(cfg.logging.base_ckpt_dir, stage_cfg.ckpt_subdir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{stage}-{{epoch:02d}}-{{val_loss:.4f}}-{{val_auroc:.4f}}',
        monitor=cfg.logging.checkpoint.monitor,
        mode=cfg.logging.checkpoint.mode,
        save_top_k=cfg.logging.checkpoint.save_top_k,
        save_last=cfg.logging.checkpoint.save_last
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor=cfg.training.early_stop.monitor,
        patience=patience,
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


def train_stage1(cfg):
    """Train on full CelebA dataset."""
    print("="*60)
    print("Stage 1: Training on CelebA Dataset")
    print("="*60)
    
    # Create model
    model = AttrClassifier(
        # Data paths
        celeba_lmdb=cfg.data.celeba.lmdb_file,
        celeba_attr=cfg.data.celeba.attr_file,
        celeba_partition=cfg.data.celeba.partition_file,
        celebahq_lmdb=cfg.data.celebahq.lmdb_file,
        celebahq_attr=cfg.data.celebahq.attr_file,
        # Model params
        img_size=cfg.model.img_size,
        backbone=cfg.model.backbone,
        dropout=cfg.model.dropout,
        # Training params
        lr=cfg.training.learning_rate.first_stage,
        weight_decay=cfg.training.weight_decay,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.dataloader.num_workers,
        lr_scheduler=cfg.optim.scheduler.first_stage,
        # Other params
        stage='celeba',
        use_augmentation=cfg.training.use_augmentation,
        # Pass full config for other settings
        cfg=cfg
    )

    # CRITICAL FIX: Freeze backbone to prevent feature destruction
    # The pretrained ResNet features are being destroyed by high LR before classifier learns
    print("Freezing backbone - training classifier head only...")
    for param in model.model.net.parameters():
        param.requires_grad = False
    # CRITICAL: Put backbone in eval mode so BatchNorm uses pretrained statistics
    model.model.net.eval()
    # Count trainable params (should be ~82K for classifier only)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} (classifier head only)")
    print("Backbone in eval mode (BatchNorm uses pretrained running stats)")
    
    # Setup callbacks
    callbacks, checkpoint_callback = setup_callbacks(cfg, "first_stage")
    
    # Logger
    log_dir = os.path.join(cfg.logging.base_log_dir, cfg.logging.first_stage.log_subdir)
    logger = TensorBoardLogger(log_dir, name='', version='')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs.first_stage,
        accelerator='gpu',
        devices=1,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.logging.first_stage.log_every_n_steps,
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=cfg.training.deterministic,
        val_check_interval=cfg.logging.first_stage.val_check_interval
    )
    
    # Train
    trainer.fit(model)
    
    return checkpoint_callback.best_model_path


def train_stage2(cfg, pretrained_checkpoint):
    """Fine-tune on CelebA-HQ dataset."""
    print("\n" + "="*60)
    print("Stage 2: Fine-tuning on CelebA-HQ Dataset")
    print("="*60)
    
    # Load pretrained model with updated parameters
    # Note: weights_only=False needed for PyTorch 2.6+ due to ConfigDict in checkpoint
    import torch
    torch.serialization.add_safe_globals([type(cfg)])  # Allow ConfigDict

    model = AttrClassifier.load_from_checkpoint(
        pretrained_checkpoint,
        # Update paths and settings for CelebA-HQ
        celebahq_lmdb=cfg.data.celebahq.lmdb_file,
        celebahq_attr=cfg.data.celebahq.attr_file,
        stage='celebahq',
        lr=cfg.training.learning_rate.fine_tune,
        lr_scheduler=cfg.optim.scheduler.fine_tune,
        use_augmentation=cfg.training.use_augmentation,
        cfg=cfg
    )
    
    # Optional: Freeze early layers for fine-tuning
    # Uncomment to freeze first two layers
    # for name, param in model.model.net.named_parameters():
    #     if 'layer1' in name or 'layer2' in name:
    #         param.requires_grad = False
    
    # Setup callbacks
    callbacks, checkpoint_callback = setup_callbacks(cfg, "fine_tune")
    
    # Logger
    log_dir = os.path.join(cfg.logging.base_log_dir, cfg.logging.fine_tune.log_subdir)
    logger = TensorBoardLogger(log_dir, name='', version='')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs.fine_tune,
        accelerator='gpu',
        devices=1,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.logging.fine_tune.log_every_n_steps,
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=cfg.training.deterministic,
        val_check_interval=cfg.logging.fine_tune.val_check_interval
    )
    
    # Fine-tune
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
    
    # Stage 1: Train on CelebA
    best_celeba_checkpoint = train_stage1(cfg)
    print(f"\nBest CelebA checkpoint: {best_celeba_checkpoint}")
    
    # Stage 2: Fine-tune on CelebA-HQ
    best_celebahq_checkpoint = train_stage2(cfg, best_celeba_checkpoint)
    print(f"\nBest CelebA-HQ checkpoint: {best_celebahq_checkpoint}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final model saved at: {best_celebahq_checkpoint}")
    print("="*60)


if __name__ == '__main__':
    main()