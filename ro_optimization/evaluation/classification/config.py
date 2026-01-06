# config.py
from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()
    
    # ----------------------------------------------------------------
    # Data paths
    # ----------------------------------------------------------------
    cfg.data = ConfigDict()
    
    # CelebA paths (relative to repo root - run from ~/diffae on server)
    cfg.data.celeba = ConfigDict()
    cfg.data.celeba.lmdb_file = "datasets/celeba.lmdb"
    cfg.data.celeba.attr_file = "datasets/celeba_anno/list_attr_celeba.txt"
    cfg.data.celeba.partition_file = "datasets/celeba_anno/list_eval_partition.txt"

    # CelebA-HQ paths (relative to repo root)
    cfg.data.celebahq = ConfigDict()
    cfg.data.celebahq.lmdb_file = "datasets/celebahq256.lmdb"
    cfg.data.celebahq.attr_file = "datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt"
    
    # ----------------------------------------------------------------
    # Model architecture
    # ----------------------------------------------------------------
    cfg.model = ConfigDict()
    cfg.model.backbone = "resnet50"  # resnet18, resnet50, resnet101, efficientnet_b0, efficientnet_b1
    cfg.model.pretrained = True
    cfg.model.img_size = 128
    cfg.model.num_classes = 40
    cfg.model.dropout = 0.5
    
    # ----------------------------------------------------------------
    # Training hyperparameters
    # ----------------------------------------------------------------
    cfg.training = ConfigDict()
    cfg.training.batch_size = 128
    cfg.training.learning_rate = ConfigDict()
    cfg.training.learning_rate.first_stage = 5e-4
    cfg.training.learning_rate.fine_tune = 1e-4
    cfg.training.weight_decay = 1e-5
    cfg.training.precision = 16  # mixed precision
    cfg.training.gradient_clip_val = 1.0
    cfg.training.seed = 42
    cfg.training.deterministic = True
    
    # Two-stage training schedule
    cfg.training.max_epochs = ConfigDict()
    cfg.training.max_epochs.first_stage = 50
    cfg.training.max_epochs.fine_tune = 30
    
    # Early stopping
    cfg.training.early_stop = ConfigDict()
    cfg.training.early_stop.patience_first_stage = 15
    cfg.training.early_stop.patience_fine_tune = 20
    cfg.training.early_stop.monitor = "val_auroc"
    cfg.training.early_stop.mode = "max"
    
    # Data augmentation
    cfg.training.use_augmentation = True
    
    # ----------------------------------------------------------------
    # Logging & checkpointing
    # ----------------------------------------------------------------
    cfg.logging = ConfigDict()
    cfg.logging.base_log_dir = "checkpoints/attribute_classifier"
    cfg.logging.base_ckpt_dir = "checkpoints/attribute_classifier"
    
    # First stage (CelebA)
    cfg.logging.first_stage = ConfigDict()
    cfg.logging.first_stage.log_subdir = "celeba_training"
    cfg.logging.first_stage.ckpt_subdir = "celeba"
    cfg.logging.first_stage.log_every_n_steps = 50
    cfg.logging.first_stage.val_check_interval = 0.5  # twice per epoch
    
    # Fine-tuning stage (CelebA-HQ)
    cfg.logging.fine_tune = ConfigDict()
    cfg.logging.fine_tune.log_subdir = "celebahq_finetuning"
    cfg.logging.fine_tune.ckpt_subdir = "celebahq"
    cfg.logging.fine_tune.log_every_n_steps = 20
    cfg.logging.fine_tune.val_check_interval = 1.0  # once per epoch
    
    # Checkpoint settings
    cfg.logging.checkpoint = ConfigDict()
    cfg.logging.checkpoint.save_top_k = 3
    cfg.logging.checkpoint.save_last = True
    cfg.logging.checkpoint.monitor = "val_auroc"
    cfg.logging.checkpoint.mode = "max"
    
    # ----------------------------------------------------------------
    # Optimizer / Scheduler
    # ----------------------------------------------------------------
    cfg.optim = ConfigDict()
    cfg.optim.optimizer = "adamw"  # adam or adamw
    cfg.optim.scheduler = ConfigDict()
    cfg.optim.scheduler.first_stage = "cosine"  # cosine, plateau, none
    cfg.optim.scheduler.fine_tune = "plateau"
    cfg.optim.scheduler.cosine_eta_min = 1e-6
    cfg.optim.scheduler.plateau_factor = 0.5
    cfg.optim.scheduler.plateau_patience = 5
    
    # ----------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------
    cfg.loss = ConfigDict()
    cfg.loss.type = "bce_logits"  # binary cross-entropy with logits
    cfg.loss.pos_weight_cap = 10.0  # clamp pos_weight to avoid extremes
    
    # ----------------------------------------------------------------
    # DataLoader settings
    # ----------------------------------------------------------------
    cfg.dataloader = ConfigDict()
    cfg.dataloader.num_workers = 0  # Must be 0 - LMDB causes segfaults with multiprocessing
    cfg.dataloader.pin_memory = True
    cfg.dataloader.persistent_workers = False  # Can't use with num_workers=0
    cfg.dataloader.drop_last = True  # for training
    
    # ----------------------------------------------------------------
    # CelebA-HQ specific settings
    # ----------------------------------------------------------------
    cfg.celebahq = ConfigDict()
    cfg.celebahq.train_split_ratio = 0.9  # 90% train, 10% val
    
    return cfg