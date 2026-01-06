# ro_optimization/evaluation/configs/contrastive_alignment_config.py

from ml_collections import ConfigDict

def get_config():
    cfg = ConfigDict()
    
    # ===================================================================
    # Data Configuration
    # ===================================================================
    cfg.dataset_names = ["celebahq"]  # CelebA-HQ only (add "ffhqlmdb" if FFHQ available)
    cfg.img_size = 128  # Should match your encoder's expected input size
    cfg.encoder_config = "ffhq128"  # "ffhq128" or "ffhq256"
    cfg.augmentation_strength = 1.0  # Scale factor for augmentation intensity
    
    # ===================================================================
    # Model Architecture
    # ===================================================================
    cfg.architecture = "mlp"  # Currently only "mlp" with skip connections
    cfg.input_dim = 512  # Custom encoder latent dimension
    cfg.output_dim = 128  # Output dimension for distance space (can be smaller)
    cfg.hidden_dim = 1024  # Hidden layer dimension
    cfg.num_layers = 6  # Number of layers
    cfg.skip_layers = []  # No skip connections - avoids dimension mismatch bug
    cfg.activation = "silu"  # Activation function
    cfg.last_act = "none"  # No activation before normalization
    cfg.use_norm = True  # Layer normalization
    cfg.dropout = 0.1  # Dropout rate (lower for contrastive learning)
    
    # ===================================================================
    # Contrastive Learning Configuration
    # ===================================================================
    cfg.temperature = 0.07  # Temperature for InfoNCE loss (SimCLR default)
    cfg.ema_decay = 0.999  # EMA decay rate
    
    # ===================================================================
    # Training Configuration
    # ===================================================================
    cfg.batch_size = 256  # Larger batch size is better for contrastive learning
    cfg.learning_rate = 3e-4  # Base learning rate
    cfg.min_lr = 1e-6  # Minimum learning rate for cosine schedule
    cfg.weight_decay = 1e-4  # Weight decay
    cfg.max_epochs = 200  # Maximum epochs
    cfg.warmup_epochs = 10  # Warmup epochs
    cfg.gradient_clip_val = 1.0  # Gradient clipping
    
    # Optimizer and scheduler
    cfg.optimizer = "adamw"  # Options: "adam", "adamw", "lars"
    cfg.scheduler = "cosine_warmup"  # Options: "cosine", "cosine_warmup", "none"
    cfg.scheduler_min_lr = 0.0  # Minimum LR ratio for cosine schedule
    
    # ===================================================================
    # Logging and Checkpointing
    # ===================================================================
    cfg.log_dir = "checkpoints/contrastive_alignment/logs"
    cfg.checkpoint_dir = "checkpoints/contrastive_alignment"
    cfg.log_every_n_steps = 50
    cfg.val_check_interval = 1.0  # Validate every epoch
    cfg.save_top_k = 3  # Save top 3 checkpoints
    cfg.monitor_metric = "val_loss_ema"  # Metric to monitor
    cfg.patience = 20  # Early stopping patience
    
    # ===================================================================
    # Hardware Configuration
    # ===================================================================
    cfg.num_workers = 0  # Must be 0 - dataset encodes on CUDA in __getitem__
    cfg.accelerator = "gpu"  # Accelerator type
    cfg.devices = 1  # Number of GPUs
    cfg.device = 0  # GPU device index for encoder
    cfg.precision = "16-mixed"  # Mixed precision training
    
    # ===================================================================
    # Miscellaneous
    # ===================================================================
    cfg.seed = 42  # Random seed
    
    return cfg


def get_config_large_batch():
    """Configuration with larger batch size for better contrastive learning."""
    cfg = get_config()
    
    # Larger batch size (if you have enough GPU memory)
    cfg.batch_size = 512
    cfg.learning_rate = 5e-4  # Scale learning rate with batch size
    cfg.warmup_epochs = 5
    
    return cfg


def get_config_strong_augmentation():
    """Configuration with stronger augmentations for more robust learning."""
    cfg = get_config()
    
    # Stronger augmentations
    cfg.augmentation_strength = 1.5
    cfg.temperature = 0.1  # Slightly higher temperature
    
    return cfg


def get_config_deeper_network():
    """Configuration with deeper network architecture."""
    cfg = get_config()
    
    # Deeper architecture
    cfg.num_layers = 12
    cfg.hidden_dim = 3072
    cfg.skip_layers = list(range(2, cfg.num_layers, 2))  # Skip every 2 layers
    cfg.dropout = 0.2  # More dropout for deeper network
    
    # Adjust training
    cfg.max_epochs = 300
    cfg.learning_rate = 2e-4
    
    return cfg


def get_config_compact():
    """Configuration for a more compact model with faster training."""
    cfg = get_config()
    
    # Smaller architecture
    cfg.output_dim = 64  # Smaller output dimension
    cfg.hidden_dim = 1024
    cfg.num_layers = 6
    cfg.skip_layers = [2, 4]
    
    # Faster training
    cfg.batch_size = 128
    cfg.max_epochs = 100
    
    return cfg