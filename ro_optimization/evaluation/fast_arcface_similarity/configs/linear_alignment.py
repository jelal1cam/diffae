# ro_optimization/evaluation/configs/linear_alignment_config.py

from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()
    
    # Data paths
    config.paired_latents_dir = "datasets/paired_ArcFace_latents"
    
    # Model architecture - Simple linear transformation
    config.architecture = "mlp"
    config.hidden_dims = []  # No hidden layers - just a linear transformation
    config.activation = "none"  # No activation needed
    config.dropout = 0.0  # No dropout for linear model
    config.residual_connections = False  # Since input_dim == output_dim = 512
    
    # Input/output dimensions
    config.custom_latent_dim = 512
    config.arcface_latent_dim = 512
    
    # Training parameters - Adjusted for simpler model
    config.batch_size = 512  # Larger batch size for linear model
    config.learning_rate = 5e-3  # Higher LR for linear model
    config.weight_decay = 1e-6  # Lower weight decay
    config.ema_decay = 0.995  # Slightly lower EMA for faster adaptation
    config.max_epochs = 400  # Fewer epochs needed for linear model
    config.gradient_clip_val = 5.0  # Higher clip value
    
    # Optimizer
    config.optimizer = "adamw"  # Adam works well for linear models
    config.scheduler = "none"  # Cosine annealing
    config.warmup_steps = 500  # Shorter warmup
    
    # Loss function
    config.loss_type = "cosine"  # Cosine similarity loss
    config.mse_weight = 0.01  # Not used with cosine loss
    
    # Logging and checkpointing
    config.log_dir = "checkpoints/alignment_linear/logs"
    config.checkpoint_dir = "checkpoints/alignment_linear"
    config.log_every_n_steps = 50
    config.val_check_interval = 0.5  # Check validation twice per epoch
    config.save_top_k = 3
    config.monitor_metric = "val_loss"
    
    # Hardware
    config.num_workers = 4
    config.accelerator = "gpu"
    config.devices = 1
    config.precision = 32
    
    # Seed
    config.seed = 42
    
    return config