# sweep_config.py (example for running hyperparameter sweeps)
from ml_collections import ConfigDict
from config import get_config


def get_sweep_config():
    """Example sweep configuration for hyperparameter tuning."""
    base_config = get_config()
    
    sweep_config = ConfigDict()
    
    # Define parameter ranges to sweep
    sweep_config.parameters = ConfigDict()
    
    # Model architecture sweeps
    sweep_config.parameters.backbone = {
        'values': ['resnet18', 'resnet50', 'efficientnet_b0']
    }
    
    sweep_config.parameters.dropout = {
        'distribution': 'uniform',
        'min': 0.2,
        'max': 0.7
    }
    
    # Training hyperparameter sweeps
    sweep_config.parameters.learning_rate = {
        'distribution': 'log_uniform',
        'min': 1e-5,
        'max': 1e-3
    }
    
    sweep_config.parameters.batch_size = {
        'values': [64, 128, 256]
    }
    
    sweep_config.parameters.weight_decay = {
        'distribution': 'log_uniform',
        'min': 1e-6,
        'max': 1e-4
    }
    
    # Optimizer sweeps
    sweep_config.parameters.optimizer = {
        'values': ['adam', 'adamw']
    }
    
    sweep_config.parameters.scheduler = {
        'values': ['cosine', 'plateau', 'none']
    }
    
    # Fixed parameters
    sweep_config.method = 'bayes'  # or 'grid', 'random'
    sweep_config.metric = ConfigDict()
    sweep_config.metric.name = 'val_auroc'
    sweep_config.metric.goal = 'maximize'
    
    return sweep_config, base_config


# run_sweep.py (example script for running sweeps with wandb)
import wandb
import pytorch_lightning as pl
from train import train_stage1, train_stage2
from config import get_config
from sweep_config import get_sweep_config
import os


def train_with_config():
    """Training function that uses wandb config."""
    # Initialize wandb
    run = wandb.init()
    
    # Get base config
    cfg = get_config()
    
    # Override with sweep parameters
    if hasattr(wandb.config, 'backbone'):
        cfg.model.backbone = wandb.config.backbone
    if hasattr(wandb.config, 'dropout'):
        cfg.model.dropout = wandb.config.dropout
    if hasattr(wandb.config, 'learning_rate'):
        cfg.training.learning_rate.first_stage = wandb.config.learning_rate
        cfg.training.learning_rate.fine_tune = wandb.config.learning_rate * 0.2
    if hasattr(wandb.config, 'batch_size'):
        cfg.training.batch_size = wandb.config.batch_size
    if hasattr(wandb.config, 'weight_decay'):
        cfg.training.weight_decay = wandb.config.weight_decay
    if hasattr(wandb.config, 'optimizer'):
        cfg.optim.optimizer = wandb.config.optimizer
    if hasattr(wandb.config, 'scheduler'):
        cfg.optim.scheduler.first_stage = wandb.config.scheduler
        cfg.optim.scheduler.fine_tune = wandb.config.scheduler
    
    # Set seed
    pl.seed_everything(cfg.training.seed)
    
    # Create directories
    os.makedirs(cfg.logging.base_log_dir, exist_ok=True)
    os.makedirs(cfg.logging.base_ckpt_dir, exist_ok=True)
    
    try:
        # Train stage 1
        best_celeba_checkpoint = train_stage1(cfg)
        
        # Train stage 2
        best_celebahq_checkpoint = train_stage2(cfg, best_celeba_checkpoint)
        
        # Log final metric
        # You might want to load the best model and evaluate it here
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        wandb.log({'error': str(e)})
        raise


def main():
    # Get sweep configuration
    sweep_cfg, base_cfg = get_sweep_config()
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_cfg.to_dict(),
        project="celeba-attribute-classifier"
    )
    
    # Run sweep
    wandb.agent(sweep_id, function=train_with_config, count=20)  # Run 20 experiments


if __name__ == '__main__':
    main()