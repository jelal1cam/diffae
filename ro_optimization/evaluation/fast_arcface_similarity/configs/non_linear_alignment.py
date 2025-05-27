# ro_optimization/evaluation/configs/skip_mlp_alignment_config.py

from ml_collections import ConfigDict

def get_config():
    cfg = ConfigDict()

    # ------------------------------------------------------------------ data
    cfg.paired_latents_dir = "datasets/paired_ArcFace_latents"

    # ----------------------------------------------------------------- model
    cfg.architecture   = "mlp"        # <- triggers MLPSkipAlignNet
    cfg.custom_latent_dim = 512
    cfg.arcface_latent_dim = 512

    # MLP hyper-parameters (good starting point for 512->512 mapping)
    cfg.hidden_dims    = [1024] #2048 is the best      # Base hidden size (see build_alignment_net)
    cfg.num_layers     = 8
    cfg.skip_layers    = list(range(1, cfg.num_layers))
    cfg.activation     = "silu"
    cfg.last_act       = "none"
    cfg.use_norm       = True
    cfg.dropout        = 0.25
    cfg.residual_connections = False  # Not used but kept for parity
    cfg.output_norm = "l2"

    # ---------------------------------------------------------------- training
    cfg.batch_size         = 512          # Lower than linear (model larger)
    cfg.learning_rate      = 5e-4         # Typical for 1-2 M param MLP
    cfg.weight_decay       = 1e-4
    cfg.ema_decay          = 0.999
    cfg.max_epochs         = 800          # Give it time to converge
    cfg.gradient_clip_val  = 1.0

    cfg.optimizer  = "adamw"
    cfg.scheduler  = "cosine_warmup"
    cfg.warmup_steps = 2000

    # ---------------------------------------------------------------- loss
    cfg.loss_type   = "mse"  # cosine + small MSE
    cfg.mse_weight  = 0.05

    # ------------------------------------------------------------ logging
    cfg.log_dir          = "checkpoints/alignment_skipmlp/logs"
    cfg.checkpoint_dir   = "checkpoints/alignment_skipmlp"
    cfg.log_every_n_steps = 50
    cfg.val_check_interval = None
    cfg.save_top_k       = 3
    cfg.monitor_metric   = "val_loss"

    # ------------------------------------------------------------- hardware
    cfg.num_workers  = 8
    cfg.accelerator  = "gpu"
    cfg.devices      = 1          # or >1 if you run DDP
    cfg.precision    = '16-mixed'         # fp16 w/ EMA is safe

    # ------------------------------------------------------------- misc
    cfg.seed = 42

    return cfg
