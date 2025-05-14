CONFIG = {
    # Random seed for reproducibility
    "random_seed": 42,

    #general parameters
    "num_samples": 9, #number of negative samples for manipulation.
    "median_samples": 1000, #num of samples used for calculation of the median logit of the target attribute.
    "median_batch_size": 128, #batchsize for calculation of median target logit
    
    #General RO parameters parameters
    'ro_type': 'multi-stage',
    "target_attr": 'Eyeglasses',
    "multistage_steps": 16,
    "start_diffusion_timestep": 15,
    "jump_method": 'ddpm',
    "schedule": 'linear',

    # Riemannian optimization parameters
    "ro_SNR": 124, #124, #SNR at which Riemannian optimization takes place
    "reg_lambda": 1e-5,
    "riemannian_steps": 2,
    "riemannian_lr_init": 5e-3, #5e-3,
    
    # Optimizer selection:
    "optimizer_type": "gradient_descent",  # Choices:["gradient_descent", "trust_region"]

    # Optimization function
    "classifier_weight": 1.,
    "reg_norm_weight": 0.2, 
    "reg_norm_type": "L2",

    # Trust-region parameters
    "trust_region_delta0": 0.1,
    "trust_region_eta_success": 0.75,
    "trust_region_eta_fail": 0.25,
    "trust_region_gamma_inc": 2.0,
    "trust_region_gamma_dec": 0.5,

    # Line search parameters (used by gradient descent branch)
    "line_search": "strong_wolfe",
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.7,
    "max_bracket": 12,
    "max_zoom": 10,
    "max_alpha": 300,
    "armijo_rho": 1e-6,
    "armijo_beta": 0.1,

    # Retraction operator options: "identity" or "denoiser"
    "retraction_operator": "denoiser",

    # Momentum settings (if used in gradient descent)
    "use_momentum": False,  # (set to False for now)
    "momentum_coeff": 0.6,

    # Settings for fast calculation of Riemannian gradient via CG
    "cg_preconditioner": 'diagonal',
    "cg_precond_diag_samples": 10, 
    "cg_tol": 1e-6, 
    "cg_max_iter": 15, #20

    #Settings for computing the conditioning number of the metric
    "estimate_cond": False,   # <-- toggle condition‐number estimation

    # Logging
    "log_dir": "ro_optimization/ro_results/multistage_optimization",
    "plot_filename": "combined_plot.png",
}
