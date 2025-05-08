CONFIG = {
    # Random seed for reproducibility
    "random_seed": 42,

    #Multistage parameters
    "multistage_steps": 8,
    "start_diffusion_timestep": 20,
    "jump_method": 'ddpm',


    # Riemannian optimization parameters
    "ro_SNR": 124, #SNR at which Riemannian optimization takes place
    "reg_lambda": 1e-5,
    "riemannian_steps": 3,
    "riemannian_lr_init": 5e-3, #5e-3,
    
    # Optimizer selection:
    "optimizer_type": "gradient_descent",  # Choices:["gradient_descent", "trust_region"]

    # Optimization function
    "classifier_weight": 1.,
    "reg_norm_weight": 0.35, 
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
