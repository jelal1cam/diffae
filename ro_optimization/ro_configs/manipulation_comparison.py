CONFIG = {
    # Random seed for reproducibility
    "random_seed": 42,
    "verbose": False,

    #general parameters
    "target_attr": 'Male', #Attibute for Manipulation
    "median_samples": 2000, #num of samples used for calculation of the median logit of the target attribute.
    "median_batch_size": 128, #batchsize for calculation of median target logit

    "num_samples": 50, #50, #number of negative samples for manipulation.
    'ro_type': 'multi-stage', #Riemannian Optimization scheme.

    #Multistage RO parameters
    'num_ro_seeds': 3,
    "multistage_steps": 11,
    "start_diffusion_timestep": 20,
    "jump_method": 'ddpm',
    "schedule": 'rescaled_entropy',

    # RO parameters
    "ro_SNR": 124, #SNR at which Riemannian optimization takes place
    "reg_lambda": 1e-5, #regularizes the Riemannian metric
    "riemannian_steps": 2,
    "riemannian_lr_init": 1e-2, #5e-3,
    
    # ROptimizer selection:
    "optimizer_type": "gradient_descent",  # Choices:["gradient_descent", "trust_region"]

    # Optimization function
    "classifier_weight": 1.,
    "reg_norm_weight": 0.9, #['Eyeglasses':0.45, 'Similing':0.8]
    "reg_norm_type": "L2",

    # Line search parameters (used by gradient descent branch)
    "line_search": "strong_wolfe",
    "wolfe_c1": 5e-3,
    "wolfe_c2": 0.45,
    "max_bracket": 13,
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
    "cg_max_iter": 13, #20

    #Settings for computing the conditioning number of the metric
    "estimate_cond": False,   # <-- toggle condition‐number estimation

    # Logging
    "log_dir": "ro_optimization/ro_results/multistage_optimization",
    "plot_filename": "combined_plot.png",
}
