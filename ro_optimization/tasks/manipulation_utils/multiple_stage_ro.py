import copy
import torch 
from ..utils import flatten_tensor
from ...diffusion_utils import (
    DiffusionWrapper,
    get_score_fn,
    get_denoiser_fn,
    get_classifier_fn,
    compute_discrete_time_from_target_snr,

)
from ...objectives import get_opt_fn, get_opt_fn_debug
from data_geometry.riemannian_optimization import get_riemannian_optimizer
from data_geometry.riemannian_optimization.retraction import create_retraction_fn

from ..multistage_optimization import (
    get_cross_retraction_fn,
    forward_perturb,
    reverse_jump_explicit,
    load_schedule_from_rescaled_entropy,
    linear_timesteps
)

def multiple_stage_ro(
    ae,
    cls_nl,
    batch,
    median_logit: float,
    cfg: dict,
    cid: int,
    device: torch.device,
    debug: bool = False
):
    """
    Multi‐stage Riemannian attribute manipulation, now with optional debug printing
    of per‐stage SNR, classification loss, and regularization loss.

    Args:
      ae           : autoencoder (LitModel)
      cls_nl       : nonlinear (Riemannian) classifier (ClsModel)
      batch        : Tensor, shape (B,C,H,W)
      median_logit : float, the target logit to hit at each stage
      cfg          : dict, config parameters
      cid          : int, class index for the attribute
      device       : torch.device
      debug        : bool, if True print diagnostic losses each stage

    Returns:
      z_riem       : Tensor (B, latent_dim), the manipulated flattened code
    """
    # --- Initial encode & normalization ---
    cond         = ae.encode(batch).to(device)
    latent_shape = cond.shape[1:]
    x0           = flatten_tensor(cond)
    x0n          = cls_nl.normalize(x0)
    x            = x0n.clone()

    if debug:
        print(f"[Init] batch size={batch.size(0)}, latent dim={x0.shape[1]}")
        print(f"[Init] x0 head (first 10 dims): {x0[0, :10].cpu().numpy()}")

    # --- Diffusion sampler & wrapper ---
    diff      = ae.conf.make_latent_eval_diffusion_conf().make_sampler()
    wrap      = DiffusionWrapper(diff)
    alpha_fn  = wrap.get_alpha_fn()
    sigma_fn  = wrap.get_sigma_fn()

    # --- Riemannian retractor factory ---
    retractor = get_cross_retraction_fn(alpha_fn, sigma_fn, wrap, ae.ema_model, latent_shape)

    # --- Build schedule ---
    t_val      = compute_discrete_time_from_target_snr(cfg, ae.conf)
    start_t    = cfg["start_diffusion_timestep"]
    num_stages = cfg.get("multistage_steps")
    if cfg.get("schedule", "linear") == "rescaled_entropy":
        stages = load_schedule_from_rescaled_entropy(cfg)
    else:
        stages = linear_timesteps(start_t, t_val, num_stages)[::-1]

    if debug:
        print(f"[Schedule] t_val={t_val}")
        print(f"[Schedule] stages={stages}")

    steps        = cfg.get("riemannian_steps")
    reverse_flag = cfg.get("do_reverse_jump", False)

    # --- Loop over stages ---
    for i, t in enumerate(stages):
        # forward perturb at first stage
        if i == 0:
            t0 = torch.full((batch.size(0),), t, dtype=torch.float32, device=device)
            x  = forward_perturb(x, t0, alpha_fn, sigma_fn)
            if debug:
                print(f"[Stage {i}] After forward perturb, x head: {x[0, :10].cpu().numpy()}")

        # prepare t_tensor
        t_tensor = torch.full((1,), t, dtype=torch.float32, device=device)

        # build classifier‐ and score‐based sub‐functions
        cls_fn    = get_classifier_fn(cls_nl, t_tensor, latent_shape)
        opt_fn    = get_opt_fn(
            cls_fn,
            cls_id=cid,
            latent_shape=latent_shape,
            x0_flat=x0n,
            classifier_weight=cfg.get("classifier_weight", 1.0),
            reg_norm_weight=cfg.get("reg_norm_weight", 0.5),
            reg_norm_type=cfg.get("reg_norm_type", "L2"),
            target_logit=median_logit,
        )
        score_fn  = get_score_fn(wrap, ae.ema_model.latent_net, t_tensor, latent_shape)
        retract_fn= retractor(t, t + 1, batch.size(0), device)

        # run Riemannian GD 
        cfg_stage                   = copy.deepcopy(cfg)
        cfg_stage["initial_point"]  = x.detach().clone().requires_grad_(True).to(device)
        cfg_stage["riemannian_steps"]= steps
        riem_opt = get_riemannian_optimizer(score_fn, opt_fn, cfg_stage, retract_fn)
        traj, _  = riem_opt.run()
        x        = traj[-1]

        # diagnostic losses
        if debug:
            opt_dbg = get_opt_fn_debug(
                cls_fn,
                cls_id=cid,
                latent_shape=latent_shape,
                x0_flat=x0n,
                classifier_weight=cfg.get("classifier_weight", 1.0),
                reg_norm_weight=cfg.get("reg_norm_weight", 0.5),
                reg_norm_type=cfg.get("reg_norm_type", "L2"),
                target_logit=None,
            )
            with torch.no_grad():
                _, cls_loss, reg_loss = opt_dbg(x.to(device))
                snr = wrap.snr(t_tensor).mean().item()
                print(
                    f"[Stage {i}] t={t} | SNR={snr:.2f} | "
                    f"Cls loss={cls_loss.mean():.4f} | "
                    f"{cfg.get('reg_norm_type','L2')} loss={reg_loss.mean():.4f}"
                )

        # optional reverse jump
        if reverse_flag and i < len(stages) - 1:
            next_t = stages[i + 1]
            x = reverse_jump_explicit(
                x, t, next_t, alpha_fn, sigma_fn,
                ae.ema_model.latent_net, latent_shape
            )
            if debug:
                print(f"[Stage {i}] After reverse jump, x head: {x[0, :10].cpu().numpy()}")

    # --- Denormalize and return ---
    z_riem = cls_nl.denormalize(x.to(device))

    debug_outputs = None
    if debug:
        opt_dbg = get_opt_fn_debug(
            cls_fn,
            cls_id=cid,
            latent_shape=latent_shape,
            x0_flat=x0n,
            classifier_weight=cfg.get("classifier_weight",1.0),
            reg_norm_weight=cfg.get("reg_norm_weight",0.5),
            reg_norm_type=cfg.get("reg_norm_type","L2"),
            target_logit=None,
        )
        with torch.no_grad():
            total_vals, cls_vals, reg_vals = opt_dbg(z_riem)

        debug_outputs = {
            "total": total_vals,
            "cls": cls_vals,
            "reg": reg_vals,
        }

    return z_riem, debug_outputs