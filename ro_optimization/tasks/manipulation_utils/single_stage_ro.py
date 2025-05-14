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


def single_stage_ro(
    ae,
    cls,
    batch,
    median_logit,
    cfg: dict,
    cid: int,
    device: torch.device,
    debug: bool = False
):
    """
    Single-stage Riemannian optimization, returning the best‐per‐sample latent
    and printing per-sample diagnostics in a neat ASCII table when debug=True.
    """
    # 1) Encode & normalize
    cond         = ae.encode(batch).to(device)
    latent_shape = cond.shape[1:]
    x0_flat = flatten_tensor(cond).clone()
    x0n     = cls.normalize(x0_flat).clone()
    cfg["initial_point"] = x0n

    # 2) Pick SNR timestep
    t_val    = compute_discrete_time_from_target_snr(cfg, ae.conf)
    t_tensor = torch.full((1,), t_val, dtype=torch.float32, device=device)

    # 3) Build diffusion + score + denoiser + retraction
    diff     = ae.conf.make_latent_eval_diffusion_conf().make_sampler()
    wrap     = DiffusionWrapper(diff)
    score_fn = get_score_fn(wrap, ae.ema_model.latent_net, t_tensor, latent_shape)
    denoiser = get_denoiser_fn(wrap, ae.ema_model.latent_net, t_tensor, latent_shape)
    retr_fn  = create_retraction_fn(
        retraction_type=cfg.get("retraction_operator","identity"),
        denoiser_fn=denoiser
    )

    # 4) Build classifier‐based objective
    classifier_fn = get_classifier_fn(cls, torch.zeros_like(t_tensor), latent_shape)
    opt_kwargs = {
        "classifier_fn":     classifier_fn,
        "cls_id":            cid,
        "latent_shape":      latent_shape,
        "x0_flat":           x0n,
        "classifier_weight": cfg.get("classifier_weight",1.0),
        "reg_norm_weight":   cfg.get("reg_norm_weight",0.5),
        "reg_norm_type":     cfg.get("reg_norm_type","L2"),
    }
    if median_logit is not None:
        opt_kwargs["target_logit"] = median_logit
    opt_fn = get_opt_fn(**opt_kwargs)

    # 5) Run Riemannian‐GD and get full trajectory + processed f‐values
    riem_opt, _         = get_riemannian_optimizer(score_fn, opt_fn, cfg, retr_fn), None
    trajectory, metrics_proc = riem_opt.run()
    # - `trajectory`: list of T tensors [B, D]
    # - `metrics_proc["function_values"]`: list of B lists, each length T

    # 6) Stack trajectory → [T, B, D], build [B, T] f‐value tensor
    traj_stack = torch.stack(trajectory, dim=0).cpu()           # [T, B, D]
    fvals      = torch.tensor(metrics_proc["function_values"])  # [B, T]

    # 7) Pick best step per sample and extract that latent
    best_steps   = fvals.argmin(dim=1)      # [B]
    idx_samples  = torch.arange(best_steps.shape[0])
    best_latents = traj_stack[best_steps, idx_samples, :].to(device)  # [B, D]

    # 8) Denormalize
    z_denorm = cls.denormalize(best_latents)

    debug_outputs = None
    if debug:
        opt_dbg = get_opt_fn_debug(
            classifier_fn,
            cls_id=cid,
            latent_shape=latent_shape,
            x0_flat=x0n,
            classifier_weight=cfg.get("classifier_weight",1.0),
            reg_norm_weight=cfg.get("reg_norm_weight",0.5),
            reg_norm_type=cfg.get("reg_norm_type","L2"),
            target_logit=median_logit,
        )
        with torch.no_grad():
            total_vals, cls_vals, reg_vals = opt_dbg(best_latents)
        debug_outputs = {
            "steps": best_steps.cpu(),
            "total": total_vals,
            "cls": cls_vals,
            "reg": reg_vals,
        }

    return z_denorm, debug_outputs