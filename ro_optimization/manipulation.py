import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Utilities
from .utils import flatten_tensor, unflatten_tensor, encode_xt_in_chunks
from .diffusion_utils import (
    DiffusionWrapper,
    get_score_fn,
    get_denoiser_fn,
    get_classifier_fn,
    compute_discrete_time_from_target_snr,

)
from data_geometry.riemannian_optimization import get_riemannian_optimizer
from data_geometry.riemannian_optimization.retraction import create_retraction_fn
from .objectives import get_opt_fn, get_opt_fn_debug
from .multistage_optimization import (
    get_cross_retraction_fn,
    forward_perturb,
    reverse_jump_explicit,
    load_schedule_from_rescaled_entropy,
    linear_timesteps
)
from .config_loader import load_riemannian_config
from templates_latent import ffhq128_autoenc_latent
from templates_cls import (
    ffhq128_autoenc_non_linear_cls,
    ffhq128_autoenc_cls,
)
from experiment import LitModel
from experiment_classifier import ClsModel
from dataset import CelebAttrDataset


def load_shared_resources(config_path, device=None):
    """
    Load models, dataset, and configuration.

    Returns:
      ae          – autoencoder model
      cls_nl      – nonlinear classifier (Riemannian)
      cls_lin     – linear classifier (baseline)
      dataset     – full manipulation dataset
      pos_dataset – list of positive examples (labels == 1)
      neg_indices – list of indices of negative examples (labels == -1)
      cfg         – configuration dict
      cid         – target attribute index
      device      – torch device
    """
    # Determine device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load config and target attribute
    cfg  = load_riemannian_config(config_path)
    attr = cfg["target_attr"]

    # --- Autoencoder ---
    ae_conf = ffhq128_autoenc_latent()
    ae_conf.T_eval = 1000
    ae_conf.latent_T_eval = 1000
    ae = LitModel(ae_conf).to(device)
    ckpt = torch.load(
        os.path.join("checkpoints", ae_conf.name, "last.ckpt"),
        map_location="cpu"
    )
    ae.load_state_dict(ckpt["state_dict"], strict=False)
    ae.ema_model.eval().to(device)

    # --- Nonlinear classifier (Riemannian) ---
    cls_nl_conf = ffhq128_autoenc_non_linear_cls()
    cls_nl = ClsModel(cls_nl_conf).to(device)
    ckpt_nl = torch.load(
        os.path.join("checkpoints", cls_nl_conf.name, "last.ckpt"),
        map_location="cpu"
    )
    cls_nl.load_state_dict(ckpt_nl["state_dict"], strict=False)
    cls_nl.eval()

    # --- Linear classifier (Baseline) ---
    cls_lin_conf = ffhq128_autoenc_cls()
    cls_lin = ClsModel(cls_lin_conf).to(device)
    ckpt_lin = torch.load(
        os.path.join("checkpoints", cls_lin_conf.name, "last.ckpt"),
        map_location="cpu"
    )
    cls_lin.load_state_dict(ckpt_lin["state_dict"], strict=False)
    cls_lin.eval()

    # --- Dataset and positive/negative split (single pass) ---
    dataset     = cls_lin.load_dataset()
    cid         = CelebAttrDataset.cls_to_id[attr]
    pos_dataset = []
    neg_indices = []
    for idx, item in enumerate(dataset):
        label = item['labels'][cid]
        if label == 1:
            pos_dataset.append(item)
        elif label == -1:
            neg_indices.append(idx)

    return ae, cls_nl, cls_lin, dataset, pos_dataset, neg_indices, cfg, cid, device


def compute_median_logit(ae, classifier, cls_id, pos_dataset, cfg, device):
    """
    Compute the median logit score for positive examples.
    """
    # How many positives to use and batch size
    num_samples = cfg.get("median_samples", 1000)
    batch_size  = cfg.get("median_batch_size", 16)

    # Subset of positives
    items = pos_dataset[:num_samples] if num_samples and num_samples < len(pos_dataset) else pos_dataset
    logits = []

    # Process in mini-batches
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        imgs = torch.stack([it['img'] for it in batch_items]).to(device)

        with torch.no_grad():
            conds     = ae.encode(imgs)
            flat_z    = flatten_tensor(conds)
            norm_z    = classifier.normalize(flat_z)
            batch_log = classifier.classifier(norm_z)[:, cls_id]
            logits.extend(batch_log.cpu().tolist())

    if not logits:
        raise ValueError("No positive samples for median logit.")

    return float(np.median(logits))


def linear_manipulation(
    ae,
    cls_lin,
    batch,
    median_logit: float,
    cfg: dict,
    cid: int,
    debug: bool = False
):
    """
    Baseline linear attribute manipulation, with optional debug printing.

    Args:
      ae          : autoencoder (LitModel)
      cls_lin     : linear classifier (ClsModel)
      batch       : Tensor, shape (B,C,H,W)
      median_logit: float, the target logit to hit
      cid         : int, the class index for the attribute
      cfg         : dict, config (for reg weights/types)
      debug       : bool, if True print classification & L2 losses
    Returns:
      z_lin       : Tensor (B, latent_dim), the manipulated flattened code
    """
    # --- encode & normalize ---
    cond = ae.encode(batch)            # (B, latent_dims...)
    latent_shape = cond.shape[1:]
    z0   = flatten_tensor(cond)        # (B, D)
    z0n  = cls_lin.normalize(z0)       # (B, D)

    # --- compute shift along classifier normal ---
    w      = cls_lin.classifier.weight[cid]    # (D,)
    b      = cls_lin.classifier.bias[cid]      # ()
    w_norm = (w.pow(2)).sum()                  # ()
    proj   = z0n @ w                            # (B,)
    s      = (median_logit - b - proj) / w_norm # (B,)
    if debug:
        print(f"[Linear] shift s: {s}")

    # --- apply shift and denormalize ---
    z_lin = cls_lin.denormalize(z0n + s.unsqueeze(1) * w.unsqueeze(0))

    # --- optional debug losses ---
    if debug:
        z_lin_norm = cls_lin.normalize(z_lin)
        # build a small classifier_fn for debug
        cls_fn_debug = lambda z: cls_lin.classifier(z)
        debug_fn = get_opt_fn_debug(
            cls_fn_debug,
            cls_id=cid,
            latent_shape=latent_shape,                           # unused in debug
            x0_flat=z0n,                                  # original flattened codes
            classifier_weight=cfg.get("classifier_weight", 1.0),
            reg_norm_weight=cfg.get("reg_norm_weight", 0.5),
            reg_norm_type=cfg.get("reg_norm_type", "L2"),
            target_logit=median_logit,
        )
        total_loss, cls_loss, reg_loss = debug_fn(z_lin_norm)
        print(
            f"[Linear Edit] CLS loss: {cls_loss.mean().item():.4f}, "
            f"{cfg.get('reg_norm_type','L2')} loss={reg_loss.mean():.4f}"
        )

    return z_lin


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
                target_logit=median_logit,
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
    if debug:
        print(f"[Done] final z_riem head: {z_riem[0, :10].cpu().numpy()}")
    return z_riem

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
    Single-stage (SNR-based) Riemannian optimization, optionally targeting a median logit,
    with final diagnostic losses if debug=True.
    """
    # encode & normalize
    cond         = ae.encode(batch).to(device)
    latent_shape = cond.shape[1:]
    x0_flat      = flatten_tensor(cond)
    x0n          = cls.normalize(x0_flat)
    cfg["initial_point"] = x0n

    # pick timestep from target SNR
    t_val     = compute_discrete_time_from_target_snr(cfg, ae.conf)
    t_tensor  = torch.full((1,), t_val, dtype=torch.float32, device=device)

    # build diffusion, score, and denoiser
    diff      = ae.conf.make_latent_eval_diffusion_conf().make_sampler()
    wrap      = DiffusionWrapper(diff)
    score_fn  = get_score_fn(wrap, ae.ema_model.latent_net, t_tensor, latent_shape)
    denoiser  = get_denoiser_fn(wrap, ae.ema_model.latent_net, t_tensor, latent_shape)

    # retraction
    retr_fn   = create_retraction_fn(
        retraction_type=cfg.get("retraction_operator","identity"),
        denoiser_fn=denoiser
    )

    # classifier objective (at zero noise), with optional target_logit
    classifier_fn = get_classifier_fn(cls, torch.zeros_like(t_tensor), latent_shape)
    opt_kwargs = {
        "classifier_fn": classifier_fn,
        "cls_id":        cid,
        "latent_shape":  latent_shape,
        "x0_flat":       x0n,
        "classifier_weight": cfg.get("classifier_weight",1.0),
        "reg_norm_weight":   cfg.get("reg_norm_weight",0.5),
        "reg_norm_type":     cfg.get("reg_norm_type","L2"),
    }
    if median_logit is not None:
        opt_kwargs["target_logit"] = median_logit

    opt_fn    = get_opt_fn(**opt_kwargs)

    # run the single Riemannian optimizer step
    riem_opt  = get_riemannian_optimizer(score_fn, opt_fn, cfg, retr_fn)
    traj, _   = riem_opt.run()
    z_final   = traj[-1].to(device)
    z_denorm  = cls.denormalize(z_final)

    if debug:
        # compute final losses
        opt_dbg = get_opt_fn_debug(
            classifier_fn,
            cls_id=cid,
            latent_shape=latent_shape,
            x0_flat=x0n,
            classifier_weight=cfg.get("classifier_weight", 1.0),
            reg_norm_weight=cfg.get("reg_norm_weight", 0.5),
            reg_norm_type=cfg.get("reg_norm_type", "L2"),
            target_logit=median_logit,
        )
        with torch.no_grad():
            _, cls_loss, reg_loss = opt_dbg(z_final)
            snr_val = wrap.snr(t_tensor).mean().item()
            print(
                f"[single_stage] t={t_val} | SNR={snr_val:.2f} | "
                f"Cls loss={cls_loss.mean():.4f} | "
                f"{cfg.get('reg_norm_type','L2')} loss={reg_loss.mean():.4f}"
            )

    return z_denorm


def save_side_by_side_comparison(original, linear, riemannian, save_path):
    """
    Plot a row of [Original | Linear | Riemannian] images for each sample.
    """
    B = original.size(0)
    fig, axes = plt.subplots(B, 3, figsize=(9, 3 * B))
    if B == 1:
        axes = axes.reshape(1, 3)

    for i in range(B):
        for j, img in enumerate((original, linear, riemannian)):
            ax = axes[i, j]
            im = img[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            ax.imshow(im)
            ax.axis('off')
            if i == 0:
                ax.set_title(["Original", "Linear", "Riemannian"][j])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    """
    Entry point: loads config, runs both methods, and saves comparison.
    """
    parser = ArgumentParser()
    parser.add_argument("--ro-config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    ae, cls_nl, cls_lin, dataset, pos_dataset, neg_indices, cfg, cid, device = \
        load_shared_resources(args.ro_config)

    # Select negative examples for manipulation
    num_neg  = cfg.get("num_samples", 5)
    neg_idxs = neg_indices[:num_neg]
    batch    = torch.stack([dataset[i]['img'] for i in neg_idxs]).to(device)

    # Compute both median logits
    median_logit_lin = compute_median_logit(ae, cls_lin, cid, pos_dataset, cfg, device)
    median_logit_nl  = compute_median_logit(ae, cls_nl,  cid, pos_dataset, cfg, device)

    print(f"Median logit (linear): {median_logit_lin:.4f}, "
          f"(non-linear): {median_logit_nl:.4f}")
    
    # Run both manipulation methods
    manipulated_linear  = linear_manipulation(
        ae, cls_lin, batch, median_logit_lin, cfg, cid, debug=True
    )

    ro_type = cfg.get("ro_type", "multistage")
    if ro_type == 'single-stage':
        manipulated_riemannian = single_stage_ro(
            ae, cls_nl, batch, median_logit_nl, cfg, cid, device, debug=True
        )
    elif ro_type == 'multi-stage':
        manipulated_riemannian = multiple_stage_ro(
        ae, cls_nl, batch, median_logit_nl, cfg, cid, device, debug=True
    )

    # Decode and render final images
    latent_shape = ae.encode(batch).shape[1:]
    T_render     = cfg.get("T_render", 250)
    chunk        = cfg.get("chunk", 25)
    xT           = encode_xt_in_chunks(ae, batch, ae.encode(batch), T_render, chunk)

    orig    = (batch * 0.5) + 0.5
    lin_out = ae.render(xT, unflatten_tensor(manipulated_linear, latent_shape), T_render)
    riem_out= ae.render(xT, unflatten_tensor(manipulated_riemannian, latent_shape), T_render)

    # Save comparison plot
    out_dir  = os.path.join(cfg.get("log_dir", "logs"), cfg.get("target_attr"))
    save_path= os.path.join(out_dir, "comparison.png")
    save_side_by_side_comparison(orig, lin_out, riem_out, save_path)


if __name__ == "__main__":
    main()
