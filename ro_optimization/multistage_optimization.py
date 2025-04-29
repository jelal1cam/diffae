#!/usr/bin/env python
"""
Multi-stage Riemannian optimization on autoencoder latent space
with forward perturbation, explicit reverse jumps (DDIM/DDPM),
manifold retractions, and automatic stage spacing.

Configuration JSON must include either:
  - "stage_counts": list of per-section diffusion step counts (e.g. [10,15,20])
    OR
  - "stages": explicit list of discrete diffusion indices (fallback).
  - "riemannian_steps": number of Riemannian steps per stage
Optional config entries:
  - "jump_method": "ddim" or "ddpm" (default "ddim")
  - "jump_eta": noise factor for DDIM (default 0.0)
Other existing riemannian_config entries are respected as before.
"""
import os
import copy
import torch
import numpy as np
from argparse import ArgumentParser

# Project imports
from .config_loader import load_riemannian_config
from .utils import flatten_tensor, unflatten_tensor, encode_xt_in_chunks
from .diffusion_utils import (
    DiffusionWrapper,
    get_score_fn,
    get_denoiser_fn,
    get_classifier_fn,
    compute_discrete_time_from_target_snr
)
from .objectives import get_opt_fn
from .visualization_utils import render_trajectory_images, visualize_trajectory, save_gif_from_rendered_images, save_comparison_image
from data_geometry.riemannian_optimization.retraction import create_retraction_fn
from data_geometry.riemannian_optimization import get_riemannian_optimizer
from templates_latent import ffhq128_autoenc_latent
from templates_cls import ffhq128_autoenc_non_linear_time_cls_full, ffhq128_autoenc_non_linear_cls
from experiment import LitModel
from experiment_classifier import ClsModel
from dataset import CelebAttrDataset, ImageDataset

# for automatic spacing
from diffusion.diffusion import space_timesteps
import math 
from tqdm import tqdm

def forward_perturb(x0, t_tensor, alpha_fn, sigma_fn):
    """
    Perturb x0 forward to noise level t: x_t = α(t)*x0 + σ(t)*noise
    """
    batch = x0.size(0)
    
    α = alpha_fn(t_tensor)
    σ = sigma_fn(t_tensor)

    # Debug prints
    #print(f"\n[forward_perturb] Diffusion step t = {t_tensor}")
    #print(f"  alpha_fn({t_tensor}) = {α[0].item():.6f}")
    #print(f"  sigma_fn({t_tensor}) = {σ[0].item():.6f}")
    #print(f"  alpha * x0 (mean/std): {torch.mean(α * x0).item():.4f} / {torch.std(α * x0).item():.4f}")
    
    noise = torch.randn_like(x0)
    out = α * x0 + σ * noise
    #print(f"  output (mean/std): {out.mean().item():.4f} / {out.std().item():.4f}")
    #print(f"  output sample: {out[0,:10].tolist()}")

    return out

def reverse_jump_explicit(
    z_flat, t_src, t_dst,
    alpha_fn, sigma_fn,
    latent_net, latent_shape,
):
    B = z_flat.size(0)
    device = z_flat.device

    # 1) predict ε and x0_pred
    t_src_t = torch.full((B,), t_src, dtype=torch.float32, device=device)
    z_shaped = unflatten_tensor(z_flat, latent_shape)
    eps = latent_net(z_shaped, t_src_t).pred.view(B, -1)
    a_t = alpha_fn(t_src_t)   # √ᾱ_t
    s_t = sigma_fn(t_src_t)   # √(1-ᾱ_t)
    x0_pred = (z_flat - s_t*eps) / a_t

    # 2) get the inter‐step scalars
    t_dst_t = torch.full((B,), t_dst, dtype=torch.float32, device=device)
    a_s = alpha_fn(t_dst_t)   # √ᾱ_s
    s_s = sigma_fn(t_dst_t)   # √(1-ᾱ_s)
    alpha_t_s  = a_t / a_s
    sigma2_t_s = s_t.pow(2) - alpha_t_s.pow(2)*s_s.pow(2)

    # 3) compute posterior variance & mean
    sigma_Q2 = sigma2_t_s * s_s.pow(2) / s_t.pow(2)
    coef_z   = alpha_t_s * s_s.pow(2) / s_t.pow(2)
    coef_x   = a_s * sigma2_t_s    / s_t.pow(2)

    mu_Q = coef_z * z_flat + coef_x * x0_pred

    # 4) sample (always stochastic; no η here)
    noise = torch.randn_like(z_flat)
    return mu_Q + torch.sqrt(sigma_Q2) * noise

def get_cross_retraction_fn(alpha_fn, sigma_fn, wrap, ae_model, latent_shape):
    """
    Returns a factory that builds retraction_fn(s, t_src) implementing
    Eq.(29) to retract from noise level t_src back to level s.
    """
    def retractor(s, t_src, batch_size, device):
        s_tensor = torch.full((batch_size,), s, dtype=torch.float32, device=device)
        src_tensor = torch.full((batch_size,), t_src,  dtype=torch.float32, device=device)

        alpha_s = alpha_fn(s_tensor).view(batch_size, 1)
        sigma_s = sigma_fn(s_tensor).view(batch_size, 1)
        alpha_tsrc = alpha_fn(src_tensor).view(batch_size, 1)
        sigma_tsrc = sigma_fn(src_tensor).view(batch_size, 1)

        alpha_ratio = alpha_tsrc / alpha_s
        sigma2_ratio = sigma_tsrc.pow(2) - (alpha_ratio.pow(2) * sigma_s.pow(2))

        score_src = get_score_fn(wrap, ae_model.latent_net, src_tensor, latent_shape)

        def retraction_fn(z_flat, v_flat):
            z_tilde = z_flat + v_flat
            grad = score_src(z_tilde)
            return (z_tilde / alpha_ratio) + (sigma2_ratio / alpha_ratio) * grad

        return retraction_fn

    return retractor

def linear_timesteps(start_t: int, end_t: int, num_steps: int):
    """
    Pick num_steps timesteps linearly spaced (inclusive) between start_t and end_t.
    Returns a sorted list of ints.
    """
    if num_steps < 2:
        return [start_t]
    interval = (end_t - start_t) / (num_steps - 1)
    return sorted({ int(round(start_t + i * interval)) for i in range(num_steps) })

def multistage_optimization(config_path):
    cfg = load_riemannian_config(config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("random_seed", 0))

    # Autoencoder
    auto_conf = ffhq128_autoenc_latent()
    auto_conf.T_eval = 1000
    auto_conf.latent_T_eval = 1000
    ae = LitModel(auto_conf).to(device)
    state = torch.load(os.path.join("checkpoints", auto_conf.name, "last.ckpt"), map_location="cpu")
    ae.load_state_dict(state["state_dict"], strict=False)
    ae.ema_model.eval()

    # Classifier
    cls_conf = ffhq128_autoenc_non_linear_cls()
    cls = ClsModel(cls_conf).to(device)
    stc = torch.load(os.path.join("checkpoints", cls_conf.name, "last.ckpt"), map_location="cpu")
    cls.load_state_dict(stc["state_dict"], strict=False)
    cls.eval()

    # Data & encode
    print("Loading data sample ...")
    '''
    ds = cls.load_dataset()
    cid = CelebAttrDataset.cls_to_id[ cfg.get("target_attr","Smiling") ]
    L = cfg.get("num_samples", 5)
    idx = [i for i,s in enumerate(ds) if s["labels"][cid]==-1][:L]
    batch = torch.stack([ds[i]["img"] for i in idx]).to(device)
    '''

    data = ImageDataset("imgs_align", image_size=auto_conf.img_size,
                         exts=["jpg", "JPG", "png"], do_augment=False)
    L = 5
    batch = data[0]["img"].unsqueeze(0).repeat(L, 1, 1, 1).to(device)
    cid = CelebAttrDataset.cls_to_id[ cfg.get("target_attr","Smiling") ]
    

    cond = ae.encode(batch)
    latent_shape = cond.shape[1:]
    x0 = flatten_tensor(cond)
    x0_normalized = cls.normalize(x0)
    x = x0_normalized.clone()

    # Diffusion sampler & wrapper
    diff = auto_conf.make_latent_eval_diffusion_conf().make_sampler()
    wrap = DiffusionWrapper(diff)
    alpha_fn = wrap.get_alpha_fn()
    sigma_fn = wrap.get_sigma_fn()

    # Modular cross-time retractor factory
    intermediate_retractor = get_cross_retraction_fn(alpha_fn, sigma_fn, wrap, ae.ema_model, latent_shape)

    # Build `stages` purely from start_diffusion_timestep + multistage_steps:
    start_t = cfg.get("start_diffusion_timestep")
    num_stages = cfg.get("multistage_steps")
    steps_per_stage = cfg.get("riemannian_steps")

    assert isinstance(start_t, int) and 0 <= start_t < diff.num_timesteps, \
        "Need integer 'start_diffusion_timestep' in [0, T_full)"
    assert isinstance(num_stages, int) and num_stages > 0, \
        "Need positive integer 'multistage_steps'"

    # Time setup
    t_val = compute_discrete_time_from_target_snr(cfg, auto_conf)
    print(f't_val:{t_val}')

    # pick N linearly-spaced timesteps from start_t up to final (diff.num_timesteps-1)
    # reverse to get descending order for the denoising loop
    stages = linear_timesteps(start_t, t_val, num_stages)[::-1]

    print(stages)

    # read debug‐flag from config
    do_riemannian = True
    do_reverse_jump = False

    trajectory = []
    trajectory.append(x.clone().detach().cpu())

    print(x.size())
    print(f'x_start:{x[0,:20]}')
    for i, t in enumerate(stages):
        # forward perturb if first stage
        if i == 0:
            t_tensor = torch.full((batch.size(0),), t, dtype=torch.float32, device=device)
            x = forward_perturb(x, t_tensor, alpha_fn, sigma_fn)
            print(f'After forward perturbation:{x[0,:20]}')

        if do_riemannian:
            t_tensor = torch.full((1,), t, dtype=torch.float32, device=device)
            print(f'SNR:{wrap.snr(t_tensor).mean().item():.4f}')
            # always build these fns (needed for both paths)
            classifier_fn = get_classifier_fn(cls, t_tensor, latent_shape)
            opt_fn        = get_opt_fn(
                classifier_fn, cid, latent_shape, x0_normalized,
                cfg.get("classifier_weight", 1),
                cfg.get("reg_norm_weight", 0.5),
                cfg.get("reg_norm_type", "L2")
            )
            score_fn = get_score_fn(wrap, ae.ema_model.latent_net, t_tensor, latent_shape)

            # build manifold retraction at time t
            t_src = t + 1
            retraction_fn = intermediate_retractor(t, t_src, batch.size(0), device)

            # ——— run Riemannian GD for this stage ———
            cfg_stage = copy.deepcopy(cfg)
            cfg_stage["initial_point"]  = x.detach().clone().requires_grad_(True).to(device)
            cfg_stage["riemannian_steps"] = steps_per_stage
            riem = get_riemannian_optimizer(score_fn, opt_fn, cfg_stage, retraction_fn)
            traj, _ = riem.run()
            x = traj[-1]
            trajectory.append(x.clone().detach().cpu())

            # 🖥️ Print memory info at the end of stage
            print(f"[Stage {i}] After Riemannian optimization:")
            print(f"    Allocated: {torch.cuda.memory_allocated() / 2**20:.2f} MB")
            print(f"    Reserved: {torch.cuda.memory_reserved() / 2**20:.2f} MB")


        # explicit reverse jump to next t (unless last stage)
        if do_reverse_jump:
            if i < len(stages) - 1:
                next_t = stages[i+1]
                x = reverse_jump_explicit(
                    x, t_src=t, t_dst=next_t,
                    alpha_fn=alpha_fn,
                    sigma_fn=sigma_fn,
                    latent_net=ae.ema_model.latent_net,
                    latent_shape=latent_shape
                )
                print(f'After reverse jump:{x[0,:20]}')

        
    # Visualization
    denorm = [cls.denormalize(zi.to(device)) for zi in trajectory]
    T_render, chunk = cfg.get("T_render", 250), cfg.get("chunk", 25)
    xT = encode_xt_in_chunks(ae, batch, cond, T_render, chunk)

    out = cfg.get("log_dir", "logs")
    os.makedirs(out, exist_ok=True)
    imgs = render_trajectory_images(ae, xT, denorm, latent_shape, T_render, fast_mode=True, chunk_size=chunk)
    visualize_trajectory(imgs, os.path.join(out, "traj.png"))
    save_gif_from_rendered_images(imgs, os.path.join(out, "traj.gif"), duration_sec=6)
    save_comparison_image(imgs, os.path.join(out, "comparison.png"))

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--ro-config", required=True)
    args = p.parse_args()
    multistage_optimization(args.ro_config)
