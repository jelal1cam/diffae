#!/usr/bin/env python
"""
Interpolation via Riemannian optimisation on the latent data manifold.

For a randomly drawn batch **A** of *L* images we pick another random batch
**B** (also *L*) of disjoint images from the same dataset and treat the
latent codes of **B** as *targets* for **A**.  We then solve

    min_z ‖ z − z_target ‖  (L1 or L2)   s.t.   z lies on the data manifold.

The optimiser, score‑function and retraction operator are unchanged from the
attribute‑manipulation pipeline; only the objective differs.

The script appends the **target batch as the final frame** in every visual
(gif / comparison / trajectory PNG) so you can clearly see
*source → trajectory → target*.

Run, optionally selecting a GPU:

```bash
python interpolation.py --ro-config configs/your_config.yaml --gpu-id 1
```

The YAML config is the usual one for Riemannian optimisation; classifier‑specific
keys are ignored.
"""

import os
import time
import random
from argparse import ArgumentParser

import torch
import torch.multiprocessing as mp
import numpy as np # Added for linear interpolation alpha
import math # Added for calculating chunk sizes if needed

# -----------------------------------------------------------------------------
# Local project imports
# -----------------------------------------------------------------------------
from ..config_loader import load_riemannian_config
from ..utils import flatten_tensor, encode_xt_in_chunks, unflatten_tensor # unflatten_tensor needed for linear interpolation
from ..diffusion_utils import (
    DiffusionWrapper,
    get_score_fn,
    get_denoiser_fn,
    compute_discrete_time_from_target_snr,
)
from ..visualization_utils import (
    render_trajectory_images,
    visualize_trajectory,
    save_gif_from_rendered_images,
    save_comparison_image,
    visualize_comparison_trajectory # Added import
)

from templates_latent import ffhq128_autoenc_latent
from templates_cls import ffhq128_autoenc_non_linear_cls
from experiment import LitModel
from experiment_classifier import ClsModel

from data_geometry.riemannian_optimization.retraction import create_retraction_fn
from data_geometry.riemannian_optimization import get_riemannian_optimizer
from data_geometry.riemannian_optimization.geometry import efficient_riemannian_inner_product
import torch.nn.functional as F


@torch.no_grad()
def path_energy_integral(score_fn, path_flat):
    """
    Discrete ∫₀¹‖γ̇(t)‖²_{g(γ(t))}dt  for a path γ sampled at N points.
    *All* points must already live in the **same** coordinate system 
    (here: z-normalised, *before* you denormalise for rendering).

    Args
    ----
    score_fn   : callable (B,D) → (B,D)  – returned by `get_score_fn`
    path_flat  : list[Tensor] {length N} – each (B,D)  (batch × latent-dim)

    Returns
    -------
    E_int      : Tensor (B,)             – per-sample energy integral
    E_segments : list[Tensor]            – energy density per segment
    """
    B = path_flat[0].shape[0]
    Δt = 1.0 / (len(path_flat) - 1)      # segment length on [0,1]

    E_seg, E_int = [], torch.zeros(B, device=path_flat[0].device)
    for x_k, x_kp1 in zip(path_flat[:-1], path_flat[1:]):
        v_k   = x_kp1 - x_k
        e_k   = efficient_riemannian_inner_product(score_fn, x_k, v_k, v_k)  # (B,)
        E_seg.append(e_k.detach())
        E_int += Δt * e_k                # accumulate integral
    return E_int.cpu(), E_seg

# -----------------------------------------------------------------------------
# Objective:  pure distance to a *target* latent
# -----------------------------------------------------------------------------

def get_interpolation_opt_fn(
    target_flat: torch.Tensor,
    reg_norm_weight: float = 1.0,
    reg_norm_type: str = "L2",
):
    """Return a callable *opt_fn(x_flat, idx=None) → (B,)*.

    The loss for every element in the (sub‑)batch is the L1/L2 distance to
    its *paired* target latent (‖·‖ computed element‑wise then summed over
    latent dimensions).
    """

    reg_norm_type = reg_norm_type.upper()

    def opt_fn(x_flat: torch.Tensor, idx=None):
        tgt = target_flat[idx] if idx is not None else target_flat

        if reg_norm_type == "L2":
            loss = F.mse_loss(x_flat, tgt, reduction="none").mean(dim=1)
        elif reg_norm_type == "L1":
            loss = F.l1_loss(x_flat, tgt, reduction="none").mean(dim=1)
        else:
            raise ValueError(
                f"Unknown norm type: {reg_norm_type}. Use 'L1' or 'L2'."
            )

        return reg_norm_weight * loss

    return opt_fn


# -----------------------------------------------------------------------------
# Core routine
# -----------------------------------------------------------------------------

def interpolation(riem_config_path: str, gpu_id: int = 0) -> None:
    """Run the interpolation experiment on the specified GPU."""

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------
    riem_config = load_riemannian_config(riem_config_path)

    # -----------------------------------------------------------------
    # Auto‑encoder (latent‑space) model
    # -----------------------------------------------------------------
    print("Loading autoencoder model …")
    ae_conf = ffhq128_autoenc_latent()
    ae_conf.T_eval = 1000
    ae_conf.latent_T_eval = 1000

    ae = LitModel(ae_conf)
    ckpt_path = os.path.join("checkpoints", ae_conf.name, "last.ckpt")
    print(f"Loading autoencoder ckpt from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    ae.load_state_dict(state["state_dict"], strict=False)
    ae.ema_model.eval().to(device)

    if not hasattr(ae.ema_model, "latent_net"):
        raise ValueError("Autoencoder model missing latent_net.")

    # -----------------------------------------------------------------
    # Classifier model (provides z‑normalisation)
    # -----------------------------------------------------------------
    cls_conf  = ffhq128_autoenc_non_linear_cls()
    cls_model = ClsModel(cls_conf)
    cls_ckpt  = os.path.join("checkpoints", cls_conf.name, "last.ckpt")
    print(f"Loading classifier ckpt from {cls_ckpt}")
    cls_state = torch.load(cls_ckpt, map_location="cpu")
    cls_model.load_state_dict(cls_state["state_dict"], strict=False)
    cls_model.eval().to(device)

    # -----------------------------------------------------------------
    # Dataset – pick two *disjoint* random batches of size L
    # -----------------------------------------------------------------
    dataset = cls_model.load_dataset()
    L       = riem_config.get("num_samples", 4)
    assert len(dataset) >= 2 * L, "Dataset too small for requested L."

    indices         = random.sample(range(len(dataset)), 2 * L)
    source_indices  = indices[:L]
    target_indices  = indices[L:]

    print(f"Selected {L} source + {L} target images (no overlap).")

    batch        = torch.stack([dataset[i]["img"] for i in source_indices]).to(device)
    target_batch = torch.stack([dataset[i]["img"] for i in target_indices]).to(device)

    # -----------------------------------------------------------------
    # Encode images → latent space
    # -----------------------------------------------------------------
    with torch.no_grad():
        cond        = ae.encode(batch) # Semantic code (denormalized)
        target_cond = ae.encode(target_batch) # Semantic code (denormalized)

    latent_shape   = cond.shape[1:]    # (C, H, W)
    x0_flat        = flatten_tensor(cond)
    target_flat    = flatten_tensor(target_cond)

    # -----------------------------------------------------------------
    # z‑normalise latents
    # -----------------------------------------------------------------
    assert getattr(cls_model.conf, "manipulate_znormalize", False) and getattr(
        ae.conf, "latent_znormalize", False
    ), "Autoencoder & classifier must share z‑normalisation."

    x0_norm     = cls_model.normalize(x0_flat) # Normalized semantic code
    target_norm = cls_model.normalize(target_flat) # Normalized semantic code

    riem_config["initial_point"] = x0_norm

    # -----------------------------------------------------------------
    # Diffusion + score / denoiser
    # -----------------------------------------------------------------
    t_val    = compute_discrete_time_from_target_snr(riem_config, ae_conf)
    t_latent = torch.full((1,), t_val, dtype=torch.float32, device=device)

    latent_sampler = ae_conf.make_latent_eval_diffusion_conf().make_sampler()
    latent_wrapper = DiffusionWrapper(latent_sampler)

    score_fn    = get_score_fn(latent_wrapper, ae.ema_model.latent_net, t_latent, latent_shape)
    denoiser_fn = get_denoiser_fn(latent_wrapper, ae.ema_model.latent_net, t_latent, latent_shape)

    retraction_fn = create_retraction_fn(
        retraction_type=riem_config.get("retraction_operator", "identity"),
        denoiser_fn=denoiser_fn,
    )

    # -----------------------------------------------------------------
    # Optimisation objective – pure interpolation
    # -----------------------------------------------------------------
    opt_fn = get_interpolation_opt_fn(
        target_norm,
        reg_norm_weight=riem_config.get("interp_weight", 1.0),
        reg_norm_type=riem_config.get("interp_norm", "L2"),
    )

    # -----------------------------------------------------------------
    # Optimise on the manifold
    # -----------------------------------------------------------------
    print("Running Riemannian optimisation …")
    riem_opt   = get_riemannian_optimizer(score_fn, opt_fn, riem_config, retraction_fn)
    start_time = time.time()
    # Get the trajectory *before* adding the target frame
    riem_trajectory_norm, metrics = riem_opt.run()
    print(f"Optimisation finished in {time.time() - start_time:.2f} s.")

    # Append target frame to the normalized Riemannian trajectory
    full_riem_traj_norm = riem_trajectory_norm + [target_norm]
    full_riem_traj_norm_gpu = [z.to(device) for z in full_riem_traj_norm]
    E_riem, E_riem_segs = path_energy_integral(score_fn, full_riem_traj_norm_gpu)

    # -----------------------------------------------------------------
    # Linear interpolation of denormalized semantic codes
    # -----------------------------------------------------------------
    n_opt_steps = len(riem_trajectory_norm)
    n_frames = n_opt_steps + 1
    alpha_linear = torch.linspace(0, 1, n_frames, device=device).unsqueeze(1) # (n_frames, 1)
    # Cond/target_cond shape is (B, C, H, W). We need (n_frames, B, C, H, W)
    # alpha_linear_bc needs to broadcast against (B, C, H, W)
    alpha_linear_bc = alpha_linear.view(n_frames, *([1] * len(cond.shape))) # (n_frames, 1, 1, 1, 1) if latent is 4D
    linear_semantic_traj_denorm = cond.unsqueeze(0) * (1 - alpha_linear_bc) + target_cond.unsqueeze(0) * alpha_linear_bc # (n_frames, B, C, H, W)

    # Convert to list of flattened tensors format expected by render_trajectory_images
    linear_semantic_traj_denorm_list = [flatten_tensor(step_tensor) for step_tensor in linear_semantic_traj_denorm]
    linear_semantic_traj_norm_list = [cls_model.normalize(z)                 # ① z-normalise
                                  for z in linear_semantic_traj_denorm_list]
    E_lin, _ = path_energy_integral(score_fn, linear_semantic_traj_norm_list) # ② energy integral

    # -----------------------------------------------------------------
    # ----- REPORT energies ---------------------------------------------------
    # -----------------------------------------------------------------
    for i, (er, el) in enumerate(zip(E_riem, E_lin)):
        print(f"sample {i:02d}:  ∫‖γ̇‖²_Riem = {er:.3f}   |   ∫‖γ̇‖²_Linear = {el:.3f}")

    # ─── NEW: mean ± std across the batch ───
    µ_riem, σ_riem = E_riem.mean().item(), E_riem.std(unbiased=False).item()
    µ_lin , σ_lin  = E_lin.mean().item() , E_lin.std(unbiased=False).item()
    print(f"avg ± std  :  Riem = {µ_riem:.3f} ± {σ_riem:.3f}   |   Linear = {µ_lin:.3f} ± {σ_lin:.3f}")
    print("───────────────────────────────────────")


    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------

    out_dir = riem_config.get("log_dir", "logs")
    os.makedirs(out_dir, exist_ok=True)

    T_render   = riem_config.get("T_render", 250)
    chunk_size = riem_config.get("chunk", 25)
    gif_duration_sec = riem_config.get("gif_duration_sec", 6)

    # Compute stochastic subcodes for source and target
    xT_source = encode_xt_in_chunks(ae, batch, cond, T_render, chunk_size)
    xT_target = encode_xt_in_chunks(ae, target_batch, target_cond, T_render, chunk_size)

    # Determine the number of steps for visualisation (optimization steps + target frame)
    bs, C, H, W = xT_source.shape

    # Spherical interpolation of subcodes across steps for the full sequence length (n_frames)
    # This sequence of xT will be used for decoding both Riem and Linear trajectories.
    alpha_slerp = torch.linspace(0, 1, n_frames, device=device).unsqueeze(1) # (n_frames, 1)
    src_flat = xT_source.view(bs, -1) # (B, D_xT)
    tgt_flat = xT_target.view(bs, -1) # (B, D_xT)
    src_norm_slerp = F.normalize(src_flat, dim=1) # (B, D_xT)
    tgt_norm_slerp = F.normalize(tgt_flat, dim=1) # (B, D_xT)
    # Clamp for numerical stability in acos near 1 or -1
    theta_slerp = torch.acos(torch.clamp((src_norm_slerp * tgt_norm_slerp).sum(dim=1), -1.0 + 1e-6, 1.0 - 1e-6)) # (B,)

    # Calculate sin terms for each frame and each sample
    # (n_frames, 1) * (B,) broadcasts to (n_frames, B)
    sin_1_a_theta = torch.sin((1 - alpha_slerp) * theta_slerp) # (n_frames, B)
    sin_a_theta = torch.sin(alpha_slerp * theta_slerp)       # (n_frames, B)

    # sin(theta) for each sample, shape (B,)
    sin_theta_val = torch.sin(theta_slerp) # (B,)

    # Perform slerp: (n_frames, B) * (B, D_xT) + (n_frames, B) * (B, D_xT) / (B,)
    # Need to unsqueeze dimensions for broadcasting
    # (n_frames, B, 1) * (1, B, D_xT) + (n_frames, B, 1) * (1, B, D_xT) / (1, B, 1)
    slerp_xT = (sin_1_a_theta.unsqueeze(-1) * src_flat.unsqueeze(0) + sin_a_theta.unsqueeze(-1) * tgt_flat.unsqueeze(0)) / (sin_theta_val.unsqueeze(0).unsqueeze(-1) + 1e-6)
    slerp_xT = slerp_xT.view(n_frames, bs, C, H, W) # Reshape to (n_frames, B, C, H, W)


    print("\nGenerating Riemannian trajectory visualisations…")
    # Denormalize the full Riemannian trajectory
    full_riem_denorm_traj = [cls_model.denormalize(z.to(device)) for z in full_riem_traj_norm]
    # Render Riemannian trajectory images using the shared slerp_xT
    imgs_riem = render_trajectory_images(
        ae,
        slerp_xT, # Use the shared slerp_xT sequence
        full_riem_denorm_traj,
        latent_shape,
        T_render,
        fast_mode=True,
        chunk_size=chunk_size,
    )

    # Save individual Riemannian visualizations
    visualize_trajectory(imgs_riem, os.path.join(out_dir, "traj_riem.png"))
    save_gif_from_rendered_images(imgs_riem, os.path.join(out_dir, "traj_riem.gif"), duration_sec=gif_duration_sec)
    save_comparison_image(imgs_riem, os.path.join(out_dir, "comparison_riem.png")) # Still useful to see start/end Riem
    print(f"Riemannian visualisations (including target frame) saved to → {out_dir}")

    # -----------------------------------------------------------------
    # Baseline Linear + Slerp Trajectory Visualisation
    # -----------------------------------------------------------------
    print("\nGenerating baseline linear interpolation trajectory visualisations…")
    # Render Linear + Slerp trajectory images using the shared slerp_xT
    imgs_linear = render_trajectory_images(
        ae,
        slerp_xT, # Use the shared slerp_xT sequence
        linear_semantic_traj_denorm_list, # Use the linear semantic trajectory
        latent_shape,
        T_render,
        fast_mode=True,
        chunk_size=chunk_size,
    )

    # Save individual Linear visualizations
    visualize_trajectory(imgs_linear, os.path.join(out_dir, "traj_linear.png"))
    save_gif_from_rendered_images(imgs_linear, os.path.join(out_dir, "traj_linear.gif"), duration_sec=gif_duration_sec)
    save_comparison_image(imgs_linear, os.path.join(out_dir, "comparison_linear.png")) # Still useful to see start/end Linear
    print(f"Baseline linear visualisations (including target frame) saved to → {out_dir}")

    # -----------------------------------------------------------------
    # Comparison Visualisation
    # -----------------------------------------------------------------
    print("\nGenerating comparison visualisation…")
    visualize_comparison_trajectory(
        imgs_riem,
        imgs_linear,
        os.path.join(out_dir, "comparison_riem_vs_linear.png")
    )
    print(f"Comparison visualisations saved to → {out_dir}")


# -----------------------------------------------------------------------------
# CLI wrapper
# -----------------------------------------------------------------------------

def main():
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser(description="Latent interpolation via Riemannian optimisation")
    parser.add_argument("--ro-config", required=True, type=str, help="Path to the Riemannian-opt config file")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU index to use (default: 0)")
    args = parser.parse_args()

    interpolation(args.ro_config, args.gpu_id)


if __name__ == "__main__":
    main()
