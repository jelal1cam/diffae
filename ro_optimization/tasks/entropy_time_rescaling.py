import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import argparse
import os

# Project imports (assumes this file is run from project root)
# Make sure these imports correctly point to your project structure

from ..config_loader import load_riemannian_config
# Import your specific autoencoder and classifier template configs
from templates_latent import ffhq128_autoenc_latent
from experiment import LitModel
from templates_cls import ffhq128_autoenc_non_linear_cls
from experiment_classifier import ClsModel
# Assuming DiffusionWrapper, flatten_tensor are in diffusion_utils or a local file
from ..diffusion_utils import DiffusionWrapper # adjust import as needed
from ..utils import flatten_tensor # adjust import as needed
# Assuming dataset.py and CelebAttrDataset are accessible
from dataset import CelebAttrDataset # adjust import as needed

# For progress bar
from tqdm import tqdm

# --- Helper functions ---

def compute_dotH(sigma, dsigma_dt, expected_total_sq_error):
    """
    Computes the conditional entropy production rate based on the paper's formula
    dH[x0|xt]/dt = (dsigma/dt / sigma^3) * E[||x0 - x0_hat||^2].
    """
    sigma_safe = sigma + 1e-9
    return dsigma_dt / (sigma_safe**3) * expected_total_sq_error


def integrate_trap(y, x):
    """
    Numerically integrates y with respect to x using the trapezoidal rule.
    Assumes x values are ordered.
    """
    out = np.zeros_like(y)
    sort_indices = np.argsort(x)
    x_sorted = x[sort_indices]
    y_sorted = y[sort_indices]

    dx = np.diff(x_sorted)
    dx[dx == 0] = 1e-9  # Avoid division by zero

    trap = (y_sorted[:-1] + y_sorted[1:]) / 2 * dx
    out[sort_indices[1:]] = np.cumsum(trap)
    return out


def compute_entropic_times(sigma_fn_edm, timesteps_original_int, expected_total_sq_error_array):
    """
    Computes the entropic and rescaled entropic time curves by integrating the
    conditional entropy production rate.
    """
    timesteps_float = timesteps_original_int.astype(np.float32)
    σ_vals = sigma_fn_edm(torch.tensor(timesteps_float, dtype=torch.float32).unsqueeze(1))
    σ_vals = σ_vals.detach().cpu().view(-1).numpy()
    dσ_dt = np.gradient(σ_vals, timesteps_float)
    dotH = compute_dotH(σ_vals, dσ_dt, expected_total_sq_error_array)
    rescaled_dotH = σ_vals * dotH
    entropic_time_curve = integrate_trap(dotH, timesteps_float)
    rescaled_entropic_time_curve = integrate_trap(rescaled_dotH, timesteps_float)
    return entropic_time_curve, rescaled_entropic_time_curve


def derive_sampling_steps(
    cumulative_time_curve,
    original_timesteps,
    num_sampling_steps,
    total_timesteps,
    max_timestep=None
):
    """
    Derives discrete sampling steps (original timestep indices) from a cumulative
    time curve by selecting points linearly spaced in the cumulative time domain,
    but only up to max_timestep (T_prime).

    Args:
        cumulative_time_curve (np.ndarray): The calculated cumulative time curve (e.g., entropic_time).
                                            Indexed by original_timesteps (0 to T-1).
        original_timesteps (np.ndarray): The array of original timestep indices (0 to T-1).
        num_sampling_steps (int): The desired number of steps in the sampling schedule.
        total_timesteps (int): The total number of original diffusion timesteps (T).
        max_timestep (int, optional): Maximum original diffusion timestep to consider (T_prime).
                                      If None, uses the full range (up to total_timesteps - 1).

    Returns:
        np.ndarray: Integer original timestep indices for sampling,
                    ordered from high t (T_prime) to low t for reverse diffusion.
    """
    # Determine the effective upper bound (T_prime)
    if max_timestep is None:
        effective_max_timestep = total_timesteps - 1
    else:
        effective_max_timestep = int(np.clip(max_timestep, 0, total_timesteps - 1))

    # Robust handling of edge cases
    if num_sampling_steps <= 0:
        return np.array([], dtype=int)

    # Restrict timesteps and curve to [0, T_prime]
    mask = original_timesteps <= effective_max_timestep
    timesteps_for_schedule = original_timesteps[mask]
    cumulative_time_for_schedule = cumulative_time_curve[mask]
    if len(timesteps_for_schedule) == 0:
        return np.array([], dtype=int)

    if num_sampling_steps == 1:
        return np.array([effective_max_timestep], dtype=int)

    # Equally spaced points in the cumulative-time domain
    new_min = cumulative_time_for_schedule[0]
    new_max = cumulative_time_for_schedule[-1]
    new_time_points = np.linspace(new_min, new_max, num_sampling_steps)

    # Interpolate to get original timesteps
    orig_steps_float = np.interp(
        new_time_points,
        cumulative_time_for_schedule,
        timesteps_for_schedule
    )
    orig_steps_int = np.round(orig_steps_float).astype(int)
    orig_steps_int = np.clip(orig_steps_int, 0, effective_max_timestep)

    # Reverse for sampling from high to low
    sampling_schedule = orig_steps_int[::-1]
    return sampling_schedule


# --- Main execution logic ---

def main(config_path):
    print(f"Loading configuration from: {config_path}")
    try:
        cfg = load_riemannian_config(config_path)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    print("Configuration loaded successfully.")

    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("random_seed", 0))

    # Autoencoder
    auto_conf = ffhq128_autoenc_latent()
    auto_conf.T_eval = 1000
    auto_conf.latent_T_eval = 1000
    ae = LitModel(auto_conf).to(device)
    ae_ckpt_path = cfg.get(
        "autoenc_checkpoint_path",
        os.path.join("checkpoints", auto_conf.name, "last.ckpt")
    )
    print(f"Loading Autoencoder checkpoint from: {ae_ckpt_path}")
    ckpt = torch.load(ae_ckpt_path, map_location="cpu")
    ae.load_state_dict(ckpt["state_dict"], strict=False)
    ae.ema_model.eval()

    # Classifier
    cls_conf = ffhq128_autoenc_non_linear_cls()
    cls_model = ClsModel(cls_conf).to(device)
    cls_ckpt_path = cfg.get(
        "classifier_checkpoint_path",
        os.path.join("checkpoints", cls_conf.name, "last.ckpt")
    )
    try:
        ck2 = torch.load(cls_ckpt_path, map_location="cpu")
        cls_model.load_state_dict(ck2["state_dict"], strict=False)
        cls_model.eval()
    except FileNotFoundError:
        if cfg.get("require_classifier", True):
            raise
        else:
            cls_model = None

    # Dataset
    if cls_model and hasattr(cls_model, 'load_dataset'):
        ds_full = cls_model.load_dataset()
    else:
        raise NotImplementedError("Dataset loading not implemented")
    cid = CelebAttrDataset.cls_to_id[cfg.get("target_attr", "Smiling")]
    neg_idxs = [i for i, s in enumerate(ds_full) if s["labels"][cid] == -1]
    subset = Subset(ds_full, neg_idxs[: cfg.get("num_samples_estimate", 3000)])
    loader = DataLoader(
        subset,
        batch_size=cfg.get("batch_size", 500),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4)
    )

    # Diffusion wrapper
    diff = auto_conf.make_latent_eval_diffusion_conf().make_sampler()
    wrap = DiffusionWrapper(diff)
    alpha_fn = wrap.get_alpha_fn()
    sigma_fn_std = wrap.get_sigma_fn()
    sigma_fn_edm = wrap.get_edm_sigma_fn()

    T = diff.num_timesteps

    # Estimate error
    sum_sq = np.zeros(T)
    counts = np.zeros(T, int)
    ae.ema_model.eval()
    if cls_model: cls_model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Estimating errors"):
            imgs = batch["img"].to(device)
            cond = ae.encode(imgs)
            z0 = flatten_tensor(cond)
            if cls_model and hasattr(cls_model, 'normalize'):
                z0 = cls_model.normalize(z0)
            B = z0.size(0)
            for t in range(T):
                t_tensor = torch.full((B,), t, dtype=torch.float32, device=device)
                α = alpha_fn(t_tensor)
                σ = sigma_fn_std(t_tensor)
                noise = torch.randn_like(z0)
                xt = α * z0 + σ * noise
                out = ae.ema_model.latent_net(xt, t_tensor)
                eps = out.pred.view(B, -1)
                x0_pred = (xt - σ * eps) / α
                err = ((z0 - x0_pred) ** 2).sum(dim=1).cpu().numpy()
                sum_sq[t] += err.sum()
                counts[t] += B

    exp_err = sum_sq / (counts + 1e-9)
    ts = np.arange(T)
    ent_time, res_time = compute_entropic_times(sigma_fn_edm, ts, exp_err)

    # Save only entropic and rescaled entropic time curves
    out = str(cfg.get("log_dir", "logs"))
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "entropic_time_curve.npy"), ent_time)
    np.save(os.path.join(out, "rescaled_entropic_time_curve.npy"), res_time)
    print(f"Saved entropic time curves to {out}")

    # Derive schedules for a specific T' and N for plotting/demo
    N = cfg.get("num_sampling_steps", 16)
    start_t = 50  # optionally set from config
    end_t = int(np.clip(start_t, 0, T - 1))
    print(f"Computing schedule up to timestep: {end_t}/{T-1}")

    ent_schedule = derive_sampling_steps(ent_time, ts, N, T, max_timestep=end_t)
    res_schedule = derive_sampling_steps(res_time, ts, N, T, max_timestep=end_t)
    print("Entropic schedule:", ent_schedule)
    print("Rescaled schedule:", res_schedule)

    # Plotting for visualization
    plt.figure()
    plt.plot(ts[: end_t + 1], ent_time[: end_t + 1])
    plt.scatter(ent_schedule, ent_time[ent_schedule])
    plt.savefig(os.path.join(out, "entropic_time.png"), dpi=300)

    plt.figure()
    plt.plot(ts[: end_t + 1], res_time[: end_t + 1])
    plt.scatter(res_schedule, res_time[res_schedule])
    plt.savefig(os.path.join(out, "rescaled_entropic_time.png"), dpi=300)

    print(f"Saved plots to {out}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate Entropic and Rescaled Entropic Time curves and derive sampling steps.'
    )
    parser.add_argument(
        '--ro-config', type=str, required=True,
        help='Path to the Riemannian config JSON file'
    )
    args = parser.parse_args()
    main(args.ro_config)
