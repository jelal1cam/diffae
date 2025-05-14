import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Utilities
from .utils import load_shared_resources, compute_median_logit, save_side_by_side_comparison
from ..utils import flatten_tensor, unflatten_tensor, encode_xt_in_chunks
from ..diffusion_utils import (
    DiffusionWrapper,
    get_score_fn,
    get_denoiser_fn,
    get_classifier_fn,
    compute_discrete_time_from_target_snr,

)
from data_geometry.riemannian_optimization import get_riemannian_optimizer
from data_geometry.riemannian_optimization.retraction import create_retraction_fn
from ..objectives import get_opt_fn, get_opt_fn_debug
from .multistage_optimization import (
    get_cross_retraction_fn,
    forward_perturb,
    reverse_jump_explicit,
    load_schedule_from_rescaled_entropy,
    linear_timesteps
)
from ..config_loader import load_riemannian_config
from templates_latent import ffhq128_autoenc_latent
from templates_cls import (
    ffhq128_autoenc_non_linear_cls,
    ffhq128_autoenc_cls,
)
from experiment import LitModel
from experiment_classifier import ClsModel
from dataset import CelebAttrDataset
import lpips

from .manipulation_utils import linear_manipulation, multiple_stage_ro, single_stage_ro


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
    manipulated_linear, debug_linear  = linear_manipulation(
        ae, cls_lin, batch, median_logit_lin, cfg, cid, debug=True
    )

    ro_type = cfg.get("ro_type", "multistage")
    if ro_type == 'single-stage':
        #median_logit_nl = None
        manipulated_riemannian, debug_riem = single_stage_ro(
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

    # --- LPIPS Evaluation ---
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    with torch.no_grad():
        lpips_lin  = lpips_model(orig * 2 - 1, lin_out * 2 - 1).squeeze()  # Convert to [-1,1]
        lpips_riem = lpips_model(orig * 2 - 1, riem_out * 2 - 1).squeeze()
    
    # --- Print Joint Diagnostics Table ---
    if debug_linear and debug_riem:
        B = batch.size(0)
        print(f"\n{'Idx':>3} {'Step':>5} {'LPIPS_L':>9} {'Cls_L':>9} {'Reg_L':>9} | "
              f"{'Step':>5} {'LPIPS_R':>9} {'Cls_R':>9} {'Reg_R':>9}")
        print("-" * 80)
        for i in range(B):
            step_r = debug_riem["steps"][i].item() if "steps" in debug_riem else "-"
            print(
                f"{i+1:3d} "
                f"{'-':>5} "
                f"{lpips_lin[i].item():9.4f} "
                f"{debug_linear['cls'][i].item():9.4f} "
                f"{debug_linear['reg'][i].item():9.4f} | "
                f"{step_r:>5} "
                f"{lpips_riem[i].item():9.4f} "
                f"{debug_riem['cls'][i].item():9.4f} "
                f"{debug_riem['reg'][i].item():9.4f}"
            )

    # Save comparison plot
    out_dir  = os.path.join(cfg.get("log_dir", "logs"), cfg.get("target_attr"))
    save_path= os.path.join(out_dir, "comparison.png")
    save_side_by_side_comparison(orig, lin_out, riem_out, save_path)


if __name__ == "__main__":
    main()
