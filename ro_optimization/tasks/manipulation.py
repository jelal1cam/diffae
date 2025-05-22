import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Utilities
from .utils import load_shared_resources, compute_median_logit, save_side_by_side_comparison
from ..utils import unflatten_tensor, encode_xt_in_chunks
import lpips
from .manipulation_utils import linear_manipulation, multiple_stage_ro, single_stage_ro

def save_manipulation_images(orig, lin_out, riem_out_tensors, out_dir, B, S, debug_riem=None):
    """
    Save all manipulation images (original, linear, Riemannian) for later evaluation.
    All images are saved under the 'images' folder.
    
    Args:
        orig: Tensor of original images (B, C, H, W)
        lin_out: Tensor of linear edited images (B, C, H, W)
        riem_out_tensors: List of tensors containing Riemannian edited images
                          Each tensor has shape (B, C, H, W)
        out_dir: Directory to save the images
        B: Batch size
        S: Number of Riemannian seeds
        debug_riem: Dictionary with debugging info (optional)
    """
    # Create images directory structure
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    riem_dir = os.path.join(images_dir, "riemannian")
    os.makedirs(riem_dir, exist_ok=True)
    
    # 1. Save original images
    orig_path = os.path.join(images_dir, "original_imgs.pt")
    torch.save(orig, orig_path)
    print(f"[INFO] Saved original images to {orig_path}")
    
    # 2. Save linear edited images
    lin_path = os.path.join(images_dir, "linear_imgs.pt")
    torch.save(lin_out, lin_path)
    print(f"[INFO] Saved linear edited images to {lin_path}")
    
    # 3. Save Riemannian edited images
    if S > 1:
        # Stack all seeds along dimension 1
        riem_tensor = torch.stack(riem_out_tensors, dim=1)  # Shape: (B, S, C, H, W)
        riem_path = os.path.join(images_dir, "riemannian_imgs.pt")
        torch.save(riem_tensor, riem_path)
        print(f"[INFO] Saved all {S} Riemannian edited image seeds to {riem_path}")
        
        # Save each seed separately
        for i in range(S):
            seed_path = os.path.join(riem_dir, f"seed_{i}.pt")
            torch.save(riem_out_tensors[i], seed_path)
            print(f"[INFO] Saved Riemannian seed {i} to {seed_path}")
        
        # Save the best seed per image based on total loss
        if debug_riem is not None and "total" in debug_riem:
            total_losses = debug_riem["total"].view(B, S)
            best_indices = total_losses.argmin(dim=1)
            best_riem_images = torch.stack([
                riem_out_tensors[best_indices[i].item()][i] 
                for i in range(B)
            ])
            best_path = os.path.join(images_dir, "riemannian_best_imgs.pt")
            torch.save(best_riem_images, best_path)
            print(f"[INFO] Saved best Riemannian edited images to {best_path}")
            
            # Save metadata about which seed was best for each image
            best_seeds_info = {
                "best_indices": best_indices.cpu().numpy(),
                "total_losses": total_losses.detach().cpu().numpy()
            }
            best_seeds_path = os.path.join(riem_dir, "best_seeds_info.npy")
            np.save(best_seeds_path, best_seeds_info)
            print(f"[INFO] Saved best seeds info to {best_seeds_path}")
    else:
        # Single-seed case - just save the single tensor
        riem_path = os.path.join(images_dir, "riemannian_imgs.pt")
        torch.save(riem_out_tensors[0], riem_path)
        print(f"[INFO] Saved Riemannian edited images to {riem_path}")
        
        # Also save to the seed subdirectory for consistency
        seed_path = os.path.join(riem_dir, "seed_0.pt") 
        torch.save(riem_out_tensors[0], seed_path)
    
    print(f"[INFO] All manipulation images saved to {images_dir}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--ro-config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    ae, cls_nl, cls_lin, dataset, pos_dataset, neg_indices, cfg, cid, device = \
        load_shared_resources(args.ro_config)

    out_dir = os.path.join(cfg.get("log_dir", "logs"), cfg.get("target_attr"))
    os.makedirs(out_dir, exist_ok=True)

    median_path = os.path.join(out_dir, "median_logits.pt")

    if os.path.exists(median_path):
        data = torch.load(median_path, map_location=device)
        median_logit_lin = data['linear']
        median_logit_nl = data['non_linear']
        print(f"Loaded median logits from {median_path}")
    else:
        median_logit_lin = compute_median_logit(ae, cls_lin, cid, pos_dataset, cfg, device)
        median_logit_nl  = compute_median_logit(ae, cls_nl,  cid, pos_dataset, cfg, device)
        torch.save({'linear': median_logit_lin, 'non_linear': median_logit_nl}, median_path)
        print(f"Saved median logits to {median_path}")

    print(f"Median logit (linear): {median_logit_lin:.4f}, "
          f"(non-linear): {median_logit_nl:.4f}")
    
    num_neg  = cfg.get("num_samples", 5)
    neg_idxs = neg_indices[:num_neg]
    batch    = torch.stack([dataset[i]['img'] for i in neg_idxs]).to(device)

    # Linear manipulation
    manipulated_linear, debug_linear = linear_manipulation(
        ae, cls_lin, batch, median_logit_lin, cfg, cid, debug=True
    )

    # Riemannian manipulation
    ro_type = cfg.get("ro_type", "multistage")
    if ro_type == 'single-stage':
        manipulated_riemannian, debug_riem = single_stage_ro(
            ae, cls_nl, batch, median_logit_nl, cfg, cid, device, debug=True
        )
        manipulated_riemannian = manipulated_riemannian.unsqueeze(1)  # (B, 1, D)
    elif ro_type == 'multi-stage':
        manipulated_riemannian, debug_riem = multiple_stage_ro(
            ae, cls_nl, batch, median_logit_nl, cfg, cid, device, debug=True
        )

    # Decode and render images
    latent_shape = ae.encode(batch).shape[1:]
    T_render     = cfg.get("T_render", 250)
    chunk        = cfg.get("chunk", 25)
    xT           = encode_xt_in_chunks(ae, batch, ae.encode(batch), T_render, chunk)

    orig    = (batch * 0.5) + 0.5
    lin_out = ae.render(xT, unflatten_tensor(manipulated_linear, latent_shape), T_render)

    # Decode each Riemannian seed
    #B, S = manipulated_riemannian.shape[:2]
    #riem_out_list = [
    #    ae.render(xT, unflatten_tensor(manipulated_riemannian[:, i], latent_shape), T_render)
    #    for i in range(S)
    #]
    B, S = manipulated_riemannian.shape[:2]
    D = int(np.prod(latent_shape))

    # 1) flatten all latents → (B*S, D)
    all_latents = manipulated_riemannian.view(B * S, D)
    all_latents = unflatten_tensor(all_latents, latent_shape)  # → (B*S, C, H, W)

    # 2) repeat xT → (B, 1, C, H, W) → (B, S, C, H, W) → (B*S, C, H, W)
    all_xT = xT.unsqueeze(1).repeat(1, S, 1, 1, 1).view(B * S, *xT.shape[1:])

    # 3) try batch-render
    try:
        # 1) render → Tensor on device
        raw = ae.render(all_xT, all_latents, T_render)  # (B*S, C, H, W), float Tensor
        raw = raw.clamp(0,1)

        # 2a) reshape for LPIPS (Tensor)
        raw = raw.view(B, S, *raw.shape[1:])             # (B, S, C, H, W)
        riem_out_tensors = [ raw[:, i] for i in range(S) ]  # list of (B, C, H, W) Tensors

        # 2b) make NumPy for saving/visualization
        out_np = raw.mul(255).byte().cpu().permute(0,1,3,4,2).numpy()  
        # now out_np.shape == (B, S, H, W, 3)

        riem_out_list = [ out_np[:, i] for i in range(S) ]

    except RuntimeError:
        # fallback: your old loop already returns Tensors, so:
        riem_out_tensors = [
            ae.render(xT, unflatten_tensor(manipulated_riemannian[:, i], latent_shape), T_render)
            for i in range(S)
        ]
        # also build NumPy versions
        riem_out_list = [ 
            t.clamp(0,1).mul(255).byte().cpu().permute(0,2,3,1).numpy() 
            for t in riem_out_tensors
        ]

        
    # --- LPIPS Evaluation ---
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    with torch.no_grad():
        lpips_lin = lpips_model(orig*2-1, lin_out*2-1).squeeze()
        lpips_riem = torch.stack([
            lpips_model(orig*2-1, t*2-1).squeeze()
            for t in riem_out_tensors
        ], dim=1)  # correctly all Tensors on device


    # --- Print Joint Diagnostics Table ---
    if debug_linear and debug_riem:
        print("\nDiagnostics:\n")
        if S == 1:
            print(f"{'Idx':>3} {'Step':>5} {'LPIPS_L':>9} {'Cls_L':>9} {'Reg_L':>9} | "
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
                    f"{lpips_riem[i, 0].item():9.4f} "
                    f"{debug_riem['cls'][i].item():9.4f} "
                    f"{debug_riem['reg'][i].item():9.4f}"
                )
        else:
            # Header
            print(
                f"{'Idx':>3} {'Step':>5} {'LPIPS_L':>9} {'Cls_L':>9} {'Reg_L':>9} | "
                f"{'LPIPS_R':^35} | {'Cls_R':^35} | {'Reg_R':^35}"
            )
            print("-" * 125)

            for i in range(B):
                lpips_vals = lpips_riem[i]
                cls_vals = debug_riem["cls"].view(B, S)[i]
                reg_vals = debug_riem["reg"].view(B, S)[i]

                def stats(vals):
                    mean = vals.mean().item()
                    std  = vals.std().item()
                    vmin = vals.min().item()
                    vmax = vals.max().item()
                    return f"{mean:.4f}±{std:.4f}", f"{vmin:.4f}", f"{vmax:.4f}"

                lpips_stat = stats(lpips_vals)
                cls_stat   = stats(cls_vals)
                reg_stat   = stats(reg_vals)

                print(
                    f"{i+1:3d} {'-':>5} "
                    f"{lpips_lin[i].item():9.4f} "
                    f"{debug_linear['cls'][i].item():9.4f} "
                    f"{debug_linear['reg'][i].item():9.4f} | "
                    f"{lpips_stat[0]:>13} {lpips_stat[1]:>9} {lpips_stat[2]:>9} | "
                    f"{cls_stat[0]:>13} {cls_stat[1]:>9} {cls_stat[2]:>9} | "
                    f"{reg_stat[0]:>13} {reg_stat[1]:>9} {reg_stat[2]:>9}"
                )
            

            print("\nLinear vs Best Riemannian (min across seeds):")
            print(
                f"{'Idx':>3} "
                f"{'LPIPS_L':>9} {'Cls_L':>9} {'Reg_L':>9} | "
                f"{'LPIPS_R_min':>12} {'Cls_R_min':>12} {'Reg_R_min':>12}"
            )
            print("-" * 72)
            for i in range(B):
                lpips_l = lpips_lin[i].item()
                cls_l   = debug_linear['cls'][i].item()
                reg_l   = debug_linear['reg'][i].item()

                lpips_min = lpips_riem[i].min().item()
                cls_min   = debug_riem["cls"].view(B, S)[i].min().item()
                reg_min   = debug_riem["reg"].view(B, S)[i].min().item()

                print(
                    f"{i+1:3d} "
                    f"{lpips_l:9.4f} {cls_l:9.4f} {reg_l:9.4f} | "
                    f"{lpips_min:12.4f} {cls_min:12.4f} {reg_min:12.4f}"
                )


    #save the images
    save_manipulation_images(orig, lin_out, riem_out_tensors, out_dir, B, S, debug_riem)

    # Save visualization
    comparison_path = os.path.join(out_dir, "comparison.png")
    save_side_by_side_comparison(
        orig, lin_out, riem_out_list, comparison_path,
        total_losses=debug_riem["total"].view(B, S)
    )



if __name__ == "__main__":
    main()
