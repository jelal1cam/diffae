import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import unflatten_tensor
import imageio
import math
from tqdm import tqdm 

def render_trajectory_images(
    model,
    xT,
    trajectory,
    latent_shape,
    T_render,
    fast_mode=False,
    chunk_size=None
):
    """
    Renders the optimization trajectory by generating images at each optimization step.

    Args:
        model: autoencoder or diffusion model with .render(subcode, latent, T) method
        xT: stochastic subcode(s), either:
             - 4D tensor (B, C, H, W) to be reused across all steps (backward-compatible)
             - 5D tensor (n_steps, B, C, H, W) for per-step subcodes
        trajectory: list of flattened latent tensors for each step
        latent_shape: shape tuple for unflattening latents (C, H, W)
        T_render: number of diffusion steps for rendering
        fast_mode: if True, batch-render all frames (faster)
        chunk_size: optional chunk size for memory-efficient rendering

    Returns:
        rendered_images: list of numpy arrays, each of shape (B, H, W, 3) uint8
    """
    n_steps = len(trajectory)
    batch_size = trajectory[0].size(0)

    # Unflatten all trajectory latents into a single batch
    all_latents = torch.cat([
        unflatten_tensor(latent, latent_shape) for latent in trajectory
    ], dim=0)

    # Determine xT layout and prepare all_xT accordingly
    if xT.ndim == 5:
        # Per-step subcodes: (n_steps, B, C, H, W)
        all_xT = xT.view(-1, *xT.shape[2:])
    elif xT.ndim == 4:
        # Constant subcode: repeat for each step (backward-compatible)
        all_xT = xT.repeat(n_steps, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported xT shape: {xT.shape}, expected 4D or 5D tensor")

    rendered_images = []

    if not fast_mode:
        # Render frame-by-frame to use individual subcodes or constant code
        for step in range(n_steps):
            latent_unflat = unflatten_tensor(trajectory[step], latent_shape)
            subcode = xT[step] if xT.ndim == 5 else xT
            imgs = model.render(subcode, latent_unflat, T=T_render)
            imgs = imgs.clamp(0.0, 1.0)
            imgs_np = imgs.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
            rendered_images.append(imgs_np)
    else:
        # Fast batch rendering with optional chunking
        total = all_latents.shape[0]
        if chunk_size is None or chunk_size >= total:
            imgs = model.render(all_xT, all_latents, T=T_render)
        else:
            chunks = []
            n_chunks = math.ceil(total / chunk_size)
            for i in tqdm(range(0, total, chunk_size), desc="Rendering chunks", total=n_chunks):
                lat_chunk = all_latents[i : i + chunk_size]
                xT_chunk = all_xT[i : i + chunk_size]
                chunks.append(model.render(xT_chunk, lat_chunk, T=T_render))
            imgs = torch.cat(chunks, dim=0)

        # Normalize and split back into per-step frames
        imgs = imgs.clamp(0.0, 1.0)
        imgs = imgs.view(n_steps, batch_size, *imgs.shape[1:])
        for step in range(n_steps):
            imgs_np = imgs[step].mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
            rendered_images.append(imgs_np)

    return rendered_images


def visualize_trajectory(
    rendered_images,
    save_path=None
):
    """
    Plots a grid of images for each step in the optimization trajectory.

    Args:
        rendered_images: list of numpy arrays, each (B, H, W, 3)
        save_path: optional path to save the figure
    """
    n_steps = len(rendered_images)
    batch_size = rendered_images[0].shape[0]

    fig, axes = plt.subplots(
        batch_size,
        n_steps,
        figsize=(n_steps * 3, batch_size * 3)
    )
    # ensure axes is 2D
    if batch_size == 1 and n_steps == 1:
        axes = np.array([[axes]])
    elif batch_size == 1:
        axes = np.expand_dims(axes, axis=0)
    elif n_steps == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(batch_size):
        for j in range(n_steps):
            ax = axes[i, j]
            ax.imshow(rendered_images[j][i])
            ax.axis('off')
            if i == 0:
                ax.set_title(f"Step {j}")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_gif_from_rendered_images(rendered_images, gif_path, duration_sec=15):
    """
    Save a GIF showing a grid of samples evolving over optimization steps.
    Arranges each frame into a nearly square (R x C) grid of images.
    """
    num_frames = len(rendered_images)
    batch_size = rendered_images[0].shape[0]
    H, W, C = rendered_images[0][0].shape
    fps = max(1, num_frames // duration_sec)

    # Compute grid size as close to square as possible
    grid_cols = math.ceil(math.sqrt(batch_size))
    grid_rows = math.ceil(batch_size / grid_cols)

    print(f"Saving square grid GIF with {num_frames} frames of {grid_rows}x{grid_cols} layout...")

    frames = []
    for t in range(num_frames):
        imgs = rendered_images[t]  # shape: (B, H, W, 3)

        # Pad with black images if needed
        padded = list(imgs)
        while len(padded) < grid_rows * grid_cols:
            padded.append(np.zeros((H, W, C), dtype=np.uint8))

        # Build grid row by row
        rows = []
        for r in range(grid_rows):
            row_imgs = padded[r * grid_cols:(r + 1) * grid_cols]
            row = np.concatenate(row_imgs, axis=1)
            rows.append(row)
        frame = np.concatenate(rows, axis=0)

        frames.append(frame)

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"Square grid GIF saved to {gif_path}")

def save_comparison_image(rendered_images, save_path):
    """
    Save a side-by-side comparison of the first and last optimization steps.
    """
    first_imgs = rendered_images[0]   # shape (B, H, W, 3)
    last_imgs = rendered_images[-1]   # shape (B, H, W, 3)
    batch_size = first_imgs.shape[0]

    fig, axes = plt.subplots(
        batch_size,
        2,  # only first and last
        figsize=(2 * 3, batch_size * 3)
    )
    if batch_size == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(batch_size):
        axes[i, 0].imshow(first_imgs[i])
        axes[i, 0].axis('off')
        axes[i, 0].set_title("Start")

        axes[i, 1].imshow(last_imgs[i])
        axes[i, 1].axis('off')
        axes[i, 1].set_title("End")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_comparison_trajectory(
    rendered_images_riem,
    rendered_images_linear,
    save_path=None
):
    """
    Plots a comparison grid of images for Riemannian vs. Linear trajectories.
    Each row pair shows:
    Row 2*i:   Riemannian trajectory for sample i
    Row 2*i+1: Linear trajectory for sample i

    Args:
        rendered_images_riem: List of numpy arrays [n_steps], each (B, H, W, 3) uint8.
        rendered_images_linear: List of numpy arrays [n_steps], each (B, H, W, 3) uint8.
        save_path: Optional path to save the comparison figure.
    """
    n_steps = len(rendered_images_riem)
    if n_steps == 0:
        print("No images to visualize.")
        return
    assert len(rendered_images_linear) == n_steps, "Both trajectories must have the same length."
    batch_size = rendered_images_riem[0].shape[0]
    assert rendered_images_linear[0].shape[0] == batch_size, "Batch sizes must match."

    print(f"Creating comparison plot with {n_steps} steps for {batch_size} samples...")

    fig, axes = plt.subplots(
        2 * batch_size,  # Two rows per sample (Riem top, Linear bottom)
        n_steps,         # Columns represent time steps
        figsize=(n_steps * 2, 2 * batch_size * 2), # Adjust size as needed
        squeeze=False # Always return 2D array for axes
    )

    for i in range(batch_size): # Iterate through samples in the batch
        for j in range(n_steps): # Iterate through time steps
            # Axis for Riemannian trajectory, sample i, step j
            ax_riem = axes[2 * i, j]
            ax_riem.imshow(rendered_images_riem[j][i])
            ax_riem.axis('off')
            if j == 0: # Label first column
                ax_riem.text(-0.05, 0.5, f'S{i}\nRiem', horizontalalignment='right', verticalalignment='center', transform=ax_riem.transAxes, fontsize=8, rotation=0)
            if i == 0: # Add step title to the top row
                ax_riem.set_title(f"Step {j}", fontsize=10)

            # Axis for Linear trajectory, sample i, step j
            ax_linear = axes[2 * i + 1, j]
            ax_linear.imshow(rendered_images_linear[j][i])
            ax_linear.axis('off')
            if j == 0: # Label first column
                 ax_linear.text(-0.05, 0.5, f'S{i}\nLinear', horizontalalignment='right', verticalalignment='center', transform=ax_linear.transAxes, fontsize=8, rotation=0)

    # Adjust layout to prevent labels overlapping titles/images
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.05)

    if save_path is not None:
        print(f"Saving comparison plot to {save_path}")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else:
        print("[Warning] save_path not provided for comparison plot, plot not saved.")
    plt.close(fig) # Close the figure to free memory