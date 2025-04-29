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

    Returns:
        rendered_images: list of numpy arrays, each of shape (B, H, W, 3) uint8
    """
    n_steps = len(trajectory)
    batch_size = trajectory[0].size(0)
    rendered_images = []

    if not fast_mode:
        # generate images step by step
        for step in range(n_steps):
            latent_flat = trajectory[step]
            latent_unflat = unflatten_tensor(latent_flat, latent_shape)
            imgs = model.render(xT, latent_unflat, T=T_render)
            imgs = imgs.clamp(0.0, 1.0)
            imgs_np = (
                imgs
                .mul(255)
                .byte()
                .cpu()
                .permute(0, 2, 3, 1)
                .numpy()
            )
            rendered_images.append(imgs_np)

    else:
        # fast batch rendering with optional chunking
        # flatten all latents and repeat xT
        all_latents = torch.cat([
            unflatten_tensor(latent, latent_shape) for latent in trajectory
        ], dim=0)
        # all_latents: (n_steps*B, C, H, W)
        xT_repeated = xT.repeat(n_steps, 1, 1, 1)
        total = all_latents.shape[0]
        # choose chunk size if provided, else all at once
        if chunk_size is None or chunk_size >= total:
            imgs = model.render(xT_repeated, all_latents, T=T_render)
        else:
            chunks = []
            n_chunks = math.ceil(total / chunk_size)
            for i in tqdm(range(0, total, chunk_size),
                          desc="Rendering chunks",
                          total=n_chunks):
                lat_chunk = all_latents[i : i + chunk_size]
                xT_chunk = xT_repeated[i : i + chunk_size]
                chunk_imgs = model.render(xT_chunk, lat_chunk, T=T_render)
                chunks.append(chunk_imgs)
            imgs = torch.cat(chunks, dim=0)

        # normalize and reshape
        imgs = imgs.clamp(0.0, 1.0)
        imgs = imgs.view(n_steps, batch_size, *imgs.shape[1:])
        # to numpy list
        for step in range(n_steps):
            imgs_np = (
                imgs[step]
                .mul(255)
                .byte()
                .cpu()
                .permute(0, 2, 3, 1)
                .numpy()
            )
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
