import os
import random
import time
import torch
import pandas as pd
import numpy as np
from itertools import product
from argparse import ArgumentParser

# ==== Utilities imports from your project ====
from .utils import load_shared_resources, compute_median_logit
from .manipulation_utils import multiple_stage_ro


def load_or_compute_median_logits(ae, cls_lin, cls_nl, pos_dataset, cfg, cid, device):
    out_dir = os.path.join(cfg.get("log_dir", "logs"), cfg.get("target_attr"))
    os.makedirs(out_dir, exist_ok=True)
    median_path = os.path.join(out_dir, "median_logits.pt")

    if os.path.exists(median_path):
        data = torch.load(median_path, map_location=device)
        print(f"Loaded median logits from {median_path}")
        return data['linear'], data['non_linear']

    lin = compute_median_logit(ae, cls_lin, cid, pos_dataset, cfg, device)
    nl = compute_median_logit(ae, cls_nl, cid, pos_dataset, cfg, device)
    torch.save({'linear': lin, 'non_linear': nl}, median_path)
    print(f"Saved median logits to {median_path}")
    return lin, nl


def evaluate_single_run(ae, cls_lin, cls_nl, batch, nl_med, cfg, cid, device, grid_keys):
    _, debug_riem = multiple_stage_ro(
        ae, cls_nl, batch, nl_med, cfg, cid, device, debug=True
    )

    cls_loss = debug_riem['cls'].mean().item()
    reg_loss = debug_riem['reg'].mean().item()
    total_loss = debug_riem['total'].mean().item()


    # Include all swept parameters
    result = {k: cfg[k] for k in grid_keys}
    result.update({
        'cls_loss': cls_loss,
        'reg_loss': reg_loss,
        'total_loss': total_loss,
    })
    return result


def run_sweep(cfg_yaml, grid_params):
    ae, cls_nl, cls_lin, dataset, pos_dataset, neg_indices, base_cfg, cid, device = \
        load_shared_resources(cfg_yaml)
    
    _, nl_med = load_or_compute_median_logits(
        ae, cls_lin, cls_nl, None, base_cfg, cid, device
    )

    idxs = neg_indices[: base_cfg['num_samples']]
    batch = torch.stack([dataset[i]['img'] for i in idxs]).to(device)

    grid = list(product(*grid_params.values()))
    total = len(grid)
    random.shuffle(grid)  # shuffle configs to get better ETA
    print(f"Starting sweep of {total} runs...")

    times = []
    rows = []
    grid_keys = list(grid_params.keys())

    for i, vals in enumerate(grid, 1):
        start = time.time()

        cfg = base_cfg.copy()
        cfg.update(dict(zip(grid_keys, vals)))

        swept_cfg = {k: cfg[k] for k in grid_keys}
        print(f"Run {i}/{total} - Swept config: {swept_cfg}")

        result = evaluate_single_run(ae, cls_lin, cls_nl, batch, nl_med, cfg, cid, device, grid_keys)
        rows.append(result)

        times.append(time.time() - start)
        avg_time = sum(times) / len(times)
        eta = avg_time * (total - i)
        m, s = divmod(int(eta), 60)
        print(f"ETA for remaining runs: {m}m{s:02d}s")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(base_cfg['log_dir'], base_cfg['target_attr'])
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'sweep_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved sweep results to {csv_path}")
    return csv_path, df


def main():
    parser = ArgumentParser(description="Sweep runner.")
    parser.add_argument('--ro-config', required=True,
                        help="Path to Riemannian-optimization config YAML.")
    args = parser.parse_args()

    grid_params = {
        'multistage_steps': [11],  # [6, 11, 16]
        'start_diffusion_timestep': [20],  # [10, 20, 30]  
        'riemannian_steps': [2],  # [1, 2, 3]
        'reg_norm_weight': [0.4],  # [0.3, 0.35, 0.4]
        'wolfe_c1': [5e-3],  # [1e-4, 1e-3]
        'wolfe_c2': [0.4, 0.5],
        'cg_precond_diag_samples': [10],
        'cg_max_iter':[13, 15, 17],
        'reg_lambda':[1e-5, 1e-6],
        'max_bracket':[12, 13],
        'riemannian_lr_init': [2.5e-3, 5e-3, 1e-2]
    }

    run_sweep(args.ro_config, grid_params)


if __name__ == '__main__':
    main()
