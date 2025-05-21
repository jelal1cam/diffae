import os
import sys
import random
import time
import torch
import pandas as pd
import numpy as np
from itertools import product
from argparse import ArgumentParser
import torch.multiprocessing as mp

# ==== Utilities imports from your project ====
from .utils import load_shared_resources, compute_median_logit
from .manipulation_utils import multiple_stage_ro
from ..config_loader import load_riemannian_config


def load_or_compute_median_logits(ae, cls_lin, cls_nl, pos_dataset, cfg, cid, device, indent=""):
    out_dir = os.path.join(cfg.get("log_dir", "logs"), cfg.get("target_attr"))
    os.makedirs(out_dir, exist_ok=True)
    median_path = os.path.join(out_dir, "median_logits.pt")

    if os.path.exists(median_path):
        data = torch.load(median_path, map_location=device)
        print(f"{indent}[GPU {device.index}] Loaded median logits from {median_path}")
        return data['linear'], data['non_linear']

    lin = compute_median_logit(ae, cls_lin, cid, pos_dataset, cfg, device)
    nl = compute_median_logit(ae, cls_nl, cid, pos_dataset, cfg, device)
    torch.save({'linear': lin, 'non_linear': nl}, median_path)
    print(f"{indent}[GPU {device.index}] Saved median logits to {median_path}")
    return lin, nl


def evaluate_single_run(ae, cls_lin, cls_nl, batch, nl_med, cfg, cid, device, grid_keys):
    _, debug_riem = multiple_stage_ro(ae, cls_nl, batch, nl_med, cfg, cid, device, debug=True)
    cls_loss = debug_riem['cls'].mean().item()
    reg_loss = debug_riem['reg'].mean().item()
    total_loss = debug_riem['total'].mean().item()

    result = {k: cfg[k] for k in grid_keys}
    result.update({'cls_loss': cls_loss, 'reg_loss': reg_loss, 'total_loss': total_loss})
    return result


def worker(local_rank, device_id, ro_config, grid, grid_keys, shared_list):
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    indent = '\t' * local_rank

    base_cfg = load_riemannian_config(ro_config)
    base_out = os.path.join(base_cfg.get("log_dir", "logs"), base_cfg.get("target_attr"))
    logs_dir = os.path.join(base_out, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Open a dedicated log file for this GPU and redirect all output
    log_path = os.path.join(logs_dir, f"gpu_{device_id}.log")
    log_file = open(log_path, 'w', buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"{indent}[GPU {device_id}] Starting process with {len(grid)} configs...")

    ae, cls_nl, cls_lin, dataset, pos_dataset, neg_indices, base_cfg, cid, _ = \
        load_shared_resources(ro_config, device=device)

    _, nl_med = load_or_compute_median_logits(ae, cls_lin, cls_nl, pos_dataset, base_cfg, cid, device, indent)

    idxs = neg_indices[: base_cfg['num_samples']]
    batch = torch.stack([dataset[i]['img'] for i in idxs]).to(device)

    local_rows = []
    times = []
    total = len(grid)

    for i, vals in enumerate(grid, 1):
        start = time.time()
        cfg = base_cfg.copy()
        cfg.update(dict(zip(grid_keys, vals)))

        print(f"{indent}[GPU {device_id}] Run {i}/{total} - Swept config: { {k: cfg[k] for k in grid_keys} }")
        result = evaluate_single_run(ae, cls_lin, cls_nl, batch, nl_med, cfg, cid, device, grid_keys)
        local_rows.append(result)

        elapsed = time.time() - start
        times.append(elapsed)
        avg = sum(times) / len(times)
        eta = avg * (total - i)
        m, s = divmod(int(eta), 60)
        print(f"{indent}[GPU {device_id}] ETA: {m}m{s:02d}s")

    shared_list.extend(local_rows)
    print(f"{indent}[GPU {device_id}] Finished all {total} runs.")
    log_file.close()


def main():
    parser = ArgumentParser(description="Parallel sweep runner.")
    parser.add_argument('--ro-config', required=True,
                        help="Path to Riemannian-optimization config YAML.")
    parser.add_argument('--gpus', type=str, default='-1',
                        help="Comma-separated GPU indices to use, or -1 for all available.")
    args = parser.parse_args()

    # GPU selection
    if args.gpus.strip() == '-1':
        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = [int(x) for x in args.gpus.split(',')]
    assert gpus, "No GPUs specified or available."

    base_cfg = load_riemannian_config(args.ro_config)

    # Build grid
    grid_params = base_cfg.get('grid_params', {
        'multistage_steps': [11],
        'start_diffusion_timestep': [20],
        'riemannian_steps': [2],
        'reg_norm_weight': [0.9, 1.1, 1.3],
        'wolfe_c1': [5e-3],
        'wolfe_c2': [0.35, 0.45, 0.55],
        'cg_precond_diag_samples': [10],
        'cg_max_iter': [13],
        'reg_lambda': [1e-5],
        'max_bracket': [13],
        'riemannian_lr_init': [1e-2]
    })

    full_grid = list(product(*grid_params.values()))
    random.shuffle(full_grid)
    grid_keys = list(grid_params.keys())

    num_workers = len(gpus)
    chunks = [full_grid[i::num_workers] for i in range(num_workers)]

    manager = mp.Manager()
    shared_results = manager.list()

    # Spawn processes
    processes = []
    for local_rank, gpu_id in enumerate(gpus):
        p = mp.Process(
            target=worker,
            args=(local_rank, gpu_id, args.ro_config, chunks[local_rank], grid_keys, shared_results)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Save final results
    final_df = pd.DataFrame(list(shared_results))
    out_dir = os.path.join(base_cfg['log_dir'], base_cfg['target_attr'])
    os.makedirs(out_dir, exist_ok=True)
    final_csv = os.path.join(out_dir, 'sweep_results.csv')
    final_df.to_csv(final_csv, index=False)
    print(f"Main Saved combined sweep results to {final_csv}")


if __name__ == '__main__':
    main()
