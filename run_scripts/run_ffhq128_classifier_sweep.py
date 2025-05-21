# parallel_grid_search.py

import os
import sys
import json
import gc
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
import torch.multiprocessing as mp
from argparse import ArgumentParser
import time
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from choices import *
from templates_cls import *
from experiment_classifier_new import ClsModel

torch.set_float32_matmul_precision('high')

# ------------------------------------------------------------------------------
# Utility: ensure reproducibility
# ------------------------------------------------------------------------------
def set_seed(seed: int):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

# ------------------------------------------------------------------------------
# Callback to track best metrics without forcing a checkpoint
# ------------------------------------------------------------------------------
class ValidationMetricTracker(pl.Callback):
    def __init__(self):
        super().__init__()
        self.best = {}

    def on_validation_end(self, trainer, pl_module):
        # capture the "val_loss" and "val_loss_ema" at the end of each validation
        metrics = trainer.callback_metrics
        for key in ("val_loss", "val_loss_ema"):
            if key not in self.best or metrics[key] < self.best[key]:
                self.best[key] = float(metrics[key])

# ------------------------------------------------------------------------------
# Clear CUDA cache and garbage collect
# ------------------------------------------------------------------------------
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    # Force synchronize CUDA for complete memory clean-up
    torch.cuda.synchronize()

# ------------------------------------------------------------------------------
# Single-run training for one hyperparameter combo
# ------------------------------------------------------------------------------
def run_single(
    combo: dict,
    run_id: int,
    gpu_id: int,
    max_steps: int = 50_000,
    save_best_model: bool = True
) -> dict:
    # Clear memory from previous runs
    clear_memory()
    
    # 1) Build & patch the config
    conf = ffhq128_autoenc_flexibleclassifier_time_cls()
    
    # Apply standard parameters
    conf.lr = combo["lr"]
    conf.weight_decay = combo["weight_decay"]
    conf.non_linear_dropout = combo["non_linear_dropout"]
    conf.non_linear_hidden_dims = combo["non_linear_hidden_dims"]
    conf.time_embedding_dim = combo["time_embedding_dim"]
    conf.optimizer = OptimizerType(combo["optimizer"])
    conf.batch_size = combo["batch_size"]
    
    # Apply the classifier type from combo
    conf.classifier_type = combo["classifier_type"]
    
    # All classifiers are time-dependent
    conf.diffusion_time_dependent_classifier = True
    
    # Apply enhancement parameters from combo if needed
    if conf.classifier_type in ['enhanced', 'film']:
        conf.use_silu_activation = combo["use_silu_activation"]
        
    if conf.classifier_type == 'enhanced':
        conf.use_residual_connections = combo["use_residual_connections"]
    
    conf.seed = 1234 + run_id
    set_seed(conf.seed)

    # 2) Unique name and logging directory
    name = (
        f"gs_run{run_id:02d}"
        f"_{conf.classifier_type}"
        f"_lr{conf.lr}"
        f"_wd{conf.weight_decay}"
        f"_drop{conf.non_linear_dropout}"
        f"_hd{'-'.join(map(str, conf.non_linear_hidden_dims))}"
        f"_t{conf.time_embedding_dim}"
    )
    conf.name = name
    logdir = os.path.join(conf.base_dir, 'classifier_grid_search', name)
    os.makedirs(logdir, exist_ok=True)

    # 3) Logger and Callbacks
    logger = pl_loggers.TensorBoardLogger(save_dir=logdir, name="", version="")    
    early_stop = EarlyStopping(
        monitor="val_loss_ema",
        mode="min",
        patience=7,
        min_delta=1e-5,
        verbose=True,
    )
    tracker = ValidationMetricTracker()
    callbacks = [early_stop, tracker]

    if save_best_model:
        ckpt_cb = ModelCheckpoint(
            monitor="val_loss_ema",
            mode="min",
            dirpath=logdir,
            filename="best-{epoch:02d}-{val_loss_ema:.4f}",
            save_top_k=1
        )
        callbacks.append(ckpt_cb)

    # 4) Trainer with proper strategy to avoid memory leaks
    trainer = pl.Trainer(
        max_steps=max_steps,
        devices=[gpu_id],
        accelerator="gpu",
        precision="16-mixed",
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=getattr(conf, "accum_batches", 1),
        enable_checkpointing=save_best_model,
        enable_progress_bar=False,
        enable_model_summary=True,
        deterministic=True,
    )

    # 5) Fit
    model = ClsModel(conf)
    trainer.fit(model)
    
    # Get results before memory cleanup
    result = {
        "run_id": run_id,
        **combo,
        "best_val_loss": tracker.best.get("val_loss", float("inf")),
        "best_val_loss_ema": tracker.best.get("val_loss_ema", float("inf")),
        "final_step": trainer.global_step,
        "stopped_epoch": early_stop.stopped_epoch if early_stop.stopped_epoch >= 0 else None,
    }

    with open(os.path.join(logdir, "run_metrics.json"), "w") as fp:
        json.dump(result, fp, indent=2)

    return result

# ------------------------------------------------------------------------------
# Worker function for each GPU
# ------------------------------------------------------------------------------
def worker(gpu_id, combos, shared_results, completed_counter, log_dir, run_id_offset):
    # Set device
    torch.cuda.set_device(gpu_id)
    print(f"[GPU {gpu_id}] Starting worker with {len(combos)} configs")
    
    # Run all configs assigned to this GPU
    for i, combo in enumerate(combos):
        run_id = run_id_offset + i
        print(f"[GPU {gpu_id}] Running combo {run_id}: {combo}")
        try:
            # Reset CUDA before each run
            clear_memory()
            
            result = run_single(combo, run_id=run_id, gpu_id=gpu_id, max_steps=50_000, save_best_model=True)
            print(f"[GPU {gpu_id}] Completed run {run_id} with best_val_loss_ema: {result['best_val_loss_ema']:.4f}")
            shared_results.append(result)
            
            # Update completed counter
            with completed_counter.get_lock():
                completed_counter.value += 1
            
            # Explicitly clear memory again after each run
            clear_memory()
        except Exception as e:
            print(f"[GPU {gpu_id}] Error in run {run_id}: {str(e)}")
            # Try to continue with the next configuration
    
    print(f"[GPU {gpu_id}] Worker finished, processed {len(combos)} configs")

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="Parallel grid search across multiple GPUs")
    parser.add_argument('--gpus', type=str, default='0,1', 
                        help="Comma-separated GPU IDs to use (default: '0,1')")
    parser.add_argument('--save-models', action='store_true', default=True,
                        help="Save the best model for each configuration (default: True)")
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    
    # 1) Hyperparameter grid
    param_grid = {
        # Standard parameters (fixed for all runs)
        "lr": [5e-4],
        "weight_decay": [1e-6],
        "non_linear_dropout": [0.2, 0.3],
        "non_linear_hidden_dims": [[512, 256]],
        "time_embedding_dim": [64],
        "optimizer": ['adamw'],
        "batch_size": [64],
        
        # Enhancement parameters (used by enhanced/film architectures)
        "use_silu_activation": [True],
        "use_residual_connections": [True],
        
        # Architecture type - this is what we're varying
        "classifier_type": ['linear', 'flexible', 'enhanced'] #, 'film'] 
    }

    # Generate all combinations
    all_combos = list(ParameterGrid(param_grid))
    total_combos = len(all_combos)
    print(f"Generated {total_combos} hyperparameter combinations")
    
    # Create log directory
    first_conf = ffhq128_autoenc_flexibleclassifier_time_cls()
    grid_summary_dir = os.path.join(first_conf.base_dir, "classifier_grid_search")
    os.makedirs(grid_summary_dir, exist_ok=True)
    
    # Divide work among GPUs
    num_gpus = len(gpu_ids)
    combos_per_gpu = [all_combos[i::num_gpus] for i in range(num_gpus)]
    
    # Create a shared list for results and a counter for progress
    manager = mp.Manager()
    shared_results = manager.list()
    completed_counter = mp.Value('i', 0)
    
    # Start processes for each GPU
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        run_id_offset = i * (total_combos // num_gpus + 1)  # Different run IDs for each GPU
        p = mp.Process(
            target=worker,
            args=(gpu_id, combos_per_gpu[i], shared_results, completed_counter, grid_summary_dir, run_id_offset)
        )
        p.start()
        processes.append(p)
    
    # Setup progress bar in the main process
    pbar = tqdm(total=total_combos, desc="Grid search progress")
    last_count = 0
    
    # Monitor progress while workers are running
    try:
        while any(p.is_alive() for p in processes):
            current_count = completed_counter.value
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count
            time.sleep(1)  # Prevent busy waiting
    except KeyboardInterrupt:
        print("\nInterrupted by user. Waiting for workers to terminate...")
        for p in processes:
            p.terminate()
    finally:
        # Make sure progress bar shows the final state
        pbar.n = completed_counter.value
        pbar.refresh()
        pbar.close()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect and save results
    results = list(shared_results)
    df = pd.DataFrame(results)
    
    # Sort by performance
    if len(results) > 0 and 'best_val_loss_ema' in df.columns:
        df = df.sort_values('best_val_loss_ema')
    
    # Save results to CSV
    csv_path = os.path.join(grid_summary_dir, "grid_search_results.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nGrid search complete! Results saved to {csv_path}")
    print(f"Processed {len(results)}/{total_combos} configurations across {num_gpus} GPUs")
    
    # Print results by architecture type
    if len(results) > 0:
        print("\nResults by architecture type:")
        by_arch = df.groupby('classifier_type')['best_val_loss_ema'].agg(['mean', 'min', 'count'])
        by_arch = by_arch.sort_values('min')
        
        print("-" * 80)
        print(f"{'Architecture':<10} | {'Count':^6} | {'Best Val Loss':^12} | {'Mean Val Loss':^12}")
        print("-" * 80)
        
        for arch, stats in by_arch.iterrows():
            print(f"{arch:<10} | {stats['count']:^6.0f} | {stats['min']:^12.4f} | {stats['mean']:^12.4f}")
        
        # Print best configuration
        print("\nBest overall configuration:")
        best_row = df.iloc[0]
        print(f"Architecture: {best_row['classifier_type']}")
        print(f"Val Loss EMA: {best_row['best_val_loss_ema']:.4f}")

if __name__ == "__main__":
    # Required for Windows support, not needed for Linux/macOS
    mp.set_start_method('spawn', force=True)
    main()