import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70" 

import numpy as np
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm # type: ignore
import pickle
import jax
import jax.numpy as jnp
from ipdb import set_trace
import gc

from functools import partial
from jaxopt import OSQP
from glue_module.glue_analysis import run_glue_solver
import data_utils
from plastic_analysis import _compute_weight_metrics_batch, _compute_rep_metrics_batch, run_plastic_analysis_pipeline

# --- Helper to generate 2^n indices up to a limit ---
def get_pow2_indices(limit):
    indices = [0]
    i = 1
    while i < limit:
        indices.append(i)
        i *= 3
    
    # Optionally ensure the absolute last index is included
    last_idx = limit - 1
    if last_idx > 0 and (not indices or indices[-1] != last_idx):
        indices.append(last_idx)
        
    return indices


@jax.jit(static_argnames=('P', 'M'))
def prep_glue_batch_jax(reps_batch, labels_batch, P, M):
    """
    Pure JAX, fully vectorized version of _prep_glue_batch.
    Executes entirely on the GPU without CPU control flow.
    """
    R, S, H = reps_batch.shape

    def process_repeat(reps, labels):
        def process_class(p):
            # 1. Identify points belonging to class p
            is_class = (labels == p) 
            count = jnp.sum(is_class)
            
            # 2. Sort to push all valid class points to the front
            # False (0) sorts before True (1). Reversing puts True at the front.
            sorted_idxs = jnp.argsort(is_class)[::-1] 
            
            # 3. Handle Tiling dynamically (replaces np.tile and np.where)
            # jnp.arange(M) % safe_count loops over valid indices exactly up to M
            safe_count = jnp.maximum(count, 1)
            tiled_idxs = sorted_idxs[jnp.arange(M) % safe_count]
            
            class_points = reps[tiled_idxs]
            
            # 4. Handle empty class case
            # Using a tiny constant avoids needing to pass a PRNGKey just for noise
            empty_fallback = jnp.ones((M, H)) * 1e-4 
            
            return jax.lax.select(
                count == 0,
                empty_fallback,
                class_points
            )
            
        # Vectorize across all P classes
        return jax.vmap(process_class)(jnp.arange(P))

    # Vectorize across all R repeats
    return jax.vmap(process_repeat)(reps_batch, labels_batch)


def run_glue_analysis_pipeline(config):
    print("\n--- Starting GLUE Metric Analysis (PMAP) ---", flush=True)
    
    metric_names = [
        'Capacity', 'Dimension', 'Radius', 
        'Center Alignment', 'Axis Alignment', 'Center-Axis Alignment', 
        'Approx Capacity'
    ]
    
    P = 2 # always two with binary classification
    M = config.analysis_subsamples 
    N = config.hidden_dim
    n_t = config.n_t #//2 # decrease compute
    
    master_key = jax.random.PRNGKey(config.seed)
    all_devices = jax.devices()
    num_devices = len(all_devices)
    print(f"Available JAX devices for PMAP: {num_devices}")

    full_results = {}

    # --- 2. Process Training Tasks ---
    for train_task_idx in [0,1]: # range(config.num_tasks): # look at just the first two tasks fully
        train_task_name = f"task_{train_task_idx:03d}"
        task_dir = os.path.join(config.results_dir, train_task_name)
        
        reps_path = os.path.join(task_dir, "representations.npy")
        lbls_path = os.path.join(task_dir, "binary_labels.npy") # .npy for SL
        # set_trace()
        if not os.path.exists(reps_path) or not os.path.exists(lbls_path):
            print(f"Skipping {train_task_name} (Files not found)")
            continue

        print(f"Processing Training Phase: {train_task_name}...")
        
        # Load Data
        reps_data = np.load(reps_path, allow_pickle=True) # (L, R, T_eval, S, H)
        lbls_data = np.load(lbls_path, allow_pickle=True) # (R, T_eval, S)
        # set_trace()
        lbls_dict = lbls_data
        # with open(lbls_path, 'rb') as f:
        #     lbls_dict = pickle.load(f)

        # 2. Extract labels, squeeze the trailing dimension, and swap Samples/Repeats axes
        formatted_lbls = []
        for t_name in sorted(lbls_dict.keys()):
            labels = lbls_dict[t_name][1]                # Shape: (400, 32, 1)
            labels_reshaped = np.array(labels).squeeze(-1).T  # Shape: (32, 400)
            formatted_lbls.append(labels_reshaped)

        # 3. Stack tasks along axis 1 to get (Repeats, Tasks, Samples)
        lbls_data = np.stack(formatted_lbls, axis=1)     # New Shape: (32, 20, 400)
        
        L, R_original, T_eval, S, H = reps_data.shape
        
        # Determine device mapping and repeats per GPU
        R_per_dev = R_original // num_devices
        
        if R_per_dev == 0:
            # Fallback if fewer repeats than devices
            R_per_dev = 1
            num_active_devices = R_original
            R_kept = R_original
        else:
            num_active_devices = num_devices
            R_kept = R_per_dev * num_devices

        if R_original > R_kept:
            print(f"  Truncating repeats from {R_original} -> {R_kept} ({R_per_dev} per GPU) to evenly distribute.")
            
        # Select the exact devices we are mapping to
        active_devices = all_devices[:num_active_devices]
        
        # --- 1. JIT-Compiled Compute Step (PMAP + VMAP) ---
        def compute_glue_batch(keys, data):
            local_qp = OSQP(tol=1e-4)
            # vmap across the repeats assigned to this single device
            def single_repeat(k, d):
                return run_glue_solver(k, d, P=P, M=M, N=N, n_t=n_t, qp_solver=local_qp)
            return jax.vmap(single_repeat)(keys, data)
            
        pmapped_glue_step = jax.pmap(
            compute_glue_batch,
            in_axes=(0, 0),
            devices=active_devices
        )

        # Slice down to the evenly divisible amount
        reps_data = reps_data[:, :R_kept, ...]
        lbls_data = lbls_data[:R_kept, ...]
        
        full_results[train_task_name] = {}

        for eval_task_idx in [0, 1]: # was [0,2,10]
            eval_task_name = f"task_{eval_task_idx:03d}"
            
            # 1. Temporary storage for JAX device arrays
            gpu_results = {m: [] for m in metric_names}
            
            current_eval_lbls = lbls_data[:, eval_task_idx, :] # (R, S)
            
            for step in tqdm(range(L), desc=f"  {eval_task_name}", leave=False):
            # for step in [0,2,5,L-1]:
                current_reps = reps_data[step, :, eval_task_idx, :, :] # (R_kept, S, H)
                formatted_data = prep_glue_batch_jax(current_reps, current_eval_lbls, P, M) 
                
                step_key = jax.random.fold_in(master_key, train_task_idx * 10000 + step)
                batch_keys = jax.random.split(step_key, R_kept)
                
                sharded_keys = batch_keys.reshape((num_active_devices, R_per_dev) + batch_keys.shape[1:])
                sharded_data = formatted_data.reshape(num_active_devices, R_per_dev, P, M, H)
                
                # --- Dispatch to GPU (Asynchronous) ---
                metrics_tuple_pmap = pmapped_glue_step(sharded_keys, sharded_data)
                
                # 2. Extract ONLY the scalar metrics we want to keep
                step_metrics_only = {}
                for i, name in enumerate(metric_names):
                    if isinstance(metrics_tuple_pmap, tuple) and len(metrics_tuple_pmap) == 3:
                        metric_val = metrics_tuple_pmap[0][i]
                    else:
                        metric_val = metrics_tuple_pmap[i]
                    
                    # Store the JAX device array pointer (DO NOT cast to numpy here)
                    step_metrics_only[name] = metric_val

                # 3. Prevent OOM: Force evaluation of the tiny metrics, then delete the massive arrays
                # jax.block_until_ready() forces the GPU to finish calculating this specific step.
                # Because step_metrics_only uses negligible VRAM, blocking here is safe.
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), step_metrics_only)
                
                # Explicitly delete the full output tuple to free up plotting_inputs (Gs, anchor_points, etc.)
                del metrics_tuple_pmap 
                
                # Append to our tracking lists
                for name in metric_names:
                    gpu_results[name].append(step_metrics_only[name])

            # 4. End of Loop: Sync and transfer to CPU ONCE
            full_results[train_task_name][eval_task_name] = {m: [] for m in metric_names}
            
            for name in metric_names:
                # Stack all steps along axis 0 (Still on GPU)
                stacked_gpu_metric = jnp.stack(gpu_results[name], axis=0)
                
                # JAX -> NumPy transfer happens exactly ONE time per metric here
                cpu_metric = jax.device_get(stacked_gpu_metric) 
                
                # Reshape to flatten device and repeat dimensions: (Steps, R_kept, ...)
                flat_shape = (cpu_metric.shape[0], R_kept) + cpu_metric.shape[3:]
                
                full_results[train_task_name][eval_task_name][name] = cpu_metric.reshape(flat_shape)

    # --- 3. Save Results ---
    save_path = os.path.join(config.results_dir, "glue_metrics_2_tasks.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(full_results, f)
    print(f"GLUE metrics saved to {save_path}")

    # Plot
    # _plot_glue_results(full_results, metric_names, config)


def _plot_glue_results(full_results, metric_names, config):
    if not full_results:
        return

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)
    if n_metrics == 1: axes = [axes]
    
    train_tasks = sorted(list(full_results.keys()))
    timeline = {m: {t: {'mean': [], 'std': []} for t in train_tasks} for m in metric_names}
    
    task_boundaries = []
    current_x = 0
    
    for t_train in train_tasks:
        steps = full_results[t_train][t_train]['Capacity'].shape[0]
        epochs_per_step = config.log_frequency
        segment_len = steps * epochs_per_step
        
        current_x += segment_len
        task_boundaries.append(current_x)
        
        for t_eval in train_tasks:
            if t_eval not in full_results[t_train]:
                continue
                
            for m in metric_names:
                data = full_results[t_train][t_eval][m]
                mean = np.nanmean(data, axis=1)
                std = np.nanstd(data, axis=1)
                
                timeline[m][t_eval]['mean'].append(mean)
                timeline[m][t_eval]['std'].append(std)

    x_axis = np.arange(0, current_x, config.log_frequency)
    colors = plt.cm.tab10(np.linspace(0, 1, len(train_tasks)))
    
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        
        for j, t_eval in enumerate(train_tasks):
            means = timeline[metric][t_eval]['mean']
            stds = timeline[metric][t_eval]['std']
            
            if not means: continue
            
            full_mean = np.concatenate(means)
            full_std = np.concatenate(stds)
            
            limit = min(len(x_axis), len(full_mean))
            
            ax.plot(x_axis[:limit], full_mean[:limit], label=t_eval, color=colors[j], linewidth=2)
            ax.fill_between(x_axis[:limit], 
                            full_mean[:limit] - full_std[:limit], 
                            full_mean[:limit] + full_std[:limit], 
                            color=colors[j], alpha=0.1)

        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        for boundary in task_boundaries[:-1]:
            ax.axvline(x=boundary, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

        if i == 0:
            ax.set_title(f"GLUE Metrics Analysis - {config.dataset_name}")
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Evaluated Task")

    axes[-1].set_xlabel('Epochs')
    plt.tight_layout()
    
    save_path = os.path.join(config.figures_dir, f"glue_metrics_{config.dataset_name}.png")
    plt.savefig(save_path)
    print(f"GLUE plots saved to {save_path}")


def run_all_representation_analysis(experiment_path):
    """
    Orchestrates the analysis suite for an experiment.
    1. Runs Plasticity Analysis (Drift, Rank, etc.)
    2. Runs GLUE Analysis (Manifold Geometry)
    3. Aggregates CL, Plasticity, and GLUE metrics.
    """
    config_path = os.path.join(experiment_path, "config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    config.results_dir = experiment_path
    config.reps_dir = experiment_path 
    config.figures_dir = os.path.join(experiment_path, "figures")
    os.makedirs(config.figures_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STARTING FULL EXPERIMENT ANALYSIS: {config.dataset_name} / {config.algorithm}")
    print(f"Target Path: {experiment_path}")
    print(f"{'='*60}")

    # --- 1. Execute Individual Analysis Pipelines ---
    # try:
    #     run_plastic_analysis_pipeline(config)
    # except Exception as e:
    #     print(f"Error in Plasticity Pipeline: {e}")

    # try:
    #     
    # except Exception as e:
    #     print(f"Error in GLUE Pipeline: {e}")
    run_glue_analysis_pipeline(config)

    # --- 2. Aggregate All Metrics into Single Object ---
    print(f"\n{'='*60}")
    print(f"Aggregating All Metrics (No Aggregation Across Repeats)")
    print(f"{'='*60}")

    all_metrics = {
        'cl': {},
        'plasticity': {},
        'glue': {}
    }

    # A. Collect CL Metrics (Accuracy/Loss)
    hist_path = os.path.join(config.results_dir, "global_history.pkl")
    if os.path.exists(hist_path):
        with open(hist_path, 'rb') as f:
            all_metrics['cl'] = pickle.load(f)
            print(f"  [x] Loaded CL metrics (Global History)")
    else:
        print(f"  [ ] Missing global_history.pkl")

    # B. Collect Plasticity Metrics
    plast_path = os.path.join(config.reps_dir, f"plastic_analysis_{config.dataset_name}.pkl")
    if os.path.exists(plast_path):
        with open(plast_path, 'rb') as f:
            data = pickle.load(f)
            all_metrics['plasticity'] = data.get('history', {})
            print(f"  [x] Loaded Plasticity metrics")
    else:
        print(f"  [ ] Missing plastic_analysis data")

    # C. Collect GLUE Metrics
    glue_path = os.path.join(config.results_dir, "glue_metrics.pkl")
    if os.path.exists(glue_path):
        with open(glue_path, 'rb') as f:
            all_metrics['glue'] = pickle.load(f)
            print(f"  [x] Loaded GLUE metrics")
    else:
        print(f"  [ ] Missing glue_metrics.pkl")

    # --- 3. Save Master Object ---
    save_path = os.path.join(config.results_dir, "all_metrics_2_tasks.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print(f"\nSUCCESS: All aggregated metrics saved to:\n  -> {save_path}")


if __name__=='__main__':
    experiment_path = "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.0001_lr2_0.01"
    run_all_representation_analysis(experiment_path)