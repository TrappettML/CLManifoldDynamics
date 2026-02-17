import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70" 

import numpy as np
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm
import os
import pickle
import jax
import jax.numpy as jnp

from functools import partial
from jaxopt import OSQP
from glue_module.glue_analysis import run_glue_solver
import data_utils
from plastic_analysis import _compute_weight_metrics_batch, _compute_rep_metrics_batch, run_plastic_analysis_pipeline


def _prep_glue_batch(reps_batch, labels_batch, P, M, H):
    """
    Python helper to organize raw mixed data into GLUE format.
    Input:
        reps_batch: (R, S, H) - Raw representations
        labels_batch: (R, S)  - Binary labels (0 or 1)
    Output:
        formatted_data: (R, P, M, H) - Padded/Repeated to ensure M points per class
    """
    R, S, _ = reps_batch.shape
    formatted = np.zeros((R, P, M, H), dtype=np.float32)

    for r in range(R):
        for p in range(P):
            # Extract points for class p
            # Note: labels are 0.0/1.0 floats or ints
            idxs = np.where(labels_batch[r] == p)[0]
            points = reps_batch[r, idxs, :]
            
            count = len(points)
            if count == 0:
                # Fallback for empty class (rare but possible in small subsamples)
                # Fill with zeros or random noise to prevent crash, though GLUE will be junk
                formatted[r, p, :, :] = np.zeros((M, H))
            elif count >= M:
                formatted[r, p, :, :] = points[:M, :]
            else:
                # If we have fewer points than M, tile them to fill M
                # This satisfies the shape requirement for the solver
                repeats = (M // count) + 1
                tiled = np.tile(points, (repeats, 1))
                formatted[r, p, :, :] = tiled[:M, :]
                
    return jnp.array(formatted)

def run_glue_analysis_pipeline(config):
    print("\n--- Starting GLUE Metric Analysis ---")
    
    # --- 1. Setup ---
    # Metrics map based on run_glue_solver output tuple indices
    # (capacity, dimension, radius, center_align, axis_align, center_axis_align, approx_capacity)
    metric_names = [
        'Capacity', 'Dimension', 'Radius', 
        'Center Alignment', 'Axis Alignment', 'Center-Axis Alignment', 
        'Approx Capacity'
    ]
    
    # Constants
    P = 2 # Always binary classification per task in this setup
    M = config.analysis_subsamples # As requested
    N = config.hidden_dim
    n_t = config.n_t
    
    # Initialize Solver
    qp = OSQP(tol=1e-4)
    
    # Create Vmapped Solver: (Key, Data) -> Metrics
    # Data shape: (R, P, M, H) -> Vmap over R (axis 0)
    # run_glue_solver expects data: (P, M, N)
    vmapped_glue = jax.vmap(
        partial(run_glue_solver, P=P, M=M, N=N, n_t=n_t, qp_solver=qp),
        in_axes=(0, 0) # Split key and data across R
    )
    
    # Main Results Storage: results[train_task][eval_task][metric] -> (Steps, Repeats)
    full_results = {}
    
    # Master Key
    master_key = jax.random.PRNGKey(config.seed)

    # --- 2. Process Training Tasks ---
    # We iterate over the tasks as they were trained (folders task_000, task_001...)
    for train_task_idx in range(config.num_tasks):
        train_task_name = f"task_{train_task_idx:03d}"
        task_dir = os.path.join(config.results_dir, train_task_name)
        
        reps_path = os.path.join(task_dir, "representations.npy")
        lbls_path = os.path.join(task_dir, "binary_labels.npy")
        
        if not os.path.exists(reps_path) or not os.path.exists(lbls_path):
            print(f"Skipping {train_task_name} (Files not found)")
            continue

        print(f"Processing Training Phase: {train_task_name}...")
        
        # Load Data
        # reps: (L_steps, R_repeats, T_eval, S_samples, H_dim)
        reps_data = np.load(reps_path)
        # lbls: (R_repeats, T_eval, S_samples) - Labels are constant over time
        lbls_data = np.load(lbls_path)
        
        L, R, T_eval, S, H = reps_data.shape
        
        # Verify M matches config
        assert S == config.analysis_subsamples, f"Data subsamples {S} != config {config.analysis_subsamples}"
        
        full_results[train_task_name] = {}

        # --- 3. Iterate over Evaluated Tasks ---
        for eval_task_idx in range(T_eval):
            eval_task_name = f"task_{eval_task_idx:03d}"
            full_results[train_task_name][eval_task_name] = {m: [] for m in metric_names}
            
            # Extract labels for this specific evaluation task (constant across L)
            # Shape: (R, S)
            current_eval_lbls = lbls_data[:, eval_task_idx, :]
            
            # Iterate Time Steps
            for step in tqdm(range(L)):
                # Shape: (R, S, H)
                current_reps = reps_data[step, :, eval_task_idx, :, :]
                
                # Format Data: (R, P, M, H)
                # We do this in Python to handle the masking logic easily
                formatted_data = _prep_glue_batch(current_reps, current_eval_lbls, P, M, H)
                
                # Generate Keys for this batch
                step_key = jax.random.fold_in(master_key, train_task_idx * 10000 + step)
                batch_keys = jax.random.split(step_key, R)
                
                # Run GLUE (Vmapped over Repeats)
                # returns: (metrics_tuple, plot_inputs, single_metrics)
                metrics_tuple, _, _ = vmapped_glue(batch_keys, formatted_data)
                
                # Store Metrics
                # metrics_tuple index map: 0:cap, 1:dim, 2:rad, 3:cen_aln, 4:ax_aln, 5:cen_ax_aln, 6:app_cap
                # Each item in tuple is shape (R,)
                for i, name in enumerate(metric_names):
                    # Convert JAX array to numpy and append
                    full_results[train_task_name][eval_task_name][name].append(np.array(metrics_tuple[i]))

            # Convert lists to arrays: (Steps, Repeats)
            for name in metric_names:
                full_results[train_task_name][eval_task_name][name] = np.stack(
                    full_results[train_task_name][eval_task_name][name]
                )

    # --- 4. Save Results ---
    save_path = os.path.join(config.results_dir, "glue_metrics.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(full_results, f)
    print(f"GLUE metrics saved to {save_path}")

    # --- 5. Plotting (Replicating style of plastic_analysis) ---
    _plot_glue_results(full_results, metric_names, config)


def _plot_glue_results(full_results, metric_names, config):
    """
    Plots the trajectory of GLUE metrics for the CURRENT task and ALL tasks.
    Aggregates data across the training timeline.
    """
    if not full_results:
        return

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)
    if n_metrics == 1: axes = [axes]
    
    # Flatten the timeline across training tasks
    # We assume sequential training: task_0 -> task_1 -> ...
    train_tasks = sorted(list(full_results.keys()))
    
    # Build a continuous timeline dictionary
    # timeline[metric][eval_task] = [ (mean, std) array over all time ]
    timeline = {m: {t: {'mean': [], 'std': []} for t in train_tasks} for m in metric_names}
    
    task_boundaries = []
    current_x = 0
    
    for t_train in train_tasks:
        # Get one metric to determine length
        steps = full_results[t_train][t_train]['Capacity'].shape[0]
        
        # Determine X axis for this segment
        epochs_per_step = config.log_frequency
        segment_len = steps * epochs_per_step
        
        current_x += segment_len
        task_boundaries.append(current_x)
        
        # Collect data for all eval tasks during this training phase
        # Note: We plot all eval tasks available
        for t_eval in train_tasks:
            if t_eval not in full_results[t_train]:
                continue
                
            for m in metric_names:
                # Data: (Steps, Repeats)
                data = full_results[t_train][t_eval][m]
                
                # NanMean/Std over Repeats
                mean = np.nanmean(data, axis=1)
                std = np.nanstd(data, axis=1)
                
                timeline[m][t_eval]['mean'].append(mean)
                timeline[m][t_eval]['std'].append(std)

    # Stitch and Plot
    x_axis = np.arange(0, current_x, config.log_frequency)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(train_tasks)))
    
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        
        for j, t_eval in enumerate(train_tasks):
            # Concatenate segments
            means = timeline[metric][t_eval]['mean']
            stds = timeline[metric][t_eval]['std']
            
            if not means: continue
            
            full_mean = np.concatenate(means)
            full_std = np.concatenate(stds)
            
            # Ensure lengths match (handle potential slight mismatches in log freq)
            limit = min(len(x_axis), len(full_mean))
            
            ax.plot(x_axis[:limit], full_mean[:limit], label=t_eval, color=colors[j], linewidth=2)
            ax.fill_between(x_axis[:limit], 
                            full_mean[:limit] - full_std[:limit], 
                            full_mean[:limit] + full_std[:limit], 
                            color=colors[j], alpha=0.1)

        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        # Task Boundaries
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


def run_mtl_plasticity_analysis(config):
    print("\n--- Running MTL Plasticity Analysis ---")
    mtl_dir = os.path.join(config.results_dir, "multitask")
    
    # Load Data
    reps = np.load(os.path.join(mtl_dir, "representations.npy")) # (L, R, T, S, H)
    weights = np.load(os.path.join(mtl_dir, "weights.npy"))      # (L, R, D)
    init_weights = weights[0] # Approx init
    
    # Flatten T and S for global plasticity metrics
    # (L, R, T, S, H) -> (L, R, T*S, H)
    L, R, T, S, H = reps.shape
    reps_flat = reps.reshape(L, R, T*S, H)
    
    history = {
        'Dormant Neurons (Ratio)': [], 
        'Active Units (Fraction)': [], 
        'Feature Norm': [],
        'Weight Magnitude': [],
        'Weight Difference (Init)': []
    }
    
    # Compute Metrics
    for i in range(L):
        # Rep Metrics
        r_met = _compute_rep_metrics_batch(jnp.array(reps_flat[i]), H)
        history['Dormant Neurons (Ratio)'].append(np.array(r_met['Dormant Neurons (Ratio)']))
        history['Active Units (Fraction)'].append(np.array(r_met['Active Units (Fraction)']))
        history['Feature Norm'].append(np.array(r_met['Feature Norm']))
        
        # Weight Metrics
        w_met = _compute_weight_metrics_batch(jnp.array(weights[i]), jnp.array(weights[i]), jnp.array(init_weights))
        history['Weight Magnitude'].append(np.array(w_met['Weight Magnitude']))
        history['Weight Difference (Init)'].append(np.array(w_met['Weight Difference (Init)']))

    # Save
    for k in history: history[k] = np.stack(history[k])
    
    with open(os.path.join(mtl_dir, "plasticity_metrics.pkl"), 'wb') as f:
        pickle.dump(history, f)
        
    # Plot
    _plot_mtl_metrics(history, config, "Plasticity", "mtl_plasticity.png")


def run_mtl_cl_metrics(config):
    print("\n--- Running MTL CL/Performance Analysis ---")
    mtl_dir = os.path.join(config.results_dir, "multitask")
    with open(os.path.join(mtl_dir, "metrics.pkl"), 'rb') as f:
        history = pickle.load(f)
        
    print("\nFinal Multi-Task Performance:")
    avg_accs = []
    
    # Print per-task final accuracy
    for t_name, metrics in history['test_metrics'].items():
        acc = np.array(metrics['acc']) # (Epochs, Repeats)
        final_acc = acc[-1, :]
        mean = np.mean(final_acc)
        std = np.std(final_acc)
        print(f"  {t_name}: {mean:.4f} ± {std:.4f}")
        avg_accs.append(mean)
        
    print(f"  AVERAGE: {np.mean(avg_accs):.4f}")


def _plot_mtl_metrics(data_dict, config, title_prefix, filename):
    keys = list(data_dict.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 3*len(keys)), sharex=True)
    if len(keys) == 1: axes = [axes]
    
    # X-axis scaling
    total_epochs = config.epochs_per_task * config.num_tasks
    n_steps = list(data_dict.values())[0].shape[0]
    x_axis = np.linspace(0, total_epochs, n_steps)
    
    for i, k in enumerate(keys):
        vals = data_dict[k] # (L, R)
        mean = np.mean(vals, axis=1)
        std = np.std(vals, axis=1)
        
        axes[i].plot(x_axis, mean, label=k, color='purple')
        axes[i].fill_between(x_axis, mean-std, mean+std, color='purple', alpha=0.2)
        axes[i].set_ylabel(k)
        axes[i].grid(True, alpha=0.3)
        if i == 0: axes[i].set_title(f"{title_prefix} - {config.dataset_name}")
            
    axes[-1].set_xlabel("Epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(config.figures_dir, filename))
    print(f"  Saved plot to {filename}")


def run_mtl_glue_analysis(config):
    print("\n--- Starting MTL GLUE Metric Analysis ---")
    mtl_dir = os.path.join(config.results_dir, "multitask")
    
    # 1. Setup
    metric_names = [
        'Capacity', 'Dimension', 'Radius', 
        'Center Alignment', 'Axis Alignment', 'Center-Axis Alignment', 
        'Approx Capacity'
    ]
    
    # Constants
    P = 2 
    M = config.analysis_subsamples
    N = config.hidden_dim
    n_t = config.n_t
    
    # Initialize Solver
    qp = OSQP(tol=1e-4)
    vmapped_glue = jax.vmap(
        partial(run_glue_solver, P=P, M=M, N=N, n_t=n_t, qp_solver=qp),
        in_axes=(0, 0)
    )
    
    # 2. Load Data
    # Shape: (L, R, T, S, H)
    reps_path = os.path.join(mtl_dir, "representations.npy")
    # Shape: (R, T, S) -- Note: Transposed in train_multitask before saving
    lbls_path = os.path.join(mtl_dir, "binary_labels.npy")
    
    if not os.path.exists(reps_path) or not os.path.exists(lbls_path):
        print("Skipping MTL GLUE (Files not found)")
        return

    reps_data = np.load(reps_path)
    lbls_data = np.load(lbls_path)
    
    L, R, T, S, H = reps_data.shape
    
    # Storage: results[task_name][metric] -> (Steps, Repeats)
    full_results = {}
    master_key = jax.random.PRNGKey(config.seed)

    # 3. Iterate over Tasks and Time
    for t_idx in range(T):
        task_name = f"task_{t_idx:03d}"
        print(f"Processing MTL GLUE for {task_name}...")
        
        full_results[task_name] = {m: [] for m in metric_names}
        
        # Labels for this task: (R, S)
        current_eval_lbls = lbls_data[:, t_idx, :]
        
        for step in range(L):
            # Reps: (R, S, H)
            current_reps = reps_data[step, :, t_idx, :, :]
            
            # Format: (R, P, M, H)
            formatted_data = _prep_glue_batch(current_reps, current_eval_lbls, P, M, H)
            
            # Keys
            step_key = jax.random.fold_in(master_key, t_idx * 10000 + step)
            batch_keys = jax.random.split(step_key, R)
            
            # Run Solver
            metrics_tuple, _, _ = vmapped_glue(batch_keys, formatted_data)
            
            for i, name in enumerate(metric_names):
                full_results[task_name][name].append(np.array(metrics_tuple[i]))

        # Stack results: (Steps, Repeats)
        for name in metric_names:
            full_results[task_name][name] = np.stack(full_results[task_name][name])

        # Plot specific task metrics
        _plot_mtl_metrics(
            full_results[task_name], 
            config, 
            f"MTL GLUE - {task_name}", 
            f"mtl_glue_{task_name}.png"
        )

    # 4. Save
    save_path = os.path.join(mtl_dir, "glue_metrics.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(full_results, f)
    print(f"MTL GLUE metrics saved to {save_path}")


def run_all_representation_analysis():
    """
    Orchestrates the complete analysis suite for an experiment.
    
    1. Runs Plasticity Analysis (Drift, Rank, etc.)
    2. Runs GLUE Analysis (Manifold Geometry)
    3. Runs Multi-Task Learning Analysis (Plasticity & GLUE on Joint Data)
    4. Aggregates ALL raw metrics (preserving repeats) into 'all_metrics.pkl'
    
    Args:
        config: The experiment configuration object.
        experiment_path: Path to the specific experiment results directory.
    """
    """
    Orchestrates the complete analysis suite for an experiment.
    Loads config from the experiment directory.
    """
    experiment_path = "/home/users/MTrappett/manifold/binary_classification/results/mnist/RL/"
    # --- 1. Load Configuration ---
    config_path = os.path.join(experiment_path, "config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    # Override config paths to point to the specific experiment folder
    # (Crucial if folder was moved or renamed since training)
    config.results_dir = experiment_path
    config.reps_dir = experiment_path 
    config.figures_dir = os.path.join(experiment_path, "figures")
    os.makedirs(config.figures_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STARTING FULL EXPERIMENT ANALYSIS: {config.dataset_name} / {config.algorithm}")
    print(f"Target Path: {experiment_path}")
    print(f"{'='*60}")

    # --- 1. Execute Individual Analysis Pipelines ---
    
    # A. Plasticity Metrics (CL)
    # Generates: {experiment_path}/plastic_analysis_{dataset}.pkl
    try:
        run_plastic_analysis_pipeline(config)
    except Exception as e:
        print(f"Error in Plasticity Pipeline: {e}")

    # B. GLUE Metrics (CL)
    # Generates: {experiment_path}/glue_metrics.pkl
    try:
        run_glue_analysis_pipeline(config)
    except Exception as e:
        print(f"Error in GLUE Pipeline: {e}")

    # C. Multi-Task Learning Analysis (Upper Bound)
    # Generates: {experiment_path}/multitask/plasticity_metrics.pkl
    # Generates: {experiment_path}/multitask/glue_metrics.pkl
    if os.path.exists(os.path.join(config.results_dir, "multitask")):
        try:
            run_mtl_plasticity_analysis(config)
            run_mtl_glue_analysis(config)
            # Optional: Print summary to console
            run_mtl_cl_metrics(config) 
        except Exception as e:
            print(f"Error in MTL Analysis: {e}")
    else:
        print("Skipping MTL Analysis (No 'multitask' directory found).")

    # --- 2. Aggregate All Metrics into Single Object ---
    print(f"\n{'='*60}")
    print(f"Aggregating All Metrics (No Aggregation Across Repeats)")
    print(f"{'='*60}")

    all_metrics = {
        'cl': {},
        'plasticity': {},
        'glue': {},
        'mtl': {}
    }

    # --- A. Collect CL Metrics (Accuracy/Loss) ---
    # Source: global_history.pkl
    hist_path = os.path.join(config.results_dir, "global_history.pkl")
    if os.path.exists(hist_path):
        with open(hist_path, 'rb') as f:
            gh = pickle.load(f)
            # gh['test_metrics'] -> {task_name: {'acc': (E, R), 'loss': (E, R)}}
            all_metrics['cl'] = gh
            print(f"  [x] Loaded CL metrics (Global History)")
    else:
        print(f"  [ ] Missing global_history.pkl")

    # --- B. Collect Plasticity Metrics ---
    # Source: plastic_analysis_{dataset}.pkl
    plast_path = os.path.join(config.reps_dir, f"plastic_analysis_{config.dataset_name}.pkl")
    if os.path.exists(plast_path):
        with open(plast_path, 'rb') as f:
            # Contains {'history': {metric: (Steps, Repeats)}, ...}
            data = pickle.load(f)
            all_metrics['plasticity'] = data.get('history', {})
            print(f"  [x] Loaded Plasticity metrics")
    else:
        print(f"  [ ] Missing plastic_analysis data")

    # --- C. Collect GLUE Metrics ---
    # Source: glue_metrics.pkl
    glue_path = os.path.join(config.results_dir, "glue_metrics.pkl")
    if os.path.exists(glue_path):
        with open(glue_path, 'rb') as f:
            # Contains {train_task: {eval_task: {metric: (Steps, Repeats)}}}
            all_metrics['glue'] = pickle.load(f)
            print(f"  [x] Loaded GLUE metrics")
    else:
        print(f"  [ ] Missing glue_metrics.pkl")

    # --- D. Collect MTL Metrics ---
    mtl_root = os.path.join(config.results_dir, "multitask")
    if os.path.exists(mtl_root):
        all_metrics['mtl'] = {'cl': {}, 'plasticity': {}, 'glue': {}}
        
        # MTL CL
        mtl_cl_path = os.path.join(mtl_root, "metrics.pkl")
        if os.path.exists(mtl_cl_path):
            with open(mtl_cl_path, 'rb') as f:
                all_metrics['mtl']['cl'] = pickle.load(f)
        
        # MTL Plasticity
        mtl_plast_path = os.path.join(mtl_root, "plasticity_metrics.pkl")
        if os.path.exists(mtl_plast_path):
            with open(mtl_plast_path, 'rb') as f:
                all_metrics['mtl']['plasticity'] = pickle.load(f)

        # MTL GLUE
        mtl_glue_path = os.path.join(mtl_root, "glue_metrics.pkl")
        if os.path.exists(mtl_glue_path):
            with open(mtl_glue_path, 'rb') as f:
                all_metrics['mtl']['glue'] = pickle.load(f)
        
        print(f"  [x] Loaded MTL metrics (CL, Plasticity, GLUE)")
    else:
        print(f"  [ ] No MTL data found")

    # --- 3. Save Master Object ---
    save_path = os.path.join(config.results_dir, "all_metrics.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print(f"\nSUCCESS: All aggregated metrics saved to:\n  -> {save_path}")



if __name__=='__main__':
    run_all_representation_analysis()
