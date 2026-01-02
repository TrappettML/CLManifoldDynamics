import os
import numpy as np
import matplotlib.pyplot as plt
import data_utils
import config as config_module
from numpy.linalg import svd, norm

def compute_rep_metrics(representations, hidden_dim, tau=0.1):
    n_repeats, n_samples, _ = representations.shape
    
    metrics = {
        'Dormant Neurons (Ratio)': [],
        'Active Units (Fraction)': [],
        'Stable Rank': [],
        'Effective Rank': [],
        'Feature Norm': [],
        'Feature Variance': []
    }
    
    for r in range(n_repeats):
        F = representations[r]
        if np.isnan(F).any():
            for k in metrics: metrics[k].append(np.nan)
            continue
            
        # Dormant Neurons
        avg_act = np.mean(F, axis=0)
        layer_mean = np.mean(avg_act)
        scores = avg_act / layer_mean if layer_mean > 1e-9 else np.zeros_like(avg_act)
        metrics['Dormant Neurons (Ratio)'].append(np.sum(scores <= tau) / hidden_dim)
        
        # Active Units
        metrics['Active Units (Fraction)'].append(np.mean(np.mean(F > 0, axis=1)))
        
        # SVD
        try:
            _, s, _ = svd(F, full_matrices=False)
            s_sum = np.sum(s)
            if s_sum > 1e-9:
                cum_energy = np.cumsum(s) / s_sum
                stable_rank = np.argmax(cum_energy > 0.99) + 1
                p = s / s_sum
                p = p[p > 0]
                effective_rank = np.exp(-np.sum(p * np.log(p)))
            else:
                stable_rank, effective_rank = 0, 0.0
            metrics['Stable Rank'].append(stable_rank)
            metrics['Effective Rank'].append(effective_rank)
        except np.linalg.LinAlgError:
            metrics['Stable Rank'].append(0)
            metrics['Effective Rank'].append(0)

        # Norm/Var
        metrics['Feature Norm'].append(np.mean(norm(F, axis=1)))
        metrics['Feature Variance'].append(np.mean(np.var(F, axis=0)))

    return {k: np.array(v) for k, v in metrics.items()}

def compute_weight_metrics(current, ref_task, ref_init):
    n_repeats = current.shape[0]
    metrics = {'Weight Magnitude': [], 'Weight Difference (Task)': [], 'Weight Difference (Init)': []}
    
    for r in range(n_repeats):
        w, w_t, w_i = current[r], ref_task[r], ref_init[r]
        if np.isnan(w).any():
             for k in metrics: metrics[k].append(np.nan)
             continue

        metrics['Weight Magnitude'].append(np.sqrt(np.mean(w**2)))
        metrics['Weight Difference (Task)'].append(np.sqrt(np.mean((w - w_t)**2)))
        metrics['Weight Difference (Init)'].append(np.sqrt(np.mean((w - w_i)**2)))
        
    return {k: np.array(v) for k, v in metrics.items()}

def run_analysis_pipeline(config):
    print("--- Starting Plasticine Metric Analysis ---")
    
    tasks = data_utils.create_continual_tasks(config, split='train')
    task_names = [t.name for t in tasks]
    
    init_weights_path = os.path.join(config.reps_dir, "init_weights.npy")
    init_weights = np.load(init_weights_path) if os.path.exists(init_weights_path) else None

    history = {k: [] for k in [
        'Dormant Neurons (Ratio)', 'Active Units (Fraction)', 'Stable Rank', 'Effective Rank',
        'Feature Norm', 'Feature Variance', 'Weight Magnitude', 'Weight Difference (Task)', 'Weight Difference (Init)'
    ]}
    
    task_boundaries = []
    current_epoch_counter = 0
    previous_task_final_weights = None 
    
    for t_name in task_names:
        rep_path = os.path.join(config.reps_dir, f"{t_name}_reps_per_epoch.npy")
        w_path = os.path.join(config.reps_dir, f"{t_name}_weights_per_epoch.npy")
        
        if not os.path.exists(rep_path) or not os.path.exists(w_path):
            continue
            
        print(f"Processing {t_name}...")
        rep_data = np.load(rep_path)   # (n_steps, n_repeats, samples, dim)
        w_data = np.load(w_path)       # (n_steps, n_repeats, params)
        
        n_steps = rep_data.shape[0]
        # Calculate how many actual epochs this represents
        epochs_in_task = n_steps * config.log_frequency

        start_of_task_weights = init_weights if previous_task_final_weights is None else previous_task_final_weights
        if start_of_task_weights is None: start_of_task_weights = w_data[0]

        for i in range(n_steps):
            rep_metrics = compute_rep_metrics(rep_data[i], config.hidden_dim)
            for k, v in rep_metrics.items(): history[k].append(v)
                
            if init_weights is not None:
                w_metrics = compute_weight_metrics(w_data[i], start_of_task_weights, init_weights)
                for k, v in w_metrics.items(): history[k].append(v)
                
        previous_task_final_weights = w_data[-1]
        current_epoch_counter += epochs_in_task
        task_boundaries.append(current_epoch_counter)

    # Convert to arrays
    for k in history:
        if len(history[k]) > 0: history[k] = np.stack(history[k])

    # Plotting
    metric_names = [m for m in history.keys() if len(history[m]) > 0]
    n_metrics = len(metric_names)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    if n_metrics == 1: axes = [axes]
    
    # Scale X-axis to represent actual Epochs
    total_steps = len(history[metric_names[0]])
    x_axis = np.arange(1, total_steps + 1) * config.log_frequency
    
    print("Plotting results...")
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        data = history[metric]
        mean = np.nanmean(data, axis=1)
        std = np.nanstd(data, axis=1)
        
        ax.plot(x_axis, mean, label='Mean', color='blue', linewidth=2)
        ax.fill_between(x_axis, mean - std, mean + std, color='blue', alpha=0.2)
        
        ax.set_ylabel(metric)
        for boundary in task_boundaries[:-1]:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
            
        if i == 0:
            ax.set_title(f"Plasticine Metrics ({config.dataset_name}) - LogFreq={config.log_frequency}")

    axes[-1].set_xlabel('Epochs (Continual)')
    plt.tight_layout()
    save_path = os.path.join(config.figures_dir, f"plasticine_metrics_{config.dataset_name}.png")
    plt.savefig(save_path)
    print(f"Analysis plots saved to {save_path}")