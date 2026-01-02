import os
import numpy as np
import matplotlib.pyplot as plt
import data_utils
import config as config_module
from numpy.linalg import svd, norm

def compute_rep_metrics(representations, tau=0.1):
    """
    Computes metrics, handling Potential NaNs in input.
    representations: (n_repeats, n_samples, hidden_dim)
    """
    n_repeats, n_samples, hidden_dim = representations.shape
    
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
        
        # Check for NaNs (if user decided to NaN out stopped models)
        if np.isnan(F).any():
            for k in metrics: metrics[k].append(np.nan)
            continue
            
        # F.1 Dormant Neurons (Ratio)
        avg_activation_per_neuron = np.mean(F, axis=0)
        layer_mean_activation = np.mean(avg_activation_per_neuron)
        if layer_mean_activation > 1e-9:
            scores = avg_activation_per_neuron / layer_mean_activation
        else:
            scores = np.zeros_like(avg_activation_per_neuron)
        dormant_count = np.sum(scores <= tau)
        metrics['Dormant Neurons (Ratio)'].append(dormant_count / hidden_dim)
        
        # F.2 Active Units
        active_per_sample = np.mean(F > 0, axis=1)
        metrics['Active Units (Fraction)'].append(np.mean(active_per_sample))
        
        # SVD for Rank Metrics
        try:
            _, s, _ = svd(F, full_matrices=False)
            
            # F.3 Stable Rank
            s_sum = np.sum(s)
            if s_sum > 1e-9:
                cumulative_energy = np.cumsum(s) / s_sum
                stable_rank = np.argmax(cumulative_energy > 0.99) + 1 
            else:
                stable_rank = 0
            metrics['Stable Rank'].append(stable_rank)
            
            # F.4 Effective Rank
            if s_sum > 1e-9:
                p = s / s_sum
                p = p[p > 0]
                entropy = -np.sum(p * np.log(p))
                effective_rank = np.exp(entropy)
            else:
                effective_rank = 0.0
            metrics['Effective Rank'].append(effective_rank)
            
        except np.linalg.LinAlgError:
            metrics['Stable Rank'].append(0)
            metrics['Effective Rank'].append(0)

        # F.7 Feature Norm
        row_norms = norm(F, axis=1)
        metrics['Feature Norm'].append(np.mean(row_norms))
        
        # F.8 Feature Variance
        col_vars = np.var(F, axis=0)
        metrics['Feature Variance'].append(np.mean(col_vars))

    for k in metrics:
        metrics[k] = np.array(metrics[k])
        
    return metrics

def compute_weight_metrics(current_weights, start_of_task_weights, init_weights):
    n_repeats, n_params = current_weights.shape
    
    metrics = {
        'Weight Magnitude': [],
        'Weight Difference (Task)': [],
        'Weight Difference (Init)': []
    }
    
    for r in range(n_repeats):
        w = current_weights[r]
        w_task_ref = start_of_task_weights[r]
        w_init_ref = init_weights[r]
        
        if np.isnan(w).any():
             for k in metrics: metrics[k].append(np.nan)
             continue

        # --- F.5 Weight Magnitude ---
        wm = np.sqrt(np.mean(w**2))
        metrics['Weight Magnitude'].append(wm)
        
        # --- F.6 Weight Difference (Task-Level) ---
        wd_task = np.sqrt(np.mean((w - w_task_ref)**2))
        metrics['Weight Difference (Task)'].append(wd_task)

        # --- F.6 Weight Difference (From Initialization) ---
        wd_init = np.sqrt(np.mean((w - w_init_ref)**2))
        metrics['Weight Difference (Init)'].append(wd_init)
        
    for k in metrics:
        metrics[k] = np.array(metrics[k])
        
    return metrics

def run_analysis_pipeline(config):
    print("--- Starting Plasticine Metric Analysis ---")
    
    tasks = data_utils.create_continual_tasks(config, split='train')
    task_names = [t.name for t in tasks]
    
    init_weights_path = os.path.join(config.reps_dir, "init_weights.npy")
    if os.path.exists(init_weights_path):
        init_weights = np.load(init_weights_path)
    else:
        print("Warning: init_weights.npy not found.")
        init_weights = None

    history = {
        'Dormant Neurons (Ratio)': [],
        'Active Units (Fraction)': [],
        'Stable Rank': [],
        'Effective Rank': [],
        'Feature Norm': [],
        'Feature Variance': [],
        'Weight Magnitude': [],   
        'Weight Difference (Task)': [],
        'Weight Difference (Init)': []
    }
    
    task_boundaries = []
    total_epochs = 0
    previous_task_final_weights = None 
    
    for t_name in task_names:
        rep_path = os.path.join(config.reps_dir, f"{t_name}_reps_per_epoch.npy")
        w_path = os.path.join(config.reps_dir, f"{t_name}_weights_per_epoch.npy")
        
        if not os.path.exists(rep_path) or not os.path.exists(w_path):
            continue
            
        print(f"Processing {t_name}...")
        rep_data = np.load(rep_path)   
        w_data = np.load(w_path)       
        n_epochs = rep_data.shape[0]
        
        if previous_task_final_weights is None:
             if init_weights is not None:
                 start_of_task_weights = init_weights
             else:
                 start_of_task_weights = w_data[0] 
        else:
             start_of_task_weights = previous_task_final_weights

        for epoch_idx in range(n_epochs):
            rep_step_metrics = compute_rep_metrics(rep_data[epoch_idx])
            for key, val in rep_step_metrics.items():
                history[key].append(val)
                
            if init_weights is not None:
                w_step_metrics = compute_weight_metrics(
                    w_data[epoch_idx], 
                    start_of_task_weights, 
                    init_weights
                )
                for key, val in w_step_metrics.items():
                    history[key].append(val)
                
        previous_task_final_weights = w_data[-1]
        total_epochs += n_epochs
        task_boundaries.append(total_epochs)

    # Convert history to arrays
    for key in history:
        if len(history[key]) > 0:
            history[key] = np.stack(history[key]) 

    # Plotting using NanMean / NanStd
    metric_names = list(history.keys())
    metric_names = [m for m in metric_names if len(history[m]) > 0]
    n_metrics = len(metric_names)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    if n_metrics == 1: axes = [axes]
    
    x_axis = np.arange(1, total_epochs + 1)
    
    print("Plotting results...")
    
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        data = history[metric] 
        
        # USE NANMEAN / NANSTD here
        mean = np.nanmean(data, axis=1)
        std = np.nanstd(data, axis=1)
        
        ax.plot(x_axis, mean, label='Mean', color='blue', linewidth=2)
        ax.fill_between(x_axis, mean - std, mean + std, color='blue', alpha=0.2, label='Std Dev')
        
        ax.set_ylabel(metric)
        for boundary in task_boundaries[:-1]:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
            
        if i == 0:
            ax.set_title(f"Plasticine Metrics Analysis ({config.dataset_name}) - ES={config.early_stopping}")

    axes[-1].set_xlabel('Epochs (Continual)')
    plt.tight_layout()
    save_path = os.path.join(config.figures_dir, f"plasticine_metrics_{config.dataset_name}_ES.png")
    plt.savefig(save_path)
    print(f"Analysis plots saved to {save_path}")