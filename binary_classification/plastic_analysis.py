import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial
import data_utils
import config as config_module

# --- JAX Optimized Metric Calculations ---

@partial(jax.jit, static_argnames=('hidden_dim', 'tau'))
def _compute_rep_metrics_batch(representations, hidden_dim, tau=0.1):
    """
    Computes representation metrics for a batch of repeats in parallel using JAX.
    Input: (n_repeats, n_samples, dim)
    Returns: Dict of arrays of shape (n_repeats,)
    """
    def _single_rep_metrics(F):
        # F: (n_samples, dim)
        
        # Guard against NaNs (which appear in sparse evaluation or failed runs)
        is_nan = jnp.isnan(F).any()
        
        def _compute_valid(x):
            # 1. Dormant Neurons
            avg_act = jnp.mean(x, axis=0)
            layer_mean = jnp.mean(avg_act)
            
            # Safe division
            denom = jnp.where(layer_mean > 1e-9, layer_mean, 1.0)
            scores = avg_act / denom
            scores = jnp.where(layer_mean > 1e-9, scores, jnp.zeros_like(scores))
            
            dormant = jnp.sum(scores <= tau) / hidden_dim
            
            # 2. Active Units
            active = jnp.mean(jnp.mean(x > 0, axis=1))
            
            # 3. SVD-based Metrics
            _, s, _ = jnp.linalg.svd(x, full_matrices=False)
            s_sum = jnp.sum(s)
            
            # Stable Rank
            safe_s_sum = jnp.where(s_sum > 1e-9, s_sum, 1.0)
            cum_energy = jnp.cumsum(s) / safe_s_sum
            stable_r = jnp.argmax(cum_energy > 0.99) + 1.0
            stable_r = jnp.where(s_sum > 1e-9, stable_r, 0.0)

            # Effective Rank
            p = s / safe_s_sum
            p_safe = jnp.where(p > 0, p, 1.0)
            entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(p_safe), 0.0))
            eff_r = jnp.exp(entropy)
            eff_r = jnp.where(s_sum > 1e-9, eff_r, 0.0)

            # 4. Norm / Variance
            f_norm = jnp.mean(jnp.linalg.norm(x, axis=1))
            f_var = jnp.mean(jnp.var(x, axis=0))
            
            # FIX: Explicitly cast all outputs to float32 to match the NaN branch
            return (jnp.float32(dormant), jnp.float32(active), 
                    jnp.float32(stable_r), jnp.float32(eff_r), 
                    jnp.float32(f_norm), jnp.float32(f_var))

        def _compute_nan(x):
            val = jnp.nan
            return (jnp.float32(val), jnp.float32(val), jnp.float32(val), 
                    jnp.float32(val), jnp.float32(val), jnp.float32(val))

        return jax.lax.cond(is_nan, _compute_nan, _compute_valid, F)

    # Vectorize over the 'n_repeats' axis (axis 0)
    results = jax.vmap(_single_rep_metrics)(representations)
    
    keys = [
        'Dormant Neurons (Ratio)', 'Active Units (Fraction)', 
        'Stable Rank', 'Effective Rank', 
        'Feature Norm', 'Feature Variance'
    ]
    return {k: v for k, v in zip(keys, results)}

@jax.jit
def _compute_weight_metrics_batch(current, ref_task, ref_init):
    """
    Computes weight metrics for a batch of repeats in parallel.
    Input: (n_repeats, params)
    """
    def _single_w_metric(w, w_t, w_i):
        is_nan = jnp.isnan(w).any()
        
        def _calc(val_w, val_t, val_i):
            mag = jnp.sqrt(jnp.mean(val_w**2))
            diff_t = jnp.sqrt(jnp.mean((val_w - val_t)**2))
            diff_i = jnp.sqrt(jnp.mean((val_w - val_i)**2))
            # FIX: Explicitly cast to float32
            return jnp.float32(mag), jnp.float32(diff_t), jnp.float32(diff_i)

        def _nans(val_w, val_t, val_i):
            val = jnp.nan
            return jnp.float32(val), jnp.float32(val), jnp.float32(val)

        return jax.lax.cond(is_nan, _nans, _calc, w, w_t, w_i)

    results = jax.vmap(_single_w_metric)(current, ref_task, ref_init)
    
    keys = ['Weight Magnitude', 'Weight Difference (Task)', 'Weight Difference (Init)']
    return {k: v for k, v in zip(keys, results)}

# --- Main Analysis Pipeline ---

def run_analysis_pipeline(config):
    print("--- Starting Plasticine Metric Analysis (JAX Optimized) ---")
    
    tasks = data_utils.create_continual_tasks(config, split='train')
    task_names = [t.name for t in tasks]
    
    init_weights_path = os.path.join(config.reps_dir, "init_weights.npy")
    init_weights_np = np.load(init_weights_path) if os.path.exists(init_weights_path) else None

    # Storage for history
    metric_keys = [
        'Dormant Neurons (Ratio)', 'Active Units (Fraction)', 'Stable Rank', 'Effective Rank',
        'Feature Norm', 'Feature Variance', 'Weight Magnitude', 
        'Weight Difference (Task)', 'Weight Difference (Init)'
    ]
    history = {k: [] for k in metric_keys}
    
    task_boundaries = []
    current_epoch_counter = 0
    previous_task_final_weights = None 
    
    for t_name in task_names:
        rep_path = os.path.join(config.reps_dir, f"{t_name}_reps_per_epoch.npy")
        w_path = os.path.join(config.reps_dir, f"{t_name}_weights_per_epoch.npy")
        
        if not os.path.exists(rep_path) or not os.path.exists(w_path):
            print(f"Skipping {t_name} (Data not found)")
            continue
            
        print(f"Processing {t_name}...")
        
        # Load Data (NumPy)
        rep_data_np = np.load(rep_path)   # (n_steps, n_repeats, samples, dim)
        w_data_np = np.load(w_path)       # (n_steps, n_repeats, params)
        
        # Convert to JAX Arrays
        rep_data = jnp.array(rep_data_np)
        w_data = jnp.array(w_data_np)
        
        n_steps = rep_data.shape[0]
        epochs_in_task = n_steps * config.log_frequency

        # Determine Reference Weights
        if init_weights_np is not None:
            init_w_jax = jnp.array(init_weights_np)
        else:
            init_w_jax = w_data[0]

        if previous_task_final_weights is not None:
            task_ref_w_jax = jnp.array(previous_task_final_weights)
        else:
            task_ref_w_jax = init_w_jax

        # --- Iteration Loop ---
        for i in range(n_steps):
            # 1. Representation Metrics
            rep_res = _compute_rep_metrics_batch(
                rep_data[i], 
                config.hidden_dim
            )
            
            # 2. Weight Metrics
            w_res = _compute_weight_metrics_batch(
                w_data[i], 
                task_ref_w_jax, 
                init_w_jax
            )
            
            # 3. Store results (Convert back to NumPy for list storage)
            for k, v in rep_res.items():
                history[k].append(np.array(v))
            for k, v in w_res.items():
                history[k].append(np.array(v))
                
        # Update references for next task
        previous_task_final_weights = w_data_np[-1]
        current_epoch_counter += epochs_in_task
        task_boundaries.append(current_epoch_counter)

    # Clean up History
    for k in history:
        if len(history[k]) > 0: 
            history[k] = np.stack(history[k])

    # --- Plotting ---
    metric_names = [m for m in history.keys() if len(history[m]) > 0]
    n_metrics = len(metric_names)
    
    if n_metrics == 0:
        print("No metrics computed. Exiting analysis.")
        return

    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    if n_metrics == 1: axes = [axes]
    
    total_steps = len(history[metric_names[0]])
    x_axis = np.arange(1, total_steps + 1) * config.log_frequency
    
    print("Plotting results...")
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        data = history[metric]
        
        # Robust Mean/Std (handling NaNs)
        mean = np.nanmean(data, axis=1)
        std = np.nanstd(data, axis=1)
        
        ax.plot(x_axis, mean, label='Mean', color='blue', linewidth=2)
        ax.fill_between(x_axis, mean - std, mean + std, color='blue', alpha=0.2)
        
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        # Add task boundaries
        for boundary in task_boundaries[:-1]:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
            
        if i == 0:
            ax.set_title(f"Plasticine Metrics ({config.dataset_name}) - LogFreq={config.log_frequency}")

    axes[-1].set_xlabel('Epochs (Continual)')
    plt.tight_layout()
    save_path = os.path.join(config.figures_dir, f"plasticine_metrics_{config.dataset_name}.png")
    plt.savefig(save_path)
    print(f"Analysis plots saved to {save_path}")