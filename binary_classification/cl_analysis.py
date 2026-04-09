import os
import pickle
import argparse
import jax
import jax.numpy as jnp
import numpy as np


EXPERIMENT_PATH = "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.0001_lr2_0.01"

def get_metric_integrator(metric_name: str):
    """
    Returns a JAX-compatible integrator function.
    Args:
        metric_name: 'auc', 'mean', 'min', 'max', 'median', 'final'
    Returns:
        Function (curve, x_axis) -> scalar
    """
    if metric_name == 'auc':
        # Trapezoidal rule (requires moving time to last axis if multidimensional)
        return lambda y, x: jnp.trapezoid(jnp.moveaxis(y, 0, -1), x=x)
    
    # Point-estimate: takes the last value in the time sequence
    if metric_name == 'final':
        return lambda y, x: y[-1]
    
    # Simple reductions (ignore x-axis spacing)
    if metric_name == 'mean':
        return lambda y, x: jnp.mean(y, axis=0)
    if metric_name == 'min':
        return lambda y, x: jnp.min(y, axis=0)
    if metric_name == 'max':
        return lambda y, x: jnp.max(y, axis=0)
    if metric_name == 'median':
        return lambda y, x: jnp.median(y, axis=0)
        
    raise ValueError(f"Unknown metric integration method: {metric_name}")

def calculate_performance_matrix(student_tensor, n_tasks, log_freq, integrator_fn):
    """
    Calculates M[i, j]: Metric on Eval Task i while Training on Task j.
    Handles sparse sampling (NaNs) implicitly by slicing.
    """
    # student_tensor shape: (Total_Epochs, N_Repeats, N_Eval_Tasks)
    
    # 1. Filter NaNs by slicing valid log indices
    # We assume logs happen strictly at log_freq intervals
    valid_indices = jnp.arange(log_freq - 1, student_tensor.shape[0], log_freq)
    clean_history = student_tensor[valid_indices] # (Total_Steps, N_Repeats, N_Eval)

    # 2. Reshape to group by Training Task
    steps_per_task = clean_history.shape[0] // n_tasks
    # Shape: (N_Train_Tasks, Steps_Per_Task, N_Repeats, N_Eval_Tasks)
    reshaped = clean_history.reshape(n_tasks, steps_per_task, clean_history.shape[1], n_tasks)

    # 3. Integrate over the 'Steps_Per_Task' axis
    x_axis = jnp.arange(steps_per_task) * log_freq
    
    # Let's align for integrator: Input y must be (Time, Batch...)
    # Current: (Train, Time, Repeats, Eval) -> (Time, Train, Repeats, Eval)
    time_major = jnp.moveaxis(reshaped, 1, 0)
    
    # Result: (Train, Repeats, Eval)
    M_integrated = integrator_fn(time_major, x_axis)
    
    # 4. Transpose to final M[Eval, Train, Repeats]
    # Current: (Train, Repeats, Eval) -> (Eval, Train, Repeats)
    M = jnp.transpose(M_integrated, (2, 0, 1))
    
    return M

def calculate_expert_vector(expert_histories_dict, task_names, metric_type, log_freq, integrator_fn):
    """
    Calculates E[i]: Metric for Expert on Task i.
    """
    E_list = []
    for t_name in task_names:
        # (Epochs, Repeats)
        raw = jnp.array(expert_histories_dict[t_name][metric_type])
        
        # Slice valid indices
        valid_indices = jnp.arange(log_freq - 1, raw.shape[0], log_freq)
        clean = raw[valid_indices]
        
        x_axis = jnp.arange(clean.shape[0]) * log_freq
        
        # Integrate (Time is axis 0)
        # Result: (Repeats,)
        val = integrator_fn(clean, x_axis)
        E_list.append(val)
        
    return jnp.stack(E_list, axis=0) # (N_Tasks, N_Repeats)

def compute_and_log_cl_metrics(global_history, expert_histories, config):
    """
    Computes Remembering, Transfer, and Zero-Shot Learning. 
    Accepts 'm_integrator' to switch between 'auc' (Anytime) and 'final' (End-of-Task).
    """
    print(f"\n--- Computing CL Metrics ({config.metric_type.upper()} | Integrator: {config.m_integrator}) ---")
    
    task_names = list(global_history['test_metrics'].keys())
    n_tasks = len(task_names)
    
    # 1. Setup Integrator
    integrator_fn = get_metric_integrator(config.m_integrator)

    # 2. Prepare Data
    # Shape: (Total_Epochs, N_Repeats, N_Eval_Tasks)
    student_data = [jnp.array(global_history['test_metrics'][t][config.metric_type]) for t in task_names]
    student_tensor = jnp.stack(student_data, axis=-1)
    
    # 3. Compute Matrices M and E
    # M shape: (N_Eval, N_Train, N_Repeats)
    M = calculate_performance_matrix(student_tensor, n_tasks, config.log_frequency, integrator_fn)
    # E shape: (N_Eval, N_Repeats)
    E = calculate_expert_vector(expert_histories, task_names, config.metric_type, config.log_frequency, integrator_fn)
    
    # 4. Compute Metrics (Vectorized)
    # --- Forward Transfer (FT) ---
    m_diag = jnp.diagonal(M, axis1=0, axis2=1).T  # (N_Eval, N_Repeats)
    eps = 1e-9
    
    # (Expert - Student) / (Expert + Student)
    ft_per_task = (E - m_diag) / (E + m_diag + eps)
    
    # --- Remembering (Rem) ---
    m_diag_expanded = jnp.expand_dims(m_diag, 1)
    
    # (M_ii - M_ij) / (M_ii + M_ij)
    rem_matrix_full = (m_diag_expanded - M) # / (m_diag_expanded + M + eps) # remove scaling to get absolute difference
    
    # Extract Upper Triangle (j > i) for valid Remembering scores
    mask_rem = jnp.triu(jnp.ones((n_tasks, n_tasks)), k=1)
    mask_rem = jnp.expand_dims(mask_rem, -1) # Broadcast to repeats
    
    # Apply mask
    rem_values = jnp.where(mask_rem, rem_matrix_full, jnp.nan)

    # --- Zero-Shot Learning (ZSL) ---
    # Extract Lower Triangle (i > j) representing performance on task i before training on it
    mask_zsl = jnp.tril(jnp.ones((n_tasks, n_tasks)), k=-1)
    mask_zsl = jnp.expand_dims(mask_zsl, -1)
    
    # Extract absolute performance on unseen tasks
    zsl_values = jnp.where(mask_zsl, M, jnp.nan)

    # 5. Sign Correction (Metric Agnostic)
    if config.metric_type == 'acc':
        # Invert signs so that Positive always means "Good" for relative metrics
        ft_final = -ft_per_task
        rem_final = -rem_values
    else:
        ft_final = ft_per_task
        rem_final = rem_values

    # Note: ZSL is an absolute performance metric (unlike the relative Rem/FT formulas), 
    # so we do not apply sign correction to it. It remains the raw accuracy or loss.

    # 6. Aggregation
    # --- Transfer ---
    # 1. Get the mean Transfer score for each repeat independently
    ft_per_repeat = jnp.nanmean(ft_final, axis=0) # Shape: (N_Repeats,)
    
    # 2. Calculate global mean and the std deviation across the repeats
    avg_ft = jnp.nanmean(ft_per_repeat) 
    std_ft = jnp.std(ft_per_repeat)
    
    # --- Remembering ---
    # 1. Get the mean Remembering score for each repeat (averaging over both task axes)
    rem_per_repeat = jnp.nanmean(rem_final, axis=(0, 1)) # Shape: (N_Repeats,)
    
    # 2. Calculate global mean and the std deviation across the repeats
    avg_rem = jnp.nanmean(rem_per_repeat)
    std_rem = jnp.std(rem_per_repeat)

    
    zsl_per_repeat = jnp.nanmean(zsl_values, axis=(0, 1))
    avg_zsl = jnp.nanmean(zsl_per_repeat)
    std_zsl = jnp.std(zsl_per_repeat)

    # 7. Reporting
    print("-" * 65)
    print(f"{'Metric':<25} | {'Mean':<12} | {'Std':<12}")
    print("-" * 65)
    
    print(f"{'Remembering (Stability)':<25} | {float(avg_rem):<12.4f} | {float(std_rem):<12.4f}")
    print(f"{'Transfer (vs Expert)':<25} | {float(avg_ft):<12.4f} | {float(std_ft):<12.4f}")
    print(f"{'Zero-Shot Learning':<25} | {float(avg_zsl):<12.4f} | {float(std_zsl):<12.4f}")
    print("-" * 65)

    return {
        'transfer': np.array(ft_final),
        'remembering': np.array(rem_final),
        'zero_shot': np.array(zsl_values),
        'stats': {
            'rem_mean': float(avg_rem), 
            'rem_std': np.array(std_rem),
            'trans_mean': float(avg_ft),
            'trans_std': np.array(std_ft),
            'zsl_mean': float(avg_zsl),
            'zsl_std': np.array(std_zsl)
        }
    }


def run_cl_analysis(experiment_path):
    """
    Standalone execution: Loads necessary files from an experiment directory,
    reconstructs expert histories, computes CL metrics, and saves the results.
    """
    config_path = os.path.join(experiment_path, "config.pkl")
    history_path = os.path.join(experiment_path, "global_history.pkl")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Global history file not found at {history_path}")
        
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
        
    with open(history_path, 'rb') as f:
        global_history = pickle.load(f)
        
    # Reconstruct expert_histories from individual task directories
    expert_histories = {}
    for task_idx in range(config.num_tasks):
        task_name = f"task_{task_idx:03d}"
        expert_path = os.path.join(experiment_path, task_name, "expert_metrics.pkl")
        
        if os.path.exists(expert_path):
            with open(expert_path, 'rb') as f:
                exp_data = pickle.load(f)
                expert_histories[task_name] = {
                    'loss': exp_data['test_loss'], 
                    'acc': exp_data['test_acc']
                }
        else:
            print(f"Warning: Missing expert_metrics.pkl for {task_name} at {expert_path}")
            
    # Compute metrics
    cl_results = compute_and_log_cl_metrics(global_history, expert_histories, config)
    
    # Save results
    save_path = os.path.join(experiment_path, "cl_metrics.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(cl_results, f)
        
    print(f"\nStandalone CL metrics successfully saved to:\n  -> {save_path}")
    return cl_results


if __name__ == "__main__":
    run_cl_analysis(EXPERIMENT_PATH)