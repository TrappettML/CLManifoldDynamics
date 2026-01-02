import numpy as np
import pandas as pd
from typing import Dict, Tuple

def get_metric_integrator(metric_name: str):
    """Returns a function that integrates loss/accuracy over time."""
    if metric_name == 'auc':
        # Use trapezoidal rule. If x is not provided (uniform), assume step=1
        return lambda y, x: np.trapz(y, x) if len(y) > 1 else np.nan
    if metric_name == 'mean': return lambda y, x: np.mean(y)
    if metric_name == 'median': return lambda y, x: np.median(y)
    if metric_name == 'min': return lambda y, x: np.min(y)
    if metric_name == 'max': return lambda y, x: np.max(y)
    raise ValueError(f"Unknown metric integration method: {metric_name}")

def calculate_metrics_for_run(
    student_data: np.ndarray, 
    expert_data: np.ndarray,
    ntasks: int, 
    sample_rate: int, 
    metric_integrator
) -> Dict[str, np.ndarray]:
    """
    Calculates Remembering and Forward Transfer matrices for a single repeat.
    
    Args:
        student_data: Shape (total_epochs,) - Flattened history of one repeat
        expert_data: Shape (ntasks, task_epochs) - Expert history for each task
    """
    remembering = np.full((ntasks, ntasks), np.nan)
    transfer = np.full((ntasks, ntasks), np.nan)
    
    # Reshape student data: (Training_Task_Index, Epochs_Per_Task, Evaluated_Task_Index)
    # Note: The input `student_data` is a specific column from global history 
    # corresponding to the performance on *one* specific evaluation task across the whole timeline.
    # Actually, the user's logic requires M[evaluated_task, trained_task].
    # We need to construct M outside this function or adapt the inputs.
    
    # Let's adapt the inputs to match the snippet's logic exactly:
    # We assume 'student_data' passed here is actually the full matrix for one repeat:
    # Shape: (Total_Epochs, Num_Eval_Tasks)
    
    total_epochs, num_eval_tasks = student_data.shape
    epochs_per_task = total_epochs // ntasks
    
    student_data_reshaped = student_data.reshape(ntasks, epochs_per_task, num_eval_tasks)
    raw_epoch_axis = np.arange(epochs_per_task) * sample_rate

    # M[i, j]: Metric on Eval Task i while Training on Task j
    M = np.full((ntasks, ntasks), np.nan)
    
    for j in range(ntasks): # Task being trained
        for i in range(ntasks): # Task being evaluated
            # Extract curve
            curve = student_data_reshaped[j, :, i]
            is_valid = ~np.isnan(curve)
            if np.any(is_valid):
                M[i, j] = metric_integrator(curve[is_valid], raw_epoch_axis[is_valid])

    # E[i]: Metric on Eval Task i (Expert)
    E = np.full(ntasks, np.nan)
    expert_epoch_axis = np.arange(expert_data.shape[1]) * sample_rate
    
    for i in range(ntasks):
        curve = expert_data[i]
        is_valid = ~np.isnan(curve)
        if np.any(is_valid):
            E[i] = metric_integrator(curve[is_valid], expert_epoch_axis[is_valid])

    # Calculate Transfer (Diagonal) and Remembering (Off-diagonal)
    # Using formula: (Expert - Student) / (Expert + Student)
    # Note: If metric is Accuracy (higher better), this formula might need inversion 
    # or interpretation. For Loss (lower better), positive result = good transfer.
    
    for i in range(ntasks):
        # Forward Transfer
        m_ii = M[i, i]
        e_i = E[i]
        if not (np.isnan(m_ii) or np.isnan(e_i)):
            transfer[i, i] = (e_i - m_ii) / (e_i + m_ii + 1e-9)

        # Remembering (How much did training on j hurt i?)
        for j in range(i + 1, ntasks):
            m_ij = M[i, j] # Current performance on i
            m_ii = M[i, i] # Original performance on i
            
            # For Loss: if m_ij > m_ii (loss went up), we want negative score.
            # (m_ii - m_ij) / ... -> (Low - High) -> Negative. Correct.
            if not (np.isnan(m_ii) or np.isnan(m_ij)):
                remembering[i, j] = (m_ii - m_ij) / (m_ii + m_ij + 1e-9)
                
    return M, E, remembering, transfer

def compute_and_log_cl_metrics(global_history, expert_histories, config, metric_type='acc', m_integrator='auc'):
    """
    Main driver to compute CL metrics across all repeats.
    """
    print(f"\n--- Computing CL Metrics ({metric_type.upper()}) ---")
    
    task_names = list(global_history['test_metrics'].keys())
    ntasks = len(task_names)
    n_repeats = config.n_repeats
    
    # 1. Prepare Data Containers
    # We need to construct a tensor of shape: (Total_Epochs, N_Repeats, N_Eval_Tasks)
    # global_history['test_metrics'][t_name][metric] is (Total_Epochs, N_Repeats)
    
    full_student_tensor = []
    for t_name in task_names:
        # (Total_Epochs, N_Repeats)
        data = np.array(global_history['test_metrics'][t_name][metric_type]) 
        full_student_tensor.append(data)
    
    # Result: (N_Eval_Tasks, Total_Epochs, N_Repeats) -> Transpose to (Total_Epochs, N_Repeats, N_Eval_Tasks)
    full_student_tensor = np.stack(full_student_tensor, axis=0)
    full_student_tensor = np.moveaxis(full_student_tensor, [0, 1, 2], [2, 0, 1])

    # Prepare Expert Data: (N_Tasks, N_Epochs_Expert, N_Repeats)
    full_expert_tensor = []
    for t_name in task_names:
        # Expert histories might differ in length if early stopping was used, 
        # but here they are fixed epochs.
        # (Epochs, Repeats)
        exp_data = expert_histories[t_name][metric_type] 
        full_expert_tensor.append(exp_data.T) # Store as (Repeats, Epochs) for easier slicing
    
    full_expert_tensor = np.stack(full_expert_tensor, axis=0) # (N_Tasks, Repeats, Epochs)

    # 2. Iterate over Repeats
    integrator = get_metric_integrator(m_integrator)
    
    results = {
        'transfer_scores': [],
        'remember_scores': []
    }

    for r in range(n_repeats):
        # Slice for this repeat
        # (Total_Epochs, N_Eval_Tasks)
        student_slice = full_student_tensor[:, r, :]
        
        # (N_Tasks, Epochs)
        expert_slice = full_expert_tensor[:, r, :]
        
        _, _, rem_mat, trans_mat = calculate_metrics_for_run(
            student_slice, expert_slice, ntasks, config.log_frequency, integrator
        )
        
        # Aggregate matrix to scalar (Mean of Diagonal / Upper Triangle)
        # Forward Transfer: Mean of Diagonal
        ft_score = np.nanmean(np.diag(trans_mat))
        
        # Remembering: Mean of Upper Triangle
        if ntasks > 1:
            # Extracts upper triangle indices (k=1 excludes diagonal)
            tri_indices = np.triu_indices(ntasks, k=1)
            rem_score = np.nanmean(rem_mat[tri_indices])
        else:
            rem_score = np.nan
            
        results['transfer_scores'].append(ft_score)
        results['remember_scores'].append(rem_score)

    # 3. Calculate Stats
    final_stats = {}
    for key, vals in results.items():
        vals = np.array(vals)
        # Filter NaNs if any (e.g. if a run failed)
        vals = vals[~np.isnan(vals)]
        
        if len(vals) > 0:
            stats = {
                'mean': np.mean(vals),
                'median': np.median(vals),
                'std': np.std(vals),
                'iqr': np.percentile(vals, 75) - np.percentile(vals, 25)
            }
        else:
            stats = {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'iqr': np.nan}
        
        final_stats[key] = stats

    # 4. Print Table
    print(f"\n{metric_type.upper()} Based CL Metrics (Aggregation: {config.n_repeats} repeats)")
    print("-" * 65)
    print(f"{'Metric':<20} | {'Mean':<10} | {'Median':<10} | {'Std':<10} | {'IQR':<10}")
    print("-" * 65)
    for k, s in final_stats.items():
        print(f"{k:<20} | {s['mean']:.4f}     | {s['median']:.4f}     | {s['std']:.4f}     | {s['iqr']:.4f}")
    print("-" * 65)
    
    return final_stats