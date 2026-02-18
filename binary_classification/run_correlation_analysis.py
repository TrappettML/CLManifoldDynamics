import os
import pickle
import numpy as np
import warnings

def compute_metric_differences(experiment_path):
    """
    Loads experiment configuration and metrics, then computes:
    1. Delta GLUE/Plasticity: (Metric after Training Task N - Metric after Training Task M)
       specifically for the case defined: (Task 1 after Training Task 2 - Task 1 after Training Task 1).
       It generalizes this to: Delta = Val(Eval=i, Train=j) - Val(Eval=i, Train=i) for j > i.
    2. CL Metrics: Extracts 'remembering' and 'transfer'.
    3. MTL Metrics: Extracts mean GLUE and Plasticity values from the MTL baseline.

    Args:
        experiment_path (str): Path to the experiment results directory (containing config.pkl and all_metrics.pkl).

    Returns:
        dict: A dictionary containing 'deltas', 'cl_metrics', and 'mtl_metrics'.
    """
    
    # --- 1. Load Configuration ---
    config_path = os.path.join(experiment_path, "config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
        
    # --- 2. Load Aggregated Metrics ---
    metrics_path = os.path.join(experiment_path, "all_metrics.pkl")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}. Please run run_all_representation_analysis() first.")

    with open(metrics_path, 'rb') as f:
        all_metrics = pickle.load(f)

    # --- 3. Determine Time Indices ---
    # We need the index corresponding to the "final epoch" of each training task.
    # Data shape is (Steps, Repeats). Steps = Total_Epochs / log_freq.
    
    # Number of log steps per task
    steps_per_task = config.epochs_per_task // config.log_frequency
    
    # Calculate the index of the *last* log entry for each task.
    # Task 0 ends at index (1 * steps_per_task) - 1
    # Task 1 ends at index (2 * steps_per_task) - 1
    task_end_indices = [((t + 1) * steps_per_task) - 1 for t in range(config.num_tasks)]
    
    results = {
        'deltas': {'glue': {}, 'plasticity': {}},
        'cl_metrics': {},
        'mtl_metrics': {'glue': {}, 'plasticity': {}}
    }

    # =========================================================
    # PART A: GLUE Metric Differences
    # Formula: Delta = Metric(Eval=T_i, Train=T_j) - Metric(Eval=T_i, Train=T_i)
    # This represents how much the metric changed for Task i after training on Task j.
    # =========================================================
    
    if 'glue' in all_metrics and all_metrics['glue']:
        glue_data = all_metrics['glue']
        # glue_data structure: [train_task_name][eval_task_name][metric_name] -> (Steps, Repeats)
        
        # Get list of task names to iterate in order (task_000, task_001...)
        task_names = sorted([f"task_{i:03d}" for i in range(config.num_tasks)])
        
        # We need the list of metric names (e.g., Capacity, Dimension...)
        # Grab from the first available entry
        first_train = task_names[0]
        first_eval = task_names[0]
        metric_names = list(glue_data[first_train][first_eval].keys())

        for m_name in metric_names:
            results['deltas']['glue'][m_name] = {}

            # Iterate over EVALUATION tasks (The task we are measuring)
            for i, eval_task in enumerate(task_names):
                
                # We can only compute forgetting/change if there are subsequent training tasks.
                # If this is the last task, it has no "future" training to disturb it.
                if i >= config.num_tasks - 1:
                    continue

                # 1. Get Baseline: Value at the end of training Task i
                # Note: The glue_data structure is hierarchical by train_task.
                # We need to look up glue_data[eval_task][eval_task] because T_i is trained in phase T_i.
                try:
                    # Data for "Train on i, Eval on i"
                    base_series = glue_data[eval_task][eval_task][m_name] # (Steps, Repeats)
                    
                    # The steps in this specific array correspond to the training of eval_task.
                    # So we just take the last element (-1) of this specific training block.
                    base_val = base_series[-1, :] # (Repeats,)
                    
                except KeyError:
                    warnings.warn(f"Missing GLUE baseline for Eval {eval_task}, Metric {m_name}")
                    continue

                # 2. Get Subsequent Values: Value at end of training Task j (j > i)
                for j in range(i + 1, config.num_tasks):
                    train_task = task_names[j]
                    
                    try:
                        # Data for "Train on j, Eval on i"
                        curr_series = glue_data[train_task][eval_task][m_name]
                        curr_val = curr_series[-1, :] # End of training task j
                        
                        # 3. Compute Difference
                        diff = curr_val - base_val
                        
                        # Store Key: "eval_task_000_trained_on_task_001"
                        key = f"eval_{eval_task}_after_{train_task}"
                        results['deltas']['glue'][m_name][key] = diff
                        
                    except KeyError:
                        continue

    # =========================================================
    # PART B: Plasticity Metric Differences
    # Plasticity metrics are continuous (global), not per-task.
    # Formula: Delta = Metric(End of Task j) - Metric(End of Task i)
    # =========================================================
    
    if 'plasticity' in all_metrics and all_metrics['plasticity']:
        plast_data = all_metrics['plasticity'] # {metric_name: (Total_Steps, Repeats)}
        
        for m_name, series in plast_data.items():
            results['deltas']['plasticity'][m_name] = {}
            
            # Iterate through tasks to find "shifts"
            for i in range(config.num_tasks):
                idx_i = task_end_indices[i]
                
                # Ensure index exists (in case run was short)
                if idx_i >= series.shape[0]:
                    continue
                    
                val_i = series[idx_i, :]
                
                # Compare against future tasks
                for j in range(i + 1, config.num_tasks):
                    idx_j = task_end_indices[j]
                    
                    if idx_j >= series.shape[0]:
                        continue
                        
                    val_j = series[idx_j, :]
                    
                    diff = val_j - val_i
                    
                    key = f"change_from_task_{i:03d}_to_{j:03d}"
                    results['deltas']['plasticity'][m_name][key] = diff

    # =========================================================
    # PART C: CL Metrics (Transfer & Remembering)
    # =========================================================
    
    if 'cl' in all_metrics and 'remembering' in all_metrics['cl']:
        # These were already computed as per-repeat vectors in run_analysis.py
        # We assume they are stored as (Repeats,) or (N, Repeats) in the dictionary
        # Based on previous code: {'remembering': array, 'transfer': array}
        
        # Just copy them over
        cl_data = all_metrics['cl']
        for k in ['remembering', 'transfer']:
            if k in cl_data:
                results['cl_metrics'][k] = cl_data[k]
                
        # Also grab raw accuracy per task if needed, but Transfer/Remembering are the requested summaries.

    # =========================================================
    # PART D: MTL Metrics (Baseline Means)
    # =========================================================
    
    if 'mtl' in all_metrics and all_metrics['mtl']:
        mtl_data = all_metrics['mtl']
        
        # 1. MTL GLUE
        # Structure: mtl_data['glue'][train_task_name][metric] -> (Steps, Repeats)
        # Note: In MTL, "train_task_name" in the pickle usually corresponds to the evaluated task
        # because MTL representation analysis iterates over eval tasks (see run_mtl_glue_analysis).
        if 'glue' in mtl_data:
            for task_name, metrics_dict in mtl_data['glue'].items():
                for m_name, series in metrics_dict.items():
                    if m_name not in results['mtl_metrics']['glue']:
                        results['mtl_metrics']['glue'][m_name] = {}
                    
                    # We take the mean over the *entire* MTL training run or the final value?
                    # The prompt asks for "mean glue metrics".
                    # MTL runs are often stable. We will take the mean of the *final* epoch
                    # across repeats to be consistent with "final state" comparison,
                    # OR the mean over time if requested.
                    # Let's stick to the "values at the final epoch" logic for consistency.
                    final_vals = series[-1, :] # (Repeats,)
                    results['mtl_metrics']['glue'][m_name][task_name] = final_vals

        # 2. MTL Plasticity
        # Structure: mtl_data['plasticity'][metric] -> (Steps, Repeats)
        if 'plasticity' in mtl_data:
            for m_name, series in mtl_data['plasticity'].items():
                # "mean plasticity values... for each repeat"
                # Since plasticity changes during training, "mean" likely implies
                # the average value across the training trajectory for the MTL agent.
                mean_vals = np.nanmean(series, axis=0) # Mean over time -> (Repeats,)
                results['mtl_metrics']['plasticity'][m_name] = mean_vals

    return results

# --- Example Usage Block ---
if __name__ == "__main__":
    # Update this path to your actual experiment folder
    exp_path = "/home/users/MTrappett/manifold/binary_classification/results/mnist/RL/"
    
    try:
        data = compute_metric_differences(exp_path)
        
        print("\n=== Analysis Complete ===")
        
        print("\n--- CL Metrics (Mean +/- Std) ---")
        for k, v in data['cl_metrics'].items():
            print(f"{k}: {np.mean(v):.4f} +/- {np.std(v):.4f}")

        print("\n--- GLUE Deltas (Example: Capacity) ---")
        if 'Capacity' in data['deltas']['glue']:
            for k, v in data['deltas']['glue']['Capacity'].items():
                print(f"{k}: {np.mean(v):.4f}")
                
        print("\n--- MTL Metrics (Example: Plasticity) ---")
        for k, v in data['mtl_metrics']['plasticity'].items():
            print(f"{k}: {np.mean(v):.4f}")
            
    except Exception as e:
        print(f"Analysis failed: {e}")