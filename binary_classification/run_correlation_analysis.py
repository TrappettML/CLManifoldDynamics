import os
import pickle
import numpy as np
import warnings
from ipdb import set_trace

def compute_metric_differences(experiment_path):
    """
    Computes analysis matrices for GLUE, Plasticity, and CL metrics.
    
    Returns:
        results (dict): Contains:
            - 'glue': {metric_name: {'raw': Matrix, 'relative': Matrix}}
            - 'plasticity': {metric_name: Delta_Vector}
            - 'cl': {metric_name: Vector/Matrix}
    """
    print(f"--- Processing Metrics for: {experiment_path} ---")

    # --- 1. Load Data ---
    config_path = os.path.join(experiment_path, "config.pkl")
    cl_path = os.path.join(experiment_path, "cl_metrics.pkl")
    glue_path = os.path.join(experiment_path, "glue_metrics.pkl")
    plast_path = os.path.join(experiment_path, "plastic_analysis_mnist.pkl") # Tries generic name first
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f) 

    results = {
        'glue': {},
        'plasticity': {},
        'cl': {}
    }

    # --- 2. Process CL Metrics (Direct Load) ---
    if os.path.exists(cl_path):
        try:
            with open(cl_path, 'rb') as f:
                cl_data = pickle.load(f)
            # cl_data structure: {'transfer': array, 'remembering': array, 'stats': dict}
            # remembering array has nan on diag and lower left of nTask x nTask matrix, only upper right has rem values. 
            results['cl'] = cl_data
            print(f"  [x] Loaded CL Metrics from {os.path.basename(cl_path)}")
        except Exception as e:
            print(f"  [!] Error loading CL metrics: {e}")
    else:
        print(f"  [ ] CL metrics file not found: {cl_path}")

    # --- 3. Process GLUE Metrics (Full Matrix) ---
    if os.path.exists(glue_path):
        try:
            with open(glue_path, 'rb') as f:
                glue_data = pickle.load(f)
            
            # glue_data: {train_task: {eval_task: {metric: [steps, repeats]}}}
            # We want to build Matrix M[eval_task, train_task]
            
            # Get sorted task names to ensure index alignment
            # Assuming task names are like "task_000", "task_001"
            train_tasks = sorted(list(glue_data.keys()))
            eval_tasks = sorted(list(glue_data[train_tasks[0]].keys()))
            
            # Get metric names from the first entry
            metric_names = list(glue_data[train_tasks[0]][eval_tasks[0]].keys())
            
            n_train = len(train_tasks)
            n_eval = len(eval_tasks)
            
            # Determine n_repeats from data
            sample_data = glue_data[train_tasks[0]][eval_tasks[0]][metric_names[0]]
            # Handle case where it might be list of arrays or just array
            if isinstance(sample_data, list):
                n_repeats = sample_data[0].shape[0] # First step, shape (Repeats,)
            else:
                n_repeats = sample_data.shape[1] # (Steps, Repeats)

            for metric in metric_names:
                # Initialize matrices
                # Shape: (N_Eval, N_Train, N_Repeats)
                M_raw = np.zeros((n_eval, n_train, n_repeats))
                M_raw[:] = np.nan
                
                for j, t_train in enumerate(train_tasks):
                    for i, t_eval in enumerate(eval_tasks):
                        try:
                            # Extract time series
                            series = glue_data[t_train][t_eval][metric]
                            
                            # Ensure numpy array
                            if isinstance(series, list):
                                series = np.stack(series) # (Steps, Repeats)
                            
                            # Take the FINAL value of this training phase
                            val_final = series[-1] # (Repeats,)
                            
                            M_raw[i, j, :] = val_final
                        except (KeyError, IndexError) as e:
                            pass
                
                # Compute Relative Difference Matrix
                # Formula: (M_ii - M_ij) / (M_ii + M_ij)
                # This compares the state after training task j (M_ij) 
                # to the state after training task i (M_ii, the "Baseline" or "Solved" state)
                
                # Extract Diagonal (Baseline): M[i, i]
                # We need to broadcast M_ii to match M_ij columns
                M_diag = np.diagonal(M_raw, axis1=0, axis2=1) # (Repeats, N_Tasks) -> wait, numpy diagonal behavior is tricky
                # Correct way for (N, N, R):
                M_diag = np.zeros((n_eval, n_repeats))
                for k in range(min(n_eval, n_train)):
                    M_diag[k, :] = M_raw[k, k, :]
                
                # Expand dimensions for broadcasting: (N_Eval, 1, N_Repeats)
                M_diag_expanded = M_diag[:, np.newaxis, :]
                
                # Avoid division by zero
                denom = M_diag_expanded + M_raw
                eps = 1e-9
                
                # Calculate Relative Matrix
                M_rel = (M_diag_expanded - M_raw) / (denom + eps)
                
                results['glue'][metric] = {
                    'raw': M_raw,           # The absolute values
                    'relative': M_rel,      # The normalized deviation from diagonal
                    'diagonal': M_diag      # The peak performance values
                }
            
            print(f"  [x] Processed GLUE matrices for {len(metric_names)} metrics")
            
        except Exception as e:
            print(f"  [!] Error processing GLUE metrics: {e}")
            # raise e # Uncomment for debugging
    else:
        print(f"  [ ] GLUE metrics file not found: {glue_path}")

    # --- 4. Process Plasticity (Deltas) ---
    if os.path.exists(plast_path):
        try:
            with open(plast_path, 'rb') as f:
                plast_data = pickle.load(f)
            
            # plast_data['history'] -> {metric: (Total_Steps, Repeats)}
            # plast_data['task_boundaries'] -> [step_idx_1, step_idx_2, ...]
            
            history = plast_data.get('history', {})
            boundaries = plast_data.get('task_boundaries', [])
            
            # We calculate deltas between task boundaries
            # i.e., Change in Plasticity Metric during Task X
            
            for metric, data in history.items():
                if isinstance(data, list):
                    data = np.stack(data)
                
                # data shape: (Total_Steps, Repeats)
                results['plasticity'][metric] = {}
                
                # Pre-Task 0 Baseline (Step 0)
                val_start = data[0]
                
                current_step = 0
                prev_boundary_step = 0
                
                # The boundaries in plast_file are often stored as Epochs, not array indices
                # We need to rely on the array shape and number of tasks usually.
                # Assuming uniform log_freq
                
                if 'log_frequency' in config:
                    log_freq = config.log_frequency
                else:
                    log_freq = 10 # Fallback
                
                # Calculate step indices from epochs or shape
                steps_per_task = data.shape[0] // config.num_tasks
                
                for t in range(config.num_tasks):
                    # Start and End indices for Task t
                    idx_start = t * steps_per_task
                    idx_end = (t + 1) * steps_per_task - 1
                    
                    if idx_end < data.shape[0]:
                        val_initial = data[idx_start]
                        val_final = data[idx_end]
                        
                        # Delta during task
                        delta = val_final - val_initial
                        results['plasticity'][metric][f"delta_task_{t:03d}"] = delta
                        
                        # Absolute value at end of task
                        results['plasticity'][metric][f"value_task_{t:03d}"] = val_final

            print(f"  [x] Processed Plasticity metrics")
            
        except Exception as e:
            print(f"  [!] Error processing Plasticity metrics: {e}")
    else:
        print(f"  [ ] Plasticity metrics file not found: {plast_path}")

    print("--- Metrics Calculation Complete ---")
    return results

def generate_correlation_plots(computed_data):
    """For each metric, caclulate the pearson correlation and then make plots.
    for example: for remembering and glue dimension difference for task 1, 
    take the repeat values of one and find the correlation 
    value of the repeat values of the other.
    
    When plotted, this will yield a plot of 30 points, 
    where the y-axis would be glue dimension difference for task1 and 
    x-axis would be remembering.

    for each plot we can also dispaly the correlation values. 

    We'll make a new folder in the experiment path for putting the correlation plots. 

    We will save the correlations as well as produce a heatmap of them. 
    """
    pass

if __name__ == "__main__":
    # Example usage
    EXP_PATH = "/home/users/MTrappett/manifold/binary_classification/results/mnist/RL/"
    computed_data = compute_metric_differences(EXP_PATH)
    set_trace()