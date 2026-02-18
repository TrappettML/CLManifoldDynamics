import os
import pickle
import numpy as np
import warnings
from ipdb import set_trace

def compute_metric_differences(experiment_path):
    """
    Computes:
    1. CL Deltas: Change in metrics for Task A after training on Task B vs after training on Task A.
    2. CL Metrics: Extracts 'remembering' and 'transfer'.
    3. MTL Means: Time-averaged metrics for the Multi-Task Learner.

    Args:
        experiment_path (str): Path to results (must contain config.pkl and all_metrics.pkl).
    """
    print(f"--- Processing Metrics for: {experiment_path} ---")

    # --- 1. Load Data ---
    try:
        with open(os.path.join(experiment_path, "config.pkl"), 'rb') as f:
            config = pickle.load(f)
        with open(os.path.join(experiment_path, "all_metrics.pkl"), 'rb') as f:
            all_metrics = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None

    results = {
        'glue_deltas': {},         # {metric: { 'task_0_after_task_1': [repeats] }}
        'plasticity_deltas': {},   # {metric: { 'task_0_to_1': [repeats] }}
        'cl_metrics': {},          # { 'remembering': [repeats], ... }
        'mtl_glue_means': {},      # { task: { metric: [repeats] } }
        'mtl_plasticity_means': {} # { metric: [repeats] }
    }
    # set_trace()
    # --- Helper: Ensure Data is Numpy Array ---
    def ensure_stack(data_obj):
        """Fixes issue where CL metrics might be lists of arrays instead of stacked arrays."""
        if isinstance(data_obj, list):
            return np.stack(data_obj)
        return data_obj

    # --- 2. CL GLUE Deltas ---
    # Goal: Delta = Metric(Eval=T0, State=End_of_T1) - Metric(Eval=T0, State=End_of_T0)
    if 'glue' in all_metrics and all_metrics['glue']:
        cl_glue = all_metrics['glue']
        
        # We need the list of task names in chronological order
        sorted_tasks = sorted([t for t in cl_glue.keys() if t.startswith('task_')])
        
        # Extract metric names from the first valid entry
        first_valid = cl_glue[sorted_tasks[0]][sorted_tasks[0]]
        metric_names = list(first_valid.keys())

        for metric in metric_names:
            results['glue_deltas'][metric] = {}

            # Iterate over the EVALUATED task (The task being remembered)
            for i in range(len(sorted_tasks)):
                eval_task = sorted_tasks[i]
                
                # Get the baseline: State of Eval Task i immediately after training Task i
                try:
                    # Data: Train=i, Eval=i
                    base_data = ensure_stack(cl_glue[eval_task][eval_task][metric])
                    # Value at the final step of this training phase
                    val_base = base_data[-1] 
                except (KeyError, IndexError) as e:
                    # Task might not have been evaluated or data missing
                    continue

                # Iterate over FUTURE training phases (The tasks causing interference)
                for j in range(i + 1, len(sorted_tasks)):
                    train_task = sorted_tasks[j]
                    
                    try:
                        # Data: Train=j, Eval=i
                        # We want the state at the END of training task j
                        curr_data = ensure_stack(cl_glue[train_task][eval_task][metric])
                        val_curr = curr_data[-1]

                        # Compute Delta: (Post - Pre)
                        diff = val_curr - val_base
                        
                        # Key format: "task_000_after_training_task_001"
                        key = f"{eval_task}_after_training_{train_task}"
                        results['glue_deltas'][metric][key] = diff
                        
                    except (KeyError, IndexError):
                        continue

    # --- 3. CL Plasticity Deltas ---
    # Plasticity is a global time-series. We slice it at task boundaries.
    if 'plasticity' in all_metrics and all_metrics['plasticity']:
        cl_plast = all_metrics['plasticity']
        
        steps_per_task = config.epochs_per_task // config.log_frequency
        
        for metric, data_raw in cl_plast.items():
            data = ensure_stack(data_raw) # (Total_Steps, Repeats)
            results['plasticity_deltas'][metric] = {}
            
            # Iterate transitions
            for i in range(config.num_tasks - 1):
                # End index of Task i (0-based)
                # If steps_per_task=50. Task 0 ends at 49. Task 1 ends at 99.
                idx_i = ((i + 1) * steps_per_task) - 1
                
                # End index of Task i+1
                idx_j = ((i + 2) * steps_per_task) - 1
                
                # Safety check for array bounds
                if idx_j < data.shape[0]:
                    val_i = data[idx_i]
                    val_j = data[idx_j]
                    
                    diff = val_j - val_i
                    
                    key = f"delta_task_{i:03d}_to_{i+1:03d}"
                    results['plasticity_deltas'][metric][key] = diff

    # --- 4. CL Metrics (Transfer/Remembering) ---
    if 'cl' in all_metrics:
        # These are usually 1D arrays of length (Repeats)
        for key in ['remembering', 'transfer']:
            if key in all_metrics['cl']:
                results['cl_metrics'][key] = ensure_stack(all_metrics['cl'][key])

    # --- 5. MTL Means (Averaged over Time) ---
    if 'mtl' in all_metrics and all_metrics['mtl']:
        mtl_data = all_metrics['mtl']

        # A. MTL Plasticity: Mean over time for each metric
        if 'plasticity' in mtl_data:
            for metric, data_raw in mtl_data['plasticity'].items():
                data = ensure_stack(data_raw) # (Steps, Repeats)
                # Mean over time axis (0) -> Result shape (Repeats,)
                results['mtl_plasticity_means'][metric] = np.mean(data, axis=0)

        # B. MTL GLUE: Mean over time for each metric, separated by task
        if 'glue' in mtl_data:
            # MTL GLUE structure is {eval_task: {metric: (Steps, Repeats)}}
            for task_name, task_metrics in mtl_data['glue'].items():
                results['mtl_glue_means'][task_name] = {}
                
                for metric, data_raw in task_metrics.items():
                    data = ensure_stack(data_raw) # (Steps, Repeats)
                    # Mean over time axis (0) -> Result shape (Repeats,)
                    results['mtl_glue_means'][task_name][metric] = np.mean(data, axis=0)
    set_trace()
    print("--- Metrics Calculation Complete ---")

    return results

if __name__ == "__main__":
    # Example usage
    EXP_PATH = "/home/users/MTrappett/manifold/binary_classification/results/mnist/RL/"
    computed_data = compute_metric_differences(EXP_PATH)
