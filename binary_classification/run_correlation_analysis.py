import os
import pickle
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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
    plast_path = os.path.join(experiment_path, "plastic_analysis_mnist.pkl") 
    
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
            train_tasks = sorted(list(glue_data.keys()))
            if not train_tasks:
                 raise ValueError("Glue data is empty")
            
            eval_tasks = sorted(list(glue_data[train_tasks[0]].keys()))
            metric_names = list(glue_data[train_tasks[0]][eval_tasks[0]].keys())
            
            n_train = len(train_tasks)
            n_eval = len(eval_tasks)
            
            # Determine n_repeats from data
            sample_data = glue_data[train_tasks[0]][eval_tasks[0]][metric_names[0]]
            if isinstance(sample_data, list):
                n_repeats = sample_data[0].shape[0] 
            else:
                n_repeats = sample_data.shape[1] 

            for metric in metric_names:
                # Shape: (N_Eval, N_Train, N_Repeats)
                M_raw = np.zeros((n_eval, n_train, n_repeats))
                M_raw[:] = np.nan
                
                for j, t_train in enumerate(train_tasks):
                    for i, t_eval in enumerate(eval_tasks):
                        try:
                            # Extract time series
                            series = glue_data[t_train][t_eval][metric]
                            if isinstance(series, list):
                                series = np.stack(series)
                            
                            # Take the FINAL value of this training phase
                            val_final = series[-1] # (Repeats,)
                            M_raw[i, j, :] = val_final
                        except (KeyError, IndexError):
                            pass
                
                # Compute Relative Difference Matrix
                M_diag = np.zeros((n_eval, n_repeats))
                for k in range(min(n_eval, n_train)):
                    M_diag[k, :] = M_raw[k, k, :]
                
                M_diag_expanded = M_diag[:, np.newaxis, :]
                denom = M_diag_expanded + M_raw
                eps = 1e-9
                M_rel = (M_diag_expanded - M_raw) / (denom + eps)
                
                results['glue'][metric] = {
                    'raw': M_raw,           
                    'relative': M_rel,      
                    'diagonal': M_diag      
                }
            
            print(f"  [x] Processed GLUE matrices for {len(metric_names)} metrics")
            
        except Exception as e:
            print(f"  [!] Error processing GLUE metrics: {e}")
    else:
        print(f"  [ ] GLUE metrics file not found: {glue_path}")

    # --- 4. Process Plasticity (Deltas) ---
    if os.path.exists(plast_path):
        try:
            with open(plast_path, 'rb') as f:
                plast_data = pickle.load(f)
            
            history = plast_data.get('history', {})
            # Calculate deltas between task boundaries
            
            for metric, data in history.items():
                if isinstance(data, list):
                    data = np.stack(data)
                
                results['plasticity'][metric] = {}
                samples_per_task = config.epochs_per_task // config.log_frequency
                
                for t in range(config.num_tasks):
                    idx_start = t * samples_per_task
                    idx_end = (t + 1) * samples_per_task - 1
                    
                    if idx_end < data.shape[0]:
                        val_initial = data[idx_start]
                        val_final = data[idx_end]
                        
                        delta = val_final - val_initial
                        results['plasticity'][metric][f"delta_task_{t:03d}"] = delta
                        results['plasticity'][metric][f"value_task_{t:03d}"] = val_final

            print(f"  [x] Processed Plasticity metrics")
            
        except Exception as e:
            print(f"  [!] Error processing Plasticity metrics: {e}")
    else:
        print(f"  [ ] Plasticity metrics file not found: {plast_path}")

    print("--- Metrics Calculation Complete ---")
    return results

def metric_x_metric_plot(x_data, y_data, x_label, y_label, title, save_path):
    """
    For the two metrics, return a matplotlib plot
    and the correlation value displayed as a floating value in the plot.
    """
    if len(x_data) != len(y_data):
        print(f"Skipping plot {title}: Dimension mismatch ({len(x_data)} vs {len(y_data)})")
        return

    # Remove NaNs
    valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]

    if len(x_clean) < 2:
        return

    # Calculate statistics
    r_val, p_val = pearsonr(x_clean, y_clean)
    
    # Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(x_clean, y_clean, alpha=0.7, edgecolors='w', s=80, color='royalblue')
    
    # Trend line
    if np.var(x_clean) > 0:
        m, b = np.polyfit(x_clean, y_clean, 1)
        plt.plot(x_clean, m*x_clean + b, color='darkorange', linestyle='--', linewidth=2, label=f'Fit (m={m:.2f})')

    plt.title(title + f"\nR = {r_val:.3f} (p={p_val:.3e})", fontsize=11)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def correlation_heatmap(correlations, labels, save_path):
    """
    Given the correlations between all the metrics across the repeats, 
    make a heatmap using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create Heatmap
    im = ax.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    
    # Label ticks
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # Use white text for dark backgrounds (high correlation) and black for light.
    # Note: If matrix is huge, we skip text to keep it readable
    if len(labels) < 20:
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = correlations[i, j]
                if np.isnan(val):
                    text_val = "nan"
                    color = "black"
                else:
                    text_val = f"{val:.2f}"
                    color = "white" if abs(val) > 0.5 else "black"
                    
                ax.text(j, i, text_val, ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Metric Correlation Matrix (Pearson R)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def _flatten_metrics(computed_data):
    """
    Helper to flatten nested metric dictionaries into a single dictionary of 
    1D arrays (vectors of repeats).
    """
    flat_data = {}
    
    # 1. Flatten Plasticity (Already mostly flat vectors)
    if 'plasticity' in computed_data:
        for metric, subdict in computed_data['plasticity'].items():
            for subkey, vector in subdict.items():
                # subkey is like 'delta_task_000' or 'value_task_000'
                # Clean up name: "Weight Mag Delta (T0)"
                short_metric = metric.split(" ")[0] # 'Weight' from 'Weight Magnitude'
                task_id = subkey.split("_")[-1] # '000'
                type_id = subkey.split("_")[0] # 'delta' or 'value'
                
                key = f"Plast_{short_metric}_{type_id}_T{task_id}"
                flat_data[key] = np.array(vector)

    # 2. Flatten GLUE (Extracting key summaries)
    if 'glue' in computed_data:
        for metric, matrices in computed_data['glue'].items():
            # relative: (N_Eval, N_Train, N_Repeats)
            rel_mat = matrices['relative'] 
            n_eval, n_train, _ = rel_mat.shape
            
            # Scenario A: Forgetting/Stability
            # Look at Task 0 after Training Final Task
            if n_train > 0:
                final_train = n_train - 1
                for e in range(n_eval):
                    # Relative change of Task E after all training
                    vec = rel_mat[e, final_train, :]
                    key = f"Glue_{metric}_Rel_T{e}_End"
                    flat_data[key] = np.array(vec)

    # 3. Flatten CL (Updated Logic)
    if 'cl' in computed_data:
        cl = computed_data['cl']
        
        # --- Transfer (Task-Specific) ---
        if 'transfer' in cl:
            # Shape: (N_Tasks, N_Repeats)
            trans = cl['transfer']
            
            if isinstance(trans, (np.ndarray, jnp.ndarray)):
                if trans.ndim == 2:
                    n_tasks_cl = trans.shape[0]
                    # We create a metric entry for EACH task's transfer vector
                    for t in range(n_tasks_cl):
                        key = f"CL_Transfer_T{t}"
                        flat_data[key] = np.array(trans[t, :])
                elif trans.ndim == 1:
                     flat_data['CL_Transfer_Avg'] = np.array(trans)

        # --- Remembering (Pair-Specific Upper Triangle) ---
        if 'remembering' in cl:
            # Shape: (N_Eval_Tasks, N_Train_Tasks, N_Repeats)
            rem = cl['remembering']
            
            if isinstance(rem, (np.ndarray, jnp.ndarray)) and rem.ndim == 3:
                n_eval, n_train, _ = rem.shape
                
                # We want the upper right diagonal: Train Task (j) > Eval Task (i)
                # This represents how well we remember Task i after subsequently training on Task j
                # e.g., if we train T0, then T1. We want to know how T0 is doing (i=0) after training T1 (j=1).
                
                for i in range(n_eval):
                    for j in range(i + 1, n_train):
                        # Extract the vector of repeats for this specific pair
                        vec = rem[i, j, :]
                        
                        # Only add if it contains valid data (not all NaNs)
                        if not np.all(np.isnan(vec)):
                            key = f"CL_Rem_Eval{i}_Train{j}"
                            flat_data[key] = np.array(vec)
            
        # --- Stats (optional global averages) ---
        if 'stats' in cl and 'accuracy' in cl['stats']:
            acc = cl['stats']['accuracy']
            if isinstance(acc, (np.ndarray, jnp.ndarray)) and acc.ndim == 2:
                 flat_data['CL_Avg_Accuracy'] = np.mean(acc, axis=1)

    return flat_data

def generate_correlation_plots(computed_data, experiment_path):
    """
    For each metric, calculate the pearson correlation and then make plots.
    """
    print("--- Generating Correlation Analysis ---")
    
    # 1. Create Output Directory
    corr_dir = os.path.join(experiment_path, "correlations")
    os.makedirs(corr_dir, exist_ok=True)
    
    # 2. Flatten Data
    flat_data = _flatten_metrics(computed_data)
    
    # Filter out metrics with 0 variance or all NaNs to prevent errors
    valid_data = {}
    for k, v in flat_data.items():
        v = np.array(v)
        if np.any(np.isnan(v)):
            if np.sum(np.isnan(v)) > len(v) // 2:
                # print(f"  [Skipping] {k} (Too many NaNs)")
                continue
        if np.var(v) < 1e-12:
            # print(f"  [Skipping] {k} (Zero Variance)") 
            continue
        valid_data[k] = v
        
    metric_names = sorted(list(valid_data.keys()))
    n_metrics = len(metric_names)
    
    if n_metrics < 2:
        print("  [!] Not enough valid metrics for correlation analysis.")
        return

    print(f"  [i] Computing correlations for {n_metrics} metrics...")

    # 3. Compute Correlation Matrix
    corr_matrix = np.zeros((n_metrics, n_metrics))
    p_matrix = np.zeros((n_metrics, n_metrics))
    
    for i, name_i in enumerate(metric_names):
        for j, name_j in enumerate(metric_names):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue
                
            vec_i = valid_data[name_i]
            vec_j = valid_data[name_j]
            
            # Handle potential NaNs pairwise
            mask = ~np.isnan(vec_i) & ~np.isnan(vec_j)
            if np.sum(mask) < 3: # Need at least 3 points for meaningful correlation
                c, p = 0, 1
            else:
                c, p = pearsonr(vec_i[mask], vec_j[mask])
            
            corr_matrix[i, j] = c
            p_matrix[i, j] = p

    # 4. Generate Heatmap
    heatmap_path = os.path.join(corr_dir, "correlation_matrix.png")
    correlation_heatmap(corr_matrix, metric_names, heatmap_path)
    print(f"  [x] Saved Heatmap to {heatmap_path}")

    # 5. Generate Scatter Plots for Significant Correlations
    # We only plot pairs with |R| > 0.6 and p < 0.05 to reduce noise
    count = 0
    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            r_val = corr_matrix[i, j]
            p_val = p_matrix[i, j]
            
            if abs(r_val) > 0.6 and p_val < 0.05:
                name_x = metric_names[i]
                name_y = metric_names[j]
                
                # Check if we are correlating different types (e.g. GLUE vs CL)
                type_x = name_x.split("_")[0]
                type_y = name_y.split("_")[0]
                
                # Prioritize Cross-Metric correlations to avoid too many plots
                if type_x != type_y:
                    fname = f"corr_{name_x}_VS_{name_y}.png"
                    fname = fname.replace(" ", "").replace("(", "").replace(")", "")
                    # shorten filename if too long
                    if len(fname) > 255:
                        fname = f"corr_{i}_VS_{j}.png"
                        
                    save_path = os.path.join(corr_dir, fname)
                    
                    metric_x_metric_plot(
                        valid_data[name_x], 
                        valid_data[name_y],
                        name_x, name_y,
                        f"Correlation Analysis",
                        save_path
                    )
                    count += 1
    
    print(f"  [x] Generated {count} scatter plots for significant correlations")

if __name__ == "__main__":
    # EXP_PATH should be updated to your actual local path
    EXP_PATH = "/home/users/MTrappett/manifold/binary_classification/results/mnist/RL/"
    
    if os.path.exists(EXP_PATH):
        computed_data = compute_metric_differences(EXP_PATH)
        generate_correlation_plots(computed_data, EXP_PATH)
    else:
        print(f"Experiment path not found: {EXP_PATH}")