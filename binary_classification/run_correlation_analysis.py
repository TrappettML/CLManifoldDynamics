import os
import pickle
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

import os
import pickle
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

def compute_metric_differences(experiment_path):
    """
    Computes relative difference matrices focusing on the transition 
    from Task 0 to Task 1.
    
    Calculation: (Val_End_T1 - Val_End_T0) / (Val_End_T1 + Val_End_T0)
    Result: Positive value implies the metric increased during Task 1.
    
    Returns:
        results (dict): Contains:
            - 'glue': {metric_name: {eval_task_id: rel_diff_vector}}
            - 'plasticity': {metric_name: rel_diff_vector}
            - 'cl': {metric_name: Vector}
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

    # Helper for safe relative difference
    def calc_rel_diff(v0, v1):
        # (t1 - t0) / (t1 + t0)
        num = v1 - v0
        denom = v1 + v0
        # Add epsilon where denom is 0 to avoid NaNs, preserve sign elsewhere
        denom = np.where(denom == 0, 1e-9, denom) 
        return num / denom

    # --- 2. Process CL Metrics (Direct Load) ---
    if os.path.exists(cl_path):
        try:
            with open(cl_path, 'rb') as f:
                cl_data = pickle.load(f)
            results['cl'] = cl_data
            print(f"  [x] Loaded CL Metrics from {os.path.basename(cl_path)}")
        except Exception as e:
            print(f"  [!] Error loading CL metrics: {e}")

    # --- 3. Process GLUE Metrics ---
    if os.path.exists(glue_path):
        try:
            with open(glue_path, 'rb') as f:
                glue_data = pickle.load(f)
            
            # glue_data structure: {train_task: {eval_task: {metric: [steps, repeats]}}}
            if 0 not in glue_data or 1 not in glue_data:
                 print("  [!] GLUE data missing Task 0 or Task 1 data.")
            else:
                # Get list of eval tasks and metrics from the first training block
                sample_train_block = glue_data[0]
                eval_tasks = sorted(list(sample_train_block.keys()))
                metric_names = list(sample_train_block[eval_tasks[0]].keys())
                
                for metric in metric_names:
                    results['glue'][metric] = {}
                    
                    for t_eval in eval_tasks:
                        try:
                            # 1. Get value at End of Task 0 (t0)
                            series_t0 = glue_data[0][t_eval][metric]
                            if isinstance(series_t0, list): series_t0 = np.stack(series_t0)
                            val_t0 = series_t0[-1] 
                            
                            # 2. Get value at End of Task 1 (t1)
                            series_t1 = glue_data[1][t_eval][metric]
                            if isinstance(series_t1, list): series_t1 = np.stack(series_t1)
                            val_t1 = series_t1[-1] 
                            
                            # 3. Calculate Relative Difference
                            rel_diff = calc_rel_diff(val_t0, val_t1)
                            
                            results['glue'][metric][t_eval] = rel_diff
                            
                        except (KeyError, IndexError):
                            pass
                
                print(f"  [x] Processed GLUE relative differences")
            
        except Exception as e:
            print(f"  [!] Error processing GLUE metrics: {e}")
    else:
        print(f"  [ ] GLUE metrics file not found: {glue_path}")

    # Helper for safe relative difference
    # If flip_sign is True: We want Decrease to be Positive (Good) -> (v0 - v1)
    # If flip_sign is False: We want Increase to be Positive (Good) -> (v1 - v0)
    def calc_rel_diff(v0, v1, flip_sign=False):
        denom = v1 + v0
        # Add epsilon where denom is 0 to avoid NaNs
        denom = np.where(denom == 0, 1e-9, denom)
        
        if flip_sign:
            # "Lower is Better" (e.g. Dormant Units)
            # If v1 < v0 (Decrease), num is positive -> GOOD
            num = v0 - v1  
        else:
            # "Higher is Better" (e.g. Accuracy, Rank)
            # If v1 > v0 (Increase), num is positive -> GOOD
            num = v1 - v0
            
        return num / denom

    # Configuration: Map raw keys to (CleanName, Flip_Bool)
    # Flip_Bool = True means "Decrease is Good" (Positive output = metric went down)
    PLASTICITY_CONFIG = {
        # Lower is Better (Pathologies)
        'dormant':          ('DormantUnits', True),  #  High dormancy = pathology
        'weight_mag':       ('WeightMag', True),     # [cite: 332] High mag = saturation/loss of plasticity
        'l2':               ('L2Norm', True),        # Similar to Weight Mag
        
        # Higher is Better (Health Signals)
        'active':           ('ActiveUnits', False),  # [cite: 310] Active units maintain plasticity
        'stable_rank':      ('StableRank', False),   # [cite: 317] Higher rank = better representation
        'effective_rank':   ('EffectiveRank', False),# [cite: 323] Higher effective dim is better
        'gradient_norm':    ('GradNorm', False),     # [cite: 365] Avoid vanishing gradients
        'feature_norm':     ('FeatureNorm', False),  # [cite: 347] Avoid representation collapse
        'variance':         ('FeatureVar', False),   # [cite: 357] Collapse in variance = forgetting
        'entropy':          ('Entropy', False),      # [cite: 373] Low entropy = overconfident/rigid
        'weight_diff':      ('WeightDiff', False),   # [cite: 338] Zero diff = no learning
    }

    # --- 4. Process Plasticity (Relative Deltas) ---
    if os.path.exists(plast_path):
        try:
            with open(plast_path, 'rb') as f:
                plast_data = pickle.load(f)
            
            history = plast_data.get('history', {})
            samples_per_task = config.epochs_per_task // config.log_frequency
            
            idx_end_t0 = samples_per_task - 1          # End of Task 0
            idx_end_t1 = (2 * samples_per_task) - 1    # End of Task 1
            
            for metric_raw, data in history.items():
                if isinstance(data, list):
                    data = np.stack(data)
                
                # Determine clean name and direction from partial match
                clean_name = None
                should_flip = False
                
                # Simple string matching against config keys
                metric_lower = metric_raw.lower()
                for key_sub, (name, flip) in PLASTICITY_CONFIG.items():
                    if key_sub in metric_lower:
                        clean_name = name
                        should_flip = flip
                        break
                
                # Fallback if metric not in config (assume Higher is Better)
                if clean_name is None:
                    clean_name = metric_raw.replace(" ", "")
                
                # Construct final key: e.g., "DormantUnits_t1t0"
                res_key = f"{clean_name}_t1t0"

                if data.shape[0] > idx_end_t1:
                    val_t0 = data[idx_end_t0]   # End Task 0
                    val_t1 = data[idx_end_t1]   # End Task 1
                    
                    # Calculate Directional Relative Difference
                    # Positive Result ALWAYS means "Improved Plasticity/Health"
                    rel_delta = calc_rel_diff(val_t0, val_t1, flip_sign=should_flip)
                    
                    results['plasticity'][res_key] = rel_delta
                else:
                    print(f"  [!] Not enough steps in plasticity {metric_raw}")

            print(f"  [x] Processed Plasticity relative differences")
            
        except Exception as e:
            print(f"  [!] Error processing Plasticity metrics: {e}")

    return results

def _flatten_metrics(computed_data):
    """
    Helper to flatten the simplified metric dictionary into 1D arrays (repeats).
    """
    flat_data = {}
    
    # 1. Flatten Plasticity
    if 'plasticity' in computed_data:
        for metric, vector in computed_data['plasticity'].items():
            # Clean up name: "Weight Mag Delta" -> "Plast_Weight"
            short_metric = metric.split(" ")[0] 
            # RelDiff indicates (t1-t0)/(t1+t0)
            key = f"{short_metric}"
            flat_data[key] = np.array(vector)

    # 2. Flatten GLUE
    if 'glue' in computed_data:
        for metric, eval_dict in computed_data['glue'].items():
            for t_eval, vector in eval_dict.items():
                # Key: Glue_{Metric}_EvalT{TaskID}
                key = f"Glue_{metric}_EvalT{t_eval}"
                flat_data[key] = np.array(vector)

    # 3. Flatten CL (Standard Processing)
    if 'cl' in computed_data:
        cl = computed_data['cl']
        
        # Transfer
        if 'transfer' in cl:
            trans = cl['transfer']
            if isinstance(trans, (np.ndarray, jnp.ndarray)):
                if trans.ndim == 2:
                    # Only take first few tasks if desired, or all
                    for t in range(trans.shape[0]):
                        flat_data[f"CL_Transfer_T{t}"] = np.array(trans[t, :])
        
        # Remembering (Pair-Specific)
        if 'remembering' in cl:
            rem = cl['remembering']
            if isinstance(rem, (np.ndarray, jnp.ndarray)) and rem.ndim == 3:
                n_eval, n_train, _ = rem.shape
                # Usually we care about Eval 0 after Train 1 (Forgetting of T0)
                if n_eval > 0 and n_train > 1:
                     vec = rem[0, 1, :] # Eval T0 after Train T1
                     flat_data["CL_Rem_Eval0_Train1"] = np.array(vec)

        # Average Accuracy
        if 'stats' in cl and 'accuracy' in cl['stats']:
            acc = cl['stats']['accuracy']
            if isinstance(acc, (np.ndarray, jnp.ndarray)) and acc.ndim == 2:
                 flat_data['CL_Avg_Accuracy'] = np.mean(acc, axis=1)

    # --- Validation ---
    clean_data = {}
    for key, val in flat_data.items():
        if len(val) < 2: continue
        
        var = np.nanvar(val)
        if np.isnan(var) or var < 1e-12:
            pass
        else:
            clean_data[key] = val

    return clean_data


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

    # --- Sort by Cluster Analysis (Best Practice) ---
    # We use 1 - |r| as distance so that strong positive OR negative correlations 
    # are grouped together.
    try:
        dissimilarity = 1 - np.abs(corr_matrix)
        np.fill_diagonal(dissimilarity, 0) # Ensure 0 distance for self
        
        # Convert to condensed distance matrix for linkage
        # Use 'ward' variance minimization algorithm
        condensed_dist = ssd.squareform(dissimilarity, checks=False)
        linkage_matrix = sch.linkage(condensed_dist, method='ward')
        
        # Get the new order of leaves
        dendro = sch.dendrogram(linkage_matrix, no_plot=True)
        new_order = dendro['leaves']
        
        # Reorder the matrix and the labels
        corr_matrix = corr_matrix[new_order, :][:, new_order]
        metric_names = [metric_names[i] for i in new_order]
        
        print(f"  [x] Reordered metrics via hierarchical clustering")
    except Exception as e:
        print(f"  [!] Clustering failed, keeping original alphabetic order: {e}")

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