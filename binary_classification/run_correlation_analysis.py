import os
import pickle
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from ipdb import set_trace

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

            # Define the keys for Task 0 and Task 1
            key_t0 = 'task_000'
            key_t1 = 'task_001'

            # glue_data structure: keys: ['task_000', 'task_001'] (training) -> keys: ['task_000', 'task_001'] (evaluation)
            if key_t0 not in glue_data or key_t1 not in glue_data:
                 print(f"  [!] GLUE data missing {key_t0} or {key_t1} data.")
            else:
                # Get list of eval tasks and metrics from the first training block
                sample_train_block = glue_data[key_t0]
                eval_tasks = sorted(list(sample_train_block.keys()))
                
                # Grab metrics from the first available eval task
                first_eval_task = eval_tasks[0]
                metric_names = list(sample_train_block[first_eval_task].keys())
                
                for metric in metric_names:
                    results['glue'][metric] = {}
                    
                    for t_eval in eval_tasks:
                        try:
                            # 1. Get value at End of Task 0 (t0)
                            # Access via string key 'task_000'
                            series_t0 = glue_data[key_t0][t_eval][metric]
                            if isinstance(series_t0, list): series_t0 = np.stack(series_t0)
                            val_t0 = series_t0[-1] 
                            
                            # 2. Get value at End of Task 1 (t1)
                            # Access via string key 'task_001'
                            series_t1 = glue_data[key_t1][t_eval][metric]
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
        'dormant':          ('-DormantUnits', True),  #  High dormancy = pathology
        'weight_mag':       ('-WeightMag', True),     # [cite: 332] High mag = saturation/loss of plasticity
        'l2':               ('-L2Norm', True),        # Similar to Weight Mag
        
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
    flat_data = {}
    
    # 1. Plasticity
    if 'plasticity' in computed_data:
        for metric, vector in computed_data['plasticity'].items():
            short_metric = metric.split(" ")[0] 
            flat_data[short_metric] = np.array(vector)

    # 2. GLUE
    if 'glue' in computed_data:
        for metric, eval_dict in computed_data['glue'].items():
            for t_eval, vector in eval_dict.items():
                key = f"Glue_{metric}_EvalT{t_eval}"
                flat_data[key] = np.array(vector)

    # 3. CL
    if 'cl' in computed_data:
        cl = computed_data['cl']
        if 'transfer' in cl:
            trans = cl['transfer']
            if isinstance(trans, (np.ndarray, jnp.ndarray)) and trans.ndim == 2:
                for t in range(trans.shape[0]):
                    flat_data[f"CL_Transfer_T{t}"] = np.array(trans[t, :])
        
        if 'remembering' in cl:
            rem = cl['remembering']
            if isinstance(rem, (np.ndarray, jnp.ndarray)) and rem.ndim == 3:
                n_eval, n_train, _ = rem.shape
                if n_eval > 0 and n_train > 1:
                     vec = rem[0, 1, :] 
                     flat_data["CL_Rem_Eval0_Train1"] = np.array(vec)

        if 'stats' in cl and 'accuracy' in cl['stats']:
            acc = cl['stats']['accuracy']
            if isinstance(acc, (np.ndarray, jnp.ndarray)) and acc.ndim == 2:
                 flat_data['CL_Avg_Accuracy'] = np.mean(acc, axis=1)

    clean_data = {}
    for key, val in flat_data.items():
        if len(val) < 2: continue
        var = np.nanvar(val)
        if not (np.isnan(var) or var < 1e-12):
            clean_data[key] = val

    return clean_data

def correlation_heatmap(correlations, labels, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
    
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if len(labels) < 20:
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = correlations[i, j]
                text_val = "nan" if np.isnan(val) else f"{val:.2f}"
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, text_val, ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Metric Correlation Matrix (Pearson R)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def run_analysis(experiment_path):
    print("--- Generating Correlation Analysis ---")
    
    # 1. Compute Data
    computed_data = compute_metric_differences(experiment_path)
    flat_data = _flatten_metrics(computed_data)
    
    # 2. Filter Valid Data
    valid_data = {}
    for k, v in flat_data.items():
        v = np.array(v)
        # Check NaNs and Variance
        if np.any(np.isnan(v)) and np.sum(np.isnan(v)) > len(v) // 2: continue
        if np.var(v) < 1e-12: continue
        valid_data[k] = v
        
    metric_names = sorted(list(valid_data.keys()))
    n_metrics = len(metric_names)
    
    if n_metrics < 2:
        print("  [!] Not enough valid metrics for correlation analysis.")
        return

    # 3. Setup Directories
    corr_dir = os.path.join(experiment_path, "correlations")
    os.makedirs(corr_dir, exist_ok=True)
    
    # 4. Save Data for Scatter Plotter
    data_path = os.path.join(corr_dir, "correlation_data.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(valid_data, f)
    print(f"  [x] Saved correlation data dict to {data_path}")
    print(f"      Keys available: {metric_names}")

    # 5. Compute Matrix
    print(f"  [i] Computing heatmap for {n_metrics} metrics...")
    corr_matrix = np.zeros((n_metrics, n_metrics))
    
    for i, name_i in enumerate(metric_names):
        for j, name_j in enumerate(metric_names):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue
            
            vec_i, vec_j = valid_data[name_i], valid_data[name_j]
            mask = ~np.isnan(vec_i) & ~np.isnan(vec_j)
            
            if np.sum(mask) < 3:
                corr_matrix[i, j] = 0
            else:
                corr_matrix[i, j], _ = pearsonr(vec_i[mask], vec_j[mask])

    # 6. Hierarchical Clustering for Order
    try:
        dissimilarity = 1 - np.abs(corr_matrix)
        np.fill_diagonal(dissimilarity, 0)
        condensed_dist = ssd.squareform(dissimilarity, checks=False)
        linkage_matrix = sch.linkage(condensed_dist, method='ward')
        dendro = sch.dendrogram(linkage_matrix, no_plot=True)
        new_order = dendro['leaves']
        
        corr_matrix = corr_matrix[new_order, :][:, new_order]
        metric_names = [metric_names[i] for i in new_order]
        print(f"  [x] Reordered metrics via hierarchical clustering")
    except Exception as e:
        print(f"  [!] Clustering failed, keeping original order: {e}")

    # 7. Generate Heatmap
    heatmap_path = os.path.join(corr_dir, "correlation_matrix.png")
    correlation_heatmap(corr_matrix, metric_names, heatmap_path)
    print(f"  [x] Saved Heatmap to {heatmap_path}")



if __name__ == "__main__":
    # EXP_PATH should be updated to your actual local path
    EXP_PATH = "/home/users/MTrappett/manifold/binary_classification/results/mnist/RL/"
    
    if os.path.exists(EXP_PATH):
        run_analysis(EXP_PATH)
    else:
        print(f"Experiment path not found: {EXP_PATH}")