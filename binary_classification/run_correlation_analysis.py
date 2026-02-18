import os
import pickle
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from ipdb import set_trace

# Configuration: Map raw keys to (CleanName, Flip_Bool)
# Flip_Bool = True means "Decrease is Good" (Positive output = metric went down)
PLASTICITY_CONFIG = {
    # Lower is Better (Pathologies)
    'dormant':          ('-DormantUnits', True),  
    'weight_mag':       ('-WeightMag', True),     
    'l2':               ('-L2Norm', True),        
    
    # Higher is Better (Health Signals)
    'active':           ('ActiveUnits', False),  
    'stable_rank':      ('StableRank', False),   
    'effective_rank':   ('EffectiveRank', False),
    'gradient_norm':    ('GradNorm', False),     
    'feature_norm':     ('FeatureNorm', False),  
    'variance':         ('FeatureVar', False),   
    'entropy':          ('Entropy', False),      
    'weight_diff':      ('WeightDiff', False),   
}

def calc_rel_diff(v0, v1, flip_sign=False):
    denom = v1 + v0
    denom = np.where(denom == 0, 1e-9, denom)
    
    if flip_sign:
        num = v0 - v1  
    else:
        num = v1 - v0
        
    return num / denom

def compute_metric_differences(experiment_path):
    print(f"--- Processing Metrics for: {experiment_path} ---")

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

    if os.path.exists(cl_path):
        try:
            with open(cl_path, 'rb') as f:
                cl_data = pickle.load(f)
            results['cl'] = cl_data
            print(f"  [x] Loaded CL Metrics from {os.path.basename(cl_path)}")
        except Exception as e:
            print(f"  [!] Error loading CL metrics: {e}")

    if os.path.exists(glue_path):
        try:
            with open(glue_path, 'rb') as f:
                glue_data = pickle.load(f)

            key_t0 = 'task_000'
            key_t1 = 'task_001'

            if key_t0 not in glue_data or key_t1 not in glue_data:
                 print(f"  [!] GLUE data missing {key_t0} or {key_t1} data.")
            else:
                sample_train_block = glue_data[key_t0]
                eval_tasks = sorted(list(sample_train_block.keys()))
                first_eval_task = eval_tasks[0]
                metric_names = list(sample_train_block[first_eval_task].keys())
                
                for metric in metric_names:
                    results['glue'][metric] = {}
                    for t_eval in eval_tasks:
                        try:
                            series_t0 = glue_data[key_t0][t_eval][metric]
                            if isinstance(series_t0, list): series_t0 = np.stack(series_t0)
                            val_t0 = series_t0[-1] 
                            
                            series_t1 = glue_data[key_t1][t_eval][metric]
                            if isinstance(series_t1, list): series_t1 = np.stack(series_t1)
                            val_t1 = series_t1[-1] 
                            
                            rel_diff = calc_rel_diff(val_t0, val_t1, flip_sign=False)
                            results['glue'][metric][t_eval] = rel_diff
                            
                        except (KeyError, IndexError):
                            pass
                
                print(f"  [x] Processed GLUE relative differences")
            
        except Exception as e:
            print(f"  [!] Error processing GLUE metrics: {e}")
    else:
        print(f"  [ ] GLUE metrics file not found: {glue_path}")

    if os.path.exists(plast_path):
        try:
            with open(plast_path, 'rb') as f:
                plast_data = pickle.load(f)
            
            history = plast_data.get('history', {})
            samples_per_task = config.epochs_per_task // config.log_frequency
            
            idx_end_t0 = samples_per_task - 1         
            idx_end_t1 = (2 * samples_per_task) - 1   
            
            for metric_raw, data in history.items():
                if isinstance(data, list):
                    data = np.stack(data)
                
                clean_name = None
                should_flip = False
                metric_lower = metric_raw.lower()
                for key_sub, (name, flip) in PLASTICITY_CONFIG.items():
                    if key_sub in metric_lower:
                        clean_name = name
                        should_flip = flip
                        break
                
                if clean_name is None:
                    clean_name = metric_raw.replace(" ", "")
                
                res_key = f"{clean_name}_t1t0"

                if data.shape[0] > idx_end_t1:
                    val_t0 = data[idx_end_t0]   
                    val_t1 = data[idx_end_t1]   
                    rel_delta = calc_rel_diff(val_t0, val_t1, flip_sign=should_flip)
                    results['plasticity'][res_key] = rel_delta
                else:
                    print(f"  [!] Not enough steps in plasticity {metric_raw}")

            print(f"  [x] Processed Plasticity relative differences")
            
        except Exception as e:
            print(f"  [!] Error processing Plasticity metrics: {e}")

    return results

def compute_mtl_differences(experiment_path):
    print(f"\n--- Processing MTL Metrics for: {experiment_path} ---")
    mtl_dir = os.path.join(experiment_path, "multitask")
    glue_path = os.path.join(mtl_dir, "glue_metrics.pkl")
    plast_path = os.path.join(mtl_dir, "plasticity_metrics.pkl")
    cl_path = os.path.join(mtl_dir, "metrics.pkl")
    
    flat_data = {}
    
    # 1. MTL Plasticity
    if os.path.exists(plast_path):
        with open(plast_path, 'rb') as f:
            plast_data = pickle.load(f)
        for metric_raw, data in plast_data.items():
            data = np.stack(data)
            clean_name = None
            should_flip = False
            for key_sub, (name, flip) in PLASTICITY_CONFIG.items():
                if key_sub in metric_raw.lower():
                    clean_name = name
                    should_flip = flip
                    break
            
            if clean_name is None:
                clean_name = metric_raw.replace(" ", "")

            if data.shape[0] > 1:
                val_init = data[0]
                val_final = data[-1]
                rel_delta = calc_rel_diff(val_init, val_final, flip_sign=should_flip)
                flat_data[f"MTL_Plast_{clean_name}"] = rel_delta
        print("  [x] Processed MTL Plasticity relative differences")

    # 2. MTL GLUE
    if os.path.exists(glue_path):
        with open(glue_path, 'rb') as f:
            glue_data = pickle.load(f)
        for task, metrics in glue_data.items():
            for metric, data in metrics.items():
                if len(data) > 1:
                    val_init = data[0]
                    val_final = data[-1]
                    rel_diff = calc_rel_diff(val_init, val_final, flip_sign=False)
                    flat_data[f"MTL_Glue_{metric}_{task}"] = rel_diff
        print("  [x] Processed MTL GLUE relative differences")

    # 3. MTL CL (Accuracy final vs init)
    if os.path.exists(cl_path):
        with open(cl_path, 'rb') as f:
            cl_data = pickle.load(f)
        for task, metrics in cl_data['test_metrics'].items():
            if 'acc' in metrics and len(metrics['acc']) > 1:
                val_init = np.array(metrics['acc'][0])
                val_final = np.array(metrics['acc'][-1])
                rel_diff = calc_rel_diff(val_init, val_final, flip_sign=False)
                flat_data[f"MTL_Acc_{task}"] = rel_diff
        print("  [x] Processed MTL Accuracy relative differences")

    return flat_data

def _flatten_metrics(computed_data):
    flat_data = {}
    
    if 'plasticity' in computed_data:
        for metric, vector in computed_data['plasticity'].items():
            short_metric = metric.split(" ")[0] 
            flat_data[short_metric] = np.array(vector)

    if 'glue' in computed_data:
        for metric, eval_dict in computed_data['glue'].items():
            for t_eval, vector in eval_dict.items():
                key = f"Glue_{metric}_EvalT{t_eval}"
                flat_data[key] = np.array(vector)

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

def generate_correlation_artifacts(flat_data, corr_dir, prefix=""):
    valid_data = {}
    for k, v in flat_data.items():
        v = np.array(v)
        if np.any(np.isnan(v)) and np.sum(np.isnan(v)) > len(v) // 2: continue
        if np.var(v) < 1e-12: continue
        valid_data[k] = v
        
    metric_names = sorted(list(valid_data.keys()))
    n_metrics = len(metric_names)
    
    if n_metrics < 2:
        print(f"  [!] Not enough valid metrics for {prefix}correlation analysis.")
        return

    data_filename = f"{prefix}correlation_data.pkl"
    data_path = os.path.join(corr_dir, data_filename)
    with open(data_path, 'wb') as f:
        pickle.dump(valid_data, f)
    print(f"  [x] Saved correlation data dict to {data_path}")
    print(f"      Keys available: {metric_names}")

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

    heatmap_filename = f"{prefix}correlation_matrix.png"
    heatmap_path = os.path.join(corr_dir, heatmap_filename)
    correlation_heatmap(corr_matrix, metric_names, heatmap_path)
    print(f"  [x] Saved Heatmap to {heatmap_path}")

def run_analysis(experiment_path):
    print("--- Generating Correlation Analysis ---")
    corr_dir = os.path.join(experiment_path, "correlations")
    os.makedirs(corr_dir, exist_ok=True)
    
    # 1. Continual Learning Analysis
    computed_data = compute_metric_differences(experiment_path)
    flat_data = _flatten_metrics(computed_data)
    generate_correlation_artifacts(flat_data, corr_dir, prefix="")

    # 2. Multi-Task Analysis
    mtl_flat_data = compute_mtl_differences(experiment_path)
    if mtl_flat_data:
        generate_correlation_artifacts(mtl_flat_data, corr_dir, prefix="mtl_")


if __name__ == "__main__":
    EXP_PATH = "/home/users/MTrappett/manifold/binary_classification/results/mnist/RL/"
    
    if os.path.exists(EXP_PATH):
        run_analysis(EXP_PATH)
    else:
        print(f"Experiment path not found: {EXP_PATH}")