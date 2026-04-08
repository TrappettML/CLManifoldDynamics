import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import Normalize
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  
from ipdb import set_trace
import re


def get_config_val(config, key, default=None):
    """Safely fetch a value whether config is a dict or an object."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

# =====================================================================
# CONFIGURATION
# =====================================================================
BASE_DIR = "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/"
OUTPUT_DIR = os.path.join(BASE_DIR, "SL_grid_search_plots")

X_PARAM = 'num_epochs' 

# =====================================================================
# FORMATTING
# =====================================================================
def set_plot_formatting():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

# =====================================================================
# DATA EXTRACTION
# =====================================================================
def extract_immediate_acc(global_history, config):
    test_metrics = global_history.get('test_metrics', {})
    n_repeats = get_config_val(config, 'n_repeats', 1)
    epochs_per_task = get_config_val(config, 'epochs_per_task', 1)
    num_tasks = get_config_val(config, 'num_tasks', 20)
    
    extracted_accs = [] 

    for t in range(num_tasks):
        last_task_str = f"task_{num_tasks-1:03d}"
        if last_task_str not in test_metrics:
            return np.full(n_repeats, np.nan) 

        t_name = f"task_{t:03d}"
        if t_name not in test_metrics:
            continue
            
        t_acc = np.array(test_metrics[t_name]['acc'])
        if t_acc.ndim == 1:
            t_acc = t_acc.reshape(-1, n_repeats)
            
        start_idx = t * epochs_per_task
        end_idx = (t + 1) * epochs_per_task
        task_t_acc_during_training = t_acc[start_idx:end_idx]
        
        last_valid_acc = np.full(n_repeats, np.nan)
        if len(task_t_acc_during_training) > 0:
            valid_mask = ~np.isnan(task_t_acc_during_training)
            has_valid = valid_mask.any(axis=1)
            if has_valid.any():
                last_valid_idx = np.where(has_valid)[0][-1]
                last_valid_acc = task_t_acc_during_training[last_valid_idx]
                
        extracted_accs.append(last_valid_acc)
        
    if not extracted_accs:
        return np.full(n_repeats, np.nan)
        
    acc_matrix = np.array(extracted_accs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        repeat_means = np.nanmean(acc_matrix, axis=0) 
        
    return repeat_means

def extract_cl_metric(cl_metrics, metric_name):
    if metric_name not in cl_metrics:
        return None
        
    data = cl_metrics[metric_name]
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if data.ndim == 3:
            data = np.nanmean(data, axis=1)
        repeat_means = np.nanmean(data, axis=0)
        
    return repeat_means


def extract_final_acc(global_history, config):
    test_metrics = global_history.get('test_metrics', {})
    n_repeats = get_config_val(config, 'n_repeats', 1)
    num_tasks = get_config_val(config, 'num_tasks', 20)
    
    extracted_accs = [] 

    for t in range(num_tasks):
        t_name = f"task_{t:03d}"
        if t_name not in test_metrics:
            continue
            
        t_acc = np.array(test_metrics[t_name]['acc'])
        if t_acc.ndim == 1:
            t_acc = t_acc.reshape(-1, n_repeats)
            
        last_valid_acc = np.full(n_repeats, np.nan)
        if len(t_acc) > 0:
            valid_mask = ~np.isnan(t_acc)
            has_valid = valid_mask.any(axis=1)
            if has_valid.any():
                last_valid_idx = np.where(has_valid)[0][-1]
                last_valid_acc = t_acc[last_valid_idx]
                
        extracted_accs.append(last_valid_acc)
        
    if not extracted_accs:
        return np.full(n_repeats, np.nan)
        
    acc_matrix = np.array(extracted_accs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        repeat_means = np.nanmean(acc_matrix, axis=0) 
        
    return repeat_means


def process_single_experiment(exp_path, item_name):
    config_path = os.path.join(exp_path, 'config.pkl')
    gh_path = os.path.join(exp_path, 'global_history.pkl')
    cl_path = os.path.join(exp_path, 'cl_metrics.pkl')
    
    if not (os.path.exists(config_path) and os.path.exists(gh_path)):
        return None
        
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
        
    with open(gh_path, 'rb') as f:
        global_history = pickle.load(f)
        
    cl_metrics = {}
    if os.path.exists(cl_path):
        with open(cl_path, 'rb') as f:
            cl_metrics = pickle.load(f)
            
    lr1 = get_config_val(config, 'lr1')
    lr2 = get_config_val(config, 'lr2')
    x_val = get_config_val(config, X_PARAM)
    
    if lr1 is None or lr2 is None:
        lr1_match = re.search(r'lr1_([0-9\.eE\-]+)', item_name)
        lr2_match = re.search(r'lr2_([0-9\.eE\-]+)', item_name)
        if lr1_match: 
            lr1 = float(lr1_match.group(1))
        if lr2_match: 
            lr2 = float(lr2_match.group(1))

    if x_val is None and X_PARAM == 'num_epochs':
        if 'epochs_' in item_name:
            try:
                x_val = int(item_name.split('epochs_')[1].split('_')[0])
            except:
                x_val = 1000
        else:
            x_val = 1000
            
    if lr1 is None or lr2 is None or x_val is None:
        return None
        
    config_key = (lr1, lr2)
    
    imm_acc_reps = extract_immediate_acc(global_history, config)
    transfer_reps = extract_cl_metric(cl_metrics, 'transfer') if cl_metrics else np.full(1, np.nan)
    final_acc_reps = extract_final_acc(global_history, config)
    rem_reps = extract_cl_metric(cl_metrics, 'remembering') if cl_metrics else np.full(1, np.nan)
    
    return {
        'x_val': x_val,
        'config_key': config_key,
        'data': {
            'imm_acc': imm_acc_reps,
            'transfer': transfer_reps,
            'final_acc': final_acc_reps,
            'remembering': rem_reps
        }
    }

def load_grid_search_data_parallel(base_dir):
    data_dict = {}
    tasks = []
    
    for item in os.listdir(base_dir):
        if not item.startswith("SL_20_tasks"):
            continue
        exp_path = os.path.join(base_dir, item)
        if os.path.isdir(exp_path):
            tasks.append((exp_path, item))

    if not tasks:
        return data_dict

    with ProcessPoolExecutor(max_workers=90) as executor:
        futures = {executor.submit(process_single_experiment, path, name): path for path, name in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing Experiments", unit="exp"):
            result = future.result()
            if result is not None:
                x_val = result['x_val']
                config_key = result['config_key']
                
                if x_val not in data_dict:
                    data_dict[x_val] = {}
                data_dict[x_val][config_key] = result['data']
                
    return data_dict

# =====================================================================
# PLOTTING FUNCTIONS
# =====================================================================
def get_mean_sem(reps_array):
    if reps_array is None or np.all(np.isnan(reps_array)):
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(reps_array)
        valid_count = np.sum(~np.isnan(reps_array))
        sem = np.nanstd(reps_array) / np.sqrt(max(1, valid_count))
    return mean, sem

def plot_grouped_boxplots(data_dict, metric_key, output_path, title, target_configs=None):
    """
    Creates a grouped box plot. For each `x_val` (e.g. num_epochs), it plots boxes 
    for the top configurations. If `target_configs` is None, it dynamically finds 
    the top 4 configurations for each specific epoch based on the provided metric.
    """
    if not data_dict:
        return
        
    x_vals = sorted(list(data_dict.keys()))
    
    num_groups = len(x_vals)
    configs_per_group = 4 if target_configs is None else len(target_configs)
    
    # Calculate a responsive figure width based on the number of groups/boxes
    fig_width = max(12, num_groups * configs_per_group * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    
    box_width = 0.6
    group_spacing = 2.0
    
    all_positions = []
    all_data = []
    all_labels = []
    
    x_ticks_positions = []
    x_ticks_labels = []
    
    current_x = 0
    cmap = plt.get_cmap('tab10')
    colors_for_boxes = []
    
    for x in x_vals:
        if target_configs is None:
            # Dynamically sort configs at this specific epoch
            configs = data_dict[x]
            ranked = sorted(
                configs.keys(), 
                key=lambda k: get_mean_sem(configs[k][metric_key])[0], 
                reverse=True
            )
            configs_to_plot = ranked[:4]
        else:
            configs_to_plot = [cfg for cfg in target_configs if cfg in data_dict[x]]
            
        if not configs_to_plot:
            continue
            
        group_positions = []
        for i, cfg in enumerate(configs_to_plot):
            pos = current_x + i * box_width
            group_positions.append(pos)
            all_positions.append(pos)
            
            reps = data_dict[x][cfg][metric_key]
            all_data.append(reps)
            
            # Formulate the angled label for the individual bar
            all_labels.append(f"lr1={cfg[0]}\nlr2={cfg[1]}")
            colors_for_boxes.append(cmap(i % 10))
            
        x_ticks_positions.append(np.mean(group_positions))
        x_ticks_labels.append(f"{X_PARAM.replace('_', ' ').title()} = {x}")
        
        current_x += (len(configs_to_plot) * box_width) + group_spacing

    if not all_positions:
        plt.close()
        return

    # Filter out NaNs for matplotlib's boxplot to avoid errors while maintaining alignment
    valid_data = []
    valid_positions = []
    valid_colors = []
    for d, p, c in zip(all_data, all_positions, colors_for_boxes):
        if d is not None:
            d_clean = d[~np.isnan(d)]
            if len(d_clean) > 0:
                valid_data.append(d_clean)
                valid_positions.append(p)
                valid_colors.append(c)

    if valid_data:
        bp = ax.boxplot(valid_data, positions=valid_positions, widths=box_width*0.8, patch_artist=True)
        for patch, color in zip(bp['boxes'], valid_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
    # Place angled configuration labels under each bar
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
    
    # Place main epoch axis labels grouped beneath the angled configuration labels
    for pos, label in zip(x_ticks_positions, x_ticks_labels):
        ax.annotate(label, xy=(pos, 0), xytext=(0, -75),
                    xycoords=('data', 'axes fraction'), textcoords='offset points',
                    ha='center', va='top', fontweight='bold', fontsize=12, annotation_clip=False)

    plt.subplots_adjust(bottom=0.3)
    ax.set_ylabel(title)
    
    prefix = "Top 4 Configs per Epoch" if target_configs is None else "Selected Configurations"
    ax.set_title(f"{prefix}: {title}", pad=15)
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(axis='x', visible=False) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved grouped box plot to {output_path}")

def plot_heatmaps(data_dict, metric_key, output_path, title):
    if not data_dict:
        return
        
    x_vals = sorted(list(data_dict.keys()))
    
    all_lr1, all_lr2 = set(), set()
    for x in x_vals:
        for (lr1, lr2) in data_dict[x].keys():
            all_lr1.add(lr1)
            all_lr2.add(lr2)
            
    all_lr1, all_lr2 = sorted(list(all_lr1)), sorted(list(all_lr2))
    
    if not all_lr1 or not all_lr2:
        return
        
    rows = len(x_vals)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 4 * rows), squeeze=False)
    
    global_min, global_max = np.inf, -np.inf
    for x in x_vals:
        for cfg in data_dict[x].keys():
            m, _ = get_mean_sem(data_dict[x][cfg][metric_key])
            if not np.isnan(m):
                global_min = min(global_min, m)
                global_max = max(global_max, m)
                
    if np.isinf(global_min):
        global_min, global_max = 0, 1
        
    norm = Normalize(vmin=global_min, vmax=global_max)
    
    for r, x in enumerate(x_vals):
        ax = axes[r, 0]
        grid = np.full((len(all_lr2), len(all_lr1)), np.nan)
        
        for i, lr2 in enumerate(all_lr2):
            for j, lr1 in enumerate(all_lr1):
                if (lr1, lr2) in data_dict[x]:
                    m, _ = get_mean_sem(data_dict[x][(lr1, lr2)][metric_key])
                    grid[i, j] = m
                    
        im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis', norm=norm)
        
        ax.set_title(f"{X_PARAM} = {x}", fontweight='bold')
        ax.set_xticks(np.arange(len(all_lr1)))
        ax.set_yticks(np.arange(len(all_lr2)))
        ax.set_xticklabels(all_lr1, rotation=45)
        ax.set_yticklabels(all_lr2)
        ax.set_xlabel("LR 1")
        ax.set_ylabel("LR 2")
        
        if len(all_lr1) <= 10 and len(all_lr2) <= 10:
            for i in range(len(all_lr2)):
                for j in range(len(all_lr1)):
                    val = grid[i, j]
                    if not np.isnan(val):
                        text_color = "black" if norm(val) > 0.5 else "white"
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=8)

    fig.suptitle(f"{title} Heatmaps Over {X_PARAM}", fontsize=16, fontweight='bold', y=0.98)
    fig.colorbar(im, ax=axes.ravel().tolist(), label=title)
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved heatmap to {output_path}")

# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_plot_formatting()
    
    print("Gathering experiment directories from:", BASE_DIR)
    data_dict = load_grid_search_data_parallel(BASE_DIR)
    
    if not data_dict:
        print("No valid data matching the criteria was found. Exiting.")
        return
        
    print(f"\nData loaded. Processing plots...")
    
    # 1. Dynamic Box Plots (Finds Top 4 per Epoch)
    plot_grouped_boxplots(data_dict, 'imm_acc', os.path.join(OUTPUT_DIR, "boxplot_immediate_accuracy.png"), "Immediate Accuracy")
    plot_grouped_boxplots(data_dict, 'transfer', os.path.join(OUTPUT_DIR, "boxplot_transfer.png"), "Transfer")
    plot_grouped_boxplots(data_dict, 'remembering', os.path.join(OUTPUT_DIR, "boxplot_remembering.png"), "Remembering")
    plot_grouped_boxplots(data_dict, 'final_acc', os.path.join(OUTPUT_DIR, "boxplot_final_accuracy.png"), "Final Accuracy")
    
    # 2. Heatmaps
    plot_heatmaps(data_dict, 'imm_acc', os.path.join(OUTPUT_DIR, "heatmap_immediate_accuracy.png"), "Immediate Accuracy")
    plot_heatmaps(data_dict, 'transfer', os.path.join(OUTPUT_DIR, "heatmap_transfer.png"), "Transfer")
    plot_heatmaps(data_dict, 'remembering', os.path.join(OUTPUT_DIR, "heatmap_remembering.png"), "Remembering")
    plot_heatmaps(data_dict, 'final_acc', os.path.join(OUTPUT_DIR, "heatmap_final_accuracy.png"), "Final Accuracy")

    # 3. Hardcoded Box Plots
    hardcoded_configs = [(0.0001, 0.01), (0.001, 0.001), (0.01, 0.0001), (0.01, 0.01)]
    
    plot_grouped_boxplots(data_dict, 'imm_acc', os.path.join(OUTPUT_DIR, "boxplot_hardcoded_imm_acc.png"), "Immediate Accuracy", hardcoded_configs)
    plot_grouped_boxplots(data_dict, 'final_acc', os.path.join(OUTPUT_DIR, "boxplot_hardcoded_final_acc.png"), "Final Accuracy", hardcoded_configs)
    plot_grouped_boxplots(data_dict, 'transfer', os.path.join(OUTPUT_DIR, "boxplot_hardcoded_transfer.png"), "Transfer", hardcoded_configs)
    plot_grouped_boxplots(data_dict, 'remembering', os.path.join(OUTPUT_DIR, "boxplot_hardcoded_remembering.png"), "Remembering", hardcoded_configs)

    print("\nComplete! All grid search plots have been generated.")

if __name__ == "__main__":
    main()