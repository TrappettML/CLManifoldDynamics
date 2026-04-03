import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings
import re

# =====================================================================
# CONFIGURATION
# =====================================================================
EXPERIMENT_PATHS = [
    "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.01_lr2_0.01",
    "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.0001_lr2_0.01",
    "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.01_lr2_0.0001",
    '/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.001_lr2_0.001'
]

# Output settings
OUTPUT_DIR = "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_slow_fast_comparison/"
OUTPUT_FILENAME_ABS = "immediate_accuracy_comparison.png"
OUTPUT_FILENAME_DIFF = "immediate_accuracy_difference.png"
PLOT_TITLE_ABS = "Immediate Task Accuracy Comparison"

# Custom Label Mapping
# Map the experiment folder name to your desired display string.
# If an experiment isn't listed here, it defaults to label[12:]
LABEL_MAPPING = {
    "SL_20_tasks_lr1_0.01_lr2_0.01": "Faster Learner",
    "SL_20_tasks_lr1_0.0001_lr2_0.01": "Slow Feature",
    "SL_20_tasks_lr1_0.01_lr2_0.0001": "Fast Feature", 
    "SL_20_tasks_lr1_0.0001_lr2_0.0001": "Slow Learner"
}
# =====================================================================

def extract_task_num(t_name):
    """Helper to extract numbers for proper chronological sorting."""
    nums = re.findall(r'\d+', t_name)
    return int(nums[0]) if nums else 0

def set_plot_formatting():
    """Applies clean, modern, publication-ready formatting."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.titlepad': 15,
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.frameon': False,      
        'axes.spines.top': False,     
        'axes.spines.right': False,   
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.color': '#e0e0e0',
        'figure.figsize': (12, 6),
        'figure.dpi': 300             
    })

def load_and_extract_data(exp_path):
    """
    Loads global_history.pkl and config.pkl to extract the final test 
    accuracy for each task at the exact moment its own training concluded.
    """
    gh_path = os.path.join(exp_path, 'global_history.pkl')
    cfg_path = os.path.join(exp_path, 'config.pkl')
    
    if not os.path.exists(gh_path):
        print(f"Warning: {gh_path} not found. Skipping {exp_path}.")
        return None, None
        
    with open(gh_path, 'rb') as f:
        gh = pickle.load(f)
        
    if os.path.exists(cfg_path):
        with open(cfg_path, 'rb') as f:
            config = pickle.load(f)
        n_repeats = getattr(config, 'n_repeats', 1)
        epochs_per_task = getattr(config, 'epochs_per_task', 1)
        num_tasks = getattr(config, 'num_tasks', len(gh.get('test_metrics', {})))
    else:
        print(f"Warning: {cfg_path} missing in {exp_path}. Cannot reliably determine task boundaries.")
        return None, None
        
    test_metrics = gh.get('test_metrics', {})
    if not test_metrics:
        print(f"Warning: No test_metrics found in {exp_path}. Skipping.")
        return None, None
        
    extracted_accs = []
    
    for t in range(num_tasks):
        t_name = f"task_{t:03d}"
        if t_name not in test_metrics:
            break 
            
        t_acc = np.array(test_metrics[t_name]['acc'])
        
        if t_acc.ndim == 1 and n_repeats > 1:
            t_acc = t_acc.reshape(-1, n_repeats)
        elif t_acc.ndim == 1:
            t_acc = t_acc.reshape(-1, 1)
            
        start_idx = t * epochs_per_task
        end_idx = (t + 1) * epochs_per_task
        task_t_acc_during_training = t_acc[start_idx:end_idx]
        
        if len(task_t_acc_during_training) == 0:
            break 
            
        last_valid_acc = None
        for i in range(len(task_t_acc_during_training)-1, -1, -1):
            if not np.all(np.isnan(task_t_acc_during_training[i])):
                last_valid_acc = task_t_acc_during_training[i]
                break
                
        if last_valid_acc is not None:
            extracted_accs.append(last_valid_acc)
        else:
            extracted_accs.append(np.full(n_repeats, np.nan))
            
    if not extracted_accs:
        return None, None
        
    return np.array(extracted_accs), num_tasks

def plot_absolute_accuracy(data_dict, colors, save_path, label_mapping):
    fig, ax = plt.subplots()
    max_tasks_plotted = 0

    for label, data in data_dict.items():
        tasks_completed = data.shape[0]
        max_tasks_plotted = max(max_tasks_plotted, tasks_completed)
        x_axis = np.arange(1, tasks_completed + 1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_acc = np.nanmean(data, axis=1)
            std_acc = np.nanstd(data, axis=1)
            valid_counts = np.sum(~np.isnan(data), axis=1)
            valid_counts = np.where(valid_counts == 0, 1, valid_counts) 
            sem_acc = std_acc / np.sqrt(valid_counts)
            
        color = colors[label]
        display_label = label_mapping.get(label, label[12:])
        
        ax.plot(
            x_axis, mean_acc, label=display_label, color=color, 
            linewidth=2.5, markersize=6, markerfacecolor='white',
            markeredgewidth=1.5, zorder=3             
        )
        
        ax.fill_between(
            x_axis, mean_acc - sem_acc, mean_acc + sem_acc,
            color=color, alpha=0.15, linewidth=0, zorder=2
        )

    plt.suptitle(PLOT_TITLE_ABS, fontsize=20, fontweight='bold', y=0.98)
    ax.set_xlabel("Task Number")
    ax.set_ylabel(r"Final T_{i} Test Accuracy")
    if max_tasks_plotted > 0:
        ax.set_xticks(np.arange(1, max_tasks_plotted + 1))
    ax.set_ylim(-0.05, 1.05) 
    
    handles, legend_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), frameon=True, 
                   loc='upper right', bbox_to_anchor=(0.95, 0.88))
        
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Absolute Accuracy plot to: {save_path}")

def plot_difference_accuracy(data_dict, colors, save_path, baseline_algo, label_mapping):
    fig, ax = plt.subplots()
    
    if baseline_algo not in data_dict:
        print(f"Error: Baseline algorithm '{baseline_algo}' not found.")
        plt.close()
        return
        
    compare_algos = [algo for algo in data_dict.keys() if algo != baseline_algo]
    
    if not compare_algos:
        print("Not enough algorithms to plot difference.")
        plt.close()
        return

    base_data = data_dict[baseline_algo]  # Shape: (Tasks, Repeats)
    max_tasks_plotted = 0
    base_display = label_mapping.get(baseline_algo, baseline_algo[12:])

    for algo in compare_algos:
        comp_data = data_dict[algo]
        
        # Align tasks and repeats to safely compute element-wise differences
        min_tasks = min(base_data.shape[0], comp_data.shape[0])
        min_repeats = min(base_data.shape[1], comp_data.shape[1])
        max_tasks_plotted = max(max_tasks_plotted, min_tasks)
        
        b_aligned = base_data[:min_tasks, :min_repeats]
        c_aligned = comp_data[:min_tasks, :min_repeats]
        
        # Calculate difference per repeat
        diff_data = b_aligned - c_aligned
        x_axis = np.arange(1, min_tasks + 1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_diff = np.nanmean(diff_data, axis=1)
            std_diff = np.nanstd(diff_data, axis=1)
            valid_counts = np.sum(~np.isnan(diff_data), axis=1)
            valid_counts = np.where(valid_counts == 0, 1, valid_counts) 
            sem_diff = std_diff / np.sqrt(valid_counts)
            
        color = colors[algo]
        comp_display = label_mapping.get(algo, algo[12:])
        
        # Plot the main difference mean line
        ax.plot(
            x_axis, mean_diff, label=f"{base_display} - {comp_display}", color=color, 
            linewidth=2.5, markersize=6, markerfacecolor='white',
            markeredgewidth=1.5, zorder=3             
        )
        
        # Plot the SEM confidence interval
        ax.fill_between(
            x_axis, mean_diff - sem_diff, mean_diff + sem_diff,
            color=color, alpha=0.15, linewidth=0, zorder=2
        )

    # Add reference line at y=0
    ax.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)

    plt.suptitle(f"Immediate Accuracy Difference\n(Baseline: {base_display})", fontsize=16, fontweight='bold', y=0.98)
    ax.set_xlabel("Task Number")
    ax.set_ylabel(r"$\Delta$ Final T_{i} Test Accuracy")
    
    if max_tasks_plotted > 0:
        ax.set_xticks(np.arange(1, max_tasks_plotted + 1))
    
    # Handle Legends
    handles, legend_labels = ax.get_legend_handles_labels()
        
    by_label = dict(zip(legend_labels, handles))
    if by_label:
        # Place legend exactly (x, y) relative to the figure. 
        fig.legend(by_label.values(), by_label.keys(), frameon=True, 
                   loc='upper right', bbox_to_anchor=(0.95, 0.88))
        
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Difference Accuracy plot to: {save_path}")

def main():
    set_plot_formatting()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    labels = [os.path.basename(os.path.normpath(p)) for p in EXPERIMENT_PATHS]
    paired_experiments = list(zip(labels, EXPERIMENT_PATHS))
    sorted_experiments = sorted(paired_experiments, key=lambda x: x[0])
    
    algorithms = sorted(list(labels))
    cmap = plt.get_cmap('tab10')
    colors = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}

    # Collect data for all experiments first
    data_dict = {}
    for label, path in sorted_experiments:
        print(f"Extracting data for: {label}")
        data, expected_tasks = load_and_extract_data(path)
        if data is not None:
            data_dict[label] = data

    if not data_dict:
        print("No valid data found in any provided paths. Exiting.")
        return

    # Generate the plots
    abs_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_ABS)
    diff_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_DIFF)

    # Set the explicit baseline here
    baseline_name = "SL_20_tasks_lr1_0.0001_lr2_0.01"

    plot_absolute_accuracy(data_dict, colors, abs_path, LABEL_MAPPING)
    plot_difference_accuracy(data_dict, colors, diff_path, baseline_algo=baseline_name, label_mapping=LABEL_MAPPING)


if __name__ == "__main__":
    main()