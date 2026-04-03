import os
import re
import pickle
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace

# --- Formatting Best Practices ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

# Define the explicit paths you want to compare
EXPERIMENT_PATHS = [
    "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.01_lr2_0.01",
    "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.0001_lr2_0.01",
    "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.01_lr2_0.0001",
    "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.001_lr2_0.001"
]

OUTPUT_PATH = "/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_slow_fast_comparison/"

def extract_task_num(t_name):
    """Helper to extract numbers for proper chronological sorting."""
    nums = re.findall(r'\d+', t_name)
    return int(nums[0]) if nums else 0

def load_algorithm_data(experiment_paths):
    """
    Iterates through the provided list of experiment paths. 
    Strictly loads performance time-series data from global_history.pkl, 
    manifold/plasticity data from all_metrics.pkl, and cl metrics from cl_metrics.pkl.
    """
    data_by_algo = {}
    config = None
    
    for algo_dir in experiment_paths:
        if not os.path.isdir(algo_dir):
            print(f"Warning: Directory not found, skipping: {algo_dir}")
            continue
            
        # Use the folder name as the algorithm label for plots
        entry = os.path.basename(os.path.normpath(algo_dir))
            
        config_file = os.path.join(algo_dir, "config.pkl")
        gh_file = os.path.join(algo_dir, "global_history.pkl")
        glue_metrics_file = os.path.join(algo_dir, "all_metrics.pkl")
        plastic_metric_file = os.path.join(algo_dir, "plastic_analysis_imagenet_28_gray.pkl")
        cl_metrics_file = os.path.join(algo_dir, "cl_metrics.pkl")
        
        algo_data = {'performance': {}, 'plasticity': {}, 'glue': {}, 'cl_metrics': {}}
        loaded_any = False

        # Grab experiment configurations first so we have n_repeats for reshaping
        if config is None and os.path.exists(config_file):
            with open(config_file, 'rb') as f:
                config = pickle.load(f)

        # 1. Load Performance Data strictly from global_history.pkl
        if os.path.exists(gh_file):
            with open(gh_file, 'rb') as f:
                algo_data['performance'] = pickle.load(f)
            loaded_any = True
            
        # 2. Load GLUE data strictly 
        if os.path.exists(glue_metrics_file):
            with open(glue_metrics_file, 'rb') as f:
                master_data = pickle.load(f)
                algo_data['glue'] = master_data.get('glue', {})
            loaded_any = True

        # 3. Load Plasticity data strictly 
        if os.path.exists(plastic_metric_file):
            with open(plastic_metric_file, 'rb') as f:
                master_data = pickle.load(f)
                algo_data['plasticity'] = master_data.get('history', {})
            loaded_any = True

        # 4. Load CL metrics strictly
        if os.path.exists(cl_metrics_file):
            with open(cl_metrics_file, 'rb') as f:
                algo_data['cl_metrics'] = pickle.load(f)
            loaded_any = True
            

        if loaded_any:
            data_by_algo[entry] = algo_data
            print(f"Loaded metric data for algorithm: {entry}")
        else:
            print(f"Skipping {entry}: No metric files found. (Looked for global_history.pkl, all_metrics.pkl, and cl_metrics.pkl)")
            
    return data_by_algo, config

def extract_performance_data(data_by_algo, config):
    """
    Extracts task-specific Accuracy and Loss for performance comparison plots.
    Reshapes the traces using config.n_repeats exactly like single_run.py.
    """
    acc_extracted = {}
    loss_extracted = {}

    repeats = config.n_repeats
    
    for algo, data in data_by_algo.items():
        perf_data = data.get('performance', {})
        if not perf_data or 'test_metrics' not in perf_data:
            continue
        
        acc_extracted[algo] = {}
        loss_extracted[algo] = {}
        
        tasks = sorted(list(perf_data['test_metrics'].keys()))
        for t in tasks:
            acc = np.array(perf_data['test_metrics'][t]['acc'])
            loss = np.array(perf_data['test_metrics'][t]['loss'])
            
            # Reshape flattened arrays using n_repeats just like single_run.py
            if acc.ndim == 1: 
                acc = acc.reshape(-1, repeats)
            if loss.ndim == 1: 
                loss = loss.reshape(-1, repeats)
            
            # Clean task name for the subplot titles (e.g., 'task_000' -> 'Task 0')
            clean_name = t.replace('_', ' ').replace('00', '').title()
            
            acc_extracted[algo][clean_name] = acc
            loss_extracted[algo][clean_name] = loss
            
    return acc_extracted, loss_extracted


def extract_plasticity_data(data_by_algo):
    """Directly extracts the tracked network plasticity traces."""
    plast_extracted = {}
    for algo, data in data_by_algo.items():
        p_data = data.get('plasticity', {})
        if p_data:
            plast_extracted[algo] = {k: np.array(v) for k, v in p_data.items()}
    return plast_extracted

def extract_glue_data(data_by_algo):
    """
    Extracts GLUE metrics structurally by Metric and Evaluated Task.
    Stitches the timeline chronologically across all Training phases.
    """
    glue_extracted = {}
    for algo, data in data_by_algo.items():
        g_data = data.get('glue', {})
        if not g_data: continue
        
        algo_metrics = {}
        train_tasks = sorted(list(g_data.keys()))
        
        # Discover all unique metric names and evaluated tasks
        metrics = set()
        eval_tasks = set()
        for t_train in train_tasks:
            for t_eval in g_data[t_train].keys():
                eval_tasks.add(t_eval)
                metrics.update(g_data[t_train][t_eval].keys())
        
        metrics = sorted(list(metrics))
        eval_tasks = sorted(list(eval_tasks))
        
        # Aggregate the timeline per metric and per eval task
        for metric in metrics:
            algo_metrics[metric] = {}
            for t_eval in eval_tasks:
                stitched_raw = []
                for t_train in train_tasks:
                    # Find expected shape L for this train task (handle missing eval tasks gracefully)
                    expected_shape = None
                    for temp_eval in g_data[t_train]:
                        if metric in g_data[t_train][temp_eval]:
                            expected_shape = np.array(g_data[t_train][temp_eval][metric]).shape
                            break
                            
                    if expected_shape is None:
                        continue # No data at all for this metric in this train block
                        
                    if t_eval in g_data[t_train] and metric in g_data[t_train][t_eval]:
                        stitched_raw.append(np.array(g_data[t_train][t_eval][metric]))
                    else:
                        # Pad with NaNs if this specific eval task was skipped in this block
                        stitched_raw.append(np.full(expected_shape, np.nan))
                        
                if stitched_raw:
                    # Concatenate the sequential training blocks into one global timeline
                    algo_metrics[metric][t_eval] = np.concatenate(stitched_raw, axis=0) 
                    
        if algo_metrics:
            glue_extracted[algo] = algo_metrics
            
    return glue_extracted

def extract_cl_metrics(data_by_algo):
    """
    Extracts raw Continual Learning arrays (e.g., 'transfer', 'remembering')
    and explicitly ignores the pre-calculated 'stats' dictionary.
    """
    cl_extracted = {}
    for algo, data in data_by_algo.items():
        c_data = data.get('cl_metrics', {})
        if not c_data:
            continue
            
        # Filter out 'stats' and ensure we only keep numpy arrays
        algo_metrics = {k: v for k, v in c_data.items() if k != 'stats' and isinstance(v, np.ndarray)}
        
        if algo_metrics:
            cl_extracted[algo] = algo_metrics
            
    return cl_extracted


def plot_performance_grid(acc_data, loss_data, save_path, config):
    """
    Plots Accuracy and Loss in the same figure.
    Columns = Metrics (Col 0: Accuracy, Col 1: Loss)
    Rows = Tasks (Sorted chronologically)
    """
    if not acc_data and not loss_data:
        print("No performance data available. Skipping plot.")
        return

    # Discover unique evaluated tasks across all algorithms
    tasks = set()
    algorithms = sorted(list(acc_data.keys()))
    for algo in algorithms:
        tasks.update(acc_data[algo].keys())
    
    # Sort tasks numerically instead of alphabetically
    tasks = sorted(list(tasks), key=extract_task_num)
    
    rows = len(tasks)
    cols = 2 # Accuracy and Loss
    
    if rows == 0:
        return
        
    fig_width = max(12, 6 * cols)
    fig_height = max(4, 3 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    
    cmap = plt.get_cmap('tab10')
    colors = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    
    metrics_data = [
        ('Accuracy', acc_data),
        ('Loss', loss_data)
    ]
    
    for r, task in enumerate(tasks):
        for c, (metric_name, data_dict) in enumerate(metrics_data):
            ax = axes[r, c]
            max_x = 0
            has_data = False
            
            for algo in algorithms:
                if algo not in data_dict or task not in data_dict[algo]:
                    continue
                    
                data = data_dict[algo][task] # (Steps, Repeats)
                if data.size == 0: continue
                    
                has_data = True
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                        
                    mean = np.nanmean(data, axis=1)
                    std = np.nanstd(data, axis=1)
                    
                steps = len(mean)
                total_epochs = config.epochs_per_task * config.num_tasks
                epochs = np.linspace(total_epochs / steps, total_epochs, steps)
                
                mask = ~np.isnan(mean)
                if mask.any():
                    ax.plot(epochs[mask], mean[mask], label=algo, color=colors[algo])
                    ax.fill_between(
                        epochs[mask], 
                        (mean - std)[mask], 
                        (mean + std)[mask], 
                        color=colors[algo], alpha=0.15, linewidth=0
                    )
                    max_x = max(max_x, epochs[mask][-1] if len(epochs[mask]) > 0 else 0)
                    
            if has_data:
                # Setup Grid Titles and Labels
                if r == 0: # Top row gets Col Titles
                    ax.set_title(f"{metric_name}", fontweight='bold', pad=10)
                if c == 0: # Left Col gets Row Titles
                    ax.set_ylabel(f"{task}", fontweight='bold')
                if r == rows - 1: # Bottom Row gets X Labels
                    ax.set_xlabel("Epochs")
                    
                ax.set_xlim(0, max_x)
                
                # Draw task boundaries clearly
                for t in range(1, config.num_tasks):
                    boundary = t * config.epochs_per_task
                    if boundary < max_x:
                        ax.axvline(x=boundary, color='#333333', linestyle=':', alpha=0.6, linewidth=1.5)
                        if r == 0 and c == 0: # Label boundaries on first subplot only
                            ax.text(boundary + (max_x * 0.01), ax.get_ylim()[0], f"Task {t+1}", 
                                    rotation=90, verticalalignment='bottom', fontsize=10, color='#555555')
                                    
    # Collect unique legend handles and labels
    handles, labels = [], []
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        
    by_label = dict(zip(labels, handles))

    # Add the figure title
    plt.suptitle("Performance Timeline Over Epochs", fontsize=20, fontweight='bold')

    # Place the legend at the top, just below the title
    if by_label:
        fig.legend(
            by_label.values(), by_label.keys(), 
            loc='upper center', bbox_to_anchor=(0.5, 0.95), 
            ncol=len(by_label), frameon=False
        )
        
    # Use tight_layout's rect parameter to reserve the top 8% of the figure 
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Performance multi-plot to {save_path}")


def plot_glue_timeseries_grid(data_dict, category_name, save_path, config):
    """
    Creates a dynamic grid specific for GLUE metrics where:
    Rows = Metrics
    Cols = Evaluated Tasks
    """
    if not data_dict:
        print(f"No data available for {category_name}. Skipping plot.")
        return
        
    # Discover unique metrics and evaluated tasks across all algorithms
    metrics = set()
    eval_tasks = set()
    algorithms = sorted(list(data_dict.keys()))
    
    for algo in algorithms:
        metrics.update(data_dict[algo].keys())
        for metric in data_dict[algo].keys():
            eval_tasks.update(data_dict[algo][metric].keys())
            
    metrics = sorted(list(metrics))
    # Sort tasks numerically here as well
    eval_tasks = sorted(list(eval_tasks), key=extract_task_num)
    
    rows = len(metrics)
    cols = len(eval_tasks)
    
    if rows == 0 or cols == 0:
        return
        
    fig_width = max(12, 6 * cols)
    fig_height = max(6, 4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    
    cmap = plt.get_cmap('tab10')
    colors = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    
    for r, metric in enumerate(metrics):
        for c, t_eval in enumerate(eval_tasks):
            ax = axes[r, c]
            max_x = 0
            has_data = False
            
            for algo in algorithms:
                if metric not in data_dict[algo] or t_eval not in data_dict[algo][metric]:
                    continue
                    
                data = data_dict[algo][metric][t_eval] # (Steps, Repeats)
                if data.size == 0: continue
                    
                has_data = True
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                        
                    mean = np.nanmean(data, axis=1)
                    std = np.nanstd(data, axis=1)
                    
                steps = len(mean)
                total_epochs = config.epochs_per_task * config.num_tasks
                epochs = np.linspace(total_epochs / steps, total_epochs, steps)
                
                mask = ~np.isnan(mean)
                if mask.any():
                    ax.plot(epochs[mask], mean[mask], label=algo, color=colors[algo])
                    ax.fill_between(
                        epochs[mask], 
                        (mean - std)[mask], 
                        (mean + std)[mask], 
                        color=colors[algo], alpha=0.15, linewidth=0
                    )
                    max_x = max(max_x, epochs[mask][-1] if len(epochs[mask]) > 0 else 0)
                    
            if has_data:
                # Clean up names for presentation
                clean_task_name = t_eval.replace('_', ' ').replace('00', '').title()
                
                # Setup Grid Titles and Labels
                if r == 0: # Top row gets Col Titles
                    ax.set_title(f"Eval: {clean_task_name}", fontweight='bold', pad=10)
                if c == 0: # Left Col gets Row Titles
                    ax.set_ylabel(metric, fontweight='bold')
                if r == rows - 1: # Bottom Row gets X Labels
                    ax.set_xlabel("Epochs")
                    
                ax.set_xlim(0, max_x)
                
                # Draw task boundaries clearly
                for t in range(1, config.num_tasks):
                    boundary = t * config.epochs_per_task
                    if boundary < max_x:
                        ax.axvline(x=boundary, color='#333333', linestyle=':', alpha=0.6, linewidth=1.5)
                        if r == 0 and c == 0: # Label boundaries on first subplot only
                            ax.text(boundary + (max_x * 0.01), ax.get_ylim()[0], f"Task {t+1}", 
                                    rotation=90, verticalalignment='bottom', fontsize=10, color='#555555')
                                    
    # Collect unique legend handles and labels
    handles, labels = [], []
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        
    by_label = dict(zip(labels, handles))

    # Add the figure title
    plt.suptitle(f"{category_name} Timeline Over Epochs", fontsize=20, fontweight='bold')

    # Place the legend at the top, just below the title
    if by_label:
        fig.legend(
            by_label.values(), by_label.keys(), 
            loc='upper center', bbox_to_anchor=(0.5, 0.95), 
            ncol=len(by_label), frameon=False
        )
        
    # Use tight_layout's rect parameter to reserve the top 8% of the figure 
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {category_name} multi-plot to {save_path}")


def plot_timeseries_grid(data_dict, category_name, save_path, config):
    """
    Creates a master grid of subplots for all available metrics within a category.
    (Used primarily for Plasticity Metrics).
    """
    if not data_dict:
        print(f"No data available for {category_name}. Skipping plot.")
        return
        
    # Extract unique metrics to determine grid size
    metrics = set()
    for algo, m_dict in data_dict.items():
        metrics.update(m_dict.keys())
    metrics = sorted(list(metrics))

    if not metrics:
        return

    num_metrics = len(metrics)
    cols = min(2, num_metrics)
    rows = math.ceil(num_metrics / cols)

    # Dynamic scaling so large sets of metrics don't get squashed
    fig_width = max(12, 8 * cols)
    fig_height = max(6, 5 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    if num_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Assign consistent colors dynamically
    algorithms = sorted(list(data_dict.keys()))
    cmap = plt.get_cmap('tab10')
    colors = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}

    for i, metric in enumerate(metrics):
        ax = axes[i]
        max_x = 0
        has_data = False

        for algo in algorithms:
            if metric not in data_dict[algo] or data_dict[algo][metric] is None: 
                continue
            
            data = data_dict[algo][metric] # Expected shape: (Steps, Repeats)
            if data.size == 0: continue
                
            has_data = True

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Enforce 2D in case n_repeats=1 caused a flattened array elsewhere
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                
                mean = np.nanmean(data, axis=1)
                std = np.nanstd(data, axis=1)

            steps = len(mean)
            total_epochs = config.epochs_per_task * config.num_tasks
            epochs = np.linspace(total_epochs / steps, total_epochs, steps)

            mask = ~np.isnan(mean)
            if mask.any():
                ax.plot(epochs[mask], mean[mask], label=algo, color=colors[algo])
                ax.fill_between(
                    epochs[mask], 
                    (mean - std)[mask], 
                    (mean + std)[mask], 
                    color=colors[algo], alpha=0.15, linewidth=0
                )
                
                max_x = max(max_x, epochs[mask][-1] if len(epochs[mask]) > 0 else 0)


        if has_data:
            ax.set_title(metric, fontweight='bold', pad=10)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Metric Value")
            ax.set_xlim(0, max_x)

            # Draw task boundaries clearly
            for t in range(1, config.num_tasks):
                boundary = t * config.epochs_per_task
                if boundary < max_x:
                    ax.axvline(x=boundary, color='#333333', linestyle=':', alpha=0.6, linewidth=1.5)
                    if i == 0: 
                        ax.text(boundary + (max_x * 0.01), ax.get_ylim()[0], f"Task {t+1}", 
                                rotation=90, verticalalignment='bottom', fontsize=10, color='#555555')

    for j in range(len(metrics), len(axes)):
        axes[j].axis('off')

    # Collect unique legend handles and labels
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        
    by_label = dict(zip(labels, handles))

    # Add the figure title
    plt.suptitle(f"{category_name} Timeline Over Epochs", fontsize=20, fontweight='bold')

    # Place the legend at the top, just below the title
    if by_label:
        fig.legend(
            by_label.values(), by_label.keys(), 
        )
        
    # Use tight_layout's rect parameter to reserve the top 8% of the figure 
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {category_name} multi-plot to {save_path}")

def plot_cl_metrics_bar(cl_data, save_path):
    """
    Plots a grouped bar chart for Continual Learning metrics (e.g. transfer, remembering, zero-shot).
    Columns = 1, Rows = Number of metrics.
    Handles both 2D (Tasks, Repeats) and 3D (Tasks, Tasks, Repeats) metric arrays natively.
    """
    if not cl_data:
        print("No CL metrics available. Skipping plot.")
        return

    # Discover unique metrics
    metrics = set()
    algorithms = sorted(list(cl_data.keys()))
    for algo in algorithms:
        metrics.update(cl_data[algo].keys())
        
    metrics = sorted(list(metrics))
    if not metrics:
        return

    rows = len(metrics)
    cols = 1
    
    fig_width = 14
    fig_height = max(5, 5 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    
    cmap = plt.get_cmap('tab10')
    colors = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    
    num_algos = len(algorithms)
    bar_width = 0.8 / num_algos
    
    for r, metric in enumerate(metrics):
        ax = axes[r, 0]
        max_tasks = 0 
        
        for i, algo in enumerate(algorithms):
            if metric not in cl_data[algo]: 
                continue
                
            data = cl_data[algo][metric] 
            
            # Enforce at least 2D array
            if data.ndim == 1:
                data = data.reshape(-1, 1)
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                
                # Handle 3D arrays (e.g., remembering: 20x20x32, zero_shot: 20x20x32)
                # Average across the task-interaction dimension to get a per-task timeline
                if data.ndim == 3:
                    data = np.nanmean(data, axis=1)
                
                # Now data is guaranteed to be (Tasks, Repeats)
                mean_vals = np.nanmean(data, axis=1)
                
                # Use actual valid non-NaN counts for SEM instead of raw shape
                valid_counts = np.sum(~np.isnan(data), axis=1)
                # Prevent division by zero where data was entirely NaNs
                valid_counts = np.where(valid_counts == 0, 1, valid_counts) 
                
                std_error = np.nanstd(data, axis=1) / np.sqrt(valid_counts)
            
            num_tasks = len(mean_vals)
            max_tasks = max(max_tasks, num_tasks)
            indices = np.arange(num_tasks)
            
            # Calculate dynamic offsets to group the bars side-by-side
            offset = (i - num_algos / 2 + 0.5) * bar_width
            
            # Filter out pure NaN bars so matplotlib doesn't complain
            mask = ~np.isnan(mean_vals)
            
            ax.bar(
                indices[mask] + offset, 
                mean_vals[mask], 
                width=bar_width, 
                yerr=std_error[mask], 
                label=algo if r == 0 else "", # Avoid legend duplication across subplots
                color=colors[algo], 
                capsize=3,
                alpha=0.85
            )
            
        # Format the subplot for this metric specifically
        title_str = metric.replace('_', ' ').title()
        if metric == 'zero_shot':
            title_str = 'Zero-Shot Learning'
            
        ax.set_title(title_str, fontweight='bold', pad=10)
        ax.set_ylabel("Metric Value", fontweight='bold')
        
        if r == rows - 1:
            ax.set_xlabel("Task Number", fontweight='bold')
            
        if max_tasks > 0:
            ax.set_xticks(np.arange(max_tasks))
            ax.set_xticklabels([str(t+1) for t in range(max_tasks)])
        
        # Turn off vertical grid lines for cleaner bar grouping
        ax.grid(axis='x', visible=False) 
        
    # Standardized unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.suptitle("Continual Learning Metrics Per Task", fontsize=20, fontweight='bold')

    if by_label:
        fig.legend(
            by_label.values(), by_label.keys(), 
            # loc='upper center', bbox_to_anchor=(0.5, 0.98), 
            # ncol=len(by_label), frameon=False
        )
        
    # plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved CL Metrics bar chart to {save_path}")

def plot_cl_metrics_total_average_bar(cl_data, save_path):
    """
    Plots a bar chart for Continual Learning metrics averaged across all tasks.
    Correctly propagates error by averaging tasks per repeat first, then calculating
    the standard error of the mean (SEM) across independent repeats.
    """
    if not cl_data:
        return

    # Discover unique metrics
    metrics = set()
    algorithms = sorted(list(cl_data.keys()))
    for algo in algorithms:
        metrics.update(cl_data[algo].keys())
        
    metrics = sorted(list(metrics))
    if not metrics:
        return

    rows = len(metrics)
    fig_width = max(8, 2 * len(algorithms))
    fig_height = max(5, 4 * rows)
    fig, axes = plt.subplots(rows, 1, figsize=(fig_width, fig_height), squeeze=False)
    
    cmap = plt.get_cmap('tab10')
    colors = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    
    for r, metric in enumerate(metrics):
        ax = axes[r, 0]
        
        algo_means = []
        algo_sems = []
        valid_algos = []
        algo_colors = []
        
        for algo in algorithms:
            if metric not in cl_data[algo]: 
                continue
                
            data = cl_data[algo][metric] 
            
            if data.ndim == 1:
                data = data.reshape(-1, 1)
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                
                # Handle 3D arrays (e.g., remembering: 20x20x32)
                if data.ndim == 3:
                    data = np.nanmean(data, axis=1)
                
                # data is now (Tasks, Repeats). 
                # Average over tasks first to preserve within-seed covariance.
                repeat_means = np.nanmean(data, axis=0) 
                
                # Calculate final mean and SEM across the independent repeats
                final_mean = np.nanmean(repeat_means)
                valid_counts = np.sum(~np.isnan(repeat_means))
                valid_counts = max(1, valid_counts) # Prevent division by zero
                
                final_sem = np.nanstd(repeat_means) / np.sqrt(valid_counts)
            
            valid_algos.append(algo)
            algo_means.append(final_mean)
            algo_sems.append(final_sem)
            algo_colors.append(colors[algo])
            
        if not valid_algos:
            continue
            
        x_pos = np.arange(len(valid_algos))
        
        ax.bar(
            x_pos, 
            algo_means, 
            yerr=algo_sems, 
            color=algo_colors, 
            capsize=5,
            alpha=0.85,
            edgecolor='black',
            linewidth=1.2
        )
        
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold', pad=10)
        ax.set_ylabel("Task-Averaged Value", fontweight='bold')
        ax.set_xticks(x_pos)
        
        # Only add x-labels to the bottom plot to keep it clean
        if r == rows - 1:
            ax.set_xticklabels(valid_algos, rotation=15, ha="right", fontweight='bold')
        else:
            ax.set_xticklabels([])
            
        ax.grid(axis='x', visible=False) 
        
    plt.suptitle("Total Average CL Metrics Across All Tasks", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Total Average CL Metrics bar chart to {save_path}")


def main(experiment_paths):
    print(f"\nScanning the provided experiment paths...")
    data_by_algo, config = load_algorithm_data(experiment_paths)
    
    if not data_by_algo:
        print("No valid metric data found. Exiting.")
        return
        
    if config is None:
        print("Warning: Could not load config.pkl. Using fallback task values (1000 epochs, 2 tasks).")
        class MockConfig:
            epochs_per_task = 1000
            num_tasks = 2
        config = MockConfig()
    

    base_dir = OUTPUT_PATH
    output_dir = os.path.join(base_dir, "comparison_plots")


    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory ready: {output_dir}")
    
    # Process Metrics
    print("\nExtracting and formatting metric data...")
    acc_data, loss_data = extract_performance_data(data_by_algo, config)
    glue_data = extract_glue_data(data_by_algo)
    plasticity_data = extract_plasticity_data(data_by_algo)
    cl_data = extract_cl_metrics(data_by_algo)
    
    # Plot Generation (Subplot panels)
    print("\nGenerating Time-Series Plots...")
    
    # 1. Performance (Accuracy and Loss combined into one grid)
    plot_performance_grid(
        acc_data, 
        loss_data, 
        save_path=os.path.join(output_dir, "performance_timeseries_comparison.png"),
        config=config
    )
    
    # 2. GLUE Manifold Geometry
    plot_glue_timeseries_grid(
        glue_data, 
        category_name="Manifold Geometry (GLUE)", 
        save_path=os.path.join(output_dir, "glue_timeseries_comparison.png"),
        config=config
    )
    
    # 3. Plasticity
    plot_timeseries_grid(
        plasticity_data, 
        category_name="Network Plasticity", 
        save_path=os.path.join(output_dir, "plasticity_timeseries_comparison.png"),
        config=config
    )
    
    # 4. Continual Learning Bar Charts
    print("\nGenerating CL Metrics Bar Charts...")
    plot_cl_metrics_bar(
        cl_data, 
        save_path=os.path.join(output_dir, "cl_metrics_comparison_bar.png")
    )

    # 5. Total Average Continual Learning Bar Charts
    plot_cl_metrics_total_average_bar(
        cl_data,
        save_path=os.path.join(output_dir, "cl_metrics_total_average_bar.png")
    )


    
    print("\nAll comparison plots generated successfully!")

if __name__ == "__main__":
    main(EXPERIMENT_PATHS)