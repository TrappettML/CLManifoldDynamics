import os
import pickle
import argparse
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt

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

def load_algorithm_data(base_path):
    """
    Scans the base directory for algorithm subfolders. 
    Strictly loads performance time-series data from global_history.pkl, 
    and manifold/plasticity data from all_metrics.pkl.
    """
    data_by_algo = {}
    config = None
    
    for entry in os.listdir(base_path):
        # Skip the output directory and hidden system files
        if entry == "comparison_plots" or entry.startswith('.'):
            continue
            
        algo_dir = os.path.join(base_path, entry)
        if not os.path.isdir(algo_dir):
            continue
            
        config_file = os.path.join(algo_dir, "config.pkl")
        gh_file = os.path.join(algo_dir, "global_history.pkl")
        all_metrics_file = os.path.join(algo_dir, "all_metrics.pkl")
        
        algo_data = {'performance': {}, 'plasticity': {}, 'glue': {}}
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
            
        # 2. Load Plasticity and GLUE data strictly from all_metrics.pkl
        if os.path.exists(all_metrics_file):
            with open(all_metrics_file, 'rb') as f:
                master_data = pickle.load(f)
                algo_data['plasticity'] = master_data.get('plasticity', {})
                algo_data['glue'] = master_data.get('glue', {})
            loaded_any = True

        if loaded_any:
            data_by_algo[entry] = algo_data
            print(f"Loaded metric data for algorithm: {entry}")
        else:
            print(f"Skipping {entry}: No metric files found. (Looked for global_history.pkl and all_metrics.pkl)")
            
    return data_by_algo, config

def extract_performance_data(data_by_algo, config):
    """
    Extracts task-specific Accuracy and Loss for performance comparison plots.
    Reshapes the traces using config.n_repeats exactly like single_run.py.
    """
    acc_extracted = {}
    loss_extracted = {}
    
    # Fallback to 1 repeat if config is missing to prevent crashes
    repeats = getattr(config, 'n_repeats', 1)
    
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

def plot_performance_grid(acc_data, loss_data, save_path, config):
    """
    Plots Accuracy and Loss in the same figure.
    Row 0: Accuracy across tasks
    Row 1: Loss across tasks
    """
    if not acc_data and not loss_data:
        print("No performance data available. Skipping plot.")
        return

    # Discover unique evaluated tasks across all algorithms
    tasks = set()
    algorithms = sorted(list(acc_data.keys()))
    for algo in algorithms:
        tasks.update(acc_data[algo].keys())
    tasks = sorted(list(tasks))
    
    cols = len(tasks)
    rows = 2
    
    if cols == 0:
        return
        
    fig_width = max(12, 6 * cols)
    fig_height = max(8, 4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    
    cmap = plt.get_cmap('tab10')
    colors = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    
    metrics_data = [
        ('Accuracy', acc_data),
        ('Loss', loss_data)
    ]
    
    for r, (metric_name, data_dict) in enumerate(metrics_data):
        for c, task in enumerate(tasks):
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
                total_epochs = getattr(config, 'epochs_per_task', 1000) * getattr(config, 'num_tasks', 2)
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
                    ax.set_title(f"Task: {task}", fontweight='bold', pad=10)
                if c == 0: # Left Col gets Row Titles
                    ax.set_ylabel(metric_name, fontweight='bold')
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
    eval_tasks = sorted(list(eval_tasks))
    
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
                total_epochs = getattr(config, 'epochs_per_task', 1000) * getattr(config, 'num_tasks', 2)
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
            total_epochs = getattr(config, 'epochs_per_task', 1000) * getattr(config, 'num_tasks', 2)
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
            loc='upper center', bbox_to_anchor=(0.5, 0.95), 
            ncol=len(by_label), frameon=False
        )
        
    # Use tight_layout's rect parameter to reserve the top 8% of the figure 
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {category_name} multi-plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate algorithm comparison time-series subplots.")
    parser.add_argument(
        '--experiment_folder', 
        type=str, 
        required=True,
        help="Path to the dataset experiment folder containing algorithm directories (e.g., /.../results/mnist/)"
    )
    args = parser.parse_args()
    
    base_path = args.experiment_folder
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Experiment folder not found: {base_path}")
        
    print(f"\nScanning for algorithms in: {base_path}")
    data_by_algo, config = load_algorithm_data(base_path)
    
    if not data_by_algo:
        print("No valid metric data found. Exiting.")
        return
        
    if config is None:
        print("Warning: Could not load config.pkl. Using fallback task values (1000 epochs, 2 tasks).")
        class MockConfig:
            epochs_per_task = 1000
            num_tasks = 2
        config = MockConfig()
    
    # Setup Output Directory mapping where combined_box_plots writes
    output_dir = os.path.join(base_path, "comparison_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory ready: {output_dir}")
    
    # Process Metrics
    print("\nExtracting and formatting metric data...")
    acc_data, loss_data = extract_performance_data(data_by_algo, config)
    glue_data = extract_glue_data(data_by_algo)
    plasticity_data = extract_plasticity_data(data_by_algo)
    
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
    
    print("\nAll comparison plots generated successfully!")

if __name__ == "__main__":
    main()