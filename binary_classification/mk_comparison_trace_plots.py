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
    Scans the base directory for algorithm subfolders. Explicitly loads 
    performance time-series data from global_history.pkl (or all_metrics.pkl), 
    and manifold/plasticity data from their respective files.
    """
    data_by_algo = {}
    config = None
    
    for entry in os.listdir(base_path):
        algo_dir = os.path.join(base_path, entry)
        if not os.path.isdir(algo_dir):
            continue
            
        metrics_file = os.path.join(algo_dir, "all_metrics.pkl")
        config_file = os.path.join(algo_dir, "config.pkl")
        gh_file = os.path.join(algo_dir, "global_history.pkl")
        
        algo_data = {'performance': {}, 'plasticity': {}, 'glue': {}}
        loaded_any = False

        # Grab experiment configurations first so we have n_repeats for reshaping
        if config is None and os.path.exists(config_file):
            with open(config_file, 'rb') as f:
                config = pickle.load(f)

        # 1. Load Performance Data (Prioritize global_history.pkl, fallback to all_metrics['cl'])
        if os.path.exists(gh_file):
            with open(gh_file, 'rb') as f:
                algo_data['performance'] = pickle.load(f)
            loaded_any = True
            
        # 2. Load Unified Master Object (if exists)
        if os.path.exists(metrics_file):
            with open(metrics_file, 'rb') as f:
                master_data = pickle.load(f)
                if not algo_data['performance']:
                    algo_data['performance'] = master_data.get('cl', {})
                algo_data['plasticity'] = master_data.get('plasticity', {})
                algo_data['glue'] = master_data.get('glue', {})
            loaded_any = True
        else:
            # Fallback for individual metric files
            import glob
            plast_files = glob.glob(os.path.join(algo_dir, "plastic_analysis_*.pkl"))
            if plast_files:
                with open(plast_files[0], 'rb') as f:
                    algo_data['plasticity'] = pickle.load(f).get('history', {})
                loaded_any = True
                
            glue_file = os.path.join(algo_dir, "glue_metrics.pkl")
            if os.path.exists(glue_file):
                with open(glue_file, 'rb') as f:
                    algo_data['glue'] = pickle.load(f)
                loaded_any = True

        if loaded_any:
            data_by_algo[entry] = algo_data
            print(f"Loaded metric data for algorithm: {entry}")
        else:
            print(f"Skipping {entry}: No metric files found.")
            
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
    """Averages GLUE metric evaluations over all evaluated tasks and stitches over time."""
    glue_extracted = {}
    for algo, data in data_by_algo.items():
        g_data = data.get('glue', {})
        if not g_data: continue
        
        algo_metrics = {}
        train_tasks = sorted(list(g_data.keys()))
        
        # Discover all unique metric names
        metrics = set()
        for t_train in train_tasks:
            for t_eval in g_data[t_train].keys():
                metrics.update(g_data[t_train][t_eval].keys())
        
        # Aggregate the timeline per metric
        for metric in metrics:
            stitched_raw = []
            for t_train in train_tasks:
                eval_tasks = list(g_data[t_train].keys())
                arrays = []
                for t_eval in eval_tasks:
                    if metric in g_data[t_train][t_eval]:
                        arrays.append(np.array(g_data[t_train][t_eval][metric]))
                
                if arrays:
                    stacked = np.stack(arrays, axis=0)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        # Average evaluations over the current train block 
                        mean_over_eval = np.nanmean(stacked, axis=0) 
                    stitched_raw.append(mean_over_eval)
                    
            if stitched_raw:
                # Concatenate the sequential training blocks into one global timeline
                algo_metrics[metric] = np.concatenate(stitched_raw, axis=0) 
                
        if algo_metrics:
            glue_extracted[algo] = algo_metrics
    return glue_extracted

def plot_timeseries_grid(data_dict, category_name, save_path, config):
    """
    Creates a master grid of subplots for all available metrics within a category,
    overlaying multiple algorithms with a unified legend.
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

            # --- FIX: Apply NaN Mask before plotting ---
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
                    # Add task label slightly offset
                    if i == 0: # Only put text on first plot to avoid clutter
                        ax.text(boundary + (max_x * 0.01), ax.get_ylim()[0], f"Task {t+1}", 
                                rotation=90, verticalalignment='bottom', fontsize=10, color='#555555')

    for j in range(len(metrics), len(axes)):
        axes[j].axis('off')

    # --- FIX: Collect Legend handles from ALL axes, not just axes[0] ---
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        
    by_label = dict(zip(labels, handles)) # De-duplicate
    
    legend_y = 1.15 if rows == 1 else 1.05
    title_y = 1.25 if rows == 1 else 1.10
    
    if by_label:
        fig.legend(
            by_label.values(), by_label.keys(), 
            loc='upper center', bbox_to_anchor=(0.5, legend_y), 
            ncol=len(algorithms), frameon=False
        )

    plt.suptitle(f"{category_name} Timeline Over Epochs", fontsize=20, fontweight='bold', y=title_y)
    
    plt.tight_layout()
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
    
    plot_timeseries_grid(
        acc_data, 
        category_name="Performance (Accuracy)", 
        save_path=os.path.join(output_dir, "performance_comparison_plots_accuracy.png"),
        config=config
    )
    
    plot_timeseries_grid(
        loss_data, 
        category_name="Performance (Loss)", 
        save_path=os.path.join(output_dir, "performance_comparison_plots_loss.png"),
        config=config
    )
    
    plot_timeseries_grid(
        glue_data, 
        category_name="Manifold Geometry (GLUE)", 
        save_path=os.path.join(output_dir, "glue_timeseries_comparison.png"),
        config=config
    )
    
    plot_timeseries_grid(
        plasticity_data, 
        category_name="Network Plasticity", 
        save_path=os.path.join(output_dir, "plasticity_timeseries_comparison.png"),
        config=config
    )
    
    print("\nAll comparison plots generated successfully!")

if __name__ == "__main__":
    main()