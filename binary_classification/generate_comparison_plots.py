import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict
import math

# --- Configuration ---

ALGO_DIRS = {
    'Supervised (SL)': './single_runs/SL/data',
    'Reinforcement (RL)': './single_runs/RL/data',
    # Add more algorithms here:
    # 'Unsupervised (UL)': './single_runs/UL/data'
}

OUTPUT_DIR = "/home/users/MTrappett/manifold/binary_classification/aggregate_plots"
DATASET_NAME = 'kmnist' 
LOG_FREQ = 10 

COLOR_MAP = {
    'Supervised (SL)': '#1f77b4',      # Blue
    'Reinforcement (RL)': '#ff7f0e',   # Orange
    'Unsupervised (UL)': '#2ca02c',    # Green
}

# Shared Matplotlib Style
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_pickle(path):
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# --- Plotting Functions ---

def create_grouped_box_plot_html(cl_data_map, save_dir):
    """Generates Plotly HTML box plot for CL metrics."""
    print("Generating CL Metrics Box Plot (HTML)...")
    fig = go.Figure()

    metrics_to_plot = ['remembering', 'transfer']
    
    for algo_name, metrics_dict in cl_data_map.items():
        color = COLOR_MAP.get(algo_name, 'grey')
        y_values = []
        x_labels = []
        
        for m in metrics_to_plot:
            if m in metrics_dict:
                vals = np.array(metrics_dict[m]).flatten()
                y_values.extend(vals)
                label = "Stability" if m == 'remembering' else "Transfer"
                x_labels.extend([label] * len(vals))
        
        fig.add_trace(go.Box(
            y=y_values, x=x_labels, name=algo_name,
            marker_color=color, boxpoints='all', jitter=0.3, pointpos=-1.8
        ))

    fig.update_layout(
        title="<b>Continual Learning Performance</b>",
        yaxis=dict(title="Score (Normalized)", range=[-1.1, 1.1], zeroline=True),
        boxmode='group', template="plotly_white",
        font=dict(family="Arial", size=18, color="black"),
        width=1000, height=700
    )
    
    fig.write_html(os.path.join(save_dir, "CL_Metrics_Boxplot.html"))

def create_matplotlib_line_plot(metric_name, algo_data_map, task_boundaries, save_dir):
    """Generates single PNG line plot for plastic/manifold metrics."""
    print(f"Generating Line Plot for {metric_name}...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    max_x = 0
    has_data = False

    for algo_name, data_array in algo_data_map.items():
        if data_array is None or data_array.size == 0: continue
        has_data = True
        
        mean = np.nanmean(data_array, axis=1)
        std = np.nanstd(data_array, axis=1)
        epochs = np.arange(1, len(mean) + 1) * LOG_FREQ
        max_x = max(max_x, epochs[-1])
        color = COLOR_MAP.get(algo_name, 'grey')

        ax.plot(epochs, mean, label=algo_name, color=color)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

    if not has_data:
        plt.close(fig)
        return

    if task_boundaries:
        for b in task_boundaries[:-1]: 
            ax.axvline(x=b, color='black', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_title(metric_name, fontweight='bold')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Value")
    ax.set_xlim(0, max_x)
    ax.legend(loc='best')
    
    plt.tight_layout()
    safe_name = metric_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=300)
    plt.close(fig)

def create_task_comparison_subplots(algo_history_map, task_boundaries, save_dir):
    """
    Creates two PNGs: one for Accuracy, one for Loss.
    Each PNG contains subplots (one per Task).
    """
    print("Generating Task Comparison Subplots...")

    # Identify all tasks present across algorithms
    all_tasks = set()
    for history in algo_history_map.values():
        if history and 'test_metrics' in history:
            all_tasks.update(history['test_metrics'].keys())
    
    if not all_tasks:
        print("No task history found.")
        return

    sorted_tasks = sorted(list(all_tasks))
    n_tasks = len(sorted_tasks)
    
    # Calculate subplot grid (e.g., 2 tasks -> 1x2, 4 tasks -> 2x2)
    cols = min(n_tasks, 2)
    rows = math.ceil(n_tasks / cols)

    metrics_to_plot = [('acc', 'Test Accuracy'), ('loss', 'Test Loss')]

    for metric_key, metric_title in metrics_to_plot:
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), sharex=True)
        if n_tasks == 1: axes = [axes]
        axes = np.array(axes).flatten()

        max_x_epoch = 0

        for i, t_name in enumerate(sorted_tasks):
            ax = axes[i]
            
            for algo_name, history in algo_history_map.items():
                if t_name not in history['test_metrics']: continue
                
                # Extract data: List of arrays -> single array (Steps, Repeats)
                raw_data_list = history['test_metrics'][t_name][metric_key]
                data_array = np.array(raw_data_list)
                
                if data_array.size == 0: continue

                mean = np.nanmean(data_array, axis=1)
                std = np.nanstd(data_array, axis=1)
                epochs = np.arange(1, len(mean) + 1) * LOG_FREQ
                
                max_x_epoch = max(max_x_epoch, epochs[-1] if len(epochs) > 0 else 0)
                color = COLOR_MAP.get(algo_name, 'grey')

                # Filter NaNs for plotting (in case tasks start late)
                mask = ~np.isnan(mean)
                if mask.any():
                    ax.plot(epochs[mask], mean[mask], label=algo_name, color=color)
                    ax.fill_between(epochs[mask], mean[mask]-std[mask], mean[mask]+std[mask], 
                                    color=color, alpha=0.15, linewidth=0)

            ax.set_title(t_name, fontweight='bold')
            ax.set_xlabel("Epochs")
            ax.set_ylabel(metric_title)
            
            # Draw Task Boundaries
            if task_boundaries:
                for b in task_boundaries[:-1]:
                    ax.axvline(x=b, color='black', linestyle=':', alpha=0.5)

            # Only show legend on the first subplot to reduce clutter
            if i == 0:
                ax.legend(loc='best', fontsize=10)

        # Hide empty subplots if any
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Comparison: {metric_title}", fontsize=20, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90) # Make room for suptitle

        save_path = os.path.join(save_dir, f"Comparison_{metric_title.replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {save_path}")

# --- Main ---

def main():
    ensure_dir(OUTPUT_DIR)
    
    cl_metrics_data = {}
    plastic_data_agg = defaultdict(dict)
    manifold_data_agg = defaultdict(dict)
    algo_history_map = {} # Stores global_history for each algo
    
    common_task_boundaries = None

    print(f"--- Processing Algorithms: {list(ALGO_DIRS.keys())} ---")

    for algo_name, data_dir in ALGO_DIRS.items():
        # 1. Load Global History (New)
        hist_path = os.path.join(data_dir, f"global_history_{DATASET_NAME}.pkl")
        hist_data = load_pickle(hist_path)
        if hist_data:
            algo_history_map[algo_name] = hist_data

        # 2. Load CL Metrics
        cl_path = os.path.join(data_dir, f"cl_metrics_{DATASET_NAME}.pkl")
        cl_data = load_pickle(cl_path)
        if cl_data: cl_metrics_data[algo_name] = cl_data
        
        # 3. Load Plasticine Metrics
        plas_path = os.path.join(data_dir, f"plastic_analysis_{DATASET_NAME}.pkl")
        plas_pkl = load_pickle(plas_path)
        if plas_pkl:
            history = plas_pkl['history']
            boundaries = plas_pkl.get('task_boundaries', [])
            if common_task_boundaries is None and boundaries:
                common_task_boundaries = boundaries
            for metric, val_array in history.items():
                plastic_data_agg[metric][algo_name] = val_array

        # 4. Load Manifold Metrics
        man_path = os.path.join(data_dir, f"manifold_metrics_full_{DATASET_NAME}.pkl")
        man_pkl = load_pickle(man_path)
        if man_pkl:
            sorted_tasks = sorted(man_pkl.keys())
            algo_man_metrics = defaultdict(list)
            for t_name in sorted_tasks:
                task_dict = man_pkl[t_name]
                for metric, val_array in task_dict.items():
                    algo_man_metrics[metric].append(val_array)
            for metric, list_of_arrays in algo_man_metrics.items():
                if list_of_arrays:
                    manifold_data_agg[metric][algo_name] = np.concatenate(list_of_arrays, axis=0)

    # --- Generate Plots ---

    # 1. Plotly HTML Box Plot
    if cl_metrics_data:
        create_grouped_box_plot_html(cl_metrics_data, OUTPUT_DIR)

    # 2. Matplotlib Plasticity & Manifold Lines
    for metric, algo_map in plastic_data_agg.items():
        create_matplotlib_line_plot(metric, algo_map, common_task_boundaries, OUTPUT_DIR)
    for metric, algo_map in manifold_data_agg.items():
        create_matplotlib_line_plot(metric, algo_map, common_task_boundaries, OUTPUT_DIR)

    # 3. Task Comparison Subplots (Accuracy/Loss)
    if algo_history_map:
        create_task_comparison_subplots(algo_history_map, common_task_boundaries, OUTPUT_DIR)
    else:
        print("Skipping Task Comparison (No 'global_history' found. Did you update single_run.py?)")

    print(f"\nAll plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()