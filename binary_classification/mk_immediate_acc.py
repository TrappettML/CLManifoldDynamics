import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings

# =====================================================================
# CONFIGURATION
# =====================================================================
# Paste the paths to your experiment directories here
EXPERIMENT_PATHS = [
"/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.01_lr2_0.01",
"/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.0001_lr2_0.01",
"/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.01_lr2_0.0001",
'/home/users/MTrappett/manifold/binary_classification/results/imagenet_28_gray/SL_20_tasks_lr1_0.0001_lr2_0.0001'
]

# Labels for the legend (must match the order and number of paths above)
EXPERIMENT_LABELS = [
    "lr1: 0.01, lr2: 0.01",
    "lr1: 0.0001, lr2: 0.01",
    "lr1: 0.01, lr2: 0.0001",
    "lr1: 0.0001, lr2: 0.0001"
]

# Plot output settings
OUTPUT_FILENAME = "immediate_accuracy_comparison.png"
PLOT_TITLE = "Immediate Task Accuracy Comparison"
# =====================================================================


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
        'legend.frameon': False,      # Remove box around legend
        'axes.spines.top': False,     # Remove top border
        'axes.spines.right': False,   # Remove right border
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.color': '#e0e0e0',
        'figure.figsize': (10, 6),
        'figure.dpi': 300             # Higher res for crisp rendering
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
        
    # Load config to know task boundaries
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
            break  # Stop if we hit an incomplete task
            
        t_acc = np.array(test_metrics[t_name]['acc'])
        
        # Defensive reshape mirroring learner.py logic
        if t_acc.ndim == 1 and n_repeats > 1:
            t_acc = t_acc.reshape(-1, n_repeats)
        elif t_acc.ndim == 1:
            t_acc = t_acc.reshape(-1, 1)
            
        # Isolate the training epochs specific to THIS task
        start_idx = t * epochs_per_task
        end_idx = (t + 1) * epochs_per_task
        task_t_acc_during_training = t_acc[start_idx:end_idx]
        
        if len(task_t_acc_during_training) == 0:
            break  # Run terminated early
            
        # The learner pads with NaNs between log_frequencies. 
        # We find the last row that contains actual evaluation numbers.
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
        
    # Shape will be (Tasks_Completed, Repeats)
    return np.array(extracted_accs), num_tasks


def main():
    if EXPERIMENT_LABELS and len(EXPERIMENT_LABELS) != len(EXPERIMENT_PATHS):
        raise ValueError("Number of EXPERIMENT_LABELS must exactly match the number of EXPERIMENT_PATHS provided.")

    set_plot_formatting()
    fig, ax = plt.subplots()
    
    # A slightly softer, more modern colormap
    cmap = plt.get_cmap('Set1')
    max_tasks_plotted = 0

    for i, path in enumerate(EXPERIMENT_PATHS):
        label = EXPERIMENT_LABELS[i] if EXPERIMENT_LABELS else os.path.basename(os.path.normpath(path))
        print(f"Processing: {label} ({path})")
        
        data, expected_tasks = load_and_extract_data(path)
        
        if data is None:
            continue
            
        tasks_completed = data.shape[0]
        max_tasks_plotted = max(max_tasks_plotted, tasks_completed)
        x_axis = np.arange(1, tasks_completed + 1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Calculate Mean
            mean_acc = np.nanmean(data, axis=1)
            # Calculate Standard Error of the Mean (SEM)
            std_acc = np.nanstd(data, axis=1)
            valid_counts = np.sum(~np.isnan(data), axis=1)
            valid_counts = np.where(valid_counts == 0, 1, valid_counts) # Prevent div by 0
            sem_acc = std_acc / np.sqrt(valid_counts)
            
        color = cmap(i % 9)
        
        # 1. Plot the main trend line
        ax.plot(
            x_axis, 
            mean_acc, 
            label=label, 
            color=color, 
            linewidth=2.5,
            marker='o',          
            markersize=6,        
            markerfacecolor='white',
            markeredgewidth=1.5,
            zorder=3             
        )
        
        # 2. Add the shaded SEM region instead of error bars
        ax.fill_between(
            x_axis,
            mean_acc - sem_acc,
            mean_acc + sem_acc,
            color=color,
            alpha=0.15,          # Transparency of the shaded region
            linewidth=0,         # No border on the shading
            zorder=2
        )

    if max_tasks_plotted == 0:
        print("No valid data found in any provided paths. Exiting.")
        return

    # Chart Polish
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel("Task Number")
    ax.set_ylabel("Final Test Accuracy")
    
    # Force integer ticks on the X-axis
    ax.set_xticks(np.arange(1, max_tasks_plotted + 1))
    
    # Cap Y-axis cleanly (assuming accuracy is 0-1)
    ax.set_ylim(-0.05, 1.05) 
    
    # Style the legend outside the plot to prevent covering data
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.05), ncol=2, borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME)
    print(f"\nPlot successfully saved to: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()