import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import matplotlib.pyplot as plt
import numpy as np
import warnings
import data_utils
import config as config_module
from learner import ContinualLearner
import expert_trainer
from hooks import CLHook
import analysis
import cl_metrics  
import manifold_analysis


class LoggerHook(CLHook):
    def on_task_start(self, task, state):
        print(f"[Hook] Starting Task ID {task.task_id}")

def main():
    print(f"Using device: {jax.devices()}")
    config = config_module.get_config()
    print(f"Configuration Loaded (Repeats={config.n_repeats}, LogFreq={config.log_frequency})")
    
    os.makedirs(config.reps_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)

    # 1. Setup Data
    train_tasks = data_utils.create_continual_tasks(config, split='train')
    test_streams = {}
    test_tasks = data_utils.create_continual_tasks(config, split='test')
    for t in test_tasks:
        test_streams[t.name] = t.load_data()

    data_utils.save_task_samples_grid(train_tasks, config)
    
    # 2. Setup Learner & History
    learner = ContinualLearner(config, hooks=[LoggerHook()])
    
    print("Saving random initialization weights...")
    init_weights = learner.get_flat_params(learner.state)
    np.save(f"{config.reps_dir}/init_weights.npy", np.array(init_weights))
    
    global_history = {
        'train_acc': [], 'train_loss': [],
        'test_metrics': {
            name: {'acc': [], 'loss': []} 
            for name in test_streams.keys()
        }
    }
    
    task_boundaries = []
    total_epochs = 0
    task_names = [t.name for t in train_tasks] # Store names for analysis later

    # 3. CL Loop
    for task in train_tasks:
        analysis_ds = task.load_mandi_subset(samples_per_class=config.mandi_samples)
        
        _, subset_lbls = learner.preload_data(analysis_ds)

        if subset_lbls.ndim > 1: subset_lbls = subset_lbls.flatten()
        np.save(f"{config.reps_dir}/{task.name}_labels.npy", subset_lbls)

        # Train
        rep_history, w_history = learner.train_task(
            task, test_streams, global_history, analysis_subset=analysis_ds
        )
        
        # Save History (Sparse)
        if rep_history is not None:
            np.save(f"{config.reps_dir}/{task.name}_reps_per_epoch.npy", rep_history)

        if w_history is not None:
            np.save(f"{config.reps_dir}/{task.name}_weights_per_epoch.npy", w_history)
        
        total_epochs += config.epochs_per_task
        task_boundaries.append(total_epochs)

    # 4. Expert Baselines
    expert_stats = {} 
    expert_histories = {} 
    
    print("\n--- Computing Expert Baselines ---")
    for task in train_tasks:
        # expert_trainer returns: loss_mean, loss_std, acc_mean, acc_std, te_l, te_a
        # te_l and te_a are shape (Epochs, Repeats)
        _, _, _, _, exp_l, exp_acc = expert_trainer.train_single_expert(
            config, task, test_streams[task.name]
        )
        
        # Store raw histories for CL metrics
        expert_histories[task.name] = {
            'loss': exp_l,
            'acc': exp_acc
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stats = {
                'loss_mean': np.nanmean(exp_l, axis=1),
                'acc_mean': np.nanmean(exp_acc, axis=1)
            }
        expert_stats[task.name] = stats

    # 4a. CL metrics
    cl_met_results = cl_metrics.compute_and_log_cl_metrics(
        global_history, expert_histories, config
    )
    print(f"{cl_met_results}")

    # 5. Plotting
    print("\nGenerating Plots...")
    
    # --- Styling Setup ---
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'lines.linewidth': 2
    })
    
    # Use a more compact figure size (Width, Height)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    epochs_range = np.arange(1, total_epochs + 1)
    task_names = [t.name for t in train_tasks]
    
    # Use a high-quality qualitative colormap (tab10 is standard for categorical data)
    cmap = plt.get_cmap('tab10')
    color_dict = {name: cmap(i % 10) for i, name in enumerate(task_names)}

    # Helper to clean up the plot area
    def setup_axis(ax, ylabel):
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylabel(ylabel, fontweight='bold')
        # Draw Task Boundaries
        for i, boundary in enumerate(task_boundaries[:-1]):
            ax.axvline(x=boundary, color='#333333', linestyle='-', alpha=0.3, linewidth=1.5)
            if i < len(task_names) - 1:
                # Add label slightly offset from the boundary
                ax.text(boundary + (total_epochs*0.01), ax.get_ylim()[0], f"End T{i+1}", 
                        rotation=90, verticalalignment='bottom', fontsize=9, color='#555555')

    # --- Plot 1: Accuracy ---
    train_acc_raw = np.array(global_history['train_acc'])
    if train_acc_raw.ndim == 1: train_acc_raw = train_acc_raw.reshape(-1, config.n_repeats)

    train_mean = np.mean(train_acc_raw, axis=1)
    train_std = np.std(train_acc_raw, axis=1)
    
    # Current Task Performance (Grey)
    ax1.plot(epochs_range, train_mean, label='Current Task (Train)', color='grey', linestyle='-', alpha=0.3, linewidth=1)
    
    for t_name in task_names:
        metrics = global_history['test_metrics'][t_name]
        color = color_dict[t_name]
        
        # Continual Learning Curve
        acc_raw = np.array(metrics['acc'])
        if acc_raw.ndim == 1: acc_raw = acc_raw.reshape(-1, config.n_repeats)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(acc_raw, axis=1)
            std = np.nanstd(acc_raw, axis=1)
        
        mask = ~np.isnan(mean)
        if mask.any():
            ax1.plot(epochs_range[mask], mean[mask], label=f"{t_name} (CL)", color=color, linewidth=2.5)
            # Add subtle fill for variance
            ax1.fill_between(epochs_range[mask], mean[mask] - std[mask], mean[mask] + std[mask], color=color, alpha=0.1)
        
        # Expert Baseline
        if t_name in expert_stats:
            estats = expert_stats[t_name]
            task_idx = task_names.index(t_name)
            start_epoch = task_idx * config.epochs_per_task
            expert_x = np.arange(start_epoch + 1, start_epoch + len(estats['acc_mean']) + 1)
            emean = estats['acc_mean']
            emask = ~np.isnan(emean)
            
            if emask.any():
                # IMPROVED VISIBILITY: Dashed line, darker/thinner or distinct style
                ax1.plot(expert_x[emask], emean[emask], color=color, linestyle='--', linewidth=2.0, alpha=0.9, 
                         label=f"{t_name} (Expert)")

    setup_axis(ax1, 'Accuracy')
    ax1.set_ylim(-0.05, 1.05)
    
    # Consolidate legend to the right
    handles, labels = ax1.get_legend_handles_labels()
    # Filter duplicate labels if any
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), title="Legend")

    # --- Plot 2: Loss ---
    train_loss_raw = np.array(global_history['train_loss'])
    if train_loss_raw.ndim == 1: train_loss_raw = train_loss_raw.reshape(-1, config.n_repeats)
    loss_mean = np.mean(train_loss_raw, axis=1)
    
    ax2.plot(epochs_range, loss_mean, label='Current Task (Train)', color='grey', linestyle='-', alpha=0.3, linewidth=1)
    
    for t_name in task_names:
        color = color_dict[t_name]
        loss_raw = np.array(global_history['test_metrics'][t_name]['loss'])
        if loss_raw.ndim == 1: loss_raw = loss_raw.reshape(-1, config.n_repeats)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(loss_raw, axis=1)
        
        mask = ~np.isnan(mean)
        if mask.any():
            ax2.plot(epochs_range[mask], mean[mask], label=f"{t_name}", color=color, linewidth=2.5)
            
        if t_name in expert_stats:
             estats = expert_stats[t_name]
             task_idx = task_names.index(t_name)
             start_epoch = task_idx * config.epochs_per_task
             expert_x = np.arange(start_epoch + 1, start_epoch + len(estats['loss_mean']) + 1)
             emean = estats['loss_mean']
             emask = ~np.isnan(emean)
             if emask.any():
                 ax2.plot(expert_x[emask], emean[emask], color=color, linestyle='--', linewidth=2.0, alpha=0.9)

    setup_axis(ax2, 'Loss')
    ax2.set_xlabel('Total Epochs', fontweight='bold')
    
    # Adjust layout to make room for the legend on the right
    plt.tight_layout(rect=[0, 0, 0.82, 1]) 
    
    save_path = f'{config.figures_dir}/sl_{config.dataset_name}_{config.num_tasks}_tasks.png'
    plt.savefig(save_path, dpi=150) # Increase DPI for better text clarity
    print(f"Improved plots saved to {save_path}")
    
    # 6. Analysis
    analysis.run_analysis_pipeline(config)

    # 7 Manifold Analysis
    manifold_analysis.analyze_manifold_trajectory(config, task_names)
    
if __name__ == "__main__":
    main()