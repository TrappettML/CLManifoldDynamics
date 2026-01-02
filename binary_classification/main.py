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

    # 3. CL Loop
    for task in train_tasks:
        analysis_ds = task.load_mandi_subset(samples_per_class=config.mandi_samples)
        
        _, subset_lbls = learner.preload_data(analysis_ds)
        np.save(f"{config.reps_dir}/{task.name}_labels.npy", subset_lbls)

        # Train
        rep_history, w_history = learner.train_task(
            task, test_streams, global_history, analysis_subset=analysis_ds
        )
        
        # Save History (Sparse)
        if rep_history is not None:
            # rep_history is already a numpy array from learner (tree_mapped)
            np.save(f"{config.reps_dir}/{task.name}_reps_per_epoch.npy", rep_history)

        if w_history is not None:
            np.save(f"{config.reps_dir}/{task.name}_weights_per_epoch.npy", w_history)
        
        total_epochs += config.epochs_per_task
        task_boundaries.append(total_epochs)

    # 4. Expert Baselines
    expert_stats = {} 
    print("\n--- Computing Expert Baselines ---")
    for task in train_tasks:
        l_mean, l_std, a_mean, a_std, exp_l, exp_acc = expert_trainer.train_single_expert(config, task, test_streams[task.name])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stats = {
                'loss_mean': np.nanmean(exp_l, axis=1),
                'acc_mean': np.nanmean(exp_acc, axis=1)
            }
        expert_stats[task.name] = stats

    # 5. Plotting
    print("\nGenerating Plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    epochs_range = np.arange(1, total_epochs + 1)
    task_names = [t.name for t in train_tasks]
    colormap = plt.cm.jet(np.linspace(0, 1, len(task_names)))
    color_dict = {name: color for name, color in zip(task_names, colormap)}

    # Plot Train Accuracy (Dense)
    train_acc_raw = np.array(global_history['train_acc'])
    
    # Handle list extension correctly: global_history is a list of arrays (Repeats,)
    # np.array() creates (TotalEpochs, Repeats).
    if train_acc_raw.ndim == 1:
        # Fallback if flattening happened
        train_acc_raw = train_acc_raw.reshape(-1, config.n_repeats)

    train_mean = np.mean(train_acc_raw, axis=1)
    train_std = np.std(train_acc_raw, axis=1)
    
    ax1.plot(epochs_range, train_mean, label='Current Train Acc', color='grey', linestyle='--', alpha=0.4)
    ax1.fill_between(epochs_range, train_mean - train_std, train_mean + train_std, color='grey', alpha=0.2)
    
    for t_name in task_names:
        metrics = global_history['test_metrics'][t_name]
        color = color_dict[t_name]
        
        acc_raw = np.array(metrics['acc'])
        if acc_raw.ndim == 1: acc_raw = acc_raw.reshape(-1, config.n_repeats)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(acc_raw, axis=1)
            std = np.nanstd(acc_raw, axis=1)
        
        mask = ~np.isnan(mean)
        if mask.any():
            ax1.plot(epochs_range[mask], mean[mask], label=f"{t_name} (CL)", color=color, linewidth=2)
            ax1.fill_between(epochs_range[mask], mean[mask] - std[mask], mean[mask] + std[mask], color=color, alpha=0.1)
        
        # Expert
        if t_name in expert_stats:
            estats = expert_stats[t_name]
            task_idx = task_names.index(t_name)
            start_epoch = task_idx * config.epochs_per_task
            expert_x = np.arange(start_epoch + 1, start_epoch + len(estats['acc_mean']) + 1)
            emean = estats['acc_mean']
            emask = ~np.isnan(emean)
            if emask.any():
                ax1.plot(expert_x[emask], emean[emask], color=color, linestyle=':', linewidth=2.5, alpha=0.8)

    ax1.set_ylabel('Accuracy')
    # Move legend inside or adjust bbox to prevent cut-off with tight_layout
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax1.grid(True, alpha=0.3)

    # Plot Loss
    train_loss_raw = np.array(global_history['train_loss'])
    if train_loss_raw.ndim == 1: train_loss_raw = train_loss_raw.reshape(-1, config.n_repeats)
    loss_mean = np.mean(train_loss_raw, axis=1)
    
    ax2.plot(epochs_range, loss_mean, label='Current Train Loss', color='grey', linestyle='--', alpha=0.4)
    
    for t_name in task_names:
        loss_raw = np.array(global_history['test_metrics'][t_name]['loss'])
        if loss_raw.ndim == 1: loss_raw = loss_raw.reshape(-1, config.n_repeats)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(loss_raw, axis=1)
        
        mask = ~np.isnan(mean)
        if mask.any():
            ax2.plot(epochs_range[mask], mean[mask], label=f"{t_name}", color=color, linewidth=2)
            
        if t_name in expert_stats:
             estats = expert_stats[t_name]
             task_idx = task_names.index(t_name)
             start_epoch = task_idx * config.epochs_per_task
             expert_x = np.arange(start_epoch + 1, start_epoch + len(estats['loss_mean']) + 1)
             emean = estats['loss_mean']
             emask = ~np.isnan(emean)
             if emask.any():
                 ax2.plot(expert_x[emask], emean[emask], color=color, linestyle=':', linewidth=2.5, alpha=0.8)

    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Total Epochs')
    for boundary in task_boundaries[:-1]:
        ax1.axvline(x=boundary, color='black', alpha=0.5)
        ax2.axvline(x=boundary, color='black', alpha=0.5)

    # Use constrained_layout or adjust subplots manually to fit external legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.85) # Make room for legend
    plt.savefig(f'{config.figures_dir}/sl_{config.dataset_name}_{config.num_tasks}_tasks.png')
    print(f"Plots saved.")
    
    # 6. Analysis
    analysis.run_analysis_pipeline(config)
    
if __name__ == "__main__":
    main()