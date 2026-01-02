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
    print(f"Configuration Loaded (Repeats={config.n_repeats})")
    
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
    
    # --- SAVE INITIALIZATION WEIGHTS ---
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
        
        rep_stack = np.stack(rep_history) 
        np.save(f"{config.reps_dir}/{task.name}_reps_per_epoch.npy", rep_stack)

        w_stack = np.stack(w_history)
        np.save(f"{config.reps_dir}/{task.name}_weights_per_epoch.npy", w_stack)
        
        total_epochs += config.epochs_per_task
        task_boundaries.append(total_epochs)

    # 4. Expert Baselines
    expert_stats = {} 
    print("\n--- Computing Expert Baselines ---")
    for task in train_tasks:
        l_mean, l_std, a_mean, a_std, exp_l, exp_acc = expert_trainer.train_single_expert(config, task, test_streams[task.name])
        
        # Suppress "Mean of empty slice" warnings for epochs with no eval
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stats = {
                'loss_mean': np.nanmean(exp_l, axis=1),
                'loss_std': np.nanstd(exp_l, axis=1),
                'acc_mean': np.nanmean(exp_acc, axis=1),
                'acc_std': np.nanstd(exp_acc, axis=1)
            }
        expert_stats[task.name] = stats

    # 5. Plotting
    print("\nGenerating Plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Use array for boolean indexing
    epochs_range = np.arange(1, total_epochs + 1)
    
    task_names = [t.name for t in train_tasks]
    colormap = plt.cm.jet(np.linspace(0, 1, len(task_names)))
    color_dict = {name: color for name, color in zip(task_names, colormap)}

    # --- AGGREGATE METRICS ---
    train_acc_raw = np.array(global_history['train_acc'])
    train_loss_raw = np.array(global_history['train_loss'])
    
    # Train stats are dense (no NaNs)
    train_mean = np.mean(train_acc_raw, axis=1)
    train_std = np.std(train_acc_raw, axis=1)
    loss_mean = np.mean(train_loss_raw, axis=1)

    # Plot Train Accuracy
    ax1.plot(epochs_range, train_mean, label='Current Train Acc', color='grey', linestyle='--', alpha=0.4)
    ax1.fill_between(epochs_range, train_mean - train_std, train_mean + train_std, color='grey', alpha=0.2)
    
    for t_name in task_names:
        metrics = global_history['test_metrics'][t_name]
        color = color_dict[t_name]
        
        acc_raw = np.array(metrics['acc']) 
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(acc_raw, axis=1)
            std = np.nanstd(acc_raw, axis=1)
        
        # Filter NaNs for plotting so lines are connected
        mask = ~np.isnan(mean)
        if mask.any():
            ax1.plot(epochs_range[mask], mean[mask], label=f"{t_name} (CL)", color=color, linewidth=2)
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
                ax1.plot(expert_x[emask], emean[emask], color=color, linestyle=':', linewidth=2.5, alpha=0.8)
                if emask[-1]:
                    ax1.scatter(expert_x[-1], emean[-1], color=color, marker='*', s=60, zorder=5)

    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{config.dataset_name} Accuracy (Solid=CL, Dotted=Expert)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot Train Loss
    ax2.plot(epochs_range, loss_mean, label='Current Train Loss', color='grey', linestyle='--', alpha=0.4)
    
    for t_name in task_names:
        metrics = global_history['test_metrics'][t_name]
        color = color_dict[t_name]
        
        loss_raw = np.array(metrics['loss'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(loss_raw, axis=1)
        
        mask = ~np.isnan(mean)
        if mask.any():
            ax2.plot(epochs_range[mask], mean[mask], label=f"{t_name} (CL)", color=color, linewidth=2)
        
        if t_name in expert_stats:
            estats = expert_stats[t_name]
            task_idx = task_names.index(t_name)
            start_epoch = task_idx * config.epochs_per_task
            expert_x = np.arange(start_epoch + 1, start_epoch + len(estats['loss_mean']) + 1)
            
            emean = estats['loss_mean']
            emask = ~np.isnan(emean)
            
            if emask.any():
                ax2.plot(expert_x[emask], emean[emask], color=color, linestyle=':', linewidth=2.5, alpha=0.8)
    
    ax2.set_ylabel('Loss (BCE)')
    ax2.set_xlabel('Total Epochs')
    ax2.set_title(f'{config.dataset_name} Loss (Solid=CL, Dotted=Expert)')
    ax2.grid(True, alpha=0.3)

    for boundary in task_boundaries[:-1]:
        ax1.axvline(x=boundary, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.axvline(x=boundary, color='black', linestyle='-', linewidth=1, alpha=0.5)

    plt.tight_layout()
    save_path = f'{config.figures_dir}/sl_{config.dataset_name}_{config.num_tasks}_tasks.png'
    plt.savefig(save_path)
    print(f"Plots saved to {save_path}")
    
    # 6. Analysis Pipeline
    analysis.run_analysis_pipeline(config)
    
if __name__ == "__main__":
    main()