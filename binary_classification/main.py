import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
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
    tf.config.set_visible_devices([], 'GPU')
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
    
    # --- SAVE INITIALIZATION WEIGHTS (Reference for "Distance from Init") ---
    print("Saving random initialization weights...")
    init_weights = learner.get_flat_params(learner.state)
    np.save(f"{config.reps_dir}/init_weights.npy", np.array(init_weights))
    # ------------------------------------------------------------------------

    global_history = {
        'train_acc_mean': [], 'train_acc_std': [],
        'train_loss_mean': [], 'train_loss_std': [],
        'test_metrics': {
            name: {'acc_mean': [], 'acc_std': [], 'loss_mean': [], 'loss_std': []} 
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

        # Train Task and capture representations AND weights
        rep_history, w_history = learner.train_task(
            task, test_streams, global_history, analysis_subset=analysis_ds
        )
        
        # Save Representations
        rep_stack = np.stack(rep_history) 
        np.save(f"{config.reps_dir}/{task.name}_reps_per_epoch.npy", rep_stack)

        # Save Weights
        w_stack = np.stack(w_history)
        np.save(f"{config.reps_dir}/{task.name}_weights_per_epoch.npy", w_stack)
        
        total_epochs += config.epochs_per_task
        task_boundaries.append(total_epochs)

    # 4. Expert Baselines
    expert_stats = {} 
    print("\n--- Computing Expert Baselines ---")
    for task in train_tasks:
        l_mean, l_std, a_mean, a_std, exp_l, exp_acc = expert_trainer.train_single_expert(config, task, test_streams[task.name])
        stats = {
            'loss_mean': np.mean(exp_l, axis=1),
            'loss_std': np.std(exp_l, axis=1),
            'acc_mean': np.mean(exp_acc, axis=1),
            'acc_std': np.std(exp_acc, axis=1)
        }
        expert_stats[task.name] = stats

    # 5. Plotting Standard Metrics (Acc/Loss)
    print("\nGenerating Plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    epochs_range = range(1, total_epochs + 1)
    
    task_names = [t.name for t in train_tasks]
    colormap = plt.cm.jet(np.linspace(0, 1, len(task_names)))
    color_dict = {name: color for name, color in zip(task_names, colormap)}

    # --- Plot 1: Accuracies ---
    train_mean = np.array(global_history['train_acc_mean'])
    train_std = np.array(global_history['train_acc_std'])
    ax1.plot(epochs_range, train_mean, label='Current Train Acc', color='grey', linestyle='--', alpha=0.4)
    ax1.fill_between(epochs_range, train_mean - train_std, train_mean + train_std, color='grey', alpha=0.2)
    
    for t_name in task_names:
        metrics = global_history['test_metrics'][t_name]
        color = color_dict[t_name]
        mean = np.array(metrics['acc_mean'])
        std = np.array(metrics['acc_std'])
        ax1.plot(epochs_range, mean, label=f"{t_name} (CL)", color=color, linewidth=2)
        ax1.fill_between(epochs_range, mean - std, mean + std, color=color, alpha=0.1)
        
        if t_name in expert_stats:
            estats = expert_stats[t_name]
            task_idx = task_names.index(t_name)
            start_epoch = task_idx * config.epochs_per_task
            expert_x = range(start_epoch + 1, start_epoch + len(estats['acc_mean']) + 1)
            ax1.plot(expert_x, estats['acc_mean'], color=color, linestyle=':', linewidth=2.5, alpha=0.8)
            ax1.scatter(expert_x[-1], estats['acc_mean'][-1], color=color, marker='*', s=60, zorder=5)

    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{config.dataset_name} Accuracy (Solid=CL, Dotted=Expert)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # --- Plot 2: Losses ---
    loss_mean = np.array(global_history['train_loss_mean'])
    ax2.plot(epochs_range, loss_mean, label='Current Train Loss', color='grey', linestyle='--', alpha=0.4)
    
    for t_name in task_names:
        metrics = global_history['test_metrics'][t_name]
        color = color_dict[t_name]
        mean = np.array(metrics['loss_mean'])
        ax2.plot(epochs_range, mean, label=f"{t_name} (CL)", color=color, linewidth=2)
        if t_name in expert_stats:
            estats = expert_stats[t_name]
            task_idx = task_names.index(t_name)
            start_epoch = task_idx * config.epochs_per_task
            expert_x = range(start_epoch + 1, start_epoch + len(estats['loss_mean']) + 1)
            ax2.plot(expert_x, estats['loss_mean'], color=color, linestyle=':', linewidth=2.5, alpha=0.8)
    
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