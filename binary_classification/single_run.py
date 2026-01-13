import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
import argparse 

import data_utils
import config as config_module
from learner import ContinualLearner
from expert_trainer import train_single_expert
from models import CLHook
import plastic_analysis
import cl_analysis 
import glue_analysis


class LoggerHook(CLHook):
    def on_task_start(self, task, state):
        print(f"[Hook] Starting Task ID {task.task_id}")

def main():
    # --- 1. CLI Argument Parsing ---
    parser = argparse.ArgumentParser(description="Continual Learning Single Run")
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='mnist', 
        choices=['kmnist', 'mnist', 'fashion_mnist', 'emnist'],
        help="Dataset to use: kmnist, mnist, fashion_mnist, or cifar100"
    )
    parser.add_argument(
        '--algorithm', 
        type=str, 
        default='SL', 
        choices=['SL', 'RL'],
        help="Learning Algorithm: one of SL or RL, for now"
    )
    args = parser.parse_args()

    # --- 2. Load Config with Dataset ---
    config = config_module.get_config(args.dataset, args.algorithm)
    
    print(f"Algorithm: {config.algorithm}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Using device: {jax.devices()}")
    print(f"Configuration Loaded (Repeats={config.n_repeats}, LogFreq={config.log_frequency})")
    print(f"Directories:\n  Plots: {config.figures_dir}\n  Data:  {config.reps_dir}")
    
    # Create the new directory structure
    os.makedirs(config.reps_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)

    # --- 3. Setup Data ---
    train_tasks = data_utils.create_continual_tasks(config, split='train')
    test_streams = {}
    test_tasks = data_utils.create_continual_tasks(config, split='test')
    
    for t in test_tasks:
        # Optimization: Pre-load the data into JAX arrays ONCE.
        # Store the Tensors, NOT the generator.
        test_streams[t.name] = t.get_full_data()

    data_utils.save_task_samples_grid(train_tasks, config)
    
    # --- 4. Setup Learner & History ---
    learner = ContinualLearner(config, hooks=[LoggerHook()])
    
    print("Saving random initialization weights...")
    init_weights = learner.get_flat_params(learner.state)
    np.save(f"{config.reps_dir}/init_weights.npy", np.array(init_weights))
    
    # Initialize RNG for consistent subsampling
    rng = np.random.default_rng(config.seed)

    global_history = {
        'train_acc': [], 'train_loss': [],
        'test_metrics': {
            name: {'acc': [], 'loss': []} 
            for name in test_streams.keys()
        }
    }
    
    task_boundaries = []
    total_epochs = 0
    task_names = [t.name for t in train_tasks] 

    # --- 5. CL Loop ---
    for task in train_tasks:
        # Get full data: (Total, Repeats, Dim)
        full_imgs, full_lbls = test_streams[task.name]
        
        # Subsample for analysis to save memory/storage
        n_total = full_imgs.shape[0]
        n_sub = min(n_total, config.analysis_subsamples)
        
        # Randomly select indices (consistent across repeats/time)
        indices = rng.choice(n_total, size=n_sub, replace=False)
        indices.sort() # Sort for tidiness
        
        # Slice the data
        sub_imgs = full_imgs[indices]
        sub_lbls = full_lbls[indices]
        
        analysis_data = (sub_imgs, sub_lbls)
        
        # Save labels for this task (Shape: N_sub, Repeats)
        subset_lbls = sub_lbls.squeeze(-1)
        np.save(f"{config.reps_dir}/{task.name}_labels.npy", np.array(subset_lbls))

        # Train: Pass subsampled analysis_data
        rep_history, w_history = learner.train_task(
            task, test_streams, global_history, analysis_subset=analysis_data
        )
        
        # Save History (Sparse)
        if rep_history is not None:
            np.save(f"{config.reps_dir}/{task.name}_reps_per_epoch.npy", rep_history)

        if w_history is not None:
            np.save(f"{config.reps_dir}/{task.name}_weights_per_epoch.npy", w_history)
        
        total_epochs += config.epochs_per_task
        task_boundaries.append(total_epochs)

    learner.clear_test_cache()

    # --- 6. Expert Baselines ---
    expert_stats = {} 
    expert_histories = {} 
    
    print("\n--- Computing Expert Baselines ---")
    for task in train_tasks:
        _, _, _, _, exp_l, exp_acc = train_single_expert(
            config, task, test_streams[task.name]
        )
        
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

    # --- 6a. CL metrics ---
    # Computes Transfer and Forgetting, prints results
    cl_met_results = cl_analysis.compute_and_log_cl_metrics(
        global_history, expert_histories, config
    )

    cl_save_path = os.path.join(config.reps_dir, f"cl_metrics_{config.dataset_name}.pkl")
    with open(cl_save_path, 'wb') as f:
        pickle.dump(cl_met_results, f)
    print(f"CL Metrics saved to {cl_save_path}")

    # --- 6b. Save raw loss/acc
    history_save_path = os.path.join(config.reps_dir, f"global_history_{config.dataset_name}.pkl")
    with open(history_save_path, 'wb') as f:
        pickle.dump(global_history, f)
    print(f"Global history saved to {history_save_path}")

    # --- 7. Plotting ---
    print("\nGenerating Plots...")
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'lines.linewidth': 2
    })
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    epochs_range = np.arange(1, total_epochs + 1)
    task_names = [t.name for t in train_tasks]
    
    cmap = plt.get_cmap('tab10')
    color_dict = {name: cmap(i % 10) for i, name in enumerate(task_names)}

    def setup_axis(ax, ylabel):
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylabel(ylabel, fontweight='bold')
        for i, boundary in enumerate(task_boundaries[:-1]):
            ax.axvline(x=boundary, color='#333333', linestyle='-', alpha=0.3, linewidth=1.5)
            if i < len(task_names) - 1:
                ax.text(boundary + (total_epochs*0.01), ax.get_ylim()[0], f"End T{i+1}", 
                        rotation=90, verticalalignment='bottom', fontsize=9, color='#555555')

    # Plot 1: Accuracy
    train_acc_raw = np.array(global_history['train_acc'])
    if train_acc_raw.ndim == 1: train_acc_raw = train_acc_raw.reshape(-1, config.n_repeats)

    train_mean = np.mean(train_acc_raw, axis=1)
    ax1.plot(epochs_range, train_mean, label='Current Task (Train)', color='grey', linestyle='-', alpha=0.3, linewidth=1)
    
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
            ax1.plot(epochs_range[mask], mean[mask], label=f"{t_name} (CL)", color=color, linewidth=2.5)
            ax1.fill_between(epochs_range[mask], mean[mask] - std[mask], mean[mask] + std[mask], color=color, alpha=0.1)
        
        if t_name in expert_stats:
            estats = expert_stats[t_name]
            task_idx = task_names.index(t_name)
            start_epoch = task_idx * config.epochs_per_task
            expert_x = np.arange(start_epoch + 1, start_epoch + len(estats['acc_mean']) + 1)
            emean = estats['acc_mean']
            emask = ~np.isnan(emean)
            
            if emask.any():
                ax1.plot(expert_x[emask], emean[emask], color=color, linestyle='--', linewidth=2.0, alpha=0.9, 
                         label=f"{t_name} (Expert)")

    setup_axis(ax1, 'Accuracy')
    ax1.set_ylim(-0.05, 1.05)
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), title="Legend")

    # Plot 2: Loss
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
    
    plt.tight_layout(rect=[0, 0, 0.82, 1]) 
    
    # Save the main performance plot to the PLOTS directory
    save_path = os.path.join(config.figures_dir, f'{config.algorithm}_{config.dataset_name}_{config.num_tasks}_tasks.png')
    plt.savefig(save_path, dpi=150)
    print(f"Improved plots saved to {save_path}")
    
    # --- 8. Plasticity Analysis ---
    # Will save .pkl to /data/ and .png to /plots/
    plastic_analysis.run_analysis_pipeline(config)
    # --- 9. Manifold Analysis ---
    # Will save .pkl to /data/ and .png to /plots/
    manifold_results = glue_analysis.analyze_manifold_trajectory(config, task_names)
    
    if manifold_results:
        manifold_save_path = os.path.join(config.reps_dir, f"manifold_metrics_full_{config.dataset_name}.pkl")
        with open(manifold_save_path, 'wb') as f:
            pickle.dump(manifold_results, f)
        print(f"Full un-averaged manifold metrics saved to {manifold_save_path}")
    
    print("\nDone.")
    
if __name__ == "__main__":
    main()