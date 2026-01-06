import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
import argparse
import sys

import data_utils
import config as config_module
from learner import ContinualLearner
import expert_trainer
from hooks import CLHook
import plastic_analysis as plastic_analysis
import cl_analysis as cl_analysis  
import manifold_analysis


class LoggerHook(CLHook):
    def on_task_start(self, task, state):
        print(f"[Hook] Starting Task ID {task.task_id}")

def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning Single Run")
    parser.add_argument('--algorithm', type=str, default='SL', help='Algorithm name (e.g., SL, ER, AGEM)')
    # Add more args here if needed in the future (e.g., seed, dataset override)
    return parser.parse_args()

def setup_directories(config, algo_name):
    """
    Creates the folder structure:
    ./<algo_name>/<dataset>_<n_tasks>_tasks/
        ├── data/   (for .npy, .pkl, raw representations)
        └── plots/  (for .png images)
    """
    base_folder = f"{algo_name}/{config.dataset_name}_{config.num_tasks}_tasks"
    data_dir = os.path.join(base_folder, "data")
    plots_dir = os.path.join(base_folder, "plots")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Update config paths dynamically
    config.algorithm = algo_name
    config.reps_dir = data_dir
    config.figures_dir = plots_dir
    
    print(f"--- Directories Setup ---")
    print(f"Data Directory:  {config.reps_dir}")
    print(f"Plots Directory: {config.figures_dir}")
    
    return config

def main():
    # 1. Parse Args & Setup Config
    args = parse_args()
    config = config_module.get_config()
    config = setup_directories(config, args.algorithm)
    
    print(f"Algorithm: {config.algorithm}")
    print(f"Using device: {jax.devices()}")
    print(f"Configuration: Repeats={config.n_repeats}, LogFreq={config.log_frequency}")
    
    # 2. Setup Data
    train_tasks = data_utils.create_continual_tasks(config, split='train')
    test_streams = {}
    test_tasks = data_utils.create_continual_tasks(config, split='test')
    for t in test_tasks:
        test_streams[t.name] = t.load_data()

    data_utils.save_task_samples_grid(train_tasks, config)
    
    # 3. Setup Learner & History
    learner = ContinualLearner(config, hooks=[LoggerHook()])
    
    print("Saving random initialization weights...")
    init_weights = learner.get_flat_params(learner.state)
    np.save(os.path.join(config.reps_dir, "init_weights.npy"), np.array(init_weights))
    
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

    # 4. CL Loop
    for task in train_tasks:
        analysis_ds = task.load_mandi_subset(samples_per_class=config.mandi_samples)
        
        _, subset_lbls = learner.preload_data(analysis_ds)

        if subset_lbls.ndim > 1: subset_lbls = subset_lbls.flatten()
        np.save(os.path.join(config.reps_dir, f"{task.name}_labels.npy"), subset_lbls)

        # Train
        rep_history, w_history = learner.train_task(
            task, test_streams, global_history, analysis_subset=analysis_ds
        )
        
        # Save Raw Reps/Weights (Sparse)
        if rep_history is not None:
            np.save(os.path.join(config.reps_dir, f"{task.name}_reps_per_epoch.npy"), rep_history)

        if w_history is not None:
            np.save(os.path.join(config.reps_dir, f"{task.name}_weights_per_epoch.npy"), w_history)
        
        total_epochs += config.epochs_per_task
        task_boundaries.append(total_epochs)

    # 5. Expert Baselines
    expert_stats = {} 
    expert_histories = {} 
    
    print("\n--- Computing Expert Baselines ---")
    for task in train_tasks:
        _, _, _, _, exp_l, exp_acc = expert_trainer.train_single_expert(
            config, task, test_streams[task.name]
        )
        expert_histories[task.name] = {'loss': exp_l, 'acc': exp_acc}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stats = {
                'loss_mean': np.nanmean(exp_l, axis=1),
                'acc_mean': np.nanmean(exp_acc, axis=1)
            }
        expert_stats[task.name] = stats

    # 6. Save CL History Data (Full Un-averaged)
    cl_data_path = os.path.join(config.reps_dir, f"cl_history_{config.dataset_name}.pkl")
    cl_save_data = {
        'global_history': global_history,
        'expert_histories': expert_histories,
        'task_names': task_names,
        'task_boundaries': task_boundaries
    }
    with open(cl_data_path, 'wb') as f:
        pickle.dump(cl_save_data, f)
    print(f"CL History saved to {cl_data_path}")

    # 7. CL Metrics Analysis
    cl_met_results = cl_analysis.compute_and_log_cl_metrics(
        global_history, expert_histories, config
    )
    # Append computed metrics to the save file or save separately if needed. 
    # For now, we rely on the raw history saved above.

    # 8. Plotting (Acc/Loss)
    print("\nGenerating CL Plots...")
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
        'lines.linewidth': 2
    })
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    epochs_range = np.arange(1, total_epochs + 1)
    cmap = plt.get_cmap('tab10')
    color_dict = {name: cmap(i % 10) for i, name in enumerate(task_names)}

    def setup_axis(ax, ylabel):
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylabel(ylabel, fontweight='bold')
        for i, boundary in enumerate(task_boundaries[:-1]):
            ax.axvline(x=boundary, color='#333333', linestyle='-', alpha=0.3, linewidth=1.5)

    # Plot Acc
    train_acc_raw = np.array(global_history['train_acc']).reshape(-1, config.n_repeats)
    ax1.plot(epochs_range, np.mean(train_acc_raw, axis=1), color='grey', alpha=0.3)
    
    for t_name in task_names:
        metrics = global_history['test_metrics'][t_name]
        acc_raw = np.array(metrics['acc']).reshape(-1, config.n_repeats)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = np.nanmean(acc_raw, axis=1), np.nanstd(acc_raw, axis=1)
        
        mask = ~np.isnan(mean)
        if mask.any():
            ax1.plot(epochs_range[mask], mean[mask], label=f"{t_name} (CL)", color=color_dict[t_name])
            ax1.fill_between(epochs_range[mask], mean[mask]-std[mask], mean[mask]+std[mask], color=color_dict[t_name], alpha=0.1)
        
        if t_name in expert_stats:
            estats = expert_stats[t_name]
            task_idx = task_names.index(t_name)
            start_epoch = task_idx * config.epochs_per_task
            expert_x = np.arange(start_epoch + 1, start_epoch + len(estats['acc_mean']) + 1)
            emean = estats['acc_mean']
            emask = ~np.isnan(emean)
            if emask.any():
                ax1.plot(expert_x[emask], emean[emask], color=color_dict[t_name], linestyle='--', alpha=0.9)

    setup_axis(ax1, 'Accuracy')
    ax1.legend(bbox_to_anchor=(1.02, 0.5), loc='center left')

    # Plot Loss
    train_loss_raw = np.array(global_history['train_loss']).reshape(-1, config.n_repeats)
    ax2.plot(epochs_range, np.mean(train_loss_raw, axis=1), color='grey', alpha=0.3)
    
    for t_name in task_names:
        loss_raw = np.array(global_history['test_metrics'][t_name]['loss']).reshape(-1, config.n_repeats)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = np.nanmean(loss_raw, axis=1)
        mask = ~np.isnan(mean)
        if mask.any():
            ax2.plot(epochs_range[mask], mean[mask], color=color_dict[t_name])
            
    setup_axis(ax2, 'Loss')
    ax2.set_xlabel('Total Epochs', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.82, 1]) 
    
    plot_path = os.path.join(config.figures_dir, f"sl_{config.dataset_name}_{config.num_tasks}_tasks.png")
    plt.savefig(plot_path, dpi=150)
    print(f"CL plots saved to {plot_path}")
    
    # 9. Plasticine Analysis (Saves data internally in modified plastic_analysis.py)
    plastic_analysis.run_analysis_pipeline(config)

    # 10. Manifold Analysis
    # Returns structured dict: {task: {metric: array(steps, repeats)}}
    manifold_results = manifold_analysis.analyze_manifold_trajectory(config, task_names)
    
    if manifold_results:
        manifold_save_path = os.path.join(config.reps_dir, f"manifold_metrics_{config.dataset_name}.pkl")
        with open(manifold_save_path, 'wb') as f:
            pickle.dump(manifold_results, f)
        print(f"Manifold metrics saved to {manifold_save_path}")

    print("\nRun Complete.")
    
if __name__ == "__main__":
    main()