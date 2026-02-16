import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
import argparse
from ipdb import set_trace

import data_utils
import config as config_module
from learner import ContinualLearner
from expert_trainer import train_single_expert, run_random_baseline, train_multitask
from models import CLHook
import plastic_analysis
import cl_analysis
import glue_module.glue_module.glue_analysis as glue_analysis


class LoggerHook(CLHook):
    def on_task_start(self, task, state):
        print(f"[Hook] Starting Task {task['name']}")

    def on_task_end(self, task, state, metrics):
        print(f"[Hook] Finished Task {task['name']}")


def main():
    # --- 1. CLI Argument Parsing ---
    parser = argparse.ArgumentParser(description="Continual Learning Single Run")
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=['kmnist', 'mnist', 'fashion_mnist', 'emnist'],
        help="Dataset to use"
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='SL',
        choices=['SL', 'RL'],
        help="Learning Algorithm"
    )
    args = parser.parse_args()

    # --- 2. Load Config ---
    config = config_module.get_config(args.dataset, args.algorithm)
    
    print(f"\n{'='*60}")
    print(f"Continual Learning Experiment")
    print(f"{'='*60}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Device: {jax.devices()}")
    print(f"Tasks: {config.num_tasks}, Repeats: {config.n_repeats}")
    print(f"Log Frequency: {config.log_frequency}")
    print(f"Results Directory: {config.results_dir}")
    print(f"{'='*40}\n")
    
    # Create directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)

    # --- 3. PHASE 1: Pre-compute Class Pairs (Spec Section 2) ---
    num_classes = data_utils.DATASET_CONFIGS[config.dataset_name]['num_classes']
    task_class_pairs = data_utils.generate_task_class_pairs(
        config.num_tasks,
        config.n_repeats,
        num_classes,
        config.seed
    )

    # --- 4. PHASE 2: Load Global Datasets (Once) ---
    print(f"\n=== Loading Global Datasets ===")
    X_train_global, Y_train_global = data_utils.get_base_data_jax(
        config.dataset_name,
        config.data_dir,
        train=True,
        img_size=config.downsample_dim
    )
    X_test_global, Y_test_global = data_utils.get_base_data_jax(
        config.dataset_name,
        config.data_dir,
        train=False,
        img_size=config.downsample_dim
    )

    # --- 5. PHASE 3: Pre-load ALL Test Data (Spec Section 2) ---
    test_data_dict = data_utils.preload_all_test_data(
        task_class_pairs, X_test_global, Y_test_global, config
    )

    # --- 6. Visualize Tasks ---
    data_utils.save_task_samples_grid(
        task_class_pairs, X_train_global, Y_train_global, config
    )

    # --- 7. Setup Learner ---
    learner = ContinualLearner(config, hooks=[LoggerHook()])
    
    print("\nSaving random initialization weights...")
    init_weights = learner.get_flat_params(learner.state)
    init_save_path = os.path.join(config.results_dir, "init_weights.npy")
    np.save(init_save_path, np.array(init_weights))

    # Initialize RNG for subsampling
    rng = np.random.default_rng(config.seed)

    # Global history tracking
    global_history = {
        'train_acc': [],
        'train_loss': [],
        'test_metrics': {
            f"task_{t:03d}": {'acc': [], 'loss': []}
            for t in range(config.num_tasks)
        }
    }

    task_boundaries = []
    total_epochs = 0

    # --- 8. PHASE 4: Continual Learning Loop with Integrated Expert ---
    print(f"\n{'='*60}")
    print(f"Starting Continual Learning & Expert Training")
    print(f"{'='*60}\n")
    
    # Pre-allocate expert history storage for the global plotter
    expert_histories = {} 
    expert_stats = {}

    for task_idx in range(config.num_tasks):
        task_name = f"task_{task_idx:03d}"
        
        print(f"\n>>> TASK {task_idx + 1}/{config.num_tasks}: {task_name} <<<")
        
        # ---------------------------------------------------------
        # 1. LAZY LOAD: Generate training data for this task
        # ---------------------------------------------------------
        print(f"  [Lazy Loading] Generating training data for {task_name}...")
        train_X, train_Y, _ = data_utils.create_single_task_data(
            task_idx, task_class_pairs, X_train_global, Y_train_global, config, split='train'
        )
        
        # Create task dict
        task = {
            'id': task_idx,
            'name': task_name,
            'data': (train_X, train_Y), # Canonical: (N, R, D)
            'n_samples': train_X.shape[0]
        }
        
        # ---------------------------------------------------------
        # 2. PREPARE ANALYSIS SUBSET (ALL TASKS)
        # ---------------------------------------------------------
        # Spec Requirement: "subsamples from every task's test set"
        # We stack subsamples from Task 0...Task T to track representation drift
        
        analysis_X_list = []
        analysis_Y_list = []
        
        # Iterate over all tasks (not just current)
        for t in range(config.num_tasks):
            t_name_iter = f"task_{t:03d}"
            t_test_X, t_test_Y = test_data_dict[t_name_iter]
            
            n_total = t_test_X.shape[0]
            n_sub = min(n_total, config.analysis_subsamples)
            
            # Use sorted indices for consistency across time steps
            indices = rng.choice(n_total, size=n_sub, replace=False)
            indices.sort()
            
            analysis_X_list.append(t_test_X[indices])
            analysis_Y_list.append(t_test_Y[indices])
            
        # Concatenate: (Num_Tasks * S, R, D)
        analysis_data_X = jnp.concatenate(analysis_X_list, axis=0)
        analysis_data_Y = jnp.concatenate(analysis_Y_list, axis=0)
        analysis_subset = (analysis_data_X, analysis_data_Y)
        
        # Save analysis labels (reshaped for clarity: Num_Tasks, Subsamples, R)
        task_dir = os.path.join(config.results_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        analysis_lbls_reshaped = analysis_data_Y.squeeze(-1) # (T*S, R)
        analysis_lbls_reshaped = analysis_lbls_reshaped.reshape(config.num_tasks, config.analysis_subsamples, config.n_repeats).transpose(2,0,1) # match repres: (R,T,S)
        np.save(os.path.join(task_dir, "binary_labels.npy"), np.array(analysis_lbls_reshaped))
        
        # Get Random Performance
        run_random_baseline(config, test_data_dict, analysis_subset)

        # -------- -------------------------------------------------
        # 3. TRAIN LEARNER
        # ---------------------------------------------------------
        #
        # Returns rep_history: (L, Repeats, Total_Samples, Hidden)
        rep_history, weight_history = learner.train_task(
            task, test_data_dict, global_history, analysis_subset=analysis_subset
        )
        # L = logged epochs
        # Reshape Representations to match Spec: (L, Repeats, N_Eval_Tasks, N_Subsamples, Hidden_Dim)
        if rep_history is not None:
            L, R, SxT_eval, H = rep_history.shape
            T_eval = config.num_tasks
            S = config.analysis_subsamples
            
            # Reshape logic: The samples were stacked [Task0, Task1...], so we split that dim
            rep_reshaped = rep_history.reshape(L, R, T_eval, S, H)
            np.save(os.path.join(task_dir, "representations.npy"), rep_reshaped)
        
        if weight_history is not None:
            np.save(os.path.join(task_dir, "weights.npy"), weight_history)
        
        # Save Learner Metrics
        learner_metrics = {
            'train_acc': global_history['train_acc'][-config.epochs_per_task:],
            'train_loss': global_history['train_loss'][-config.epochs_per_task:],
            'test_acc': global_history['test_metrics'][task_name]['acc'][-config.epochs_per_task:],
            'test_loss': global_history['test_metrics'][task_name]['loss'][-config.epochs_per_task:]
        }
        with open(os.path.join(task_dir, "metrics.pkl"), 'wb') as f:
            pickle.dump(learner_metrics, f)
            
        data_utils.save_task_metadata(task_idx, task_class_pairs, config)

        # ---------------------------------------------------------
        # 4. TRAIN EXPERT (Integrated)
        # ---------------------------------------------------------
        # Uses the exact same train_X/train_Y currently in memory
        
        expert_task_wrapper = {'name': task_name, 'data': (train_X, train_Y)}
        expert_test_data = test_data_dict[task_name]
        
        # Returns: loss_mean, loss_std, acc_mean, acc_std, test_losses, test_accs
        _, _, _, _, exp_loss, exp_acc = train_single_expert(config, expert_task_wrapper, expert_test_data)
        
        # Save Expert Metrics locally
        expert_metrics_dict = {
            'test_loss': exp_loss, # (Epochs, Repeats)
            'test_acc': exp_acc    # (Epochs, Repeats)
        }
        with open(os.path.join(task_dir, "expert_metrics.pkl"), 'wb') as f:
            pickle.dump(expert_metrics_dict, f)
            
        # Update global stats for plotting at the end
        expert_histories[task_name] = {'loss': exp_loss, 'acc': exp_acc}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            expert_stats[task_name] = {
                'loss_mean': np.nanmean(exp_loss, axis=1),
                'acc_mean': np.nanmean(exp_acc, axis=1)
            }

        # ---------------------------------------------------------
        # 5. CLEANUP
        # ---------------------------------------------------------
        del train_X, train_Y, task, expert_task_wrapper, analysis_data_X, analysis_data_Y
        
        total_epochs += config.epochs_per_task
        task_boundaries.append(total_epochs)
        
        print(f"  [Memory] Training data for {task_name} cleared")

    learner.clear_test_cache()
    

    # ---------------------------------------------------------
    # Train Multi-Task Learner
    # ---------------------------------------------------------
    # Uses the global training data and the analysis subset created earlier
    train_multitask(
        config, 
        task_class_pairs, 
        X_train_global, 
        Y_train_global, 
        test_data_dict, 
        analysis_subset
    )
    # ---------------------------------------------------------
    # CL Metrics
    # ---------------------------------------------------------
    cl_results = cl_analysis.compute_and_log_cl_metrics(
        global_history, expert_histories, config
    )
    
    cl_save_path = os.path.join(config.results_dir, "cl_metrics.pkl")
    with open(cl_save_path, 'wb') as f:
        pickle.dump(cl_results, f)
    print(f"\nCL metrics saved to {cl_save_path}")

    # Save global history
    history_save_path = os.path.join(config.results_dir, "global_history.pkl")
    with open(history_save_path, 'wb') as f:
        pickle.dump(global_history, f)
    print(f"Global history saved to {history_save_path}")

    # --- 10. Plotting ---
    print(f"\n{'='*60}")
    print(f"Generating Plots")
    print(f"{'='*60}\n")
    
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
    task_names = [f"task_{t:03d}" for t in range(config.num_tasks)]
    
    cmap = plt.get_cmap('tab10')
    color_dict = {name: cmap(i % 10) for i, name in enumerate(task_names)}

    def setup_axis(ax, ylabel):
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylabel(ylabel, fontweight='bold')
        for i, boundary in enumerate(task_boundaries[:-1]):
            ax.axvline(x=boundary, color='#333333', linestyle='-', alpha=0.3, linewidth=1.5)
            if i < len(task_names) - 1:
                ax.text(boundary + (total_epochs*0.01), ax.get_ylim()[0], f"T{i+1}",
                        rotation=90, verticalalignment='bottom', fontsize=9, color='#555555')

    # Plot Accuracy
    train_acc = np.array(global_history['train_acc'])
    if train_acc.ndim == 1:
        train_acc = train_acc.reshape(-1, config.n_repeats)
    
    train_mean = np.mean(train_acc, axis=1)
    ax1.plot(epochs_range, train_mean, label='Current Task', color='grey', linestyle='-', alpha=0.3)
    
    for task_name in task_names:
        color = color_dict[task_name]
        acc = np.array(global_history['test_metrics'][task_name]['acc'])
        if acc.ndim == 1:
            acc = acc.reshape(-1, config.n_repeats)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(acc, axis=1)
            std = np.nanstd(acc, axis=1)
        
        mask = ~np.isnan(mean)
        if mask.any():
            ax1.plot(epochs_range[mask], mean[mask], label=f"{task_name} (CL)", color=color, linewidth=2.5)
            ax1.fill_between(epochs_range[mask], mean[mask] - std[mask], mean[mask] + std[mask], color=color, alpha=0.1)
        
        if task_name in expert_stats:
            task_idx = task_names.index(task_name)
            start_epoch = task_idx * config.epochs_per_task
            expert_x = np.arange(start_epoch + 1, start_epoch + len(expert_stats[task_name]['acc_mean']) + 1)
            emean = expert_stats[task_name]['acc_mean']
            emask = ~np.isnan(emean)
            if emask.any():
                ax1.plot(expert_x[emask], emean[emask], color=color, linestyle='--', linewidth=2.0, alpha=0.9, label=f"{task_name} (Expert)")

    setup_axis(ax1, 'Accuracy')
    ax1.set_ylim(-0.05, 1.05)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5))

    # Plot Loss
    train_loss = np.array(global_history['train_loss'])
    if train_loss.ndim == 1:
        train_loss = train_loss.reshape(-1, config.n_repeats)
    loss_mean = np.mean(train_loss, axis=1)
    ax2.plot(epochs_range, loss_mean, label='Current Task', color='grey', linestyle='-', alpha=0.3)
    
    for task_name in task_names:
        color = color_dict[task_name]
        loss = np.array(global_history['test_metrics'][task_name]['loss'])
        if loss.ndim == 1:
            loss = loss.reshape(-1, config.n_repeats)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(loss, axis=1)
        
        mask = ~np.isnan(mean)
        if mask.any():
            ax2.plot(epochs_range[mask], mean[mask], label=task_name, color=color, linewidth=2.5)
        
        if task_name in expert_stats:
            task_idx = task_names.index(task_name)
            start_epoch = task_idx * config.epochs_per_task
            expert_x = np.arange(start_epoch + 1, start_epoch + len(expert_stats[task_name]['loss_mean']) + 1)
            emean = expert_stats[task_name]['loss_mean']
            emask = ~np.isnan(emean)
            if emask.any():
                ax2.plot(expert_x[emask], emean[emask], color=color, linestyle='--', linewidth=2.0, alpha=0.9)

    setup_axis(ax2, 'Loss')
    ax2.set_xlabel('Epochs', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    
    plot_path = os.path.join(config.figures_dir, f'{config.algorithm}_{config.dataset_name}_{config.num_tasks}tasks.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Performance plot saved to {plot_path}")
    plt.close()

    # --- 12. Analysis Pipelines ---
    if False:
        print(f"\n{'='*60}")
        print(f"Running Analysis Pipelines")
        print(f"{'='*60}\n")
        
        plastic_analysis.run_analysis_pipeline(config)
        
        manifold_results = glue_analysis.analyze_manifold_trajectory(config, task_names)
        if manifold_results:
            manifold_path = os.path.join(config.results_dir, "manifold_metrics.pkl")
            with open(manifold_path, 'wb') as f:
                pickle.dump(manifold_results, f)
            print(f"Manifold metrics saved to {manifold_path}")
        
        print(f"\n{'='*60}")
        print(f"Experiment Complete!")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()