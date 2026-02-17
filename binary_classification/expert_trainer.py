from copy import deepcopy
import jax
import jax.numpy as jnp
import numpy as np
from learner import ContinualLearner
import pickle
import os
import data_utils


def train_single_expert(config, train_task, test_data):
    """
    Trains an expert model on a single task from scratch.
    
    Args:
        config: Configuration object
        train_task: Dict with 'data' key containing (train_X, train_Y) tuple
        test_data: Tuple (test_X, test_Y) in Canonical format
    
    Returns:
        loss_mean, loss_std, acc_mean, acc_std, test_losses, test_accs
    """
    task_name = train_task['name']
    print(f"\n--- Training Expert on {task_name} ---")
    
    # Initialize fresh learner
    learner = ContinualLearner(config)
    
    # Preload data (Canonical: Total, Repeats, Dim)
    train_imgs, train_lbls = learner.preload_data(train_task['data'])
    test_imgs, test_lbls = learner.preload_data(test_data)
    
    # Reshape for batching
    n_samples = train_imgs.shape[0]
    n_batches = n_samples // config.batch_size
    limit = n_batches * config.batch_size
    
    train_imgs = train_imgs[:limit]
    train_lbls = train_lbls[:limit]
    
    # (Limit, R, D) -> (Batches, Batch_Size, R, D) -> (Batches, R, Batch_Size, D)
    train_imgs_reshaped = train_imgs.reshape(
        n_batches, config.batch_size, config.n_repeats, -1
    )
    train_imgs_reshaped = jnp.swapaxes(train_imgs_reshaped, 1, 2)
    
    train_lbls_reshaped = train_lbls.reshape(
        n_batches, config.batch_size, config.n_repeats, -1
    )
    train_lbls_reshaped = jnp.swapaxes(train_lbls_reshaped, 1, 2)

    # JIT-compiled training loop
    @jax.jit
    def run_training_loop(state, t_imgs, t_lbls, test_i, test_l):
        """
        Scans over epochs, evaluating at log_frequency intervals.
        """
        def epoch_step(carry, epoch_idx):
            curr_state = carry
            
            # Train one epoch
            new_state, train_losses, train_accs = learner._train_epoch_jit(
                curr_state, t_imgs, t_lbls
            )
            
            # Conditionally evaluate
            is_eval_step = ((epoch_idx + 1) % config.log_frequency == 0)
            
            def true_eval_fn(s):
                return learner._eval_jit(s, test_i, test_l)
            
            def false_eval_fn(s):
                dummy_shape = train_losses.shape
                return (
                    jnp.full(dummy_shape, jnp.nan, dtype=train_losses.dtype),
                    jnp.full(dummy_shape, jnp.nan, dtype=train_accs.dtype)
                )
            
            test_losses, test_accs = jax.lax.cond(
                is_eval_step, true_eval_fn, false_eval_fn, new_state
            )
            
            metrics = (train_losses, train_accs, test_losses, test_accs)
            return new_state, metrics
        
        # Scan over all epochs
        epochs_range = jnp.arange(config.epochs_per_task)
        final_state, (tr_l, tr_a, te_l, te_a) = jax.lax.scan(
            epoch_step, state, epochs_range
        )
        
        return final_state, tr_l, tr_a, te_l, te_a

    # Execute
    print(f"  Compiling and training...")
    final_state, tr_l, tr_a, te_l, te_a = run_training_loop(
        learner.state,
        train_imgs_reshaped,
        train_lbls_reshaped,
        test_imgs,
        test_lbls
    )
    
    # Convert to numpy
    tr_l = np.array(tr_l)
    tr_a = np.array(tr_a)
    te_l = np.array(te_l)
    te_a = np.array(te_a)
    
    # Compute final statistics
    final_losses = te_l[-1]  # Last evaluation
    final_accs = te_a[-1]
    
    loss_mean = np.nanmean(final_losses)
    loss_std = np.nanstd(final_losses)
    acc_mean = np.nanmean(final_accs)
    acc_std = np.nanstd(final_accs)
    
    print(f"  Expert Final: Acc={acc_mean:.4f}±{acc_std:.4f}, Loss={loss_mean:.4f}±{loss_std:.4f}")
    
    return loss_mean, loss_std, acc_mean, acc_std, te_l, te_a



def run_random_baseline(config, test_data_dict, analysis_subset):
    """
    Evaluates a randomly initialized network (no training).
    Saves representations and metrics to results/.../random/
    """
    print(f"\n{'='*40}\nRunning Random Baseline\n{'='*40}")
    
    # 1. Initialize Learner (Random Weights)
    learner = ContinualLearner(config)
    
    # 2. Setup Output Directory
    save_dir = os.path.join(config.results_dir, "random")
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. Extract Representations (Analysis Subset)
    # analysis_subset is (Images, Labels)
    if analysis_subset is not None:
        print("  Extracting random representations...")
        # shape: (Repeats, Samples, Hidden)
        reps = learner._extract_features_jit(learner.state, analysis_subset[0])
        np.save(os.path.join(save_dir, "representations.npy"), reps)
        
    # 4. Evaluate on All Tasks
    print("  Evaluating on test sets...")
    metrics = {'acc': {}, 'loss': {}}
    
    for task_name, (t_imgs, t_lbls) in test_data_dict.items():
        loss, acc = learner._eval_jit(learner.state, t_imgs, t_lbls)
        # Convert to numpy and store
        metrics['loss'][task_name] = np.array(loss) # (Repeats,)
        metrics['acc'][task_name] = np.array(acc)   # (Repeats,)
        
    # 5. Save Metrics
    with open(os.path.join(save_dir, "metrics.pkl"), 'wb') as f:
        pickle.dump(metrics, f)
        
    print(f"  Random baseline saved to {save_dir}")



def train_multitask(config, task_class_pairs, X_global, Y_global, test_data_dict, analysis_subset):
    """
    Trains Multi-Task Learner and saves representations/labels in a format
    compatible with Plasticity and GLUE analysis pipelines.
    """
    print(f"\n{'='*40}\nRunning Multi-Task Learning (Upper Bound)\n{'='*40}")
    
    # 1. Aggregate Data
    all_images = []
    all_labels = []
    
    print("  Aggregating data from all tasks...")
    for t in range(config.num_tasks):
        t_X, t_Y, _ = data_utils.create_single_task_data(
            t, task_class_pairs, X_global, Y_global, config, split='train'
        )
        all_images.append(t_X)
        all_labels.append(t_Y)
        
    mtl_X = jnp.concatenate(all_images, axis=0)
    mtl_Y = jnp.concatenate(all_labels, axis=0)
    
    # 2. Shuffle
    rng_mtl = np.random.default_rng(config.seed)
    perm = rng_mtl.permutation(mtl_X.shape[0])
    mtl_X = mtl_X[perm]
    mtl_Y = mtl_Y[perm]
    
    # 3. Configure
    mtl_config = deepcopy(config)
    mtl_config.epochs_per_task = config.epochs_per_task * config.num_tasks
    
    # 4. Train
    learner = ContinualLearner(mtl_config)
    mtl_task = {'name': 'multi_task_joint', 'data': (mtl_X, mtl_Y)}
    
    mtl_history = {
        'train_acc': [], 'train_loss': [],
        'test_metrics': {t: {'acc': [], 'loss': []} for t in test_data_dict}
    }
    
    print(f"  Training for {mtl_config.epochs_per_task} epochs...")
    # rep_history: (L, R, Total_Samples, H)
    rep_history, weight_history = learner.train_task(
        mtl_task, test_data_dict, mtl_history, analysis_subset=analysis_subset
    )
    
    # 5. Save Results (Formatted for Analysis)
    save_dir = os.path.join(config.results_dir, "multitask")
    os.makedirs(save_dir, exist_ok=True)
    
    # A. Reshape Representations: (L, R, T*S, H) -> (L, R, T, S, H)
    # This enables per-task GLUE analysis
    if rep_history is not None:
        L, R, Total_S, H = rep_history.shape
        T = config.num_tasks
        S = config.analysis_subsamples
        
        # Ensure exact match before reshaping
        if Total_S == T * S:
            reps_reshaped = rep_history.reshape(L, R, T, S, H)
            np.save(os.path.join(save_dir, "representations.npy"), reps_reshaped)
            print(f"  Saved representations: {reps_reshaped.shape}")
        else:
            print(f"  WARNING: shape mismatch. Expected {T*S} samples, got {Total_S}. Saving raw.")
            np.save(os.path.join(save_dir, "representations.npy"), rep_history)

    # B. Save Weights
    np.save(os.path.join(save_dir, "weights.npy"), weight_history)
    
    # C. Save Metrics
    with open(os.path.join(save_dir, "metrics.pkl"), 'wb') as f:
        pickle.dump(mtl_history, f)
        
    # D. Save Analysis Labels (Critical for GLUE)
    # analysis_subset[1] is (Total_Samples, R, 1) -> (T*S, R)
    # We need (R, T, S) to match the GLUE pipeline expectation
    if analysis_subset is not None:
        analysis_Y = analysis_subset[1]
        lbls_flat = analysis_Y.squeeze(-1) # (T*S, R)
        # Reshape: (T, S, R) -> Transpose: (R, T, S)
        lbls_reshaped = lbls_flat.reshape(T, S, R).transpose(2, 0, 1)
        np.save(os.path.join(save_dir, "binary_labels.npy"), np.array(lbls_reshaped))
        
    print(f"  Multi-Task results saved to {save_dir}")