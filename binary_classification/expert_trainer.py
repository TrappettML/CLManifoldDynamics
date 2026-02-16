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
    Aggregates ALL task data into one joint dataset and trains a Multi-Task learner.
    Saves representations and metrics to results/.../multitask/
    """
    print(f"\n{'='*40}\nRunning Multi-Task Learning (Upper Bound)\n{'='*40}")
    
    # 1. Aggregate Data from ALL Tasks
    # We use create_single_task_data to ensure EXACT sample/class identity with CL run
    all_images = []
    all_labels = []
    
    print("  Aggregating data from all tasks...")
    for t in range(config.num_tasks):
        # "train" split ensures we get the training data
        t_X, t_Y, _ = data_utils.create_single_task_data(
            t, task_class_pairs, X_global, Y_global, config, split='train'
        )
        all_images.append(t_X)
        all_labels.append(t_Y)
        
    # Concatenate: (Total_Samples, R, D)
    mtl_X = jnp.concatenate(all_images, axis=0)
    mtl_Y = jnp.concatenate(all_labels, axis=0)
    
    # 2. Global Shuffle
    # Essential for MTL: shuffle samples so tasks are mixed in every batch
    rng_mtl = np.random.default_rng(config.seed)
    perm = rng_mtl.permutation(mtl_X.shape[0])
    
    mtl_X = mtl_X[perm]
    mtl_Y = mtl_Y[perm]
    
    print(f"  Joint Dataset Shape: {mtl_X.shape}")
    
    # 3. Configure Training
    # Scale epochs so MTL gets roughly same total updates as CL (optional, but fair)
    # We modify the config strictly for this learner instance
    mtl_config = config
    mtl_config.epochs_per_task = config.epochs_per_task * config.num_tasks
    
    # 4. Initialize & Train
    learner = ContinualLearner(mtl_config)
    
    # We wrap the joint data as a single "task" named 'multi_task'
    mtl_task = {'name': 'multi_task_joint', 'data': (mtl_X, mtl_Y)}
    
    # Placeholder history to capture the training curve
    mtl_history = {
        'train_acc': [], 
        'train_loss': [],
        'test_metrics': {t: {'acc': [], 'loss': []} for t in test_data_dict}
    }
    
    # Reuse train_task to get representations over time
    # This returns (Time, Repeats, Samples, Hidden)
    print(f"  Training for {mtl_config.epochs_per_task} epochs...")
    reps, weights = learner.train_task(
        mtl_task, 
        test_data_dict, 
        mtl_history, 
        analysis_subset=analysis_subset
    )
    
    # 5. Save Results
    save_dir = os.path.join(config.results_dir, "multitask")
    os.makedirs(save_dir, exist_ok=True)
    
    if reps is not None:
        np.save(os.path.join(save_dir, "representations.npy"), reps)
    
    # We don't necessarily need the massive weight history for MTL, but you can save if needed
    # np.save(os.path.join(save_dir, "weights.npy"), weights)
    
    with open(os.path.join(save_dir, "metrics.pkl"), 'wb') as f:
        pickle.dump(mtl_history, f)
        
    print(f"  Multi-Task results saved to {save_dir}")