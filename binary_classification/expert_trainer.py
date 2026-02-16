import jax
import jax.numpy as jnp
import numpy as np
from learner import ContinualLearner


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


def train_multi_task_learner(config, all_data):
    """
    Take the entire data set to shuffle all data for a multi-task learning set up
    If we have two tasks, then 2 class for each binary classification, 3 task=3 classes. 
    We want to capture hidden representations in the same way as our CL method and expert. 
    """

    return 