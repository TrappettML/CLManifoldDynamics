import jax
import jax.numpy as jnp
import numpy as np
from learner import ContinualLearner

def train_single_expert(config, train_task, test_ds):
    """
    Trains n_repeats models on a single task from scratch using a fully 
    JIT-compiled scan loop, evaluating on the test set periodically.
    
    UPDATED: Now synchronizes evaluation with config.log_frequency to match 
    the ContinualLearner's checkpointing cadence.
    """
    print(f"--- Training Expert on {train_task.name} (Synchronized to LogFreq: {config.log_frequency}) ---")
    
    learner = ContinualLearner(config)
    
    # 1. Preload Data
    train_imgs, train_lbls = learner.preload_data(train_task.load_data())
    test_imgs, test_lbls = learner.preload_data(test_ds)
    
    # 2. Reshape for Batching
    n_samples = train_imgs.shape[0]
    n_batches = n_samples // config.batch_size
    if n_batches == 0:
        raise ValueError("Data too small for batch size")
        
    train_imgs = train_imgs[:n_batches*config.batch_size]
    train_lbls = train_lbls[:n_batches*config.batch_size]
    
    train_imgs_reshaped = train_imgs.reshape(n_batches, config.batch_size, -1)
    train_lbls_reshaped = train_lbls.reshape(n_batches, config.batch_size, -1)

    # 3. Define the Super-JIT Loop with Conditional Eval
    @jax.jit
    def run_training_loop(state, t_imgs, t_lbls, test_i, test_l):
        
        def epoch_step(carry, epoch_idx):
            curr_state = carry
            
            # A. Train Step (Always runs)
            # Returns means over batches: shape (n_repeats,)
            new_state, train_losses, train_accs = learner._train_epoch_jit(curr_state, t_imgs, t_lbls)
            
            # B. Conditional Eval Step
            # Mirror Learner: Evaluate at the END of a block.
            # Learner evaluates after `log_frequency` steps.
            # We use (epoch_idx + 1) because epoch_idx is 0-based.
            # Example: log_freq=10. Learner evals at end of epoch 10.
            # Here: idx 9 -> (9+1)%10 == 0 -> True.
            is_eval_step = ((epoch_idx + 1) % config.log_frequency == 0)
            
            def true_eval_fn(s):
                # Returns (n_repeats,) arrays
                l, a = learner._eval_jit(s, test_i, test_l)
                return l, a

            def false_eval_fn(s):
                # Returns NaNs of same shape as eval -> (n_repeats,)
                # We use train_losses shape to determine n_repeats
                dummy_shape = train_losses.shape 
                return jnp.full(dummy_shape, jnp.nan), jnp.full(dummy_shape, jnp.nan)

            test_losses, test_accs = jax.lax.cond(
                is_eval_step,
                true_eval_fn,
                false_eval_fn,
                new_state
            )

            # Return state and (Train Metrics, Test Metrics)
            metrics = (train_losses, train_accs, test_losses, test_accs)
            return new_state, metrics

        # Scan over range(epochs)
        epochs_range = jnp.arange(config.epochs_per_task)
        
        final_state, (tr_loss_hist, tr_acc_hist, te_loss_hist, te_acc_hist) = jax.lax.scan(
            epoch_step, 
            state, 
            epochs_range
        )
        return final_state, tr_loss_hist, tr_acc_hist, te_loss_hist, te_acc_hist

    print(f"    > Compiling and running {config.epochs_per_task} epochs...")
    
    # 4. Execute
    final_state, tr_l, tr_a, te_l, te_a = run_training_loop(
        learner.state, 
        train_imgs_reshaped, 
        train_lbls_reshaped,
        test_imgs,
        test_lbls
    )
    
    # 5. Process Results
    # Convert to Numpy
    tr_l = np.array(tr_l)   # (Epochs, Repeats)
    tr_a = np.array(tr_a)
    te_l = np.array(te_l)   # (Epochs, Repeats) - contains NaNs
    te_a = np.array(te_a)

    # Calculate final stats
    # Since config enforces epochs % log_frequency == 0, the last epoch is always an eval step.
    final_losses = te_l[-1]
    final_accs = te_a[-1]
    
    loss_mean = np.nanmean(final_losses)
    loss_std = np.nanstd(final_losses)
    acc_mean = np.nanmean(final_accs)
    acc_std = np.nanstd(final_accs)
    
    print(f"    > Expert Final: Acc {acc_mean:.4f} | Loss {loss_mean:.4f}")

    # Return the Test History (with NaNs) to match Main's expectations for plotting
    return loss_mean, loss_std, acc_mean, acc_std, te_l, te_a