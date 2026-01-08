import jax
import jax.numpy as jnp
import numpy as np
from learner import ContinualLearner

def train_single_expert(config, train_task, test_loader):
    print(f"--- Training Expert on {train_task.name} ---")
    
    learner = ContinualLearner(config)
    
    # 1. Preload Data (Returns [Total, Repeats, Dim])
    train_imgs, train_lbls = learner.preload_data(train_task.load_data())
    test_imgs, test_lbls = learner.preload_data(test_loader)
    
    # 2. Reshape for Batching
    n_samples = train_imgs.shape[0]
    n_batches = n_samples // config.batch_size
    limit = n_batches * config.batch_size
    
    train_imgs = train_imgs[:limit]
    train_lbls = train_lbls[:limit]
    
    # Reshape: (Batches, B_Size, Repeats, Dim) -> (Batches, Repeats, B_Size, Dim)
    train_imgs_reshaped = train_imgs.reshape(n_batches, config.batch_size, config.n_repeats, -1)
    train_imgs_reshaped = jnp.swapaxes(train_imgs_reshaped, 1, 2)
    
    train_lbls_reshaped = train_lbls.reshape(n_batches, config.batch_size, config.n_repeats, -1)
    train_lbls_reshaped = jnp.swapaxes(train_lbls_reshaped, 1, 2)

    # 3. Define JIT Loop
    @jax.jit
    def run_training_loop(state, t_imgs, t_lbls, test_i, test_l):
        
        def epoch_step(carry, epoch_idx):
            curr_state = carry
            
            # Use the UPDATED _train_epoch_jit (vmap over data)
            new_state, train_losses, train_accs = learner._train_epoch_jit(curr_state, t_imgs, t_lbls)
            
            is_eval_step = ((epoch_idx + 1) % config.log_frequency == 0)
            
            def true_eval_fn(s):
                l, a = learner._eval_jit(s, test_i, test_l)
                return l, a

            def false_eval_fn(s):
                dummy_shape = train_losses.shape 
                return (jnp.full(dummy_shape, jnp.nan, dtype=train_losses.dtype), 
                        jnp.full(dummy_shape, jnp.nan, dtype=train_accs.dtype))

            test_losses, test_accs = jax.lax.cond(
                is_eval_step, true_eval_fn, false_eval_fn, new_state
            )

            metrics = (train_losses, train_accs, test_losses, test_accs)
            return new_state, metrics

        epochs_range = jnp.arange(config.epochs_per_task)
        final_state, (tr_l, tr_a, te_l, te_a) = jax.lax.scan(epoch_step, state, epochs_range)
        return final_state, tr_l, tr_a, te_l, te_a

    # 4. Execute
    final_state, tr_l, tr_a, te_l, te_a = run_training_loop(
        learner.state, 
        train_imgs_reshaped, 
        train_lbls_reshaped,
        test_imgs,
        test_lbls
    )
    
    # 5. Process
    tr_l = np.array(tr_l)
    tr_a = np.array(tr_a)
    te_l = np.array(te_l)
    te_a = np.array(te_a)

    final_losses = te_l[-1]
    final_accs = te_a[-1]
    
    loss_mean = np.nanmean(final_losses)
    loss_std = np.nanstd(final_losses)
    acc_mean = np.nanmean(final_accs)
    acc_std = np.nanstd(final_accs)
    
    print(f"    > Expert Final: Acc {acc_mean:.4f} | Loss {loss_mean:.4f}")

    return loss_mean, loss_std, acc_mean, acc_std, te_l, te_a