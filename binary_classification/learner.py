import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import flatten_util
import algorithms 

class ContinualLearner:
    def __init__(self, config, hooks=None):
        self.config = config
        self.hooks = hooks if hooks else []
        
        self.algo = algorithms.get_algorithm(config)

        rng = jax.random.key(config.seed)
        self.state = self.algo.init_vectorized_state(rng, config.input_dim)
        
        self._flat_fn = lambda p: flatten_util.ravel_pytree(p)[0]

    def preload_data(self, data_loader):
        """
        Consumes the MultiRepeatDataLoader.
        Returns jnp.arrays of shape: (Total_Samples, Repeats, Dim)
        """
        # MultiRepeatDataLoader yields (Repeats, Batch, Dim)
        # We want to concatenate along the Batch dimension (axis 1)
        # resulting in (Repeats, Total_Samples, Dim)
        
        images_list = []
        labels_list = []
        
        for batch_imgs, batch_lbls in data_loader:
            images_list.append(batch_imgs) # (R, B, D)
            labels_list.append(batch_lbls) # (R, B)
            
        if not images_list:
            raise ValueError("DataLoader returned no data.")

        # Concatenate along axis 1 (Batch dim)
        images = np.concatenate(images_list, axis=1) # (R, Total, D)
        labels = np.concatenate(labels_list, axis=1) # (R, Total)
        
        if labels.ndim == 2: labels = labels[..., None] # (R, Total, 1)

        # IMPORTANT: The training loop (scan) iterates over Time.
        # We need dim 0 to be Time (Samples/Batches) and dim 1 to be Repeats.
        # Current: (Repeats, Total, ...) -> Swap to (Total, Repeats, ...)
        
        images = np.swapaxes(images, 0, 1)
        labels = np.swapaxes(labels, 0, 1)

        return jnp.array(images), jnp.array(labels)

    def get_flat_params(self, state):
        return jax.vmap(self._flat_fn)(state.params)

    @partial(jax.jit, static_argnums=(0,))
    def _train_epoch_jit(self, state, batch_images, batch_labels):
        """
        state: (Repeats, ...)
        batch_images: (Batches, Repeats, B_Size, Dim) - Scanned over dim 0
        batch_labels: (Batches, Repeats, B_Size, 1)
        """
        
        def scan_fn(carry_state, batch_data):
            # batch_data is a tuple of (img_slice, lbl_slice)
            # img_slice shape: (Repeats, B_Size, Dim)
            imgs, lbls = batch_data
            
            # We want to map over Repeats (dim 0)
            # parallel_scan takes (single_state, single_batch_img, single_batch_lbl)
            # vmap(..., in_axes=(0, 0, 0)) splits the state and the batch data by repeat
            
            def parallel_update(s, i, l):
                return self.algo.train_step(s, (i, l))

            step_update = jax.vmap(parallel_update, in_axes=(0, 0, 0))
            new_state, metrics = step_update(carry_state, imgs, lbls)
            
            return new_state, metrics

        # Scan over the Batches dimension
        final_state, (losses, accs) = jax.lax.scan(scan_fn, state, (batch_images, batch_labels))
        
        # losses shape: (Batches, Repeats)
        return final_state, jnp.mean(losses, axis=0), jnp.mean(accs, axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def _eval_jit(self, state, images, labels):
        # state: (Repeats, ...)
        # images: (Total_Samples, Repeats, Dim) -> Need to reshuffle?
        # Typically eval is done on full set.
        # Eval logic: vmap over Repeats. Each repeat has its own State and its own Data (images[:, r, :])
        
        # Transpose images to (Repeats, Total_Samples, Dim) for vmap
        images_t = jnp.swapaxes(images, 0, 1)
        labels_t = jnp.swapaxes(labels, 0, 1)
        
        def eval_single(curr_state, curr_imgs, curr_lbls):
            return self.algo.eval_step(curr_state, (curr_imgs, curr_lbls))

        # vmap over (State, Images, Labels)
        return jax.vmap(eval_single, in_axes=(0, 0, 0))(state, images_t, labels_t)

    @partial(jax.jit, static_argnums=(0,))
    def _extract_features_jit(self, state, images):
        # images: (Total, Repeats, Dim)
        images_t = jnp.swapaxes(images, 0, 1)
        
        if images_t.ndim == 4: 
             images_t = images_t.reshape(images_t.shape[0], -1, images_t.shape[-1])
        
        def extract_single(curr_state, curr_imgs):
            return self.algo.get_features(curr_state, curr_imgs)

        return jax.vmap(extract_single, in_axes=(0, 0))(state, images_t)

    def train_task(self, task, test_streams, global_history, analysis_subset=None):
        print(f"--- CL Training on {task.name} ---")
        
        for h in self.hooks: h.on_task_start(task, self.state)

        # 1. Load Data (Now returns [Total, Repeats, Dim])
        train_loader = task.load_data()
        train_imgs_raw, train_lbls_raw = self.preload_data(train_loader)
        
        # Reshape for Scan: (Batches, Repeats, Batch_Size, Dim)
        # train_imgs_raw is (Total, Repeats, Dim)
        n_samples = train_imgs_raw.shape[0] # Total samples (min length across repeats)
        batches_per_epoch = n_samples // self.config.batch_size
        limit = batches_per_epoch * self.config.batch_size
        
        train_imgs = train_imgs_raw[:limit]
        train_lbls = train_lbls_raw[:limit]

        # Reshape: (Batches, B_Size, Repeats, Dim) -> Swap to (Batches, Repeats, B_Size, Dim)
        epoch_data_imgs = train_imgs.reshape(batches_per_epoch, self.config.batch_size, self.config.n_repeats, -1)
        epoch_data_imgs = jnp.swapaxes(epoch_data_imgs, 1, 2)
        
        epoch_data_lbls = train_lbls.reshape(batches_per_epoch, self.config.batch_size, self.config.n_repeats, -1)
        epoch_data_lbls = jnp.swapaxes(epoch_data_lbls, 1, 2)

        epoch_data_imgs = jax.device_put(epoch_data_imgs)
        epoch_data_lbls = jax.device_put(epoch_data_lbls)

        # 2. Analysis Data
        analysis_imgs = None
        if analysis_subset:
            analysis_imgs, _ = self.preload_data(analysis_subset)
            # analysis_imgs is (Total, Repeats, Dim) -> OK for _extract_features_jit

        # 3. Test Data Cache
        if not hasattr(self, 'cached_test_data'):
            self.cached_test_data = {}
        for t_name, t_loader in test_streams.items():
            if t_name not in self.cached_test_data:
                self.cached_test_data[t_name] = self.preload_data(t_loader)
        test_data_jax = {k: v for k, v in self.cached_test_data.items()}

        @jax.jit
        def nested_task_scan(initial_state):
            log_freq = self.config.log_frequency
            total_epochs = self.config.epochs_per_task
            n_outer_steps = total_epochs // log_freq
            
            def inner_loop(carry, _):
                curr_state = carry
                new_state, tr_loss, tr_acc = self._train_epoch_jit(curr_state, epoch_data_imgs, epoch_data_lbls)
                return new_state, (tr_loss, tr_acc)

            def outer_step(carry, _):
                state_start = carry
                state_end, (block_losses, block_accs) = jax.lax.scan(
                    inner_loop, state_start, None, length=log_freq
                )
                
                # Eval
                def run_eval(s):
                    results = {}
                    for t_name in sorted(test_data_jax.keys()):
                        ti, tl = test_data_jax[t_name]
                        l, a = self._eval_jit(s, ti, tl)
                        results[t_name] = (l, a)
                    return results

                test_metrics_sparse = run_eval(state_end)
                flat_w = self.get_flat_params(state_end)
                
                if analysis_imgs is not None:
                    reps = self._extract_features_jit(state_end, analysis_imgs)
                    # Result: (Repeats, Samples, Dim)
                    # We usually want (Samples, Repeats, Dim) for analysis pipeline
                    reps = jnp.swapaxes(reps, 0, 1)
                else:
                    reps = jnp.zeros((1,))

                metrics = {
                    'tr_loss': block_losses,
                    'tr_acc': block_accs,
                    'test': test_metrics_sparse,
                    'weights': flat_w,
                    'reps': reps
                }
                return state_end, metrics

            outer_range = jnp.arange(n_outer_steps)
            final_state, history = jax.lax.scan(outer_step, initial_state, outer_range)
            return final_state, history

        self.state, history_tree = nested_task_scan(self.state)

        # Post-processing
        history_np = jax.tree_util.tree_map(np.array, history_tree)
        
        # tr_loss shape: (Outer, Inner, Repeats) -> (Total_Epochs, Repeats)
        tr_loss_dense = history_np['tr_loss'].reshape(-1, self.config.n_repeats)
        tr_acc_dense = history_np['tr_acc'].reshape(-1, self.config.n_repeats)
        global_history['train_loss'].extend(tr_loss_dense)
        global_history['train_acc'].extend(tr_acc_dense)

        for t_name in history_np['test'].keys():
            sparse_loss = history_np['test'][t_name][0]
            sparse_acc = history_np['test'][t_name][1]
            n_outer = sparse_loss.shape[0]
            n_repeats = sparse_loss.shape[1]
            total_block_len = n_outer * self.config.log_frequency
            dense_loss = np.full((total_block_len, n_repeats), np.nan)
            dense_acc = np.full((total_block_len, n_repeats), np.nan)
            eval_indices = np.arange(self.config.log_frequency - 1, total_block_len, self.config.log_frequency)
            dense_loss[eval_indices] = sparse_loss
            dense_acc[eval_indices] = sparse_acc
            global_history['test_metrics'][t_name]['loss'].extend(dense_loss)
            global_history['test_metrics'][t_name]['acc'].extend(dense_acc)

        weight_history = history_np['weights']
        rep_history = history_np['reps'] if analysis_subset else None
        
        for h in self.hooks: h.on_task_end(task, self.state, metrics=None)
        
        return rep_history, weight_history
    
    def clear_test_cache(self):
        """Clear cached test data to free memory."""
        if hasattr(self, 'cached_test_data'):
            self.cached_test_data.clear()