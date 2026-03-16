import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import flatten_util
import algorithms
from ipdb import set_trace


class ContinualLearner:
    def __init__(self, config, hooks=None):
        self.config = config
        self.hooks = hooks if hooks else []

        # --- Multi-GPU Setup ---
        self.num_devices = jax.local_device_count()
        if self.config.n_repeats % self.num_devices != 0:
            raise ValueError(
                f"n_repeats ({self.config.n_repeats}) must be divisible "
                f"by available GPUs ({self.num_devices})."
            )
        self.r_per_dev = self.config.n_repeats // self.num_devices
        # -----------------------
        
        self.algo = algorithms.get_algorithm(config)
        
        rng = jax.random.key(config.seed)
        self.state = self.algo.init_vectorized_state(rng, config.input_side)
        
        self._flat_fn = lambda p: flatten_util.ravel_pytree(p)[0]

    def preload_data(self, data_source):
        """
        Optimized data loader.
        Expects Canonical Format: (Total_Samples, Repeats, Side, Side)
        
        Args:
            data_source: Either a tuple (X, Y) or an object with get_full_data()
        
        Returns:
            (images, labels) in Canonical format
    
        """
        # Fast path: Direct tuple
        if isinstance(data_source, tuple) and len(data_source) == 2:
            return data_source
        
        # Object with method
        if hasattr(data_source, 'get_full_data'):
            return data_source.get_full_data()
        
        # Generator path (legacy support)
        batch_list = list(data_source)
        if not batch_list:
            raise ValueError("DataLoader returned no data.")
        
        images_list, labels_list = zip(*batch_list)
        images = jnp.concatenate(images_list, axis=0)
        labels = jnp.concatenate(labels_list, axis=0)
        
        return images, labels

    def get_flat_params(self, state):
        """Returns flattened parameters across all repeats."""
        return jax.vmap(self._flat_fn)(state.params)

    @partial(jax.jit, static_argnums=(0,))
    def _train_epoch_jit(self, state, batch_images, batch_labels):
        """
        Trains one epoch using jax.lax.scan over batches.
        
        Args:
            state: Vectorized state (Repeats, ...)
            batch_images: (Num_Batches, Repeats, Batch_Size, Side, Side)
            batch_labels: (Num_Batches, Repeats, Batch_Size, 1)
        
        Returns:
            final_state: Updated state
            mean_loss: (Repeats,) averaged over batches
            mean_acc: (Repeats,) averaged over batches
        """
        def scan_fn(carry_state, batch_data):
            imgs, lbls = batch_data  # (Repeats, Batch_Size, Side, Side)
            
            def parallel_update(s, i, l):
                return self.algo.train_step(s, (i, l))
            
            # vmap over repeats dimension
            step_update = jax.vmap(parallel_update, in_axes=(0, 0, 0))
            new_state, metrics = step_update(carry_state, imgs, lbls)
            
            return new_state, metrics
        
        # Scan over batches
        final_state, (losses, accs) = jax.lax.scan(scan_fn, state, (batch_images, batch_labels))
        
        # Average over batches: (Batches, Repeats) -> (Repeats,)
        return final_state, jnp.mean(losses, axis=0), jnp.mean(accs, axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def _eval_jit(self, state, images, labels):
        """
        Evaluates on full dataset.
        
        Args:
            state: Vectorized state (Repeats, ...)
            images: (Total_Samples, Repeats, Dim) - Canonical format
            labels: (Total_Samples, Repeats, 1)
        
        Returns:
            loss: (Repeats,)
            acc: (Repeats,)
            reps: (total_samples, repeats, Dim)
        """
        # Transpose to (Repeats, Total_Samples, Dim) for vmap
        images_t = jnp.swapaxes(images, 0, 1)
        labels_t = jnp.swapaxes(labels, 0, 1)
        
        def eval_single(curr_state, curr_imgs, curr_lbls):
            return self.algo.eval_step(curr_state, (curr_imgs, curr_lbls))
        
        evals, representations = jax.vmap(eval_single, in_axes=(0, 0, 0))(state, images_t, labels_t)
        return evals, representations


    def train_task(self, task, test_data_dict, global_history):
        """
        Trains on a single task using nested jax.lax.scan.
        
        Args:
            task: Dict with keys 'id', 'name', 'data' (tuple of X, Y)
            test_data_dict: {task_name: (test_X, test_Y)} for all tasks
            global_history: Dict accumulating metrics across tasks
            analysis_subset: Optional (sub_X, sub_Y) for representation extraction
        """
        task_name = task['name']
        print(f"\n=== Training on {task_name} ===")
        
        for h in self.hooks:
            h.on_task_start(task, self.state)

        # Load training data (Canonical: N, R, Side, Side)
        train_imgs, train_lbls = self.preload_data(task['data'])
        
        # Prepare batches for scan: (Num_Batches, Repeats, Batch_Size, Side, Side)
        # TODO: fix how this data is split up for training
        n_samples = train_imgs.shape[0]
        n_batches = n_samples // self.config.batch_size
        limit = n_batches * self.config.batch_size
        # set_trace()
        # Truncate to divisible limit
        train_imgs = train_imgs[:limit]
        train_lbls = train_lbls[:limit]
        
        # Reshape: (Limit, R, D) -> (Batches, Batch_Size, R, Side, Side) -> (Batches, R, Batch_Size, Side, Side)
        train_imgs = train_imgs.reshape(n_batches, self.config.batch_size, self.config.n_repeats, self.config.input_side, self.config.input_side)
        train_lbls = train_lbls.reshape(n_batches, self.config.batch_size, self.config.n_repeats, -1)
        
        # Transpose for efficient vmap: (Batches, R, Batch_Size, Side, Side)
        epoch_data_imgs = jnp.swapaxes(train_imgs, 1, 2)
        epoch_data_lbls = jnp.swapaxes(train_lbls, 1, 2)
        
        # Move to device
        epoch_data_imgs = jax.device_put(epoch_data_imgs)
        epoch_data_lbls = jax.device_put(epoch_data_lbls)

        # Cache test data
        if not hasattr(self, 'cached_test_data'):
            self.cached_test_data = {}
        
        for t_name, t_data in test_data_dict.items():
            self.cached_test_data[t_name] = t_data
        
        test_data_jax = self.cached_test_data

        # --- Sharding Utilities for pmap ---
        def shard_data(x, repeat_axis):
            """Reshapes (..., R, ...) to (num_devices, ..., R // num_devices, ...)"""
            shape = list(x.shape)
            shape[repeat_axis:repeat_axis+1] = [self.num_devices, self.r_per_dev]
            reshaped = x.reshape(*shape)
            return jnp.swapaxes(reshaped, 0, repeat_axis)

        def unshard_data(x, r_axis):
            """Reverses sharding back to (..., R, ...)"""
            x = jnp.moveaxis(x, 0, r_axis)
            shape = list(x.shape)
            shape[r_axis : r_axis + 2] = [shape[r_axis] * shape[r_axis + 1]]
            return x.reshape(*shape)

        # 1. Shard all inputs across devices
        sharded_state = jax.tree_util.tree_map(lambda x: shard_data(x, 0), self.state)
        sharded_imgs = shard_data(epoch_data_imgs, 1)
        sharded_lbls = shard_data(epoch_data_lbls, 1)
        sharded_test = jax.tree_util.tree_map(lambda x: shard_data(x, 1), test_data_jax)

        # 2. Define the training loop WITHOUT @jax.jit (pmap replaces it)
        # Note: All arguments here now process 'r_per_dev' instead of 'n_repeats'
        def nested_task_scan(initial_state, e_imgs, e_lbls, t_data):
            log_freq = self.config.log_frequency
            total_epochs = self.config.epochs_per_task
            n_outer_steps = total_epochs // log_freq
            
            def inner_loop(carry, _):
                new_state, tr_loss, tr_acc = self._train_epoch_jit(carry, e_imgs, e_lbls)
                return new_state, (tr_loss, tr_acc)
            
            def outer_step(carry, _):
                state_start = carry
                state_end, (block_losses, block_accs) = jax.lax.scan(
                    inner_loop, state_start, None, length=log_freq
                )
                
                def run_eval(s):
                    results, all_reps = {}, []
                    for t_name in sorted(t_data.keys()):
                        ti, tl = t_data[t_name]
                        (l, a), reps = self._eval_jit(s, ti, tl)
                        results[t_name] = (l, a)
                        all_reps.append(reps)
                    return results, jnp.stack(all_reps)
                
                test_metrics_sparse, reps_sparse = run_eval(state_end)
                flat_w = self.get_flat_params(state_end)
                
                metrics = {
                    'tr_loss': block_losses,
                    'tr_acc': block_accs,
                    'test': test_metrics_sparse,
                    'weights': flat_w,
                    'reps': reps_sparse
                }
                return state_end, metrics
            
            return jax.lax.scan(outer_step, initial_state, jnp.arange(n_outer_steps))
        
        # 3. Apply pmap and execute
        print(f"  Mapping scan across {self.num_devices} devices...")
        pmap_scan = jax.pmap(nested_task_scan, in_axes=(0, 0, 0, 0))
        sharded_final_state, sharded_history = pmap_scan(
            sharded_state, sharded_imgs, sharded_lbls, sharded_test
        )
        
        # 4. Unshard back to normal shapes for the rest of your pipeline
        self.state = jax.tree_util.tree_map(lambda x: unshard_data(x, 0), sharded_final_state)
        
        # history_tree needs manual unsharding based on where the repeat axis ended up
        history_tree = {
            'tr_loss': unshard_data(sharded_history['tr_loss'], 2),
            'tr_acc': unshard_data(sharded_history['tr_acc'], 2),
            'weights': unshard_data(sharded_history['weights'], 1),
            'reps': unshard_data(sharded_history['reps'], 2),
            'test': {}
        }
        for t_name in sharded_history['test']:
            history_tree['test'][t_name] = (
                unshard_data(sharded_history['test'][t_name][0], 1),
                unshard_data(sharded_history['test'][t_name][1], 1)
            )
        
        
        # Post-processing: Convert to numpy and format
        history_np = jax.tree_util.tree_map(np.array, history_tree)
        
        # Flatten training metrics: (Outer, Inner, Repeats) -> (Total_Epochs, Repeats)
        tr_loss_dense = history_np['tr_loss'].reshape(-1, self.config.n_repeats)
        tr_acc_dense = history_np['tr_acc'].reshape(-1, self.config.n_repeats)
        
        global_history['train_loss'].extend(tr_loss_dense)
        global_history['train_acc'].extend(tr_acc_dense)
        
        # Expand sparse test metrics to dense format
        for t_name in history_np['test'].keys():
            sparse_loss = history_np['test'][t_name][0]  # (Outer, Repeats)
            sparse_acc = history_np['test'][t_name][1]
            
            n_outer = sparse_loss.shape[0]
            n_repeats = sparse_loss.shape[1]
            total_block_len = n_outer * self.config.log_frequency
            
            # Create dense arrays with NaN padding
            dense_loss = np.full((total_block_len, n_repeats), np.nan)
            dense_acc = np.full((total_block_len, n_repeats), np.nan)
            
            # Place sparse values at log positions
            eval_indices = np.arange(
                self.config.log_frequency - 1,
                total_block_len,
                self.config.log_frequency
            )
            dense_loss[eval_indices] = sparse_loss
            dense_acc[eval_indices] = sparse_acc
            
            global_history['test_metrics'][t_name]['loss'].extend(dense_loss)
            global_history['test_metrics'][t_name]['acc'].extend(dense_acc)
        
        # Extract histories for saving
        weight_history = history_np['weights']  # (Outer, Repeats, Param_Dim)
        rep_history = history_np['reps']  # TODO: check this shape
        set_trace()
        for h in self.hooks:
            h.on_task_end(task, self.state, metrics=None)
        
        print(f"  Training complete for {task_name}")
        
        return rep_history, weight_history
    
    def clear_test_cache(self):
        """Clears cached test data to free memory."""
        if hasattr(self, 'cached_test_data'):
            self.cached_test_data.clear()
            print("  Test cache cleared")