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

    def preload_data(self, data_source):
        """
        Optimized data loader.
        Expects Canonical Format: (Total_Samples, Repeats, Dim)
        
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
            batch_images: (Num_Batches, Repeats, Batch_Size, Dim)
            batch_labels: (Num_Batches, Repeats, Batch_Size, 1)
        
        Returns:
            final_state: Updated state
            mean_loss: (Repeats,) averaged over batches
            mean_acc: (Repeats,) averaged over batches
        """
        def scan_fn(carry_state, batch_data):
            imgs, lbls = batch_data  # (Repeats, Batch_Size, Dim)
            
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
        
        def extract_single(curr_state, curr_imgs):
            return self.algo.get_features(curr_state, curr_imgs)
        
        evals = jax.vmap(eval_single, in_axes=(0, 0, 0))(state, images_t, labels_t)
        representations = jax.vmap(extract_single, in_axes=(0, 0))(state, images_t)
        return evals, representations

    @partial(jax.jit, static_argnums=(0,))
    def _extract_features_jit(self, state, images):
        """
        Extracts hidden features from model.
        
        Args:
            state: Vectorized state (Repeats, ...)
            images: (Total_Samples, Repeats, Dim) - Canonical format
        
        Returns:
            features: (Repeats, Total_Samples, Hidden_Dim)
        """
        # Transpose to (Repeats, Total_Samples, Dim) for vmap
        images_t = jnp.swapaxes(images, 0, 1)
        
        def extract_single(curr_state, curr_imgs):
            return self.algo.get_features(curr_state, curr_imgs)
        
        return jax.vmap(extract_single, in_axes=(0, 0))(state, images_t)

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

        # Load training data (Canonical: N, R, D)
        train_imgs, train_lbls = self.preload_data(task['data'])
        
        # Prepare batches for scan: (Num_Batches, Repeats, Batch_Size, Dim)
        n_samples = train_imgs.shape[0]
        n_batches = n_samples // self.config.batch_size
        limit = n_batches * self.config.batch_size
        
        # Truncate to divisible limit
        train_imgs = train_imgs[:limit]
        train_lbls = train_lbls[:limit]
        
        # Reshape: (Limit, R, D) -> (Batches, Batch_Size, R, D) -> (Batches, R, Batch_Size, D)
        train_imgs = train_imgs.reshape(n_batches, self.config.batch_size, self.config.n_repeats, -1)
        train_lbls = train_lbls.reshape(n_batches, self.config.batch_size, self.config.n_repeats, -1)
        
        # Transpose for efficient vmap: (Batches, R, Batch_Size, D)
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

        @jax.jit
        def nested_task_scan(initial_state):
            """
            Nested scan structure:
            - Outer: Log intervals
            - Inner: Training epochs within each interval
            """
            log_freq = self.config.log_frequency
            total_epochs = self.config.epochs_per_task
            n_outer_steps = total_epochs // log_freq
            
            def inner_loop(carry, _):
                """Trains for log_freq epochs."""
                curr_state = carry
                new_state, tr_loss, tr_acc = self._train_epoch_jit(
                    curr_state, epoch_data_imgs, epoch_data_lbls
                )
                return new_state, (tr_loss, tr_acc)
            
            def outer_step(carry, _):
                """Runs inner loop, then evaluates and logs."""
                state_start = carry
                
                # Train for log_freq epochs
                state_end, (block_losses, block_accs) = jax.lax.scan(
                    inner_loop, state_start, None, length=log_freq
                )
                
                # Evaluate on all test tasks
                def run_eval(s):
                    results = {}
                    all_reps = []
                    for t_name in sorted(test_data_jax.keys()):
                        ti, tl = test_data_jax[t_name]
                        (l, a), reps = self._eval_jit(s, ti, tl)
                        results[t_name] = (l, a)
                        all_reps.append(reps)
                    # stack reps as expeted of down stream processing
                    all_reps = jnp.stack(all_reps)
                    return results, all_reps
                
                test_metrics_sparse, reps_sparse = run_eval(state_end)
                
                # Flatten weights
                flat_w = self.get_flat_params(state_end)
                
                metrics = {
                    'tr_loss': block_losses,
                    'tr_acc': block_accs,
                    'test': test_metrics_sparse,
                    'weights': flat_w,
                    'reps': reps_sparse
                }
                
                return state_end, metrics
            
            # Run outer scan
            outer_range = jnp.arange(n_outer_steps)
            final_state, history = jax.lax.scan(outer_step, initial_state, outer_range)
            
            return final_state, history
        
        # Execute training
        print(f"  Compiling and running nested scan...")
        self.state, history_tree = nested_task_scan(self.state)
        
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
        rep_history = history_np['reps']  # (numTasks*nTestSamples, repeats, hiddenDim)

        for h in self.hooks:
            h.on_task_end(task, self.state, metrics=None)
        
        print(f"  Training complete for {task_name}")
        
        return rep_history, weight_history
    
    def clear_test_cache(self):
        """Clears cached test data to free memory."""
        if hasattr(self, 'cached_test_data'):
            self.cached_test_data.clear()
            print("  Test cache cleared")