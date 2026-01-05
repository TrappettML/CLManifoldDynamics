import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import flatten_util
import algorithms  # Import the new module

class ContinualLearner:
    def __init__(self, config, hooks=None):
        self.config = config
        self.hooks = hooks if hooks else []
        
        # 1. Initialize Algorithm based on Config
        self.algo = algorithms.get_algorithm(config)

        # 2. Initialize State via Algorithm
        rng = jax.random.key(config.seed)
        self.state = self.algo.init_vectorized_state(rng, config.input_dim)
        
        self._flat_fn = lambda p: flatten_util.ravel_pytree(p)[0]

    def preload_data(self, data_loader):
        images_list = []
        labels_list = []
        for batch_imgs, batch_lbls in data_loader:
            images_list.append(batch_imgs.numpy())
            labels_list.append(batch_lbls.numpy())
            
        if not images_list:
            raise ValueError("DataLoader returned no data.")

        images = np.concatenate(images_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        if labels.ndim == 1: labels = labels[..., None]
            
        return jnp.array(images), jnp.array(labels)

    def get_flat_params(self, state):
        return jax.vmap(self._flat_fn)(state.params)

    @partial(jax.jit, static_argnums=(0,))
    def _train_epoch_jit(self, state, batch_images, batch_labels):
        """Generic scan loop that delegates the update logic to self.algo.train_step"""
        
        def scan_fn(carry_state, batch_data):
            # batch_data is (img, lbl)
            new_state, metrics = self.algo.train_step(carry_state, batch_data)
            return new_state, metrics

        def parallel_scan(s, imgs, lbls):
            return jax.lax.scan(scan_fn, s, (imgs, lbls))

        # vmap over the repeats dimension
        parallel_train = jax.vmap(parallel_scan, in_axes=(0, None, None))
        final_state, (losses, accs) = parallel_train(state, batch_images, batch_labels)
        
        return final_state, jnp.mean(losses, axis=1), jnp.mean(accs, axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def _eval_jit(self, state, images, labels):
        def eval_single(curr_state):
            return self.algo.eval_step(curr_state, (images, labels))

        return jax.vmap(eval_single)(state)

    @partial(jax.jit, static_argnums=(0,))
    def _extract_features_jit(self, state, images):
        if images.ndim == 4: 
             images = images.reshape(-1, images.shape[-1])
        
        def extract_single(curr_state):
            return self.algo.get_features(curr_state, images)

        return jax.vmap(extract_single)(state)

    def train_task(self, task, test_streams, global_history, analysis_subset=None):
        # NOTE: This method remains largely the same, as it orchestrates data loading 
        # and the high-level loop, which is common across most CL setups.
        
        print(f"--- CL Training on {task.name} (Algo: {self.config.algorithm}) ---")
        
        for h in self.hooks: h.on_task_start(task, self.state)

        # 1. Load Data
        train_loader = task.load_data()
        train_imgs_raw, train_lbls_raw = self.preload_data(train_loader)
        
        n_samples = train_imgs_raw.shape[0]
        batches_per_epoch = n_samples // self.config.batch_size
        limit = batches_per_epoch * self.config.batch_size
        train_imgs_raw = train_imgs_raw[:limit]
        train_lbls_raw = train_lbls_raw[:limit]

        epoch_data_imgs = train_imgs_raw.reshape(batches_per_epoch, self.config.batch_size, -1)
        epoch_data_lbls = train_lbls_raw.reshape(batches_per_epoch, self.config.batch_size, -1)
        
        epoch_data_imgs = jax.device_put(epoch_data_imgs)
        epoch_data_lbls = jax.device_put(epoch_data_lbls)

        # 2. Analysis Data
        analysis_imgs = None
        if analysis_subset:
            analysis_imgs, _ = self.preload_data(analysis_subset)
            if analysis_imgs.ndim > 2:
                analysis_imgs = analysis_imgs.reshape(-1, analysis_imgs.shape[-1])

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
                # _train_epoch_jit now uses self.algo internally
                new_state, tr_loss, tr_acc = self._train_epoch_jit(curr_state, epoch_data_imgs, epoch_data_lbls)
                return new_state, (tr_loss, tr_acc)

            def outer_step(carry, _):
                state_start = carry
                state_end, (block_losses, block_accs) = jax.lax.scan(
                    inner_loop, state_start, None, length=log_freq
                )
                
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

        # Post-processing (converting to numpy, handling logs) remains the same
        history_np = jax.tree_util.tree_map(np.array, history_tree)
        
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