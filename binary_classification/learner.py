import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from jax import flatten_util
from state import create_vectorized_state
from models import TwoLayerMLP

class ContinualLearner:
    def __init__(self, config, hooks=None):
        self.config = config
        self.hooks = hooks if hooks else []
        
        rng = jax.random.key(config.seed)
        self.state = create_vectorized_state(config, rng)
        
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
        
        if labels.ndim == 1: 
            labels = labels[..., None]
            
        return jnp.array(images), jnp.array(labels)

    def get_flat_params(self, state):
        return jax.vmap(self._flat_fn)(state.params)

    @partial(jax.jit, static_argnums=(0,))
    def _train_epoch_jit(self, state, batch_images, batch_labels):
        def train_step(curr_state, batch):
            b_img, b_lbl = batch
            def loss_fn(params):
                logits = curr_state.apply_fn({'params': params}, b_img)
                loss = optax.sigmoid_binary_cross_entropy(logits, b_lbl).mean()
                return loss, logits
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, loss_logits), grads = grad_fn(curr_state.params)
            new_state = curr_state.apply_gradients(grads=grads)
            preds = (loss_logits > 0).astype(jnp.float32)
            acc = jnp.mean(preds == b_lbl)
            return new_state, (loss, acc)

        def scan_fn(carry_state, batch_data):
            new_state, metrics = train_step(carry_state, batch_data)
            return new_state, metrics

        def parallel_scan(s, imgs, lbls):
            return jax.lax.scan(scan_fn, s, (imgs, lbls))

        parallel_train = jax.vmap(parallel_scan, in_axes=(0, None, None))
        final_state, (losses, accs) = parallel_train(state, batch_images, batch_labels)
        
        return final_state, jnp.mean(losses, axis=1), jnp.mean(accs, axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def _eval_jit(self, state, images, labels):
        def eval_single(curr_state):
            logits = curr_state.apply_fn({'params': curr_state.params}, images)
            loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
            preds = (logits > 0).astype(jnp.float32)
            acc = jnp.mean(preds == labels)
            return loss, acc

        return jax.vmap(eval_single)(state)

    @partial(jax.jit, static_argnums=(0,))
    def _extract_features_jit(self, state, images):
        if images.ndim == 4: 
             images = images.reshape(-1, images.shape[-1])
        
        def extract_single(curr_state):
            features = curr_state.apply_fn(
                {'params': curr_state.params}, 
                images, 
                method=TwoLayerMLP.get_features
            )
            return features

        return jax.vmap(extract_single)(state)

    def train_task(self, task, test_streams, global_history, analysis_subset=None):
        print(f"--- CL Training on {task.name} (x{self.config.n_repeats}) [Optimized Nested Scan] ---")
        
        for h in self.hooks: h.on_task_start(task, self.state)

        # 1. Load and Shape Train Data
        train_loader = task.load_data()
        train_imgs_raw, train_lbls_raw = self.preload_data(train_loader)
        
        n_samples = train_imgs_raw.shape[0]
        batches_per_epoch = n_samples // self.config.batch_size
        
        if batches_per_epoch == 0:
            raise ValueError(f"Dataset too small for batch size {self.config.batch_size}")

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
            print(f"    > Preloading analysis subset...")
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
                new_state, tr_loss, tr_acc = self._train_epoch_jit(curr_state, epoch_data_imgs, epoch_data_lbls)
                return new_state, (tr_loss, tr_acc)

            def outer_step(carry, _):
                state_start = carry
                
                state_end, (block_losses, block_accs) = jax.lax.scan(
                    inner_loop, state_start, None, length=log_freq
                )
                
                def run_eval(s):
                    results = {}
                    # Ensure deterministic iteration order for JAX safety
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

        print(f"    > Running {self.config.epochs_per_task} epochs (Logging every {self.config.log_frequency})...")
        self.state, history_tree = nested_task_scan(self.state)

        history_np = jax.tree_util.tree_map(np.array, history_tree)

        # Reshape to (TotalEpochs, Repeats)
        tr_loss_dense = history_np['tr_loss'].reshape(-1, self.config.n_repeats)
        tr_acc_dense = history_np['tr_acc'].reshape(-1, self.config.n_repeats)
        
        global_history['train_loss'].extend(tr_loss_dense)
        global_history['train_acc'].extend(tr_acc_dense)

        for t_name in history_np['test'].keys():
            sparse_loss = history_np['test'][t_name][0]
            sparse_acc = history_np['test'][t_name][1]
            
            dense_loss = np.repeat(sparse_loss, self.config.log_frequency, axis=0)
            dense_acc = np.repeat(sparse_acc, self.config.log_frequency, axis=0)
            
            global_history['test_metrics'][t_name]['loss'].extend(dense_loss)
            global_history['test_metrics'][t_name]['acc'].extend(dense_acc)

        weight_history = history_np['weights']
        
        rep_history = None
        if analysis_subset:
            rep_history = history_np['reps']

        for h in self.hooks: h.on_task_end(task, self.state, metrics=None)
        
        return rep_history, weight_history