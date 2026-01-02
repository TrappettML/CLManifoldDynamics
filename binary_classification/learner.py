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
        
        # Initialize States (Vectorized)
        rng = jax.random.key(config.seed)
        self.state = create_vectorized_state(config, rng)
        
        self._flat_fn = lambda p: flatten_util.ravel_pytree(p)[0]

    def preload_data(self, data_loader):
        """
        Converts a PyTorch DataLoader (yielding Tensors) into JAX-ready numpy arrays.
        Reads the entire dataset into memory (RAM).
        """
        images_list = []
        labels_list = []
        
        # Iterate over the PyTorch DataLoader
        for batch_imgs, batch_lbls in data_loader:
            images_list.append(batch_imgs.numpy())
            labels_list.append(batch_lbls.numpy())
            
        if not images_list:
            raise ValueError("DataLoader returned no data.")

        # Concatenate all batches
        images = np.concatenate(images_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        # Ensure labels have the shape (N, 1) if they are (N,)
        if labels.ndim == 1: 
            labels = labels[..., None]
            
        return jnp.array(images), jnp.array(labels)

    def get_flat_params(self, state):
        return jax.vmap(self._flat_fn)(state.params)

    # --- JIT Compiled Core Logic ---

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

        # vmap over the 'repeats' axis (0)
        parallel_train = jax.vmap(parallel_scan, in_axes=(0, None, None))
        final_state, (losses, accs) = parallel_train(state, batch_images, batch_labels)
        
        # losses/accs shape: (n_repeats, n_batches)
        # Return mean over batches -> (n_repeats,)
        return final_state, jnp.mean(losses, axis=1), jnp.mean(accs, axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def _eval_jit(self, state, images, labels):
        # eval_single maps over 'state' (n_repeats) but shares 'images' (dataset)
        def eval_single(curr_state):
            logits = curr_state.apply_fn({'params': curr_state.params}, images)
            loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
            preds = (logits > 0).astype(jnp.float32)
            acc = jnp.mean(preds == labels)
            return loss, acc

        return jax.vmap(eval_single)(state)

    # === Feature Extraction JIT ===
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

    # --- High Level Loop ---

    def train_task(self, task, test_streams, global_history, analysis_subset=None):
        print(f"--- CL Training on {task.name} (x{self.config.n_repeats} Repeats) [SCAN Optimized] ---")
        
        for h in self.hooks: h.on_task_start(task, self.state)

        # 1. Load Train Data
        train_loader = task.load_data()
        train_imgs, train_lbls = self.preload_data(train_loader)
        
        # Reshape for JAX scan: (N_Batches, Batch_Size, Dim)
        n_samples = train_imgs.shape[0]
        n_batches = n_samples // self.config.batch_size
        if n_batches == 0:
             raise ValueError(f"Dataset {task.name} too small for batch size {self.config.batch_size}")

        train_imgs_reshaped = train_imgs[:n_batches*self.config.batch_size].reshape(n_batches, self.config.batch_size, -1)
        train_lbls_reshaped = train_lbls[:n_batches*self.config.batch_size].reshape(n_batches, self.config.batch_size, -1)

        # 2. Prepare Analysis Data (if any)
        analysis_imgs = None
        if analysis_subset:
            print(f"    > Preloading analysis subset for representation tracking...")
            analysis_imgs, _ = self.preload_data(analysis_subset)
            if analysis_imgs.ndim > 2:
                analysis_imgs = analysis_imgs.reshape(-1, analysis_imgs.shape[-1])

        # 3. Cache Test Data (Preload all streams to GPU)
        # We assume test_streams keys are stable.
        if not hasattr(self, 'cached_test_data'):
            self.cached_test_data = {}
        
        # Ensure we have all current test streams loaded
        for t_name, t_loader in test_streams.items():
            if t_name not in self.cached_test_data:
                self.cached_test_data[t_name] = self.preload_data(t_loader)
        
        # Create a snapshot of test data for JIT
        # This dictionary maps task_name -> (imgs, lbls)
        test_data_jax = {k: v for k, v in self.cached_test_data.items()}

        # 4. Define The Super-Scan
        # We define this closure inside the method to capture 'test_data_jax' and 'analysis_imgs'
        
        @jax.jit
        def full_task_scan(initial_state, t_imgs, t_lbls):
            
            def epoch_step(carry_state, epoch_idx):
                
                # A. TRAIN STEP
                new_state, tr_losses, tr_accs = self._train_epoch_jit(carry_state, t_imgs, t_lbls)
                
                # B. EVAL STEP (Conditional)
                is_eval_step = (epoch_idx % self.config.eval_freq == 0) | (epoch_idx == self.config.epochs_per_task - 1)
                
                def run_eval(s):
                    # Evaluate on ALL available test streams
                    results = {}
                    for t_name, (ti, tl) in test_data_jax.items():
                        l, a = self._eval_jit(s, ti, tl)
                        results[t_name] = (l, a)
                    return results

                def skip_eval(s):
                    # Return NaNs with correct shape
                    results = {}
                    dummy_shape = tr_losses.shape # (n_repeats,)
                    for t_name in test_data_jax.keys():
                        results[t_name] = (
                            jnp.full(dummy_shape, jnp.nan),
                            jnp.full(dummy_shape, jnp.nan)
                        )
                    return results

                test_metrics_tree = jax.lax.cond(is_eval_step, run_eval, skip_eval, new_state)

                # C. CAPTURE DATA
                # Weights
                flat_w = self.get_flat_params(new_state)
                
                # Representations (only if analysis_imgs provided)
                if analysis_imgs is not None:
                    reps = self._extract_features_jit(new_state, analysis_imgs)
                else:
                    reps = jnp.zeros((1,)) # Dummy

                # Pack metrics
                metrics = {
                    'tr_loss': tr_losses,
                    'tr_acc': tr_accs,
                    'test': test_metrics_tree,
                    'weights': flat_w,
                    'reps': reps
                }
                
                return new_state, metrics

            epochs_range = jnp.arange(self.config.epochs_per_task)
            final_state, history = jax.lax.scan(epoch_step, initial_state, epochs_range)
            return final_state, history

        # 5. Execute Scan
        print(f"    > Compiling and running {self.config.epochs_per_task} epochs...")
        self.state, history_tree = full_task_scan(self.state, train_imgs_reshaped, train_lbls_reshaped)

        # 6. Unpack and Process Results to CPU
        # Convert entire history tree to numpy at once to minimize transfers
        history_np = jax.tree_util.tree_map(np.array, history_tree)

        # Update Global History
        # We iterate through the time dimension (axis 0 of the scanned arrays)
        for ep in range(self.config.epochs_per_task):
            global_history['train_loss'].append(history_np['tr_loss'][ep])
            global_history['train_acc'].append(history_np['tr_acc'][ep])
            
            # Test Metrics
            for t_name in history_np['test'].keys():
                # history_np['test'][t_name] is tuple (loss_arr, acc_arr)
                t_loss = history_np['test'][t_name][0][ep]
                t_acc = history_np['test'][t_name][1][ep]
                
                global_history['test_metrics'][t_name]['loss'].append(t_loss)
                global_history['test_metrics'][t_name]['acc'].append(t_acc)

        # Extract large analysis arrays
        weight_history = history_np['weights'] # (Epochs, Repeats, Params)
        
        representation_history = None
        if analysis_subset:
            representation_history = history_np['reps'] # (Epochs, Repeats, Samples, Dim)

        # 7. Final Hooks
        for h in self.hooks: h.on_task_end(task, self.state, metrics=None)
        
        return representation_history, weight_history