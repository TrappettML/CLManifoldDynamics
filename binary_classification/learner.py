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

        parallel_train = jax.vmap(parallel_scan, in_axes=(0, None, None))
        final_state, (losses, accs) = parallel_train(state, batch_images, batch_labels)
        return final_state, jnp.mean(losses, axis=1), jnp.mean(accs, axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def _eval_jit(self, state, images, labels):
        # We might have variable batch sizes at test time, but JAX expects fixed shapes in JIT.
        # However, if 'images' is passed as one giant array (Full Batch), vmap handles the 'seeds'.
        # We assume images is (N, Dim), labels (N, 1).
        
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
        print(f"--- CL Training on {task.name} (x{self.config.n_repeats} Repeats) ---")
        
        for h in self.hooks: h.on_task_start(task, self.state)

        # Use new logic to load from Torch DataLoader -> Numpy/Jax
        train_loader = task.load_data()
        train_imgs, train_lbls = self.preload_data(train_loader)
        
        analysis_imgs = None
        representation_history = [] 
        weight_history = []

        if analysis_subset:
            # analysis_subset is now a DataLoader
            print(f"    > Preloading analysis subset for representation tracking...")
            analysis_imgs, _ = self.preload_data(analysis_subset)
            if analysis_imgs.ndim > 2:
                analysis_imgs = analysis_imgs.reshape(-1, analysis_imgs.shape[-1])

        # Cache test data (Test streams are DataLoaders now)
        if not hasattr(self, 'cached_test_data'):
            self.cached_test_data = {}
            for t_name, t_loader in test_streams.items():
                self.cached_test_data[t_name] = self.preload_data(t_loader)

        # Batching for JAX scan
        # train_imgs is (TotalSamples, Dim). We need to reshape for JAX scan: (NumBatches, BatchSize, Dim)
        n_samples = train_imgs.shape[0]
        # Drop remainder logic was handled in Torch DataLoader, so n_samples should be divisible by batch_size
        n_batches = n_samples // self.config.batch_size
        
        train_imgs_reshaped = train_imgs[:n_batches*self.config.batch_size].reshape(n_batches, self.config.batch_size, -1)
        train_lbls_reshaped = train_lbls[:n_batches*self.config.batch_size].reshape(n_batches, self.config.batch_size, -1)

        for epoch in range(self.config.epochs_per_task):
            for h in self.hooks: h.on_epoch_start(epoch, self.state)

            # 1. TRAIN
            self.state, epoch_losses, epoch_accs = self._train_epoch_jit(
                self.state, train_imgs_reshaped, train_lbls_reshaped
            )
            
            global_history['train_loss_mean'].append(np.mean(epoch_losses))
            global_history['train_loss_std'].append(np.std(epoch_losses))
            global_history['train_acc_mean'].append(np.mean(epoch_accs))
            global_history['train_acc_std'].append(np.std(epoch_accs))

            # 2. EVAL
            # Logic Update: Always append to history. If skipping eval, append NaNs.
            # This ensures the history length matches the total epochs for plotting.
            if epoch % self.config.eval_freq == 0 or epoch == self.config.epochs_per_task - 1:
                for t_name, (t_imgs, t_lbls) in self.cached_test_data.items():
                    t_loss, t_acc = self._eval_jit(self.state, t_imgs, t_lbls)
                    global_history['test_metrics'][t_name]['acc_mean'].append(np.mean(t_acc))
                    global_history['test_metrics'][t_name]['acc_std'].append(np.std(t_acc))
                    global_history['test_metrics'][t_name]['loss_mean'].append(np.mean(t_loss))
                    global_history['test_metrics'][t_name]['loss_std'].append(np.std(t_loss))
                    
                    if t_name == task.name and epoch == self.config.epochs_per_task - 1:
                        print(f"    > Epoch {epoch+1} | {t_name} Acc: {np.mean(t_acc):.4f} ± {np.std(t_acc):.4f}")
            else:
                # NaN padding for alignment
                for t_name in self.cached_test_data.keys():
                    global_history['test_metrics'][t_name]['acc_mean'].append(np.nan)
                    global_history['test_metrics'][t_name]['acc_std'].append(np.nan)
                    global_history['test_metrics'][t_name]['loss_mean'].append(np.nan)
                    global_history['test_metrics'][t_name]['loss_std'].append(np.nan)
                
            # 3. CAPTURE DATA
            flat_w_gpu = self.get_flat_params(self.state)
            weight_history.append(flat_w_gpu)

            if analysis_imgs is not None:
                reps_gpu = self._extract_features_jit(self.state, analysis_imgs)
                representation_history.append(reps_gpu)

            for h in self.hooks: h.on_epoch_end(epoch, self.state, metrics=None)

        for h in self.hooks: h.on_task_end(task, self.state, metrics=None)
        weight_history = np.array(jax.device_get(weight_history))
        if analysis_imgs is not None:
            representation_history = np.array(jax.device_get(representation_history))
        
        return representation_history, weight_history