import jax
import jax.numpy as jnp
import optax
import numpy as np
import tensorflow_datasets as tfds
from functools import partial
from state import create_vectorized_state
from models import TwoLayerMLP

class ContinualLearner:
    def __init__(self, config, hooks=None):
        self.config = config
        self.hooks = hooks if hooks else []
        
        # Initialize States (Vectorized)
        rng = jax.random.key(config.seed)
        self.state = create_vectorized_state(config, rng)

    def preload_data(self, tfds_dataset):
        """Converts TFDS to JAX-ready numpy arrays."""
        ds_numpy = tfds.as_numpy(tfds_dataset)
        images, labels = [], []
        for batch in ds_numpy:
            images.append(batch[0])
            labels.append(batch[1])
        
        images = np.stack(images)
        labels = np.stack(labels)
        if labels.ndim == 2: labels = labels[..., None]
        return jnp.array(images), jnp.array(labels)

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
        n_batches, b_size = images.shape[0], images.shape[1]
        flat_img = images.reshape(n_batches * b_size, -1)
        flat_lbl = labels.reshape(n_batches * b_size, -1)

        def eval_single(curr_state):
            logits = curr_state.apply_fn({'params': curr_state.params}, flat_img)
            loss = optax.sigmoid_binary_cross_entropy(logits, flat_lbl).mean()
            preds = (logits > 0).astype(jnp.float32)
            acc = jnp.mean(preds == flat_lbl)
            return loss, acc

        return jax.vmap(eval_single)(state)

    # === NEW: Feature Extraction JIT ===
    @partial(jax.jit, static_argnums=(0,))
    def _extract_features_jit(self, state, images):
        """
        Extracts hidden representations for all seeds.
        Output Shape: (n_repeats, n_images, hidden_dim)
        """
        # Flatten batch dimension if present, as we just want a list of representations
        # Handle case where images are (batches, batch_size, dim) vs (total_samples, dim)
        if images.ndim == 4: # (Batches, H, W, C) or similar, depends on flattening
             # If data loader returns (Num_Batches, Batch_Size, Input_Dim)
             images = images.reshape(-1, images.shape[-1])
        
        def extract_single(curr_state):
            # Access the 'get_features' method defined in models.py
            features = curr_state.apply_fn(
                {'params': curr_state.params}, 
                images, 
                method=TwoLayerMLP.get_features
            )
            return features

        return jax.vmap(extract_single)(state)

    # --- High Level Loop ---

    def train_task(self, task, test_streams, global_history, analysis_subset=None):
        """
        Args:
            analysis_subset: (Optional) tf.data.Dataset or tuple of arrays 
                             containing the 'mandi' subset for representation tracking.
        """
        print(f"--- CL Training on {task.name} (x{self.config.n_repeats} Repeats) ---")
        
        # Hooks: Task Start
        for h in self.hooks: h.on_task_start(task, self.state)

        # Data Loading
        train_ds = task.load_data()
        train_imgs, train_lbls = self.preload_data(train_ds)
        
        # Prepare Analysis Data (Preload to GPU once)
        analysis_imgs = None
        representation_history = [] # Store CPU arrays here
        if analysis_subset:
            print(f"    > Preloading analysis subset for representation tracking...")
            analysis_imgs, _ = self.preload_data(analysis_subset)
            # Reshape analysis_imgs to be 2D (Total_Samples, Input_Dim) if it came in batches
            if analysis_imgs.ndim > 2:
                analysis_imgs = analysis_imgs.reshape(-1, analysis_imgs.shape[-1])

        # Ensure test cache exists
        if not hasattr(self, 'cached_test_data'):
            self.cached_test_data = {}
            for t_name, t_ds in test_streams.items():
                self.cached_test_data[t_name] = self.preload_data(t_ds)

        for epoch in range(self.config.epochs_per_task):
            for h in self.hooks: h.on_epoch_start(epoch, self.state)

            # 1. TRAIN
            self.state, epoch_losses, epoch_accs = self._train_epoch_jit(
                self.state, train_imgs, train_lbls
            )
            
            # Record Train Metrics
            global_history['train_loss_mean'].append(np.mean(epoch_losses))
            global_history['train_loss_std'].append(np.std(epoch_losses))
            global_history['train_acc_mean'].append(np.mean(epoch_accs))
            global_history['train_acc_std'].append(np.std(epoch_accs))

            # 2. EVAL
            for t_name, (t_imgs, t_lbls) in self.cached_test_data.items():
                t_loss, t_acc = self._eval_jit(self.state, t_imgs, t_lbls)
                
                global_history['test_metrics'][t_name]['acc_mean'].append(np.mean(t_acc))
                global_history['test_metrics'][t_name]['acc_std'].append(np.std(t_acc))
                global_history['test_metrics'][t_name]['loss_mean'].append(np.mean(t_loss))
                
                if t_name == task.name and epoch == self.config.epochs_per_task - 1:
                     print(f"    > Epoch {epoch+1} | {t_name} Acc: {np.mean(t_acc):.4f} ± {np.std(t_acc):.4f}")
            
            # 3. CAPTURE REPRESENTATIONS (New)
            if analysis_imgs is not None:
                # Extract on GPU
                reps_gpu = self._extract_features_jit(self.state, analysis_imgs)
                # Transfer to CPU immediately to free VRAM
                reps_cpu = np.array(reps_gpu)
                representation_history.append(reps_cpu)

            for h in self.hooks: h.on_epoch_end(epoch, self.state, metrics=None)

        # Hooks: Task End
        for h in self.hooks: h.on_task_end(task, self.state, metrics=None)
        
        return representation_history