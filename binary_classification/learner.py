import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import flatten_util
import algorithms
from ipdb import set_trace
import math


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
            batch_labels: (Num_Batches, Repeats, Batch_Size, output_dim)
        
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
            labels: (Total_Samples, Repeats, Out_dim)
        
        Returns:
            loss: (Repeats,)
            acc: (Repeats,)
        """
        # Transpose to (Repeats, Total_Samples, Dim) for vmap
        images_t = jnp.swapaxes(images, 0, 1)
        labels_t = jnp.swapaxes(labels, 0, 1)
        
        def eval_single(curr_state, curr_imgs, curr_lbls):
            return self.algo.eval_step(curr_state, (curr_imgs, curr_lbls))
        
        evals, _ = jax.vmap(eval_single, in_axes=(0, 0, 0))(state, images_t, labels_t)
        return evals


    def train_task(self, task, test_data_dict, global_history):
        task_name = task['name']
        print(f"\n=== Training on {task_name} ===", flush=True)
        
        for h in self.hooks:
            h.on_task_start(task, self.state)

        train_imgs, train_lbls = self.preload_data(task['data'])
        
        n_samples = train_imgs.shape[0]
        n_batches = n_samples // self.config.batch_size
        limit = n_batches * self.config.batch_size
        
        train_imgs = train_imgs[:limit]
        train_lbls = train_lbls[:limit]
        
        train_imgs = train_imgs.reshape(n_batches, self.config.batch_size, self.config.n_repeats, self.config.input_side, self.config.input_side)
        train_lbls = train_lbls.reshape(n_batches, self.config.batch_size, self.config.n_repeats, -1)
        
        epoch_data_imgs = jax.device_put(jnp.swapaxes(train_imgs, 1, 2))
        epoch_data_lbls = jax.device_put(jnp.swapaxes(train_lbls, 1, 2))

        if not hasattr(self, 'cached_test_data'):
            self.cached_test_data = {}

        for t_name, t_data in test_data_dict.items():
            self.cached_test_data[t_name] = t_data
        
        test_data_jax = self.cached_test_data

        def shard_data(x, repeat_axis):
            shape = list(x.shape)
            shape[repeat_axis:repeat_axis+1] = [self.num_devices, self.r_per_dev]
            reshaped = x.reshape(*shape)
            return jnp.swapaxes(reshaped, 0, repeat_axis)

        def unshard_data(x, r_axis):
            x = jnp.moveaxis(x, 0, r_axis)
            shape = list(x.shape)
            shape[r_axis : r_axis + 2] = [shape[r_axis] * shape[r_axis + 1]]
            return x.reshape(*shape)

        sharded_state = jax.tree_util.tree_map(lambda x: shard_data(x, 0), self.state)
        sharded_imgs = shard_data(epoch_data_imgs, 1)
        sharded_lbls = shard_data(epoch_data_lbls, 1)
        sharded_test = jax.tree_util.tree_map(lambda x: shard_data(x, 1), test_data_jax)

        # --- MEMORY CHUNKING ---
        log_freq = self.config.log_frequency
        total_outer_steps = self.config.epochs_per_task // log_freq
        
        # Get exactly how many parameters we have
        dummy_flat_params = self.get_flat_params(self.state)[0]
        num_params = len(dummy_flat_params)
        

        sharded_t_imgs = self.sharded_t_imgs
        sharded_t_lbls = self.sharded_t_lbls
        test_task_names = self.test_task_names

        # 2. Define the chunked task scan OUTSIDE the loop
        def get_pmap_scan(steps):
            def chunked_task_scan(initial_state, e_imgs, e_lbls, t_imgs_stack, t_lbls_stack):
                def inner_loop(carry, _):
                    new_state, tr_loss, tr_acc = self._train_epoch_jit(carry, e_imgs, e_lbls)
                    return new_state, (tr_loss, tr_acc)
                
                def outer_step(carry, _):
                    state_start = carry
                    state_end, (block_losses, block_accs) = jax.lax.scan(
                        inner_loop, state_start, None, length=log_freq
                    )
                    
                    # SEQUENTIAL EVALUATION: Prevents XLA from evaluating 20 tasks simultaneously
                    def eval_single_task(carry_dummy, task_data):
                        ti, tl = task_data
                        (l, a) = self._eval_jit(state_end, ti, tl)
                        return None, (l, a)
                    
                    _, (losses_all, accs_all) = jax.lax.scan(
                        eval_single_task, None, (t_imgs_stack, t_lbls_stack)
                    )
                    
                    flat_w = self.get_flat_params(state_end)
                    
                    metrics = {
                        'tr_loss': block_losses,
                        'tr_acc': block_accs,
                        'test_loss': losses_all,
                        'test_acc': accs_all,
                        'weights': flat_w
                    }
                    return state_end, metrics
                
                return jax.lax.scan(outer_step, initial_state, jnp.arange(steps))
            
            return jax.pmap(chunked_task_scan, in_axes=(0, 0, 0, 0, 0), donate_argnums=(0,))

        # 3. Create dummy structures representing the input shapes and dtypes for AOT compilation
        state_struct = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), sharded_state
        )
        img_struct = jax.ShapeDtypeStruct(sharded_imgs.shape, sharded_imgs.dtype)
        lbl_struct = jax.ShapeDtypeStruct(sharded_lbls.shape, sharded_lbls.dtype)
        t_img_struct = jax.ShapeDtypeStruct(sharded_t_imgs.shape, sharded_t_imgs.dtype)
        t_lbl_struct = jax.ShapeDtypeStruct(sharded_t_lbls.shape, sharded_t_lbls.dtype)

        # 4. Lower a 1-step version of the pmap to extract the base HLO memory footprint
        base_pmap = get_pmap_scan(steps=1)
        lowered_base = base_pmap.lower(state_struct, img_struct, lbl_struct, t_img_struct, t_lbl_struct)

        # 5. Execute the memory calculator with the exact compilation data
        chunk_size_steps = min([
            calculate_safe_chunk_size(
                device=d,
                lowered_scan_fn=lowered_base,
                safety_margin=0.75 
            ) for d in jax.local_devices()
        ])
    
        num_chunks = math.ceil(total_outer_steps / chunk_size_steps)
        print(f"  [Memory Manager] Executing {total_outer_steps} total logs across {num_chunks} chunk(s).", flush=True)

        # Pre-populate cache to prevent recompiling the base map if chunk_size_steps == 1
        cached_pmaps = {1: base_pmap}
        cpu_history_trees = []
        curr_state = sharded_state

        for chunk_idx in range(num_chunks):
            # Calculate static size
            steps_in_this_chunk = min(chunk_size_steps, total_outer_steps - chunk_idx * chunk_size_steps)
            
            if steps_in_this_chunk not in cached_pmaps:
                cached_pmaps[steps_in_this_chunk] = get_pmap_scan(steps_in_this_chunk)
                
            pmap_scan = cached_pmaps[steps_in_this_chunk]

            # Execute Chunk
            curr_state, sharded_history_chunk = pmap_scan(
                curr_state, sharded_imgs, sharded_lbls, sharded_t_imgs, sharded_t_lbls
            )

            # Move to CPU RAM immediately
            chunk_cpu = jax.device_get(sharded_history_chunk)
            del sharded_history_chunk
            
            def unshard_numpy(x, r_axis):
                x = np.moveaxis(x, 0, r_axis)
                shape = list(x.shape)
                shape[r_axis : r_axis + 2] = [shape[r_axis] * shape[r_axis + 1]]
                return x.reshape(*shape)

            # Unshard metrics
            unsharded_test_loss = unshard_numpy(chunk_cpu['test_loss'], 2)
            unsharded_test_acc = unshard_numpy(chunk_cpu['test_acc'], 2)

            history_tree_cpu = {
                'tr_loss': unshard_numpy(chunk_cpu['tr_loss'], 2),
                'tr_acc': unshard_numpy(chunk_cpu['tr_acc'], 2),
                'weights': unshard_numpy(chunk_cpu['weights'], 1),
                'test': {}
            }
            
            # Reconstruct the dictionary for the rest of your script
            for i, t_name in enumerate(test_task_names):
                history_tree_cpu['test'][t_name] = (
                    unsharded_test_loss[:, i, :],
                    unsharded_test_acc[:, i, :]
                )
                
            cpu_history_trees.append(history_tree_cpu)
            del chunk_cpu

        self.state = jax.tree_util.tree_map(lambda x: unshard_data(x, 0), curr_state)

        del sharded_imgs, sharded_lbls, sharded_test, curr_state
        del epoch_data_imgs, epoch_data_lbls

        # Re-assemble CPU history trees across chunks
        history_np = {
            'tr_loss': np.concatenate([c['tr_loss'] for c in cpu_history_trees], axis=0),
            'tr_acc': np.concatenate([c['tr_acc'] for c in cpu_history_trees], axis=0),
            'weights': np.concatenate([c['weights'] for c in cpu_history_trees], axis=0),
            'test': {}
        }
        for t_name in cpu_history_trees[0]['test'].keys():
            history_np['test'][t_name] = (
                np.concatenate([c['test'][t_name][0] for c in cpu_history_trees], axis=0),
                np.concatenate([c['test'][t_name][1] for c in cpu_history_trees], axis=0)
            )

        del cpu_history_trees
        jax.clear_caches()

        # --- Post-processing logic remains exactly the same ---
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
        
        for h in self.hooks:
            h.on_task_end(task, self.state, metrics=None)
        
        print(f"  Training complete for {task_name}", flush=True)
        return weight_history
    
    def clear_test_cache(self):
        """Clears cached test data to free memory."""
        if hasattr(self, 'cached_test_data'):
            self.cached_test_data.clear()
            print("  Test cache cleared")
    
    def prepare_static_test_data(self, test_data_dict):
        """Stacks and shards static test data once to prevent memory fragmentation."""
        test_task_names = sorted(test_data_dict.keys())
        
        # 1. Stack
        stacked_test_imgs = jnp.stack([test_data_dict[t][0] for t in test_task_names])
        stacked_test_lbls = jnp.stack([test_data_dict[t][1] for t in test_task_names])
        
        # 2. Shard
        def shard_stacked_test(stacked_arr):
            s = list(stacked_arr.shape)
            s[2:3] = [self.num_devices, self.r_per_dev] 
            reshaped = stacked_arr.reshape(*s)
            return jnp.moveaxis(reshaped, 2, 0)
            
            self.sharded_t_imgs = shard_stacked_test(stacked_test_imgs)
            self.sharded_t_lbls = shard_stacked_test(stacked_test_lbls)
            self.test_task_names = test_task_names


def calculate_safe_chunk_size(device, lowered_scan_fn, safety_margin=0.85):
    """
    Deterministically calculates how many logging steps can safely fit in available VRAM.
    
    Args:
        device: The JAX device to query.
        num_params: Total number of flattened parameters in the model.
        hidden_dim: The dimensionality of the hidden representations.
        r_per_dev: Number of repeats assigned to this specific device.
        test_data_jax: Dictionary of cached test data arrays.
        safety_margin: Fraction of free VRAM to use (leaves room for XLA compute buffers).
    """
    try:
        stats = device.memory_stats()
        # Assume the memory XLA preallocated (90%) is fully available for our arrays
        free_bytes = stats['bytes_limit'] * 0.90 
    except Exception as e:
        print(f"  [Warning] Could not read memory stats ({e}). Defaulting to 40GB margin.")
        free_bytes = 40 * (1024**3) * 0.90

    # 1. Compile the lowered function ahead-of-time to unlock exact memory statistics
    compiled_fn = lowered_scan_fn.compile()
    
    # 2. Extract exact HLO memory footprint
    mem_analysis = compiled_fn.memory_analysis()
    
    # Safely extract the attributes (JAX exposes these as properties on the memory_analysis object)
    temp_size = getattr(mem_analysis, 'temp_size_in_bytes', 0)
    code_size = getattr(mem_analysis, 'generated_code_size_in_bytes', 0)
    
    bytes_per_step = temp_size + code_size

    if bytes_per_step == 0:
        return 50 

    # Calculate how many steps fit into the safe VRAM allocation
    safe_free_bytes = free_bytes * safety_margin
    chunk_size = int(safe_free_bytes / bytes_per_step)
    
    mb_per_step = bytes_per_step / (1024**2)
    free_mb = free_bytes / (1024**2)
    print(f"  [Memory Calc] Free VRAM: {free_mb:.0f} MB | Cost per log step: {mb_per_step:.2f} MB")
    print(f"  [Memory Calc] Determined max safe chunk size: {chunk_size} logs")

    return max(1, chunk_size)