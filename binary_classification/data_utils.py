import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp

# --- Configuration ---
DATASET_CONFIGS = {
    'mnist': {'input_dim': 784, 'num_classes': 10, 'channels': 1, 'cls': datasets.MNIST},
    'kmnist': {'input_dim': 784, 'num_classes': 10, 'channels': 1, 'cls': datasets.KMNIST},
    'fashion_mnist': {'input_dim': 784, 'num_classes': 10, 'channels': 1, 'cls': datasets.FashionMNIST},
    'cifar100': {'input_dim': 3072, 'num_classes': 100, 'channels': 3, 'cls': datasets.CIFAR100},
    'emnist': {'input_dim': 784, 'num_classes': 26, 'channels': 1, 'cls': datasets.EMNIST, 'kwargs': {'split': 'letters'}},
}

def get_base_data_jax(dataset_name, root, train, img_size):
    """
    Loads dataset via Torch/Numpy, then immediately converts to JAX Arrays.
    Returns:
        X: (N, Flattened_Dim) jax.numpy.array
        Y: (N,) jax.numpy.array
    """
    cfg = DATASET_CONFIGS[dataset_name]
    
    # 1. Define Transforms (Resize + Grayscale + Flatten)
    t_list = []
    if cfg['channels'] == 3:
        t_list.append(transforms.Grayscale(num_output_channels=1))
        
    if img_size:
        t_list.append(transforms.Resize((img_size, img_size)))
    
    t_list.append(transforms.ToTensor())
    transform = transforms.Compose(t_list)

    # 2. Instantiate Base Dataset
    ds_kwargs = cfg.get('kwargs', {})
    ds = cfg['cls'](root=root, train=train, download=True, transform=transform, **ds_kwargs)

    # 3. Fast Load to Numpy first
    print(f"Loading {dataset_name} (Train={train}) into JAX arrays...")
    loader = torch.utils.data.DataLoader(ds, batch_size=4096, num_workers=0, shuffle=False)
    
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
        
    # Concatenate in Numpy first (faster CPU operation)
    X_np = np.concatenate(all_x)
    Y_np = np.concatenate(all_y)
    
    # Flatten: (N, C, H, W) -> (N, Dim)
    X_np = X_np.reshape(X_np.shape[0], -1)

    # EMNIST adjustment (labels 1-26 -> 0-25)
    if dataset_name == 'emnist':
        Y_np = Y_np - 1

    # 4. Convert to JAX Arrays
    X_jax = jnp.array(X_np)
    Y_jax = jnp.array(Y_np)
    
    return X_jax, Y_jax

class FastVectorizedTask:
    def __init__(self, task_id, name, X_jax, Y_jax, batch_size):
        """
        X_jax: (Repeats, Total_Samples, Dim) JAX Array
        Y_jax: (Repeats, Total_Samples, 1)   JAX Array
        """
        self.task_id = task_id
        self.name = name
        self.X = X_jax
        self.Y = Y_jax
        self.batch_size = batch_size
        
        # Shape: (Repeats, N, Dim)
        self.n_repeats = self.X.shape[0]
        self.n_samples = self.X.shape[1]

    def load_data(self):
        """
        Yields batches of JAX arrays.
        Output Shape: (Repeats, Batch_Size, Dim)
        
        Crucial: We shuffle INDICES, then apply the same indices to all repeats.
        This keeps the 'time' dimension aligned across vectorization, which is 
        better for gradient stability and JIT compilation.
        """
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices) 
        
        # Apply shuffle to the data upfront (or index on the fly)
        # Indexing on the fly in JAX is fast enough for this scale.
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            
            # Drop last partial batch to ensure fixed shapes for JIT
            if end_idx - start_idx < self.batch_size and self.n_samples > self.batch_size:
                continue
                
            batch_idx = indices[start_idx:end_idx]
            
            # Slice: (Repeats, Batch, Dim)
            # Since self.X is (Repeats, Total, Dim), we slice dim 1.
            batch_x = self.X[:, batch_idx, :]
            batch_y = self.Y[:, batch_idx, :]
            
            yield batch_x, batch_y

    def load_mandi_subset(self, samples_per_class):
        """
        Returns a simplified generator for analysis (manifold geometry).
        Yields exactly ONE batch containing (Repeats, 2*samples_per_class, Dim).
        """
        # We take the first 2*N samples. 
        # Since the data construction (create_continual_tasks) balances 0s and 1s 
        # and shuffles them, taking the first K samples is a random subset.
        
        limit = samples_per_class * 2
        limit = min(limit, self.n_samples)

        subset_x = self.X[:, :limit, :]
        subset_y = self.Y[:, :limit, :]
        
        yield subset_x, subset_y

def create_continual_tasks(config, split='train'):
    img_size = getattr(config, 'downsample_dim', None)
    
    # 1. Load FULL data as JAX arrays ONCE
    # X_global: (Total_N, Dim), Y_global: (Total_N,)
    X_global, Y_global = get_base_data_jax(config.dataset_name, config.data_dir, (split=='train'), img_size)
    
    # We move Y_global back to Numpy for the 'where' logic (finding indices)
    # as boolean indexing is often easier to debug in standard numpy.
    Y_global_np = np.array(Y_global)
    
    tasks = []
    num_classes = DATASET_CONFIGS[config.dataset_name]['num_classes']
    rng = np.random.default_rng(config.seed)
    
    print(f"Generating {config.num_tasks} tasks (JAX backed)...")

    for t_i in range(config.num_tasks):
        # We need to store raw arrays first, then stack them.
        # Because different repeats might pick classes with different sample counts,
        # we must find the MINIMUM sample count across all repeats to ensure a rectangular tensor.
        
        repeat_data_cache = [] # List of tuples (x_comb, y_comb)
        min_samples_in_task = float('inf')

        # --- 1. Sample Data for Each Repeat ---
        for r in range(config.n_repeats):
            # A. Sample two distinct classes
            c0, c1 = rng.choice(num_classes, size=2, replace=False)
            
            # B. Find indices in global data
            idx_c0 = np.where(Y_global_np == c0)[0]
            idx_c1 = np.where(Y_global_np == c1)[0]
            
            # C. Balance classes (take minimum common amount within this repeat)
            min_c = min(len(idx_c0), len(idx_c1))
            
            # D. Extract Data (Use JAX slicing)
            x_c0 = X_global[idx_c0[:min_c]]
            x_c1 = X_global[idx_c1[:min_c]]
            
            # E. Make Binary Labels (0 and 1)
            # Ensure float32 for Binary Cross Entropy
            y_0 = jnp.zeros((min_c, 1), dtype=jnp.float32)
            y_1 = jnp.ones((min_c, 1), dtype=jnp.float32)
            
            # F. Concatenate
            x_comb = jnp.concatenate([x_c0, x_c1], axis=0)
            y_comb = jnp.concatenate([y_0, y_1], axis=0)
            
            # G. Internal Shuffle (for this repeat)
            # We shuffle here so that "Sample 0" isn't always Class 0.
            perm = np.random.permutation(len(x_comb))
            x_comb = x_comb[perm]
            y_comb = y_comb[perm]
            
            repeat_data_cache.append((x_comb, y_comb))
            
            if len(x_comb) < min_samples_in_task:
                min_samples_in_task = len(x_comb)

        # --- 2. Truncate and Stack ---
        # Now we force all repeats to have 'min_samples_in_task' so jnp.stack works.
        final_x_list = []
        final_y_list = []
        
        for (x, y) in repeat_data_cache:
            final_x_list.append(x[:min_samples_in_task])
            final_y_list.append(y[:min_samples_in_task])
            
        # Stack into (Repeats, N, Dim)
        task_X = jnp.stack(final_x_list, axis=0) 
        task_Y = jnp.stack(final_y_list, axis=0)
        
        task_name = f"T{t_i+1}_{config.dataset_name}"
        
        print(f"  [{task_name}] Created. Shape: {task_X.shape}. Classes varied per repeat.")
        
        tasks.append(FastVectorizedTask(t_i+1, task_name, task_X, task_Y, config.batch_size))
        
    return tasks

def save_task_samples_grid(tasks, config, output_file="task_samples_grid.png"):
    """
    Visualizes the first few samples of the generated tasks to verify data correctness.
    """
    n_disp_repeats = min(config.n_repeats, 5)
    num_tasks = len(tasks)
    
    # Handle single task edge case for subplot dimensions
    if num_tasks == 0: return

    fig, axes = plt.subplots(n_disp_repeats, num_tasks, figsize=(num_tasks * 2.5, n_disp_repeats * 2.5))
    
    # Handle array standardizations for consistent indexing
    if n_disp_repeats == 1 and num_tasks == 1: 
        axes = np.array([[axes]])
    elif n_disp_repeats == 1: 
        axes = axes.reshape(1, -1)
    elif num_tasks == 1: 
        axes = axes.reshape(-1, 1)

    print(f"\nSaving task grid to {output_file}...")
    
    for r in range(n_disp_repeats):
        for t_idx, task in enumerate(tasks):
            # Task.X is (Repeats, N, Dim) JAX Array
            # We must convert to Numpy for Matplotlib
            img_flat = np.array(task.X[r, 0, :])
            lbl = np.array(task.Y[r, 0, 0])
            
            # Infer Sqrt for square image reconstruction
            side = int(np.sqrt(img_flat.shape[0]))
            img = img_flat.reshape(side, side)
            
            ax = axes[r, t_idx]
            ax.imshow(img, cmap='gray')
            
            if r == 0:
                ax.set_title(f"{task.name}", fontsize=10, fontweight='bold')
            
            ax.set_xlabel(f"Rep {r} | L: {int(lbl)}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(config.figures_dir, output_file)
    plt.savefig(save_path)
    plt.close()