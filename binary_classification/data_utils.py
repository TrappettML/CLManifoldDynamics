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
    loader = torch.utils.data.DataLoader(ds, batch_size=2048, num_workers=4, shuffle=False)
    
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
        
    # Concatenate in Numpy first (faster CPU operation)
    X_np = np.concatenate(all_x)
    Y_np = np.concatenate(all_y)
    
    # Flatten
    X_np = X_np.reshape(X_np.shape[0], -1)

    if dataset_name == 'emnist':
        Y_np = Y_np - 1

    # 4. Convert to JAX Arrays
    # This pushes data to the default backend (GPU if available)
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
        
        # We store shape info from the JAX array
        # Shape: (Repeats, N, Dim)
        self.n_repeats = self.X.shape[0]
        self.n_samples = self.X.shape[1]

    def load_data(self):
        """
        Yields batches of JAX arrays.
        Output Shape: (Repeats, Batch_Size, Dim)
        """
        # We use numpy for index manipulation (it's better at integer arithmetic/shuffling)
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices) # In-place shuffle of indices
        
        # We index the JAX array using the numpy indices
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            
            # Skip incomplete last batch to maintain shape consistency
            if end_idx - start_idx < self.batch_size and self.n_samples > self.batch_size:
                continue
                
            batch_idx = indices[start_idx:end_idx]
            
            # JAX Indexing:
            # self.X is (Repeats, N, Dim). 
            # We want all repeats, specific sample indices.
            # Syntax: [:, batch_idx, :]
            batch_x = self.X[:, batch_idx, :]
            batch_y = self.Y[:, batch_idx, :]
            
            yield batch_x, batch_y

    def load_mandi_subset(self, samples_per_class):
        """
        Returns a simplified generator for analysis.
        """
        limit = samples_per_class * 2
        
        # Slice the JAX array directly
        subset_x = self.X[:, :limit, :]
        subset_y = self.Y[:, :limit, :]
        
        def generator():
            yield subset_x, subset_y
            
        return generator

def create_continual_tasks(config, split='train'):
    img_size = getattr(config, 'downsample_dim', None)
    
    # Load FULL data as JAX arrays
    X_global, Y_global = get_base_data_jax(config.dataset_name, config.data_dir, (split=='train'), img_size)
    
    # We move Y_global back to Numpy just for the 'where' logic (finding indices)
    # because boolean indexing logic is easier to orchestrate on CPU
    Y_global_np = np.array(Y_global)
    
    tasks = []
    num_classes = DATASET_CONFIGS[config.dataset_name]['num_classes']
    rng = np.random.default_rng(config.seed)
    
    print(f"Generating {config.num_tasks} tasks (JAX backed)...")

    for t_i in range(config.num_tasks):
        indices_per_repeat = []
        
        # 1. Determine Indices using Numpy (CPU)
        for r in range(config.n_repeats):
            c0, c1 = rng.choice(num_classes, size=2, replace=False)
            
            idx_c0 = np.where(Y_global_np == c0)[0]
            idx_c1 = np.where(Y_global_np == c1)[0]
            
            min_c = min(len(idx_c0), len(idx_c1))
            idx_c0 = idx_c0[:min_c]
            idx_c1 = idx_c1[:min_c]
            
            # Combine
            current_indices = np.concatenate([idx_c0, idx_c1])
            np.random.shuffle(current_indices)
            
            indices_per_repeat.append(current_indices)
            
        # 2. Rectify Lengths
        min_len = min(len(x) for x in indices_per_repeat)
        final_indices = [x[:min_len] for x in indices_per_repeat]
        
        # Shape: (Repeats, Min_Len)
        indices_array = np.stack(final_indices, axis=0) 
        
        # 3. Create JAX Arrays for this Task
        # We use advanced indexing on the Global JAX array
        # X_global: (Total_Data, Dim)
        # indices_array: (Repeats, Task_Samples)
        # Result task_X: (Repeats, Task_Samples, Dim)
        task_X = X_global[indices_array] 
        
        # Labels need to be 0/1. We cannot just index Y_global.
        # We need to construct the 0/1 labels manually.
        # But wait, we shuffled the indices, so we don't know which is which easily unless we track it.
        # Alternative: We pull the real labels, then map them to 0/1.
        
        real_labels = Y_global[indices_array] # (Repeats, Task_Samples)
        
        # Map c0 -> 0, c1 -> 1. 
        # Since c0/c1 change per repeat, we need to be careful.
        # Actually, simpler approach: Construct the Y array manually during the loop, like before.
        
        # Re-doing the loop to build Y properly:
        y_list = []
        x_list = []
        
        for r in range(config.n_repeats):
            # Recalculate c0, c1 or store them? Let's just store the indices above? 
            # Optimization: Let's do the extraction in the loop to be safe.
            
            # Note: We need to re-run the random choice to match the logic or just do it here.
            # Let's restart the loop logic to be clean:
            c0, c1 = rng.choice(num_classes, size=2, replace=False)
            idx_c0 = np.where(Y_global_np == c0)[0]
            idx_c1 = np.where(Y_global_np == c1)[0]
            min_c = min(len(idx_c0), len(idx_c1))
            
            # Get JAX Data
            x_c0 = X_global[idx_c0[:min_c]]
            x_c1 = X_global[idx_c1[:min_c]]
            
            # Make JAX Labels
            y_0 = jnp.zeros((min_c, 1))
            y_1 = jnp.ones((min_c, 1))
            
            # Concat
            x_comb = jnp.concatenate([x_c0, x_c1], axis=0)
            y_comb = jnp.concatenate([y_0, y_1], axis=0)
            
            # Shuffle (Need to use indices for synchronous shuffle)
            perm = np.random.permutation(len(x_comb))
            x_comb = x_comb[perm]
            y_comb = y_comb[perm]
            
            x_list.append(x_comb)
            y_list.append(y_comb)

        # Truncate
        min_len = min(len(x) for x in x_list)
        x_list = [x[:min_len] for x in x_list]
        y_list = [y[:min_len] for y in y_list]
        
        # Stack into (Repeats, N, Dim)
        task_X = jnp.stack(x_list, axis=0) 
        task_Y = jnp.stack(y_list, axis=0)
        
        task_name = f"T{t_i+1}_{config.dataset_name}"
        tasks.append(FastVectorizedTask(t_i+1, task_name, task_X, task_Y, config.batch_size))
        
    return tasks

def save_task_samples_grid(tasks, config, output_file="task_samples_grid.png"):
    n_disp_repeats = min(config.n_repeats, 5)
    num_tasks = len(tasks)
    
    fig, axes = plt.subplots(n_disp_repeats, num_tasks, figsize=(num_tasks * 2, n_disp_repeats * 2))
    if n_disp_repeats == 1 and num_tasks == 1: axes = np.array([[axes]])
    elif n_disp_repeats == 1: axes = axes.reshape(1, -1)
    elif num_tasks == 1: axes = axes.reshape(-1, 1)

    print(f"\nSaving task grid to {output_file}...")
    
    for r in range(n_disp_repeats):
        for t_idx, task in enumerate(tasks):
            # Task.X is (Repeats, N, Dim) JAX Array
            # We must convert to Numpy for Matplotlib
            img_flat = np.array(task.X[r, 0, :])
            lbl = np.array(task.Y[r, 0, 0])
            
            side = int(np.sqrt(img_flat.shape[0]))
            img = img_flat.reshape(side, side)
            
            ax = axes[r, t_idx]
            ax.imshow(img, cmap='gray')
            
            if r == 0:
                ax.set_title(task.name, fontsize=10, fontweight='bold')
            
            ax.set_xlabel(f"Lbl: {int(lbl)}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(config.figures_dir, output_file)
    plt.savefig(save_path)
    plt.close()