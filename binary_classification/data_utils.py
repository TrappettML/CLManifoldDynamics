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
    # Increased batch size slightly for loading speed
    loader = torch.utils.data.DataLoader(ds, batch_size=4096, num_workers=0, shuffle=False)
    
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
        """
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices) 
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            
            if end_idx - start_idx < self.batch_size and self.n_samples > self.batch_size:
                continue
                
            batch_idx = indices[start_idx:end_idx]
            
            batch_x = self.X[:, batch_idx, :]
            batch_y = self.Y[:, batch_idx, :]
            
            yield batch_x, batch_y

    def load_mandi_subset(self, samples_per_class):
        """
        Returns a simplified generator for analysis.
        FIX: Uses direct yield instead of returning a nested function.
        """
        limit = samples_per_class * 2
        
        # Check to ensure we don't slice more than exists
        limit = min(limit, self.n_samples)

        # Slice the JAX array directly
        subset_x = self.X[:, :limit, :]
        subset_y = self.Y[:, :limit, :]
        
        # FIX: Directly yield the data. 
        # When called, this method now returns a generator object (iterable),
        # matching the behavior expected by the learner loop.
        yield subset_x, subset_y

def create_continual_tasks(config, split='train'):
    img_size = getattr(config, 'downsample_dim', None)
    
    # Load FULL data as JAX arrays
    X_global, Y_global = get_base_data_jax(config.dataset_name, config.data_dir, (split=='train'), img_size)
    
    # We move Y_global back to Numpy for the 'where' logic (finding indices)
    Y_global_np = np.array(Y_global)
    
    tasks = []
    num_classes = DATASET_CONFIGS[config.dataset_name]['num_classes']
    rng = np.random.default_rng(config.seed)
    
    print(f"Generating {config.num_tasks} tasks (JAX backed)...")

    for t_i in range(config.num_tasks):
        y_list = []
        x_list = []
        
        # Consolidated Loop: Sample classes, extract data, build labels per repeat
        for r in range(config.n_repeats):
            # 1. Sample two distinct classes
            c0, c1 = rng.choice(num_classes, size=2, replace=False)
            
            # 2. Find indices in global data
            idx_c0 = np.where(Y_global_np == c0)[0]
            idx_c1 = np.where(Y_global_np == c1)[0]
            
            # 3. Balance classes (take minimum common amount)
            min_c = min(len(idx_c0), len(idx_c1))
            
            # 4. Extract JAX Data
            x_c0 = X_global[idx_c0[:min_c]]
            x_c1 = X_global[idx_c1[:min_c]]
            
            # 5. Make Binary Labels (0 and 1)
            y_0 = jnp.zeros((min_c, 1))
            y_1 = jnp.ones((min_c, 1))
            
            # 6. Concat
            x_comb = jnp.concatenate([x_c0, x_c1], axis=0)
            y_comb = jnp.concatenate([y_0, y_1], axis=0)
            
            # 7. Shuffle (Synchronous X and Y)
            perm = np.random.permutation(len(x_comb))
            x_comb = x_comb[perm]
            y_comb = y_comb[perm]
            
            x_list.append(x_comb)
            y_list.append(y_comb)

        # Truncate all repeats to the shortest length found to ensure valid stacking
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
    
    # Handle single task/repeat edge cases for axes array
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