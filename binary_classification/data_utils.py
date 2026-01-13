import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
import json

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
    Loads dataset via Torch/Numpy, then converts to JAX Arrays.
    
    Returns:
        X: (N, Flattened_Dim) jax.numpy.array
        Y: (N,) jax.numpy.array
    """
    cfg = DATASET_CONFIGS[dataset_name]
    
    # Define Transforms (Grayscale if needed + Resize + ToTensor)
    t_list = []
    if cfg['channels'] == 3:
        t_list.append(transforms.Grayscale(num_output_channels=1))
    if img_size:
        t_list.append(transforms.Resize((img_size, img_size)))
    t_list.append(transforms.ToTensor())
    transform = transforms.Compose(t_list)

    # Instantiate Dataset
    ds_kwargs = cfg.get('kwargs', {})
    ds = cfg['cls'](root=root, train=train, download=True, transform=transform, **ds_kwargs)

    # Fast Load to Numpy
    print(f"Loading {dataset_name} (Train={train}) into JAX arrays...")
    loader = torch.utils.data.DataLoader(ds, batch_size=4096, num_workers=0, shuffle=False)
    
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
        
    X_np = np.concatenate(all_x)
    Y_np = np.concatenate(all_y)
    
    # Flatten: (N, C, H, W) -> (N, Dim)
    X_np = X_np.reshape(X_np.shape[0], -1)

    # EMNIST adjustment (labels 1-26 -> 0-25)
    if dataset_name == 'emnist':
        Y_np = Y_np - 1

    # Convert to JAX
    X_jax = jnp.array(X_np)
    Y_jax = jnp.array(Y_np)
    
    return X_jax, Y_jax


def generate_task_class_pairs(num_tasks, n_repeats, num_classes, seed):
    """
    PRE-COMPUTES class pairs for all tasks and repeats following the spec.
    
    Spec Requirement (Section 1):
    "Each repeat samples a random permutation of all available classes without replacement."
    
    Args:
        num_tasks: Number of sequential tasks (T)
        n_repeats: Number of parallel repeats (R)
        num_classes: Total classes in dataset
        seed: Random seed for reproducibility
        
    Returns:
        task_class_pairs: Shape (T, R, 2) - [task_idx, repeat_idx] = (class_A, class_B)
        
    Example (MNIST, 10 classes, T=5):
        Repeat 0: [(0,1), (2,3), (4,5), (6,7), (8,9)]
        Repeat 1: [(3,7), (0,5), (1,8), (2,6), (4,9)]
    """
    rng = np.random.default_rng(seed)
    
    # Validate that we can form num_tasks pairs from num_classes
    max_possible_tasks = num_classes // 2
    if num_tasks > max_possible_tasks:
        raise ValueError(
            f"Cannot create {num_tasks} tasks from {num_classes} classes. "
            f"Maximum possible: {max_possible_tasks} tasks."
        )
    
    task_class_pairs = np.zeros((num_tasks, n_repeats, 2), dtype=np.int32)
    
    print(f"\n=== Pre-computing Task Class Pairs ===")
    print(f"Tasks: {num_tasks}, Repeats: {n_repeats}, Classes: {num_classes}")
    
    for r in range(n_repeats):
        # Generate random permutation of all classes for this repeat
        perm = rng.permutation(num_classes)
        
        # Pair consecutive classes: [perm[0], perm[1]], [perm[2], perm[3]], ...
        for t in range(num_tasks):
            class_A = perm[2 * t]
            class_B = perm[2 * t + 1]
            task_class_pairs[t, r, 0] = class_A
            task_class_pairs[t, r, 1] = class_B
            
        # print(f"  Repeat {r}: {[(task_class_pairs[t, r, 0], task_class_pairs[t, r, 1]) for t in range(num_tasks)]}")
    
    return task_class_pairs


def create_single_task_data(task_idx, task_class_pairs, X_global, Y_global, config, split='train'):
    """
    LAZY LOADING: Creates data for a single task on-demand.
    
    Spec Requirement (Section 2):
    "Training data for Task t should be generated/loaded to GPU VRAM immediately 
    before Task t begins and cleared immediately after."
    
    Args:
        task_idx: Task index (0-based)
        task_class_pairs: Pre-computed pairs, shape (T, R, 2)
        X_global: Global dataset images (N, Dim)
        Y_global: Global dataset labels (N,)
        config: Configuration object
        split: 'train' or 'test'
        
    Returns:
        task_X: (N_samples, R, Dim) - Canonical format
        task_Y: (N_samples, R, 1) - Binary labels {0, 1}
        task_name: String identifier
    """
    n_repeats = config.n_repeats
    Y_global_np = np.array(Y_global)
    
    repeat_data_cache = []
    min_samples_in_task = float('inf')

    # Build data for each repeat using pre-computed class pairs
    for r in range(n_repeats):
        class_A = int(task_class_pairs[task_idx, r, 0])
        class_B = int(task_class_pairs[task_idx, r, 1])
        
        # Find indices for both classes
        idx_A = np.where(Y_global_np == class_A)[0]
        idx_B = np.where(Y_global_np == class_B)[0]
        
        # Balance classes
        min_c = min(len(idx_A), len(idx_B))
        
        # Extract data
        x_A = X_global[idx_A[:min_c]]
        x_B = X_global[idx_B[:min_c]]
        
        # Binary labels: class_A -> 0, class_B -> 1 (Spec Section 1)
        y_0 = jnp.zeros((min_c, 1), dtype=jnp.float32)
        y_1 = jnp.ones((min_c, 1), dtype=jnp.float32)
        
        # Concatenate
        x_comb = jnp.concatenate([x_A, x_B], axis=0)
        y_comb = jnp.concatenate([y_0, y_1], axis=0)
        
        # Shuffle within repeat
        perm = np.random.permutation(len(x_comb))
        x_comb = x_comb[perm]
        y_comb = y_comb[perm]
        
        repeat_data_cache.append((x_comb, y_comb))
        min_samples_in_task = min(min_samples_in_task, len(x_comb))

    # Truncate to minimum and stack
    final_x_list = [x[:min_samples_in_task] for x, _ in repeat_data_cache]
    final_y_list = [y[:min_samples_in_task] for _, y in repeat_data_cache]
    
    # Stack: (R, N, D) -> Transpose to Canonical: (N, R, D)
    task_X = jnp.stack(final_x_list, axis=0)
    task_Y = jnp.stack(final_y_list, axis=0)
    
    task_X = jnp.swapaxes(task_X, 0, 1)
    task_Y = jnp.swapaxes(task_Y, 0, 1)
    
    task_name = f"task_{task_idx:03d}"
    
    return task_X, task_Y, task_name


def preload_all_test_data(task_class_pairs, X_global, Y_global, config):
    """
    PRE-LOADS all test data as required by spec.
    
    Spec Requirement (Section 2):
    "Test sets for all T tasks are pre-loaded and kept in memory throughout 
    the entire experiment."
    
    Returns:
        test_data_dict: {task_name: (images, labels)} in Canonical format
    """
    print(f"\n=== Pre-loading ALL Test Data ===")
    test_data_dict = {}
    
    for t in range(config.num_tasks):
        task_X, task_Y, task_name = create_single_task_data(
            t, task_class_pairs, X_global, Y_global, config, split='test'
        )
        test_data_dict[task_name] = (task_X, task_Y)
        print(f"  [{task_name}] Loaded. Shape: {task_X.shape}")
    
    return test_data_dict


def save_task_metadata(task_idx, task_class_pairs, config, additional_info=None):
    """
    Saves metadata.json as specified in Section 4.
    
    Args:
        task_idx: Task index
        task_class_pairs: Pre-computed class pairs array
        config: Configuration object
        additional_info: Optional dict with extra metadata
    """
    task_dir = os.path.join("results", config.dataset_name, config.algorithm, f"task_{task_idx:03d}")
    os.makedirs(task_dir, exist_ok=True)
    
    # Extract class pairs for this specific task
    pairs_for_task = []
    for r in range(config.n_repeats):
        pairs_for_task.append({
            "repeat": r,
            "class_A": int(task_class_pairs[task_idx, r, 0]),
            "class_B": int(task_class_pairs[task_idx, r, 1])
        })
    
    metadata = {
        "task_id": task_idx,
        "task_name": f"task_{task_idx:03d}",
        "dataset": config.dataset_name,
        "algorithm": config.algorithm,
        "n_repeats": config.n_repeats,
        "task_class_pairs": pairs_for_task,
        "seed": config.seed,
        "hidden_dim": config.hidden_dim,
        "learning_rate1": config.learning_rate1,
        "learning_rate2": config.learning_rate2,
        "batch_size": config.batch_size,
        "epochs_per_task": config.epochs_per_task
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    save_path = os.path.join(task_dir, "metadata.json")
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def save_task_samples_grid(task_class_pairs, X_global, Y_global, config, output_file="task_samples_grid.png"):
    """
    Visualizes first samples from generated tasks for verification.
    """
    n_disp_repeats = min(config.n_repeats, 5)
    num_tasks = config.num_tasks
    
    if num_tasks == 0:
        return

    fig, axes = plt.subplots(n_disp_repeats, num_tasks, figsize=(num_tasks * 2.5, n_disp_repeats * 2.5))
    
    # Handle subplot dimensions
    if n_disp_repeats == 1 and num_tasks == 1:
        axes = np.array([[axes]])
    elif n_disp_repeats == 1:
        axes = axes.reshape(1, -1)
    elif num_tasks == 1:
        axes = axes.reshape(-1, 1)

    print(f"\nGenerating task visualization grid...")
    
    for t_idx in range(num_tasks):
        task_X, task_Y, task_name = create_single_task_data(
            t_idx, task_class_pairs, X_global, Y_global, config, split='train'
        )
        
        for r in range(n_disp_repeats):
            img_flat = np.array(task_X[0, r, :])
            lbl = np.array(task_Y[0, r, 0])
            
            side = int(np.sqrt(img_flat.shape[0]))
            img = img_flat.reshape(side, side)
            
            ax = axes[r, t_idx]
            ax.imshow(img, cmap='gray')
            
            if r == 0:
                ax.set_title(f"{task_name}", fontsize=10, fontweight='bold')
            
            # Show actual class pair for this repeat
            class_A = task_class_pairs[t_idx, r, 0]
            class_B = task_class_pairs[t_idx, r, 1]
            ax.set_xlabel(f"R{r} | ({class_A},{class_B}) L:{int(lbl)}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(config.figures_dir, output_file)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {save_path}")