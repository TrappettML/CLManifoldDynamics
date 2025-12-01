import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# --- Dataset Metadata ---
DATASET_CONFIGS = {
    'mnist': {'input_dim': 784, 'num_classes': 10},
    'fashion_mnist': {'input_dim': 784, 'num_classes': 10},
    'cifar100': {'input_dim': 3072, 'num_classes': 100},
}

def get_dataset_dims(dataset_name):
    """Returns the flattened input dimension for the chosen dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]['input_dim']

class CLTask:
    def __init__(self, task_id, name, class_definitions, split='train', batch_size=128, seed=42, data_dir="./data"):
        self.task_id = task_id
        self.name = name
        self.class_definitions = class_definitions
        self.split = split
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir

    def _load_single_source(self, dataset_name, original_label, binary_label):
        # Load dataset. as_supervised=True returns (image, label)
        ds = tfds.load(dataset_name, split=self.split, data_dir=self.data_dir, as_supervised=True, shuffle_files=False)
        
        def filter_fn(image, label):
            return label == original_label
        
        def map_fn(image, label):
            # Normalize to 0-1
            image = tf.cast(image, tf.float32) / 255.0
            # Flatten dynamically (works for 28x28x1 or 32x32x3)
            image = tf.reshape(image, [-1])
            new_label = tf.constant(binary_label, dtype=tf.float32)
            return image, new_label

        ds = ds.filter(filter_fn)
        ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    def load_data(self):
        datasets = []
        for comp in self.class_definitions:
            ds = self._load_single_source(comp['dataset'], comp['original_label'], comp['binary_label'])
            datasets.append(ds)
            
        full_ds = datasets[0]
        for next_ds in datasets[1:]:
            full_ds = full_ds.concatenate(next_ds)
            
        if self.split == 'train':
            full_ds = full_ds.shuffle(2048, seed=self.seed)
            
        full_ds = full_ds.batch(self.batch_size, drop_remainder=True)
        full_ds = full_ds.prefetch(tf.data.AUTOTUNE)
        return full_ds
    
    def load_mandi_subset(self, samples_per_class):
        """
        Loads exactly samples_per_class for each class definition in this task.
        Returns a single unbatched dataset of size (2 * samples_per_class).
        """
        datasets = []
        for comp in self.class_definitions:
            # Load raw stream
            ds = self._load_single_source(comp['dataset'], comp['original_label'], comp['binary_label'])
            # Take exactly N samples
            ds = ds.take(samples_per_class)
            datasets.append(ds)

        # Merge them
        full_ds = datasets[0]
        for next_ds in datasets[1:]:
            full_ds = full_ds.concatenate(next_ds)
        
        # Batch effectively creates one large array since we want them all at once
        total_samples = samples_per_class * len(datasets)
        full_ds = full_ds.batch(total_samples)
        return full_ds
    

def create_continual_tasks(config, split='train'):
    """
    Generates a list of CLTask objects based on the config.dataset_name.
    """
    tasks = []
    dataset_name = config.dataset_name
    total_classes = DATASET_CONFIGS[dataset_name]['num_classes']
    
    # Validation
    if config.num_tasks * 2 > total_classes:
        raise ValueError(f"Too many tasks requested ({config.num_tasks}) for dataset {dataset_name} "
                         f"which has {total_classes} classes (needs 2 classes per task).")

    for i in range(config.num_tasks):
        # Determine class indices (0 vs 1, 2 vs 3, etc.)
        idx_0 = (i * 2) % total_classes
        idx_1 = (i * 2 + 1) % total_classes
        
        task_name = f"T{i+1}_{dataset_name}_C{idx_0}v{idx_1}"
        
        class_defs = [
            {'dataset': dataset_name, 'original_label': idx_0, 'binary_label': 0},
            {'dataset': dataset_name, 'original_label': idx_1, 'binary_label': 1}
        ]
        
        task = CLTask(
            task_id=i+1, 
            name=task_name, 
            class_definitions=class_defs, 
            split=split, 
            batch_size=config.batch_size, 
            seed=config.seed, 
            data_dir=config.data_dir
        )
        tasks.append(task)
        
    return tasks


def save_task_samples_grid(tasks, config, output_file="task_samples_grid.png"):
    """
    Generates a grid of example images: Rows = Tasks, Columns = Binary Labels (0, 1).
    """
    import matplotlib.pyplot as plt
    import tensorflow_datasets as tfds

    num_tasks = len(tasks)
    # Create subplots: Rows = tasks, Cols = 2 (Label 0 and Label 1)
    fig, axes = plt.subplots(num_tasks, 2, figsize=(6, 3 * num_tasks))
    
    # Handle single task edge case (axes is 1D array)
    if num_tasks == 1:
        axes = axes.reshape(1, -1)

    print(f"\nGenerating sample grid for {num_tasks} tasks...")

    for i, task in enumerate(tasks):
        # 1. Load exactly 1 sample per class using existing utility
        # This returns a single batch containing [Class 0 Image, Class 1 Image]
        subset_ds = task.load_mandi_subset(samples_per_class=1)
        ds_numpy = tfds.as_numpy(subset_ds)
        
        # Extract the batch (images, labels)
        batch_images, batch_labels = next(iter(ds_numpy))
        
        # 2. Iterate through the binary labels (0 and 1)
        for binary_label in [0, 1]:
            ax = axes[i, binary_label]
            
            # Get the image corresponding to this label
            # batch_labels is shape (2, 1) or (2,), find index of current label
            idx = np.where(batch_labels.flatten() == binary_label)[0][0]
            flat_img = batch_images[idx]
            
            # 3. Reshape Flat Vector -> Image
            # Logic based on config.dataset_name
            if config.dataset_name in ['mnist', 'fashion_mnist']:
                img = flat_img.reshape(28, 28)
                cmap = 'gray'
            elif config.dataset_name == 'cifar100':
                img = flat_img.reshape(32, 32, 3)
                cmap = None
            else:
                # Fallback for unknown dims, try square root
                side = int(np.sqrt(flat_img.shape[0]))
                img = flat_img.reshape(side, side)
                cmap = 'gray'

            # 4. Plot
            ax.imshow(img, cmap=cmap)
            
            # Get original class ID for the title (e.g., "7 vs 9")
            orig_label = task.class_definitions[binary_label]['original_label']
            
            ax.set_title(f"Task {task.task_id} | Label {binary_label}\n(Original Class: {orig_label})", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{config.figures_dir}/{output_file}")
    print(f"Saved task sample grid to {output_file}")
    plt.close()