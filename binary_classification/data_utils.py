import tensorflow as tf
import tensorflow_datasets as tfds

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