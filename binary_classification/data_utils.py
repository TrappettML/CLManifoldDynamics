import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import warnings

# --- Dataset Metadata ---
DATASET_CONFIGS = {
    'mnist': {'input_dim': 784, 'num_classes': 10, 'channels': 1},
    'kmnist': {'input_dim': 784, 'num_classes': 10, 'channels': 1},
    'fashion_mnist': {'input_dim': 784, 'num_classes': 10, 'channels': 1},
    'cifar100': {'input_dim': 3072, 'num_classes': 100, 'channels': 1},
}

class PatchedCIFAR100(datasets.CIFAR100):
    """
    A patched version of CIFAR100 that safely handles the NumPy/Pickle 
    VisibleDeprecationWarning during data loading.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        # Initialize base with download=False to control the loading process manually
        super(datasets.CIFAR100, self).__init__(root, transform=transform, 
                                                target_transform=target_transform, download=False)
        
        self.train = train
        
        # FIX: Download MUST happen before integrity check
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # Reimplements the loading loop with a targeted warning filter
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                with warnings.catch_warnings():
                    # Filter the specific warning caused by the align=0 flag in old pickles
                    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
                    entry = pickle.load(f, encoding='latin1')
                    
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

DATASET_CLASS_MAP = {
    'mnist': datasets.MNIST,
    'kmnist': datasets.KMNIST,
    'fashion_mnist': datasets.FashionMNIST,
    'cifar100': PatchedCIFAR100
}

def get_dataset_dims(dataset_name, downsample_shape=None):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    meta = DATASET_CONFIGS[dataset_name]
    
    if downsample_shape:
        # Calculate new flat dimension: H * W * Channels
        new_dim = downsample_shape[0] * downsample_shape[1] * meta['channels']
        return int(new_dim)
    
    return meta['input_dim']

class FilteredMappedDataset(Dataset):
    """
    Wraps a standard torchvision dataset.
    1. Filters for a specific 'original_label'.
    2. Remaps that label to 'binary_label'.
    3. Applies transforms (Resize, Normalize, Flatten).
    """
    def __init__(self, dataset_cls, root, train, original_label, binary_label, img_size=None):
        # Load the base dataset (download if needed)
        self.base_dataset = dataset_cls(root=root, train=train, download=True, transform=None)
        
        # Find indices matching the original label
        targets = np.array(self.base_dataset.targets)
        self.indices = np.where(targets == original_label)[0]
        
        self.binary_label = torch.tensor(binary_label, dtype=torch.float32)
        
        # Define Transforms
        transform_list = [transforms.ToTensor()] # Converts [0,255] -> [0.0, 1.0]
        
        # Add Grayscale to ensure 1 channel if input is PIL (FashionMNIST default)
        transform_list.insert(0, transforms.Grayscale(num_output_channels=1))
        
        # Downsample if requested
        if img_size is not None:
            transform_list.insert(0, transforms.Resize((img_size, img_size)))
            
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, _ = self.base_dataset[real_idx]
        
        img = self.transform(img)
        img = torch.flatten(img)
        
        return img.numpy(), self.binary_label.numpy()

class CLTask:
    def __init__(self, task_id, name, class_definitions, split='train', 
                 batch_size=128, seed=42, data_dir="./data", img_size=None):
        self.task_id = task_id
        self.name = name
        self.class_definitions = class_definitions
        self.is_train = (split == 'train')
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir
        self.img_size = img_size 

    def _get_single_dataset(self, def_dict):
        d_name = def_dict['dataset']
        if d_name not in DATASET_CLASS_MAP:
             raise ValueError(f"Unknown dataset {d_name}")
             
        ds_cls = DATASET_CLASS_MAP[d_name]
        return FilteredMappedDataset(
            dataset_cls=ds_cls,
            root=self.data_dir,
            train=self.is_train,
            original_label=def_dict['original_label'],
            binary_label=def_dict['binary_label'],
            img_size=self.img_size
        )

    def load_data(self):
        """Returns a PyTorch DataLoader."""
        datasets_list = [self._get_single_dataset(d) for d in self.class_definitions]
        full_ds = ConcatDataset(datasets_list)
        
        shuffle = self.is_train
        
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        loader = DataLoader(
            full_ds, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            drop_last=True, 
            generator=g,
            num_workers=0 
        )
        return loader
    
    def load_mandi_subset(self, samples_per_class):
        datasets_list = []
        for d in self.class_definitions:
            full_sub_ds = self._get_single_dataset(d)
            indices = range(min(len(full_sub_ds), samples_per_class))
            small_ds = Subset(full_sub_ds, indices)
            datasets_list.append(small_ds)

        full_ds = ConcatDataset(datasets_list)
        total_samples = len(full_ds)
        
        loader = DataLoader(full_ds, batch_size=total_samples, shuffle=False)
        return loader

def create_continual_tasks(config, split='train'):
    tasks = []
    dataset_name = config.dataset_name
    total_classes = DATASET_CONFIGS[dataset_name]['num_classes']
    img_size = getattr(config, 'downsample_dim', None)
    
    if config.num_tasks * 2 > total_classes:
        raise ValueError(f"Too many tasks ({config.num_tasks}) for dataset {dataset_name}.")

    for i in range(config.num_tasks):
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
            data_dir=config.data_dir,
            img_size=img_size
        )
        tasks.append(task)
        
    return tasks

def save_task_samples_grid(tasks, config, output_file="task_samples_grid.png"):
    num_tasks = len(tasks)
    fig, axes = plt.subplots(num_tasks, 2, figsize=(6, 3 * num_tasks))
    if num_tasks == 1: axes = axes.reshape(1, -1)

    print(f"\nGenerating sample grid for {num_tasks} tasks...")
    channels = DATASET_CONFIGS[config.dataset_name].get('channels', 1)

    for i, task in enumerate(tasks):
        loader = task.load_mandi_subset(samples_per_class=1)
        batch_images, batch_labels = next(iter(loader))
        
        batch_images = batch_images.numpy()
        batch_labels = batch_labels.numpy()
        
        for binary_label in [0, 1]:
            ax = axes[i, binary_label]
            idx = np.where(batch_labels == binary_label)[0][0]
            flat_img = batch_images[idx]
            
            flat_len = flat_img.shape[0]
            side = int(np.sqrt(flat_len // channels))
            
            if channels == 1:
                img = flat_img.reshape(side, side)
                cmap = 'gray'
            else:
                img_chw = flat_img.reshape(channels, side, side)
                img = np.moveaxis(img_chw, 0, -1)
                cmap = None

            ax.imshow(img, cmap=cmap)
            orig_label = task.class_definitions[binary_label]['original_label']
            ax.set_title(f"T{task.task_id} | L{binary_label} (Orig: {orig_label})", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{config.figures_dir}/{output_file}")
    print(f"Saved task sample grid to {output_file}")
    plt.close()