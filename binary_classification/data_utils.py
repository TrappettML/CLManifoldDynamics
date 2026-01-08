import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import warnings
import jax.numpy as jnp

# --- Dataset Metadata ---
# num_classes defines the pool size for random sampling.
DATASET_CONFIGS = {
    'mnist': {'input_dim': 784, 'num_classes': 10, 'channels': 1},
    'kmnist': {'input_dim': 784, 'num_classes': 10, 'channels': 1},
    'fashion_mnist': {'input_dim': 784, 'num_classes': 10, 'channels': 1},
    'cifar100': {'input_dim': 3072, 'num_classes': 100, 'channels': 1},
    'emnist': {'input_dim': 784, 'num_classes': 26, 'channels': 1},
}

class PatchedCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(datasets.CIFAR100, self).__init__(root, transform=transform, 
                                                target_transform=target_transform, download=False)
        self.train = train
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

class PatchedEMNIST(datasets.EMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        # EMNIST 'letters' split provides 26 classes (A-Z).
        super().__init__(root, split='letters', train=train, transform=transform,
                         target_transform=target_transform, download=download)
        
        # EMNIST Letters are inherently 1-based (1-26). 
        # We shift them to 0-based (0-25) to align with standard indexing.
        self.targets = self.targets - 1

DATASET_CLASS_MAP = {
    'mnist': datasets.MNIST,
    'kmnist': datasets.KMNIST,
    'fashion_mnist': datasets.FashionMNIST,
    'cifar100': PatchedCIFAR100,
    'emnist': PatchedEMNIST,
}

class FilteredMappedDataset(Dataset):
    def __init__(self, dataset_cls, root, train, original_label, binary_label, img_size=None):
        self.base_dataset = dataset_cls(root=root, train=train, download=True, transform=None)
        
        targets = np.array(self.base_dataset.targets)
        self.indices = np.where(targets == original_label)[0]
        
        self.binary_label = torch.tensor(binary_label, dtype=torch.float32)
        
        transform_list = [transforms.ToTensor()] 
        transform_list.insert(0, transforms.Grayscale(num_output_channels=1))
        
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

class MultiRepeatDataLoader:
    def __init__(self, repeat_datasets, batch_size, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        
        self.images_per_repeat = []
        self.labels_per_repeat = []
        
        min_len = float('inf')
        
        for ds in repeat_datasets:
            dl = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)
            imgs, lbls = next(iter(dl))
            self.images_per_repeat.append(imgs.numpy())
            self.labels_per_repeat.append(lbls.numpy())
            if len(ds) < min_len:
                min_len = len(ds)
                
        self.n_samples = min_len
        self.n_repeats = len(repeat_datasets)
        
        self.images = np.stack([x[:self.n_samples] for x in self.images_per_repeat])
        self.labels = np.stack([x[:self.n_samples] for x in self.labels_per_repeat])
        
    def __iter__(self):
        indices = np.zeros((self.n_repeats, self.n_samples), dtype=int)
        for r in range(self.n_repeats):
            idxs = np.arange(self.n_samples)
            if self.shuffle:
                self.rng.shuffle(idxs)
            indices[r] = idxs
            
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            
            if start_idx >= end_idx:
                break
                
            if self.batch_size < self.n_samples:
                 if end_idx - start_idx < self.batch_size:
                     continue
            
            batch_indices = indices[:, start_idx:end_idx] 
            
            batch_imgs = []
            batch_lbls = []
            for r in range(self.n_repeats):
                batch_imgs.append(self.images[r, batch_indices[r]])
                batch_lbls.append(self.labels[r, batch_indices[r]])
                
            yield np.stack(batch_imgs), np.stack(batch_lbls)

    def get_full_data(self):
        return self.images, self.labels

class VectorizedCLTask:
    def __init__(self, task_id, name, repeat_configs, split='train', 
                 batch_size=128, seed=42, data_dir="./data", img_size=None):
        self.task_id = task_id
        self.name = name
        self.repeat_configs = repeat_configs 
        self.is_train = (split == 'train')
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir
        self.img_size = img_size
        
    def _get_dataset_for_repeat(self, config_list):
        datasets_list = []
        for def_dict in config_list:
            d_name = def_dict['dataset']
            ds_cls = DATASET_CLASS_MAP[d_name]
            d = FilteredMappedDataset(
                dataset_cls=ds_cls,
                root=self.data_dir,
                train=self.is_train,
                original_label=def_dict['original_label'],
                binary_label=def_dict['binary_label'],
                img_size=self.img_size
            )
            datasets_list.append(d)
        return ConcatDataset(datasets_list)

    def load_data(self):
        per_repeat_datasets = []
        for r_conf in self.repeat_configs:
            per_repeat_datasets.append(self._get_dataset_for_repeat(r_conf))
            
        loader = MultiRepeatDataLoader(
            per_repeat_datasets, 
            batch_size=self.batch_size, 
            shuffle=self.is_train,
            seed=self.seed
        )
        return loader

    def load_mandi_subset(self, samples_per_class):
        per_repeat_datasets = []
        
        for r_conf in self.repeat_configs:
            class_datasets = []
            for def_dict in r_conf:
                d_name = def_dict['dataset']
                ds_cls = DATASET_CLASS_MAP[d_name]
                full = FilteredMappedDataset(
                    dataset_cls=ds_cls, root=self.data_dir, train=self.is_train,
                    original_label=def_dict['original_label'],
                    binary_label=def_dict['binary_label'],
                    img_size=self.img_size
                )
                indices = range(min(len(full), samples_per_class))
                class_datasets.append(Subset(full, indices))
            
            per_repeat_datasets.append(ConcatDataset(class_datasets))
        
        loader = MultiRepeatDataLoader(per_repeat_datasets, batch_size=999999, shuffle=False)
        return loader

def create_continual_tasks(config, split='train'):
    tasks = []
    dataset_name = config.dataset_name
    total_classes = DATASET_CONFIGS[dataset_name]['num_classes']
    img_size = getattr(config, 'downsample_dim', None)
    n_repeats = config.n_repeats
    rng = np.random.default_rng(config.seed)

    task_configs = [ [] for _ in range(config.num_tasks) ]
    
    for r in range(n_repeats):
        for t_i in range(config.num_tasks):
            # Select 2 distinct classes from the FULL pool (0 to total_classes-1)
            # This allows reuse of classes across tasks (e.g. Task 1: 0 vs 1, Task 2: 0 vs 2)
            c0, c1 = rng.choice(total_classes, size=2, replace=False)
            
            def_list = [
                {'dataset': dataset_name, 'original_label': int(c0), 'binary_label': 0},
                {'dataset': dataset_name, 'original_label': int(c1), 'binary_label': 1}
            ]
            task_configs[t_i].append(def_list)

    for i in range(config.num_tasks):
        task_name = f"T{i+1}_{dataset_name}"
        task = VectorizedCLTask(
            task_id=i+1,
            name=task_name,
            repeat_configs=task_configs[i],
            split=split,
            batch_size=config.batch_size,
            seed=config.seed,
            data_dir=config.data_dir,
            img_size=img_size
        )
        tasks.append(task)
        
    return tasks

def save_task_samples_grid(tasks, config, output_file="task_samples_grid.png"):
    n_disp_repeats = min(config.n_repeats, 5)
    num_tasks = len(tasks)
    cols_per_task = 2
    total_cols = num_tasks * cols_per_task
    
    fig, axes = plt.subplots(n_disp_repeats, total_cols, figsize=(total_cols * 1.5, n_disp_repeats * 1.5))
    if n_disp_repeats == 1: axes = axes.reshape(1, -1)
    
    print(f"\nGenerating randomized class grid (showing first {n_disp_repeats} repeats)...")
    channels = DATASET_CONFIGS[config.dataset_name].get('channels', 1)
    
    for r in range(n_disp_repeats):
        col_idx = 0
        for task in tasks:
            defs = task.repeat_configs[r] 
            for cls_def in defs:
                temp_ds = FilteredMappedDataset(
                    DATASET_CLASS_MAP[cls_def['dataset']],
                    root=config.data_dir, train=True,
                    original_label=cls_def['original_label'],
                    binary_label=cls_def['binary_label'],
                    img_size=task.img_size
                )
                img_flat, _ = temp_ds[0] 
                
                flat_len = img_flat.shape[0]
                side = int(np.sqrt(flat_len // channels))
                
                if channels == 1:
                    img = img_flat.reshape(side, side)
                    cmap = 'gray'
                else:
                    img_chw = img_flat.reshape(channels, side, side)
                    img = np.moveaxis(img_chw, 0, -1)
                    cmap = None

                ax = axes[r, col_idx]
                ax.imshow(img, cmap=cmap)
                if r == 0:
                    ax.set_title(f"T{task.task_id}-C{int(cls_def['binary_label'])}", fontsize=10, fontweight='bold')
                
                ax.set_xlabel(f"Orig: {cls_def['original_label']}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                col_idx += 1

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    save_path = os.path.join(config.figures_dir, output_file)
    plt.savefig(save_path)
    print(f"Saved class grid to {save_path}")
    plt.close()