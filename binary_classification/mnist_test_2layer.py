import os

# --- CRITICAL: JAX GPU Configuration ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".60" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import matplotlib.pyplot as plt
import numpy as np

# --- CRITICAL: TensorFlow GPU Configuration ---
import tensorflow as tf
import tensorflow_datasets as tfds

# --- DEBUGGING: Check for File Shadowing ---
try:
    if not hasattr(tfds, 'load'):
        raise AttributeError
except (AttributeError, ImportError):
    raise ImportError(
        "\n\n!!! CRITICAL ERROR !!!\n"
        "The imported 'tensorflow_datasets' module does not have a 'load' function.\n"
    )

tf.config.set_visible_devices([], 'GPU')

# --- Configuration ---
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10
HIDDEN_DIM = 512
SEED = 42
DATA_DIR = "./data/"
IMG_SIZE = 16 
INPUT_DIM = IMG_SIZE * IMG_SIZE  # 16x16x1 = 256 inputs

# --- Data Loading ---
def get_dataset(name, split, batch_size):
    ds = tfds.load(
        name,
        split=split,
        data_dir=DATA_DIR,
        as_supervised=True,
        shuffle_files=True if split == 'train' else False
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        
        # --- HANDLE CIFAR-100 (RGB -> Grayscale) ---
        # If image has 3 channels, convert to 1 channel so it fits our model
        if image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)
            
        # Resize using Area interpolation (Best practice for downsampling)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method='area')
        
        # Flatten: (16, 16, 1) -> (256,)
        image = tf.reshape(image, [-1])
        return image, label

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if split == 'train':
        ds = ds.shuffle(buffer_size=1024, seed=SEED)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- Visualization Function ---
def display_dataset_samples(dataset_names, num_samples=5):
    print(f"\nGenerating Sample Visualization ({IMG_SIZE}x{IMG_SIZE})...")
    
    # Custom labels for KMNIST to avoid font issues
    kmnist_labels = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]

    fig, axes = plt.subplots(len(dataset_names), num_samples, figsize=(12, 8)) # Increased height
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    for row_idx, ds_name in enumerate(dataset_names):
        # Load dataset with info to get label names
        ds, info = tfds.load(ds_name, split='train', with_info=True, as_supervised=True)
        ds = ds.shuffle(1000, seed=SEED)
        
        found_images = []
        found_labels = []
        seen_classes = set()

        for img, label in ds:
            lbl_int = int(label.numpy())
            
            # For CIFAR (100 classes), we might see the same class rarely, 
            # so we relax the "unique class" constraint if needed, 
            # but for 5 samples it should be fine.
            if lbl_int not in seen_classes:
                seen_classes.add(lbl_int)
                
                # --- APPLY SAME PREPROCESSING (Grayscale + Resize) ---
                img = tf.cast(img, tf.float32) / 255.0
                if img.shape[-1] == 3:
                    img = tf.image.rgb_to_grayscale(img)
                img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method='area')
                
                found_images.append(img.numpy())
                found_labels.append(lbl_int)
                
            if len(found_images) == num_samples:
                break
        
        # Plotting
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx]
            img = found_images[col_idx]
            lbl = found_labels[col_idx]

            # Get Label String
            if ds_name == 'kmnist':
                lbl_str = kmnist_labels[lbl]
            else:
                # Dynamically get label from TFDS info
                # truncating CIFAR labels if they are too long (e.g. "household_electrical_device")
                full_name = info.features['label'].int2str(lbl)
                lbl_str = (full_name[:15] + '..') if len(full_name) > 15 else full_name

            # Display
            ax.imshow(img, cmap='gray', interpolation='nearest')
            
            if col_idx == 0:
                ax.set_ylabel(ds_name, fontsize=11, fontweight='bold')
            
            ax.set_title(lbl_str, fontsize=9)
            ax.axis('off')

    plt.suptitle(f"Random Samples (Resized to {IMG_SIZE}x{IMG_SIZE}, Grayscale)", fontsize=16)
    plt.savefig('dataset_samples.png')
    print("Sample visualization saved to 'dataset_samples.png'")

# --- Model Definition (Flax NNX) ---
class TwoLayerMLP(nnx.Module):
    def __init__(self, din, hidden, dout, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, hidden, rngs=rngs)
        self.linear2 = nnx.Linear(hidden, dout, rngs=rngs)
        self.relu = nnx.relu

    def __call__(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# --- Training Steps ---
@staticmethod
@nnx.jit(static_argnums=(3,))
def train_step(model: TwoLayerMLP, optimizer: nnx.Optimizer, batch, filter_spec):
    images, labels = batch
    
    def loss_fn(model):
        logits = model(images)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()
        return loss

    diff_state = nnx.DiffState(0, filter_spec)
    grad = nnx.grad(loss_fn, argnums=diff_state)(model)
    optimizer.update(model, grad)
    
    logits = model(images)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return accuracy

@staticmethod
@nnx.jit
def eval_step(model: TwoLayerMLP, batch):
    images, labels = batch
    logits = model(images)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy

# --- Main Experiment ---
def run_experiment(dataset_name):
    print(f"\n--- Starting Training for {dataset_name} ---")
    
    # Determine number of classes dynamically (CIFAR100=100, others=10)
    if dataset_name == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10
        
    train_ds = get_dataset(dataset_name, 'train', BATCH_SIZE)
    test_ds = get_dataset(dataset_name, 'test', BATCH_SIZE)
    
    rngs = nnx.Rngs(SEED)
    model = TwoLayerMLP(din=INPUT_DIM, hidden=HIDDEN_DIM, dout=num_classes, rngs=rngs)
    
    train_filter = nnx.All(nnx.Param, nnx.PathContains('linear1'))
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=train_filter)

    history = { 'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': [] }

    for epoch in range(EPOCHS):
        train_iter = tfds.as_numpy(train_ds)
        test_iter = tfds.as_numpy(test_ds)

        batch_accs = []
        for batch in train_iter:
            acc = train_step(model, optimizer, batch, train_filter)
            batch_accs.append(acc)
        train_acc = np.mean(batch_accs)
        
        test_losses = []
        test_accs = []
        for batch in test_iter:
            loss, acc = eval_step(model, batch)
            test_losses.append(loss)
            test_accs.append(acc)
            
        test_loss = np.mean(test_losses)
        test_acc = np.mean(test_accs)
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)

    return history

# --- Execution ---
# Added cifar100 to the list
dataset_strs = ['mnist', 'kmnist', 'fashion_mnist', 'cifar100']
results = {}

try:
    # 1. Visualize
    display_dataset_samples(dataset_strs)

    # 2. Train
    for ds_name in dataset_strs:
        results[ds_name] = run_experiment(ds_name)

    print("\nGenerating Performance Plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Accuracy
    for ds_name, hist in results.items():
        epochs = range(1, EPOCHS + 1)
        ax1.plot(epochs, hist['test_acc'], label=f'{ds_name} (Test)', marker='o')
        
    ax1.set_title(f'Model Accuracy ({IMG_SIZE}x{IMG_SIZE})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.legend()

    # Plot Loss
    for ds_name, hist in results.items():
        epochs = range(1, EPOCHS + 1)
        ax2.plot(epochs, hist['test_loss'], label=f'{ds_name}', marker='x')

    ax2.set_title(f'Test Loss ({IMG_SIZE}x{IMG_SIZE})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('dataset_performance_cifar.png')
    print("Performance plots saved to 'dataset_performance_cifar.png'")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")