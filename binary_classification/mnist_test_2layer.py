import os

# --- CRITICAL: JAX GPU Configuration ---
# Set these before importing JAX to ensure it manages memory correctly.
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
# Import TensorFlow after JAX configuration.
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
        "Check for a local file named 'tensorflow_datasets.py' and rename it.\n"
    )

# Hide GPU from TensorFlow so it doesn't conflict with JAX
# TF will run on CPU only (efficient for data loading pipeline)
tf.config.set_visible_devices([], 'GPU')

# --- Configuration ---
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10
HIDDEN_DIM = 512
SEED = 42
DATA_DIR = "./data/"

# --- Data Loading (using tensorflow_datasets) ---
def get_dataset(name, split, batch_size):
    """
    Creates a tf.data.Dataset pipeline.
    """
    
    # Load the dataset
    ds = tfds.load(
        name,
        split=split,
        data_dir=DATA_DIR,
        as_supervised=True,
        shuffle_files=True if split == 'train' else False
    )

    # --- Preprocessing Pipeline ---
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        # Flatten: (28, 28, 1) -> (784,)
        image = tf.reshape(image, [-1])
        return image, label

    # 1. Map (Preprocess)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # 2. Cache
    # We cache here so we don't re-read/re-process files every epoch
    ds = ds.cache()

    # 3. Shuffle (only for training)
    if split == 'train':
        ds = ds.shuffle(buffer_size=1024, seed=SEED)

    # 4. Batch
    ds = ds.batch(batch_size, drop_remainder=True)
    
    # 5. Prefetch
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Return the TF Dataset object (not the iterator yet)
    return ds

# --- Model Definition (Flax NNX) ---
class TwoLayerMLP(nnx.Module):
    def __init__(self, din, hidden, dout, rngs: nnx.Rngs):
        # Two layer network: Input -> Hidden -> Output
        self.linear1 = nnx.Linear(din, hidden, rngs=rngs)
        self.linear2 = nnx.Linear(hidden, dout, rngs=rngs)
        self.relu = nnx.relu

    def __call__(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# --- Training & Evaluation Steps ---
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
    # Compute gradients
    grad = nnx.grad(loss_fn, argnums=diff_state)(model)
    # jax.debug.print("grad: {grd}", grd=grad.items())
    optimizer.update(model, grad)
    
    # Calculate simple accuracy for reporting
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

# --- Main Experiment Loop ---
def run_experiment(dataset_name):
    print(f"\n--- Starting Training for {dataset_name} ---")
    
    # 1. Initialize Data (Create Pipeline ONCE)
    # This fixes the cache warning by keeping the dataset object alive
    train_ds = get_dataset(dataset_name, 'train', BATCH_SIZE)
    test_ds = get_dataset(dataset_name, 'test', BATCH_SIZE)
    
    # 2. Initialize Model & Optimizer
    rngs = nnx.Rngs(SEED)
    model = TwoLayerMLP(din=784, hidden=HIDDEN_DIM, dout=10, rngs=rngs)
    train_filter = nnx.All(nnx.Param, nnx.PathContains('linear1'))
    
    optimizer = nnx.Optimizer(model,
                            optax.adam(LEARNING_RATE), 
                            wrt=train_filter )

    # Metrics storage
    history = {
        'train_acc': [], 'test_acc': [],
        'train_loss': [], 'test_loss': []
    }

    # 3. Training Loop
    for epoch in range(EPOCHS):
        
        # Create fresh iterators for this epoch from the existing pipeline
        train_iter = tfds.as_numpy(train_ds)
        test_iter = tfds.as_numpy(test_ds)

        # Train
        batch_accs = []
        for batch in train_iter:
            acc = train_step(model, optimizer, batch, train_filter)
            batch_accs.append(acc)
        train_acc = np.mean(batch_accs)
        
        # Evaluate
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
        
        # print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    return history

# --- Execution & Plotting ---
dataset_strs = ['mnist', 'kmnist', 'fashion_mnist']
results = {}

try:
    for ds_name in dataset_strs:
        results[ds_name] = run_experiment(ds_name)

    print("\nGenerating Plots...")

    # Setup plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Accuracy
    for ds_name, hist in results.items():
        epochs = range(1, EPOCHS + 1)
        ax1.plot(epochs, hist['test_acc'], label=f'{ds_name} (Test)', marker='o')
        ax1.plot(epochs, hist['train_acc'], label=f'{ds_name} (Train)', linestyle='--', alpha=0.5)

    ax1.set_title('Model Accuracy per Dataset')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Test Loss
    for ds_name, hist in results.items():
        epochs = range(1, EPOCHS + 1)
        ax2.plot(epochs, hist['test_loss'], label=f'{ds_name}', marker='x')

    ax2.set_title('Test Loss per Dataset')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('dataset_comparison.png')
    print("Plots saved to 'dataset_comparison.png'")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")