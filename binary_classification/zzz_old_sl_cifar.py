import os

# --- JAX GPU Configuration ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".60"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import ml_collections

# Import our new data handler
import data_utils

# TF CPU-only for data pipeline
tf.config.set_visible_devices([], 'GPU')

# --- Configuration Management ---

def get_config():
    config = ml_collections.ConfigDict()
    
    # === SWITCH DATASET HERE ===
    # Options: 'cifar100', 'mnist', 'fashion_mnist'
    config.dataset_name = 'cifar100' 
    
    config.seed = 42
    config.data_dir = "./data"
    config.num_tasks = 2
    
    # Dynamically set input dim based on dataset choice
    config.input_dim = data_utils.get_dataset_dims(config.dataset_name)
    
    config.hidden_dim = 512
    config.learning_rate = 0.001
    config.batch_size = 128
    config.epochs_per_task = 10
    return config

# --- Model Definition ---

class TwoLayerMLP(nnx.Module):
    def __init__(self, din, hidden, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, hidden, rngs=rngs)
        self.relu = nnx.relu
        self.linear2 = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, x, capture_hidden=False):
        h = self.linear1(x)
        h = self.relu(h)
        out = self.linear2(h)
        if capture_hidden:
            return out, h
        return out

# --- Learner ---

class ContinualLearner:
    def __init__(self, config):
        self.config = config
        rngs = nnx.Rngs(config.seed)
        self.model = TwoLayerMLP(din=config.input_dim, hidden=config.hidden_dim, rngs=rngs)
        self.train_filter = nnx.All(nnx.Param, nnx.PathContains('linear1'))
        
        self.optimizer = nnx.Optimizer(
            self.model, 
            optax.adam(config.learning_rate), 
            wrt=self.train_filter 
        )

    @staticmethod
    @nnx.jit(static_argnums=(3,))
    def train_step(model, optimizer, batch, filter_spec):
        images, labels = batch
        labels = labels[..., None]

        def loss_fn(model):
            logits = model(images)
            loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
            return loss

        diff_state = nnx.DiffState(0, filter_spec)
        grads = nnx.grad(loss_fn, argnums=diff_state)(model)
        optimizer.update(model, grads)
        
        logits = model(images)
        preds = (logits > 0).astype(jnp.float32)
        acc = jnp.mean(preds == labels)
        return acc, loss_fn(model)

    @staticmethod
    @nnx.jit
    def eval_step(model, batch):
        images, labels = batch
        labels = labels[..., None]
        
        logits = model(images, capture_hidden=False)
        loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
        preds = (logits > 0).astype(jnp.float32)
        acc = jnp.mean(preds == labels)
        return loss, acc

    def train_task(self, task, test_streams: dict, global_history: dict):
        train_ds = task.load_data()
        print(f"--- Training on {task.name} ---")

        for epoch in range(self.config.epochs_per_task):
            # 1. Train Loop
            train_iter = tfds.as_numpy(train_ds)
            batch_accs = []
            batch_losses = []
            
            for batch in train_iter:
                acc, loss = self.train_step(self.model, self.optimizer, batch, self.train_filter)
                batch_accs.append(acc)
                batch_losses.append(loss)
            
            avg_train_loss = np.mean(batch_losses)
            avg_train_acc = np.mean(batch_accs)
            
            global_history['train_loss'].append(avg_train_loss)
            global_history['train_acc'].append(avg_train_acc)

            # 2. Universal Evaluation Loop
            # print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
            
            for t_name, t_ds in test_streams.items():
                test_iter = tfds.as_numpy(t_ds)
                t_accs = []
                t_losses = []
                
                for batch in test_iter:
                    l, a = self.eval_step(self.model, batch)
                    t_accs.append(a)
                    t_losses.append(l)
                
                avg_t_acc = np.mean(t_accs)
                avg_t_loss = np.mean(t_losses)
                
                global_history['test_metrics'][t_name]['acc'].append(avg_t_acc)
                global_history['test_metrics'][t_name]['loss'].append(avg_t_loss)
                
                if t_name == task.name:
                    print(f"    > Eval Current ({t_name}): Acc: {avg_t_acc:.4f}")

# --- Main ---

def main():
    config = get_config()
    print(f"Configuration Loaded: {config}")
    print(f"Dataset Selected: {config.dataset_name} (Input Dim: {config.input_dim})")

    # 1. Setup Data using external module
    # We pass 'split=train' for training tasks
    train_tasks = data_utils.create_continual_tasks(config, split='train')
    
    # 2. Prepare Test Streams
    test_streams = {}
    print("Preparing Test Streams...")
    # We pass 'split=test' to reuse logic but load test data
    test_tasks = data_utils.create_continual_tasks(config, split='test')
    
    for t in test_tasks:
        test_streams[t.name] = t.load_data()

    # 3. Setup Learner & History
    learner = ContinualLearner(config)
    
    global_history = {
        'train_acc': [], 
        'train_loss': [],
        'test_metrics': {name: {'acc': [], 'loss': []} for name in test_streams.keys()}
    }
    
    task_boundaries = []
    total_epochs = 0

    # 4. Training Loop
    for task in train_tasks:
        learner.train_task(task, test_streams, global_history)
        total_epochs += config.epochs_per_task
        task_boundaries.append(total_epochs)

    # 5. Plotting
    print("\nGenerating Analysis Plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    epochs_range = range(1, total_epochs + 1)
    
    # Plot 1: Accuracies
    ax1.plot(epochs_range, global_history['train_acc'], label='Current Train Acc', color='black', linestyle='--', alpha=0.6)
    
    colors = plt.cm.jet(np.linspace(0, 1, len(test_streams)))
    for (t_name, metrics), color in zip(global_history['test_metrics'].items(), colors):
        ax1.plot(epochs_range, metrics['acc'], label=f"{t_name}", color=color, linewidth=2)
        ax2.plot(epochs_range, metrics['loss'], label=f"{t_name}", color=color, linewidth=2)

    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{config.dataset_name} CL (Evaluating {len(test_streams)} Tasks)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Losses
    ax2.plot(epochs_range, global_history['train_loss'], label='Current Train Loss', color='black', linestyle='--', alpha=0.6)
    ax2.set_ylabel('Loss (BCE)')
    ax2.set_xlabel('Total Epochs')
    ax2.set_title('Continual Learning Loss')
    ax2.grid(True, alpha=0.3)

    for boundary in task_boundaries[:-1]:
        ax1.axvline(x=boundary, color='grey', linestyle=':', alpha=0.8, linewidth=2)
        ax2.axvline(x=boundary, color='grey', linestyle=':', alpha=0.8, linewidth=2)

    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    save_string = f"sl_{config.dataset_name}_cls"
    plt.savefig(f'{save_string}.png')
    print(f"Plots saved to '{save_string}.png'")

if __name__ == "__main__":
    main()