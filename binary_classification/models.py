from flax import linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state
from typing import Any, Callable, Dict, Optional, Tuple, Union

class TrainState(train_state.TrainState):
    """
    Custom TrainState to support extended functionality if needed.
    """
    pass


class CNN(nn.Module):
    """A simple CNN model using setup for shared layers."""
    hidden_dim: int  
    output_dim: int = 1

    def setup(self):
        self.conv1 = nn.Conv(features=8, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=16, kernel_size=(3, 3))
        self.conv3 = nn.Conv(features=32, kernel_size=(3, 3))
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.classifier = nn.Dense(features=self.output_dim)

    def get_features(self, x: jax.Array) -> jax.Array:
        x = nn.relu(self.conv1(x))
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.relu(self.conv2(x))
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.relu(self.conv3(x))
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = jnp.mean(x, axis=(1, 2)) # Global average pooling
        x = nn.relu(self.dense1(x))
        return x

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.get_features(x)
        x = self.classifier(x)
        return x

class TwoLayerMLP(nn.Module):
    """
    A simple two-layer MLP for binary classification.
    Structure: Input -> Dense(Hidden) -> ReLU -> Dense(1)
    """
    hidden_dim: int
    output_dim: int = 1

    def setup(self):
        self.layer1 = nn.Dense(features=self.hidden_dim, use_bias=False)
        self.classifier = nn.Dense(features=self.output_dim, use_bias=False)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass returning logits.
        """
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.classifier(x)
        return x

    def get_features(self, x: jax.Array) -> jax.Array:
        """
        Returns the hidden representation (post-ReLU).
        """
        x = self.layer1(x)
        x = nn.relu(x)
        return x
    

class CLHook:
    """
    Base class for hooks to inject logic into the training loop.
    """
    def on_task_start(self, task: Any, state: TrainState) -> None:
        pass

    def on_task_end(self, task: Any, state: TrainState, metrics: Dict[str, Any]) -> None:
        pass