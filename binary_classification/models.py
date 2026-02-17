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

class TwoLayerMLP(nn.Module):
    """
    A simple two-layer MLP for binary classification.
    Structure: Input -> Dense(Hidden) -> ReLU -> Dense(1)
    """
    hidden_dim: int
    output_dim: int = 1

    def setup(self):
        self.layer1 = nn.Dense(features=self.hidden_dim, use_bias=False)
        self.layer2 = nn.Dense(features=self.output_dim, use_bias=False)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass returning logits.
        """
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.layer2(x)
        return x

    def get_features(self, x: jax.Array) -> jax.Array:
        """
        Returns the hidden representation (post-ReLU).
        """
        x = self.layer1(x)
        x = nn.relu(x)
        return x
    

class CNN(nn.Module):
    """A simple CNN model."""
    hidden_dim: int
    output_dim: int = 1

    def setup(self):
        self.layer1 = nn.Dense(features=self.hidden_dim, use_bias=False)
        self.layer2 = nn.Dense(features=self.output_dim, use_bias=False)

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.layer2(x)
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