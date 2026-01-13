from flax import linen as nn
import jax
from flax.training import train_state

class TrainState(train_state.TrainState):
    """
    Custom TrainState. 
    Can be extended to include batch_stats or mutable variables 
    (e.g., for BatchNorm, or Target Networks in RL).
    """
    pass


class TwoLayerMLP(nn.Module):
    hidden_dim: int
    output_dim: int = 1

    def setup(self):
        # We define layers in setup to reuse them in different methods
        self.layer1 = nn.Dense(features=self.hidden_dim, use_bias=False)
        self.layer2 = nn.Dense(features=self.output_dim, use_bias=False)

    def __call__(self, x):
        # Standard forward pass
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.layer2(x)
        return x

    def get_features(self, x):
        # Returns the hidden representation (post-ReLU)
        x = self.layer1(x)
        x = nn.relu(x)
        return x
    



class CLHook:
    """
    Base class for hooks. 
    Override these methods to inject logic into the training loop.
    """
    def on_task_start(self, task, state):
        """Called at the beginning of a new task."""
        pass

    def on_task_end(self, task, state, metrics):
        """Called at the end of a task."""
        pass

    def on_epoch_start(self, epoch, state):
        """Called at the start of an epoch."""
        pass

    def on_epoch_end(self, epoch, state, metrics):
        """Called at the end of an epoch."""
        pass
    
    # Note: adding on_batch_start/end inside JAX scans is complex 
    # due to pure function constraints, but we can return auxiliary data 
    # from the train step if needed.