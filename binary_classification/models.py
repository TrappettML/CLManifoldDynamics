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