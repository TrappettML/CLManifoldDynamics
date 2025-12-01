import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import traverse_util
import optax
from models import TwoLayerMLP

class TrainState(train_state.TrainState):
    """
    Custom TrainState. 
    Can be extended to include batch_stats or mutable variables 
    (e.g., for BatchNorm) if architecture changes.
    """
    pass

def create_vectorized_state(config, rng):
    """
    Initializes N parallel model states (for n_repeats).
    """
    model = TwoLayerMLP(hidden_dim=config.hidden_dim)
    
    # 1. Define init for a single instance
    def init_single_state(key):
        # Dummy input for shape inference
        dummy_input = jnp.ones((1, config.input_dim))
        variables = model.init(key, dummy_input)
        params = variables['params']
        partition_optimizers = {'trainable': optax.sgd(config.learning_rate),
                                'frozen': optax.set_to_zero()}
        param_partitions = traverse_util.path_aware_map(
            lambda path, v: 'frozen' if 'layer2' in path else 'trainable', params
            )
        tx = optax.multi_transform(partition_optimizers, param_partitions)
        # tx = optax.SGD(config.learning_rate)
        flat = list(traverse_util.flatten_dict(param_partitions).items())
        print(f"trainable params: {traverse_util.unflatten_dict(dict(flat[:2] + flat[-2:]))}")

        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )

    # 2. Vectorize the initialization
    keys = jax.random.split(rng, config.n_repeats)
    
    # Resulting state.params will have shape (n_repeats, layer, units...)
    state = jax.vmap(init_single_state)(keys)
    
    return state