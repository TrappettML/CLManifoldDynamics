import jax
import jax.numpy as jnp
import optax
from flax import traverse_util
from models import TwoLayerMLP, TrainState

class BaseAlgorithm:
    """Interface for learning algorithms."""
    def __init__(self, config):
        self.config = config

    def init_vectorized_state(self, rng, input_shape):
        """Initializes N parallel model states."""
        raise NotImplementedError

    def train_step(self, state, batch):
        """Performs a single training step. Returns (new_state, metrics)."""
        raise NotImplementedError

    def eval_step(self, state, batch):
        """Performs a single evaluation step. Returns metrics."""
        raise NotImplementedError

    def get_features(self, state, x):
        """Extracts internal representations."""
        raise NotImplementedError

class SupervisedLearning(BaseAlgorithm):
    def init_vectorized_state(self, rng, input_shape):
        model = TwoLayerMLP(hidden_dim=self.config.hidden_dim)
        
        def init_single_state(key):
            dummy_input = jnp.ones((1, input_shape))
            variables = model.init(key, dummy_input)
            params = variables['params']
            
            # Parameter Partitioning for Separate Learning Rates
            partition_optimizers = {
                'feature': optax.chain(
                    optax.add_decayed_weights(self.config.weight_decay),
                    optax.sgd(self.config.learning_rate1)
                ),
                'readout': optax.sgd(self.config.learning_rate2)
            }
            
            param_partitions = traverse_util.path_aware_map(
                lambda path, v: 'readout' if 'layer2' in path else 'feature', params
            )
            tx = optax.multi_transform(partition_optimizers, param_partitions)
            
            return TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=tx
            )

        keys = jax.random.split(rng, self.config.n_repeats)
        return jax.vmap(init_single_state)(keys)

    def train_step(self, state, batch):
        b_img, b_lbl = batch
        
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, b_img)
            loss = optax.sigmoid_binary_cross_entropy(logits, b_lbl).mean()
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, loss_logits), grads = grad_fn(state.params)
        
        new_state = state.apply_gradients(grads=grads)
        
        preds = (loss_logits > 0).astype(jnp.float32)
        acc = jnp.mean(preds == b_lbl)
        
        return new_state, (loss, acc)

    def eval_step(self, state, batch):
        images, labels = batch
        logits = state.apply_fn({'params': state.params}, images)
        loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
        preds = (logits > 0).astype(jnp.float32)
        acc = jnp.mean(preds == labels)
        return loss, acc

    def get_features(self, state, x):
        return state.apply_fn(
            {'params': state.params}, 
            x, 
            method=TwoLayerMLP.get_features
        )

# --- Factory ---
def get_algorithm(config):
    if config.algorithm == 'SL':
        return SupervisedLearning(config)
    elif config.algorithm == 'RL':
        raise NotImplementedError("RL Algorithm not yet implemented")
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")