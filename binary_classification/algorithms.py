import jax
import jax.numpy as jnp
import optax
from flax import traverse_util
from models import TwoLayerMLP, TrainState, CNN
from typing import Tuple, Any

class BaseAlgorithm:
    def __init__(self, config):
        self.config = config

    def _build_optimizer(self, params):
        """Shared logic for creating partitioned optimizers."""
        partition_optimizers = {
            'feature': optax.chain(
                optax.add_decayed_weights(self.config.weight_decay),
                optax.sgd(self.config.learning_rate1)
            ),
            'readout': optax.sgd(self.config.learning_rate2)
        }
        
        param_partitions = traverse_util.path_aware_map(
            lambda path, v: 'readout' if 'classifier' in path else 'feature', params
        )
        return optax.multi_transform(partition_optimizers, param_partitions)

    def eval_step(self, state: Any, batch: Tuple[jax.Array, jax.Array]) -> Tuple[jax.Array, jax.Array]:
        """Shared evaluation logic (greedy accuracy proxy for RL)."""
        images, labels = batch
        images = jnp.expand_dims(images, axis=-1)
        logits = state.apply_fn({'params': state.params}, images)
        loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
        preds = (logits > 0).astype(jnp.float32)
        acc = jnp.mean(preds == labels)
        return loss, acc

    def get_features(self, state: Any, x: jax.Array) -> jax.Array:
        """Shared feature extraction logic."""
        x = jnp.expand_dims(x, axis=-1)
        return state.apply_fn({'params': state.params}, x, method=CNN.get_features)
    
    def init_vectorized_state(self, rng: jax.Array, input_side: int) -> Any:
        raise NotImplementedError

    def train_step(self, state: Any, batch: Tuple[jax.Array, jax.Array]) -> Tuple[Any, Tuple[jax.Array, jax.Array]]:
        raise NotImplementedError


class SupervisedLearning(BaseAlgorithm):
    def init_vectorized_state(self, rng: jax.Array, input_side: int) -> TrainState:
        model = CNN(hidden_dim=self.config.hidden_dim)
        
        def init_single_state(key):
            dummy_input = jnp.ones((1, input_side, input_side, 1)) # make sure to add channel
            variables = model.init(key, dummy_input)
            params = variables['params']
            tx = self._build_optimizer(params)
            
            return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        keys = jax.random.split(rng, self.config.n_repeats)
        return jax.vmap(init_single_state)(keys)

    def train_step(self, state: TrainState, batch: Tuple[jax.Array, jax.Array]):
        b_img, b_lbl = batch
        b_img = jnp.expand_dims(b_img, axis=-1)

        def loss_fn(params):
            logits = state.apply_fn({'params': params}, b_img)
            loss = optax.sigmoid_binary_cross_entropy(logits, b_lbl).mean()
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        
        new_state = state.apply_gradients(grads=grads)
        preds = (logits > 0).astype(jnp.float32)
        acc = jnp.mean(preds == b_lbl)
        
        return new_state, (loss, acc)


class RLTrainState(TrainState):
    rng: jax.Array

class ReinforcementLearning(BaseAlgorithm):
    def init_vectorized_state(self, rng: jax.Array, input_side: int) -> RLTrainState:
        model = CNN(hidden_dim=self.config.hidden_dim)
        
        def init_single_state(key):
            init_key, state_key = jax.random.split(key)
            dummy_input = jnp.ones((1, input_side, input_side, 1))
            variables = model.init(init_key, dummy_input)
            params = variables['params']
            tx = self._build_optimizer(params)
            
            return RLTrainState.create(apply_fn=model.apply, params=params, tx=tx, rng=state_key)

        keys = jax.random.split(rng, self.config.n_repeats)
        return jax.vmap(init_single_state)(keys)

    def train_step(self, state: RLTrainState, batch: Tuple[jax.Array, jax.Array]):
        b_img, b_lbl = batch
        rng_next, key_sample = jax.random.split(state.rng)
        b_img = jnp.expand_dims(b_img, axis=-1)
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, b_img)
            probs = jax.nn.sigmoid(logits)
            
            u_rand = jax.random.uniform(key_sample, shape=logits.shape)
            actions = (u_rand < probs).astype(jnp.float32)
            rewards = (actions == b_lbl).astype(jnp.float32)
            
            neg_log_prob = optax.sigmoid_binary_cross_entropy(logits, actions)
            loss = jnp.mean(jax.lax.stop_gradient(rewards) * neg_log_prob)
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(rng=rng_next)
        
        greedy_preds = (logits > 0).astype(jnp.float32)
        acc = jnp.mean(greedy_preds == b_lbl)
        
        return new_state, (loss, acc)


def get_algorithm(config):
    if 'SL' in config.algorithm:
        return SupervisedLearning(config)
    elif 'RL' in config.algorithm:
        return ReinforcementLearning(config)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")