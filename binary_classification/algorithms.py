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
        # jax.debug.breakpoint()
        acc = jnp.mean(preds == labels)
        return loss, acc

    def get_features(self, state, x):
        return state.apply_fn(
            {'params': state.params}, 
            x, 
            method=TwoLayerMLP.get_features
        )

class RLTrainState(TrainState):
    """
    Extended TrainState for RL that maintains a random key 
    for stochastic action sampling.
    """
    rng: jax.Array

class ReinforcementLearning(BaseAlgorithm):
    """
    Policy Gradient RL (REINFORCE) extended to Multi-Layer Perceptrons.
    
    References:
        Schmid, C., & Murray, J. M. (2024). Dynamics of supervised and reinforcement 
        learning in the non-linear perceptron. 
        
    The update rule approximates: Delta w ~ Reward * (Action - p) * x
    This is achieved by minimizing Loss = - Reward * log(P(Action|x)).
    """
    def init_vectorized_state(self, rng, input_shape):
        model = TwoLayerMLP(hidden_dim=self.config.hidden_dim)
        
        def init_single_state(key):
            # Split key: one for initialization, one to store in state for sampling
            init_key, state_key = jax.random.split(key)
            
            dummy_input = jnp.ones((1, input_shape))
            variables = model.init(init_key, dummy_input)
            params = variables['params']
            
            # Parameter Partitioning (Same as SL)
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
            
            # Create state with the RNG key
            return RLTrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=tx,
                rng=state_key
            )

        keys = jax.random.split(rng, self.config.n_repeats)
        return jax.vmap(init_single_state)(keys)

    def train_step(self, state, batch):
        b_img, b_lbl = batch
        
        # Split the state's RNG key for this step
        rng_next, key_sample = jax.random.split(state.rng)
        
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, b_img)
            probs = jax.nn.sigmoid(logits)
            
            # 1. Stochastic Action Sampling: a ~ Bernoulli(p)
            u_rand = jax.random.uniform(key_sample, shape=logits.shape)
            actions = (u_rand < probs).astype(jnp.float32)
            
            # 2. Reward Calculation: R = 1 if Action == Label else 0
            # (Schmid & Murray: "correctness" reward)
            rewards = (actions == b_lbl).astype(jnp.float32)
            
            # 3. Policy Gradient Objective
            # We minimize: J = - E[ R * log P(a|x) ]
            # The gradient of this loss is equivalent to REINFORCE: (a - p) * R * x
            
            # binary_cross_entropy(logits, targets) returns -log P(target)
            # Here, our "target" is the sampled action `actions`
            neg_log_prob = optax.sigmoid_binary_cross_entropy(logits, actions)
            
            # Weight by reward (detached, as it's a scalar signal)
            loss = jnp.mean(jax.lax.stop_gradient(rewards) * neg_log_prob)
            
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        
        # Update parameters and replace the RNG key with the new one
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(rng=rng_next)
        
        # Monitor Greedy Accuracy (p > 0.5)
        greedy_preds = (logits > 0).astype(jnp.float32)
        acc = jnp.mean(greedy_preds == b_lbl)
        
        return new_state, (loss, acc)

    def eval_step(self, state, batch):
        # Evaluation remains deterministic (Greedy Policy)
        images, labels = batch
        logits = state.apply_fn({'params': state.params}, images)
        
        # We can monitor the negative log likelihood against the true label here
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
        return ReinforcementLearning(config)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")