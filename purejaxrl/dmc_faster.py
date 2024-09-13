import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import flax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import functools

import wandb

from envs.dmc_gym import make_env_dmc


import distrax

'''
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)


from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'ant', 'Environment name.')
flags.DEFINE_integer('num_seeds', 5, 'Environment name.')
'''    

class Actor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        return pi 
    
class Critic(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return jnp.squeeze(critic, axis=-1)

@flax.struct.dataclass
class OnlineBuffer:
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_obs: jax.Array
    terminals: jax.Array
    truncates: jax.Array
    log_probs: jax.Array
    advantages: jax.Array
    returns: jax.Array
    values: jax.Array
    next_values: jax.Array

@jax.jit
def input_to_memory(buffer: OnlineBuffer, idx: int, state: jax.Array, action: jax.Array, reward: jax.Array, next_state_buffer: jax.Array, terminal: jax.Array, truncate: jax.Array, logprob: jax.Array, value: jax.Array, next_value: jax.Array):
    buffer = buffer.replace(
        obs=buffer.obs.at[idx].set(state),
        actions=buffer.actions.at[idx].set(action),
        rewards=buffer.rewards.at[idx].set(reward),
        next_obs=buffer.next_obs.at[idx].set(next_state_buffer),
        terminals=buffer.terminals.at[idx].set(terminal),
        truncates=buffer.truncates.at[idx].set(truncate),
        log_probs=buffer.log_probs.at[idx].set(logprob),
        values=buffer.values.at[idx].set(value),
        next_values=buffer.next_values.at[idx].set(next_value))
    return buffer

@jax.jit
def calculate_gae_scan(buffer: OnlineBuffer, discount: float = 0.99, gae_lambda: float = 0.95):
    buffer = buffer.replace(truncates=buffer.truncates.at[-1].set(jnp.ones_like(buffer.truncates[-1])))
    delta = buffer.rewards + discount * (1 - buffer.terminals) * buffer.next_values - buffer.values
    idx = buffer.advantages.shape[0] - 1
    final_advantage = delta[idx] + discount * gae_lambda * (1 - buffer.terminals[idx]) * buffer.next_values[idx]
    buffer = buffer.replace(advantages=buffer.advantages.at[idx].set(final_advantage))        
    def calculate_advantage_scan_(current_state, next_advantage):
        idx, buffer, delta, next_values = current_state
        idx -= 1
        carried_value = (1 - buffer.truncates[idx]) * buffer.advantages[idx+1]        
        advantages_ = delta[idx] + discount * gae_lambda * (1 - buffer.terminals[idx]) * carried_value
        buffer = buffer.replace(advantages=buffer.advantages.at[idx].set(advantages_))
        return (idx, buffer, delta, next_values), None
    (idx, buffer, delta, next_values), _ = jax.lax.scan(calculate_advantage_scan_, (idx, buffer, delta, buffer.next_values), None, length=buffer.advantages.shape[0]-1)
    returns = buffer.advantages - buffer.values
    buffer = buffer.replace(returns=returns)
    return buffer

@functools.partial(jax.jit, static_argnames=('actor'))
def get_action(rng: jax.random.PRNGKey, actor: Actor, actor_state: TrainState, state: jnp.ndarray):
    rng, _rng = jax.random.split(rng)
    pi = actor.apply(actor_state.params, state)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)
    return rng, action, log_prob 

@functools.partial(jax.jit, static_argnames=('critic'))
def get_value(critic: Critic, critic_state: TrainState, state: jnp.ndarray, next_state: jnp.ndarray):
    value = critic.apply(critic_state.params, state)
    next_value = critic.apply(critic_state.params, next_state)
    return value, next_value


config = {
    "LR": 3e-4,
    "NUM_ENVS": 8,
    "NUM_STEPS": 512,
    "TOTAL_TIMESTEPS": 30000000,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 16,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_NAME": "walker-stand",
    "ANNEAL_LR": False,
    "NORMALIZE_ENV": True,
    "DEBUG": False,
    "ACTION_REPEAT": 4
}


config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)
config["MINIBATCH_SIZE"] = (
    config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
)
env = make_env_dmc(config["ENV_NAME"], 0, config["NUM_ENVS"], 1000, config["ACTION_REPEAT"])

def linear_schedule(count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac

rng = jax.random.PRNGKey(0)

# INIT NETWORK
if config["ANNEAL_LR"]:
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )
else:
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    
actor = Actor(
    env.action_space.shape[-1], activation=config["ACTIVATION"]
)
rng, _rng = jax.random.split(rng)
init_x = jnp.zeros((1, env.observation_space.shape[-1]))
actor_params = actor.init(_rng, init_x)

critic = Critic(
    activation=config["ACTIVATION"]
)
rng, _rng = jax.random.split(rng)
critic_params = critic.init(_rng, init_x)

actor_state = TrainState.create(
    apply_fn=actor.apply,
    params=actor_params,
    tx=tx,
)

critic_state = TrainState.create(
    apply_fn=critic.apply,
    params=critic_params,
    tx=tx,
)

buffer = OnlineBuffer(
    obs = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"], env.observation_space.shape[-1]), dtype=jnp.float32),
    actions = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"], env.action_space.shape[-1]), dtype=jnp.float32),
    rewards = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.float32),
    next_obs = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"], env.observation_space.shape[-1]), dtype=jnp.float32),
    log_probs = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.float32),
    terminals = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.float32),
    truncates = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.float32),
    values = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.float32),
    next_values = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.float32),
    advantages = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.float32),
    returns = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.float32))

# INIT ENV
rng, _rng = jax.random.split(rng)
state = env.reset()
# TRAIN LOOP
for update in range(config["NUM_UPDATES"]):
    for step_ in range(config["NUM_STEPS"]):
        rng, action, log_prob = get_action(rng, actor, actor_state, state)
        next_state, reward, terminal, truncate, _ = env.step(np.array(action))
        value, next_value = get_value(critic, critic_state, state, next_state)
        buffer = input_to_memory(buffer, step_, state, action, reward, next_state, terminal, truncate, log_prob, value, next_value)
        state = next_state
        state, terminal, truncate, _ = env.reset_where_done(state, terminal, truncate)
    # CALCULATE ADVANTAGE
    buffer = calculate_gae_scan(buffer, discount=config['GAMMA'], gae_lambda=config['GAE_LAMBDA'])

    # UPDATE NETWORK
    @jax.jit
    def _update_epoch(update_state, unused):
        def _update_minbatch(train_states, minibatches):
            actor_state, critic_state = train_states
            def actor_loss_fn(actor_params, minibatches):
                gae = minibatches.advantages
                pi = actor.apply(actor_params, minibatches.obs)
                log_prob = pi.log_prob(minibatches.actions)
                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob - minibatches.log_probs)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor1 = ratio * gae
                loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()
                entropy = pi.entropy().mean()
                loss_a = loss_actor - config["ENT_COEF"] * entropy
                return loss_a
            
            grad_fn_a = jax.value_and_grad(actor_loss_fn, has_aux=False)
            total_loss_a, grads_a = grad_fn_a(actor_state.params, minibatches)
            actor_state = actor_state.apply_gradients(grads=grads_a)
            
            def critic_loss_fn(critic_params, minibatches):
                targets = minibatches.returns
                value = critic.apply(critic_params, minibatches.obs)
                # CALCULATE VALUE LOSS
                value_pred_clipped = minibatches.values + (
                    value - minibatches.values
                ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = (
                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                )
                return value_loss

            grad_fn_c = jax.value_and_grad(critic_loss_fn, has_aux=False)
            total_loss_c, grads_c = grad_fn_c(critic_state.params, minibatches)
            total_loss = total_loss_a + total_loss_c
            critic_state = critic_state.apply_gradients(grads=grads_c)
            train_states = (actor_state, critic_state)
            return train_states, total_loss

        actor_state, critic_state, buffer, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
        
        assert (
            batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
        ), "batch size must be equal to number of steps * number of envs"
        
        permutation = jax.random.permutation(_rng, batch_size)
        #batch = (traj_batch, advantages, targets)
        batch = buffer
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
        )
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), batch
        )
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
            ),
            shuffled_batch,
        )
        
        train_states = (actor_state, critic_state)
        train_states, total_loss = jax.lax.scan(
            _update_minbatch, train_states, minibatches
        )
        actor_state, critic_state = train_states
        update_state = (actor_state, critic_state, buffer, rng)
        return update_state, total_loss

    update_state = (actor_state, critic_state, buffer, rng)
    update_state, loss_info = jax.lax.scan(
        _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
    )
    actor_state = update_state[0]
    critic_state = update_state[1]
    buffer = update_state[2]
    rng = update_state[3]
    print((update+1)*config["NUM_STEPS"]*config["NUM_ENVS"], buffer.rewards.mean()*1000/config["ACTION_REPEAT"])


