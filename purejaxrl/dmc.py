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

def rational(x: jax.Array, a: float, b: float, c: float, d: float, e: float, f: float):
    num = a * x ** 3 + b * x ** 2 + c * x + d
    denom = jnp.abs(e * x ** 2) + jnp.abs(f * x) + 1
    return num/denom

class RationalActorCritic(nn.Module):
    action_dim: Sequence[int]
    a: int = 1
    b: int = 1
    c: int = 1
    d: int = 1
    e: int = 1
    f: int = 1

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = rational(actor_mean, self.a, self.b, self.c, self.d, self.e, self.f)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = rational(actor_mean, self.a, self.b, self.c, self.d, self.e, self.f)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = nn.relu(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
    

class ActorCritic(nn.Module):
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
        return pi, jnp.squeeze(critic, axis=-1)

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

def make_train(config):
    
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 8,
        "NUM_STEPS": 256,
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
    network = ActorCritic(
        env.action_space.shape[-1], activation=config["ACTIVATION"]
    )
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1, env.observation_space.shape[-1]))
    network_params = network.init(_rng, init_x)
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
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
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
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    state = env.reset()
    #last_obs = obsv

    # TRAIN LOOP
    for update in range(config["NUM_UPDATES"]):
        for step_ in range(config["NUM_STEPS"]):
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, state)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            # STEP ENV
            next_state, reward, terminal, truncate, _ = env.step(np.array(action))
            _, next_value = network.apply(train_state.params, next_state)
            buffer = input_to_memory(buffer, step_, state, action, reward, next_state, terminal, truncate, log_prob, value, next_value)
            state = next_state
            state, terminal, truncate, _ = env.reset_where_done(state, terminal, truncate)
            #done = jnp.logical_or(terminal, truncate)                
            #buffer = input_to_memory(buffer, step_, done, action, value, reward, log_prob, obsv)
            
        # CALCULATE ADVANTAGE
        #_, last_val = network.apply(train_state.params, last_obs)

        buffer = calculate_gae_scan(buffer, discount=config['GAMMA'], gae_lambda=config['GAE_LAMBDA'])

        # UPDATE NETWORK
        @jax.jit
        def _update_epoch(update_state, unused):
            @jax.jit
            def _update_minbatch(train_state, minibatches):
                
                def _loss_fn(params, minibatches):
                    # RERUN NETWORK
                    gae = minibatches.advantages
                    targets = minibatches.returns
                    
                    pi, value = network.apply(params, minibatches.obs)
                    log_prob = pi.log_prob(minibatches.actions)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = minibatches.values + (
                        value - minibatches.values
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

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

                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, minibatches
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, buffer, rng = update_state
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
            
            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            update_state = (train_state, buffer, rng)
            return update_state, total_loss

        update_state = (train_state, buffer, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]
        rng = update_state[-1]
        buffer = update_state[1]
        print((update+1)*config["NUM_STEPS"]*config["NUM_ENVS"], buffer.rewards.mean()*1000/config["ACTION_REPEAT"])


def log_to_wandb_timestep(res, timestep):
    for seed in range(FLAGS.num_seeds):
        wandb.log({f'seed{seed}/timesteps': timestep, 
                   f'seed{seed}/rews': res[seed]}, step=timestep)

def main(_):
    wandb.init(
        config=FLAGS,
        entity='naumix',
        project='PPO_Parallel_new',
        group=f'{FLAGS.env_name}',
        name=f'test')
    
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 10,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": 30000000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "cheetah-run",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": False,
    }
    
    
    config["ENV_NAME"] = FLAGS.env_name
    #env_list = ['halfcheetah', 'hopper']
    seed = jnp.arange(2)
    seed = jnp.arange(FLAGS.num_seeds)
    res = vmap_train(config, seed)
    #res = train_many_envs(config, env_list, seed)
    res = np.array(res)
    for timestep in range(res.shape[-1]):
        #print(res[:,timestep])
        log_to_wandb_timestep(res[:,timestep], timestep)
        
        

if __name__ == "__main__":
    app.run(main)    
