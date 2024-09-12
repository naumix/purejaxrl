from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.base import State, Wrapper

import gym.spaces as spaces

import jax
import jax.numpy as jnp
import numpy as np

class AutoResetWrapper(Wrapper):
    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        state.info['next_state'] = state.obs
        return state
    
    def step(self, state: State, action: jax.Array) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        state.info.update(next_state=state.obs)
        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)
        pipeline_state = jax.tree.map(where_done, state.info['first_pipeline_state'], state.pipeline_state)
        obs = where_done(state.info['first_obs'], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)

class make_env_brax():
    def __init__(self, env_name='humanoid', seed=0, num_envs=2):
        np.random.seed(seed)
        self.env_name = env_name
        self.num_envs = num_envs
        new_seed = np.random.randint(0,1e8)
        env = envs.create(self.env_name, batch_size=num_envs, episode_length=1000, auto_reset=False)
        env = AutoResetWrapper(env)
        self.env = gym_wrapper.VectorGymWrapper(env, seed=new_seed)
        self.timesteps = jnp.zeros(self.num_envs)
        self.action_space = spaces.Box(low=-np.ones(self.env.action_space.shape), high=np.ones(self.env.action_space.shape), shape=(self.env.action_space.shape), dtype=self.env.action_space.dtype)
        self.observation_space = self.env.observation_space
        self.action_dim = self.env.action_space.shape[-1]
        self.original_min = jnp.array(self.env.action_space.low[0,0])
        self.original_max = jnp.array(self.env.action_space.high[0,0])
        self.low = -jnp.ones(1)
        self.high = jnp.ones(1)
            
    def reset(self):
        observation = self.env.reset()
        return observation
    
    def hard_reset(self):
        new_seed = np.random.randint(0,1e8)
        env = envs.create(self.env_name, batch_size=self.num_envs, episode_length=1000, auto_reset=False)
        env = AutoResetWrapper(env)
        self.env = gym_wrapper.VectorGymWrapper(env, seed=new_seed)
        self.timesteps = jnp.zeros(self.num_envs)
        self.action_space = spaces.Box(low=-np.ones(self.env.action_space.shape), high=np.ones(self.env.action_space.shape), shape=(self.env.action_space.shape), dtype=jnp.float32)
        self.observation_space = self.env.observation_space
        self.action_dim = self.env.action_space.shape[-1]
        self.original_min = jnp.array(self.env.action_space.low[0,0])
        self.original_max = jnp.array(self.env.action_space.high[0,0])
        self.low = -jnp.ones(1)
        self.high = jnp.ones(1)
        return self.reset()
    
    def rescale_action(self, actions):
        actions = self.low + (self.high - self.low) * ((actions - self.original_min) / (self.original_max - self.original_min))
        actions = jnp.clip(actions, a_min=-1.0, a_max=1.0)
        return actions
        
    def step(self, actions):
        actions = self.rescale_action(actions)
        #actions = jnp.clip(actions, a_min=-1.0, a_max=1.0)
        observation, rewards, terminal, info = self.env.step(actions)
        truncate = info['truncation']
        terminal = jnp.where(truncate, jnp.zeros_like(truncate), terminal)
        return observation, rewards, terminal, truncate, info['next_state'], info

'''
env = make_env_brax()
observation = env.reset()
action = env.action_space.sample()
next_observation, rewards, terminals, truncates, next_observation_buffer, info = env.step(action)
state = env.hard_reset()
env.rescale_action(jnp.array(0.4))
env = envs.create('halfcheetah', batch_size=2, episode_length=2, auto_reset=False)
env = AutoResetWrapper(env)
env = gym_wrapper.VectorGymWrapper(env, seed=0)
'''





