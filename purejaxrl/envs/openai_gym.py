import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

def wrap_env(env):
    env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    #env = gym.wrappers.NormalizeReward(env, gamma=0.99)
    #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env
        
class make_env_gym(gym.Env):
    def __init__(self, env_name='Ant-v4', seed=0, num_envs=2):
        np.random.seed(seed)
        seeds = np.random.randint(0,1e6,(num_envs))
        #self.envs = [gym.make(env_name) for seed in seeds]
        self.envs = [gym.wrappers.RescaleAction(gym.make(env_name), -1.0, 1.0) for seed in seeds]
        self.num_envs = len(self.envs)
        self.timesteps = np.zeros(self.num_envs)
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        self.observation_space = spaces.Box(low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
                                            high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
                                            shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
                                            dtype=self.envs[0].observation_space.dtype)
        self.action_dim = self.envs[0].action_space.shape[0]

    def _reset_idx(self, idx):
        seed_ = np.random.randint(0,1e6)
        obs, _ = self.envs[idx].reset(seed=seed_)
        return obs
    
    def reset_where_done(self, observations, terms, truns):
        resets = np.zeros((terms.shape))
        for j, (term, trun) in enumerate(zip(terms, truns)):
            if (term == True) or (trun == True):
                observations[j], terms[j], truns[j] = self._reset_idx(j), False, False
                resets[j] = 1
                self.timesteps[j] = 0
        return observations, terms, truns, resets
    
    def reset(self):
        obs = []
        for env in self.envs:
            seed_ = np.random.randint(0,1e6)
            ob, _ = env.reset(seed=seed_)
            obs.append(ob)
        return np.stack(obs)
    
    def generate_masks(self, terms, truns):
        masks = []
        for term, trun in zip(terms, truns):
            if not term or trun:
                mask = 1.0
            else:
                mask = 0.0
            masks.append(mask)
        masks = np.array(masks)
        return masks
    
    def step(self, actions):
        obs, rews, terms, truns = [], [], [], []
        actions = np.clip(actions, -1.0, 1.0)
        for env, action in zip(self.envs, actions):
            ob, reward, term, trun, _ = env.step(action)
            obs.append(ob)
            rews.append(reward)
            terms.append(term)
            truns.append(trun)
        self.timesteps += 1
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), None

    def random_step(self):
        actions = np.random.uniform(-1, 1, (self.num_envs, self.action_dim))
        obs, rews, terms, truns = self.step(actions)
        return obs, rews, terms, truns, None, actions
    
    def evaluate(self, agent, num_episodes=5, device='cpu'):
        returns_mean = np.zeros(self.num_envs)
        returns = np.zeros(self.num_envs)
        episode_count = np.zeros(self.num_envs)
        state = self.reset()
        while episode_count.min() < num_episodes:
            actions, _ = agent.get_action(torch.from_numpy(state).to(device).float())
            new_state, reward, term, trun, _ = self.step(actions.detach().numpy())
            returns += reward
            state = new_state
            state, term, trun, reset_mask = self.reset_where_done(state, term, trun)
            episode_count += reset_mask
            returns_mean += reset_mask * returns
            returns *= (1 - reset_mask)
        returns_mean = returns_mean.mean()
        return {"sum_of_rewards": returns_mean/episode_count}
    