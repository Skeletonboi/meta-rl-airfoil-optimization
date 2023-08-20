import gym
from gym import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import numpy as np

class PointEnv(gym.Env):
    '''
    Toy env to test your implementation
    The state is fixed (bandit setup)
    Action space: gym.spaces.Discrete(10)
    Note that the action takes integer values
    '''

    def __init__(self):
        self.action_space = gym.spaces.Box(low=-np.inf*np.ones(3,), high=np.inf*np.ones(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf*np.ones(6,), high=np.inf*np.ones(6,), dtype=np.float32)
        self.scale = 1
        self.reset()
        self.eps = 1e-3
        
    def get_obs(self):
        state = np.concatenate([self._state, self.target])
        return state

    def reset(self):
        self.target = np.random.randn(3) * self.scale
        self._state = np.random.randn(3) * self.scale
        return self.get_obs()

    def step(self, action: np.ndarray):
        self._state += action
        dist = np.linalg.norm(self.target - self._state)
        reward = -dist/ 10.0
        done = dist < self.eps
        if done:
            reward = 10
        info = {}
        
        return self.get_obs(), reward, done, info

    def render(self):
        pass
