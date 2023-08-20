from typing import Any, Dict
import os 

import gym
import torch

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class ProgressCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, save_freq, render_freq: int, save_path: str, name_prefix: str = '', 
                deterministic: bool = True, verbose: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__(verbose)
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._is_render = render_freq > 0
        if self.verbose:
            if self._is_render: print("Rendering results ever %d steps" % render_freq)
            else: print("Not Rendering results")
        self._num_render_episodes = 3  # only show 3 videos of evaluation
        self._deterministic = deterministic
        self.n_rollout_calls = 0 

        self._is_save = save_freq > 0
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _init_callback(self):
        print("INIT CALLBACK!")

    def save_weights(self):
        path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
        self.model.save(path)
        if self.verbose > 1:
            print(f"Saving model checkpoint to {path}")

    def _on_step(self):
        pass

    def _on_rollout_end(self) -> bool:
        self.n_rollout_calls += 1

        if self._is_save and (self.n_rollout_calls % self.save_freq == 0):
            self.save_weights()

        return True