from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

import numpy as np
import math
import matplotlib.pyplot as plt

from grasp_env import GraspEnv
from reach_task import ReachTargetCustom

class Agent(object):
    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs: np.ndarray):
        cur_pos = obs[:3]
        target_pos = obs[3:]
        vec = target_pos - cur_pos
        vec = vec / np.linalg.norm(vec)
        arm = np.zeros(self.action_size-1)
        # arm[0] = 5 * math.pi / 180.0
        arm[0:3] = vec * 0.1
        arm[3:] = [0, 0, 0, 1]
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)

if __name__ == "__main__":
    # import gym
    # import rlbench.gym
    # full_pos = np.load("full_pos.npy")
    # plt.plot(full_pos[:, 0], label="x")
    # plt.plot(full_pos[:, 1], label="y")
    # plt.plot(full_pos[:, 2], label="z")
    # plt.legend()
    # plt.show()
    # exit()

    # temp_env = gym.make('reach_target-state-v0')
    # print(temp_env.action_space)

    act_mode = ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME
    env = GraspEnv(task_class=ReachTargetCustom, render_mode="human", act_mode=act_mode)
    agent = Agent(env.action_space.shape[0])

    iterations = 5
    max_steps = 500
    obs = None
    full_pos = []
    for i in range(iterations):
        obs = env.reset()
        for j in range(max_steps):
            action = agent.act(obs)
            obs, reward, terminate, _ = env.step(action)
            full_pos.append(obs)
            if terminate: break

    full_pos = np.vstack(full_pos)
    np.save("full_pos", full_pos)

    env.close()



