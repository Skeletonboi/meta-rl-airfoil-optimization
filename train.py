import sys
import os
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG, TD3, A2C, SAC
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from finenv import finEnv
from nacaenv import nacaEnv
from optimize import Optimize
import time
curr_dir = os.getcwd()
# sys.path.append(curr_dir)
# sys.path.append(os.path.join(curr_dir, 'eval'))


# Specify hyperparameters
runName = 'naca_ppo_rep1_26m_2'
model_params = {
    'train': True,
    'evalEpochs': 100, 
    'modelType': 'ppo',
    'time_steps' : 16000,
    'n_steps' : 40,    # buffer_size for TD3/DDPG
    'batch_size' : 5,
    'n_epochs' : 10,
    'train_freq' : (1, 'episode'), # only used for TD3/DDPG (i.e. (10, 'step'))
    'loadModelType' : 'best',
    'finetune' : True,
    # 'finetune_path' : 'meta4_state_dict_100'
    # 'finetune_path' : 'meta1_state_dict'
    # 'finetune_path' : 'meta3_state_dict'
    # 'finetune_path' : 'meta5_state_dict'
    'finetune_path' : 'reptile1_state_dict_300'
}
env_params = {
    'envType' : nacaEnv,
    'Vinf' : 1,
    'AOA' : 2.0,
    'Ma' : 0.3,
    'Re' : 2600000,
    'nPoints' : 50,
    'max_steps_ep' : 5
}

a = Optimize(runName, curr_dir, model_params, env_params)
a.run()
