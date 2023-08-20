import sys
import os
import gym
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG, TD3, A2C, SAC
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from finenv import finEnv
from nacaenv import nacaEnv
from fomaml import MAML
from optimize import Optimize
import time
curr_dir = os.getcwd()



# seed = 0
# random.seed(seed)       # python random seed
# torch.manual_seed(seed)  # pytorch random seed
# np.random.seed(seed)  # numpy random seed
# torch.backends.cudnn.deterministic = True

# PPO Adaptation Parameters
# episode_length = 200  # horizon H
# num_episodes = 5  # "K" in K-shot learning
# n_steps = num_episodes * episode_length
# n_epochs = 2
# batch_size = 64
# num_iters = 300

# num_iters = num_iters * batch_size * num_episodes * n_epochs
# num_episodes = 1
# batch_size = None
# n_epochs = 1

model_params = {
    'train': True, # for the meta model
    'evalEpochs': 100, 
    'modelType': PPO,
    'time_steps' : 5,
    'n_steps' : 5,    # buffer_size for TD3/DDPG
    'batch_size' : 5,
    'n_epochs' : 1,
    'train_freq' : (1, 'episode'), # only used for TD3/DDPG (i.e. (10, 'step'))
    'loadModelType' : 'best',
    'verbose' : 1,
    'device' : 'cuda'
}
policy_kwargs = dict(activation_fn=torch.nn.ReLU, 
                net_arch=[dict(pi=[32, 32], vf=[32, 32])])
env_params = {
    'envType' : nacaEnv,
    'nPoints' : 50,
    'max_steps_ep' : 5
}
maml_params = {
    'num_iters' : 300,
    'num_tasks' : 50,
    'task_batch_size' : 10,
    'meta_lr' : 1e-3,
    'meta_save_path' : curr_dir,
    'meta_save_name' : 'reptile1_state_dict',
    'finetune_meta' : False,
    'finetune_randinit' : False,
    'reptile' : True
}
# if maml_params['reptile']:
#     maml_params['task_batch_size'] = 2
# Create random task environment set
mamlModel = MAML(env_params, model_params, policy_kwargs, maml_params)
if model_params['train']:
    a = time.time()
    mamlModel.createEnvs()
    b = time.time()
    mamlModel.learn()
    c = time.time()
    print(f'Time to initialize envs: {b-a}')
    print(f'Time to train: {c-b}')
else:
    # mamlModel.evaluate() # Test rollout on env w/ meta init
    # mamlModel.finetuneModel(test_model_params, policy_kwargs, load_meta=True) # Train on env w/ meta init.
    # print('Orig Env Params:', mamlModel.orig_env.AOA, mamlModel.orig_env.Re, mamlModel.orig_env.Ma)
    override_env_params = {
                        'Vinf' : 1,
                        'AOA' : 0.0,
                        'Ma' : 0.1,
                        'Re' : 850000,
                        'nPoints' : 50,
                        'max_steps_ep' : 5
                        }
    test_model_params = {
                        # 'evalEpochs': 10, 
                        'modelType': PPO,
                        'time_steps' : 8000,
                        'n_steps' : 40,    # buffer_size for TD3/DDPG
                        'batch_size' : 5,
                        'n_epochs' : 10,
                        'train_freq' : (1, 'episode'), # only used for TD3/DDPG (i.e. (10, 'step'))
                        'loadModelType' : 'best',
                        'verbose' : 1,
                        'device' : 'cuda'
    }
    mamlModel.finetuneModel(test_model_params, policy_kwargs, override_env_params) # Train on env w/ rand. init.

# MAML Iterations:
# for iter = 1:num_iters
#     for task in task_batch (size=task_batch_size)
#         During PPO's Adaption to a specific task:
#         for t = 1:total_timesteps
#             for e = 1:num_episodes:
#                 Gen + add new episode of max episode_length
#             for epoch = 1:n_epochs
#                 for batch (size=batch_size) in batches
#                     Calc loss over collected episodes, step gradients
#             Post-update collection of new data and gradients:
#             for e = 1:num_episodes:
#                 Gen + add new episode of max episode_length
#             Gradients += this task's PPO Gradients
#     Step with summed gradients