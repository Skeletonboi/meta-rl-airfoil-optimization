from eval_utils import *
from fomaml import (MAML,
                  MAML_ID, REPTILE_ID, REPTILIAN_MAML_ID,
                  ID_TO_NAME, NAME_TO_ID)
from multitask_env import MultiTaskEnv
from grasp_env import GraspEnv
from reach_task import ReachTargetCustom
from progress_callback import ProgressCallback
from utils import parse_arguments
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3 import PPO, HER
from rlbench.action_modes import ArmActionMode
import torch
import numpy as np
import time
import datetime
import copy
import sys
import ipdb
import wandb
import random
import ray
ray.init()


"""Commands:
Train:
python run_maml.py --train

Eval:
python run_maml.py --eval --model_path=models/reptile_randomized_targets/320_iters.zip
"""


# class CustomPolicy(MlpPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomPolicy, self).__init__(*args, **kwargs,
#                                            net_arch=[64, 64, dict(pi=[64, 64], vf=[64, 64])])

#         # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

#     def _get_torch_save_params(self):
#         state_dicts = ["policy", "policy.optimizer", "policy.lr_scheduler"]

#         return state_dicts, []


# if __name__ == "__main__":

    # sys.stdout = open("outputs.txt", "w")
    # seed = 12345
    # seed = 320
    # seed = 420  # lel

# Args
random.seed(seed)       # python random seed
torch.manual_seed(seed)  # pytorch random seed
np.random.seed(seed)  # numpy random seed
torch.backends.cudnn.deterministic = True
render = False
is_train = True
model_path = ""
num_episodes = 10
algo_name = "MAML"
# lr_scheduler = None


# PPO Adaptation Parameters
episode_length = 200  # horizon H
num_episodes = 5  # "K" in K-shot learning
n_steps = num_episodes * episode_length
n_epochs = 2
batch_size = 64
num_iters = 300


algo_type = NAME_TO_ID[algo_name]

if (algo_type == MAML_ID):
    num_iters = num_iters * batch_size * num_episodes * n_epochs
    episode_length = 200
    num_episodes = 1
    batch_size = None
    n_epochs = 1

total_timesteps = 1 * n_steps  # number of "epochs"
action_size = 3  # only control EE position
manual_terminate = True
penalize_illegal = True

# Logistical parameters
verbose = 1
save_targets = True  # save the train targets (loaded or generated)
save_freq = 1  # save model weights every save_freq iteration

# MAML parameters
# MAML_ID, REPTILE_ID, REPTILIAN_MAML_ID
num_tasks = 10
task_batch_size = 8  # Reptile uses 1 during training automatically
act_mode = ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME
alpha = 1e-3
beta = 1e-3
vf_coef = 0.5
ent_coef = 0.01
base_init_kwargs = {'policy': CustomPolicy, 'n_steps': n_steps, 'n_epochs': n_epochs, 'learning_rate': alpha,
                    'batch_size': batch_size, 'verbose': verbose, 'vf_coef': vf_coef, 'ent_coef': ent_coef}
base_adapt_kwargs = {'total_timesteps': total_timesteps, "n_steps": n_steps}
render_mode = "human" if render else None
env_kwargs = {'task_class': ReachTargetCustom, 'act_mode': act_mode, "render_mode": render_mode,
                'epsiode_length': episode_length, 'action_size': action_size,
                'manual_terminate': manual_terminate, 'penalize_illegal': penalize_illegal}

# log results
config = {
    "num_tasks": num_tasks,
    "task_batch_size": task_batch_size,
    "alpha": alpha,
    "beta": beta,
    "algo_name": algo_name,
    "seed": args.seed,
}
run_title = "IDL - Train" if is_train else "IDL - Eval"
run = wandb.init(project=run_title, entity="idl-project", config=config)
wandb.save("maml.py")
wandb.save("run_maml.py")
wandb.save("multitask_env.py")
wandb.save("grasp_env.py")

print("Run Name:", run.name)

save_path = "models/" + str(run.name)
save_kwargs = {'save_freq': save_freq,
                'save_path': save_path, 'tensorboard_log': save_path, 'save_targets': save_targets}

# load in targets
train_targets = MultiTaskEnv.targets
task_batch_size = min(task_batch_size, len(train_targets))
test_targets = MultiTaskEnv.test_targets

if is_train:
    # create maml class that spawns multiple agents and sim environments
    model = MAML(BaseAlgo=PPO, EnvClass=GraspEnv, algo_type=algo_type,
                    num_tasks=num_tasks, task_batch_size=task_batch_size, targets=train_targets,
                    alpha=alpha, beta=beta, model_path=model_path,
                    env_kwargs=env_kwargs, base_init_kwargs=base_init_kwargs, base_adapt_kwargs=base_adapt_kwargs)
    model.learn(num_iters=num_iters, save_kwargs=save_kwargs)

else:
    # create maml class that spawns multiple agents and sim environments
    model = MAML(BaseAlgo=PPO, EnvClass=GraspEnv, algo_type=algo_type,
                    num_tasks=num_tasks, task_batch_size=task_batch_size, targets=train_targets,
                    alpha=alpha, beta=beta, model_path='',  # <--- Empty path for random weights
                    env_kwargs=env_kwargs, base_init_kwargs=base_init_kwargs, base_adapt_kwargs=base_adapt_kwargs)

    # see performance on train tasks
    assert(model.model_path == '')  # Testing randomly-initialized weights
    rand_init_metrics = model.eval_performance(
        model_type="random",
        save_kwargs=save_kwargs,
        num_iters=num_iters,
        targets=test_targets)

    rand_init_rewards = [v.reward for v in rand_init_metrics]
    rand_init_success = [v.success_rate for v in rand_init_metrics]
    rand_init_e_loss = [v.entropy_loss for v in rand_init_metrics]
    rand_init_pg_loss = [v.pg_loss for v in rand_init_metrics]
    rand_init_v_loss = [v.value_loss for v in rand_init_metrics]
    rand_init_loss = [v.loss for v in rand_init_metrics]

    # see performance on test tasks
    assert model_path != "" # maml or reptile
    pretrained_metrics = model.eval_performance(
        model_type=algo_name,  # "MAML", "RL^2"
        save_kwargs=save_kwargs,
        num_iters=num_iters,
        targets=test_targets,
        model_path=model_path)

    pretrained_rewards = [v.reward for v in pretrained_metrics]
    pretrained_success = [v.success_rate for v in pretrained_metrics]
    pretrained_e_loss = [v.entropy_loss for v in pretrained_metrics]
    pretrained_pg_loss = [v.pg_loss for v in pretrained_metrics]
    pretrained_v_loss = [v.value_loss for v in pretrained_metrics]
    pretrained_loss = [v.loss for v in pretrained_metrics]

    np.savez(f"final_results_{algo_name}",
                rand_init_rewards=rand_init_rewards,
                rand_init_success=rand_init_success,
                rand_init_e_loss=rand_init_e_loss,
                rand_init_pg_loss=rand_init_pg_loss,
                rand_init_v_loss=rand_init_v_loss,
                rand_init_loss=rand_init_loss,
                pretrained_rewards=pretrained_rewards,
                pretrained_success=pretrained_success,
                pretrained_e_loss=pretrained_e_loss,
                pretrained_v_loss=pretrained_v_loss,
                pretrained_loss=pretrained_loss,
                )

model.close()
