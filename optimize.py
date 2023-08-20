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
import time
# Add path to code
curr_dir = os.getcwd()
sys.path.append(curr_dir)
sys.path.append(os.path.join(curr_dir, 'eval'))

class Optimize():
    def __init__(self, runName, base_dir, model_params, env_params):
    # Make directories
        self.paths = self.makeDirs(base_dir, runName)
        self.base_dir = base_dir

        self.runName = runName
        self.train = model_params['train']
        self.evalEpochs = model_params['evalEpochs']
        self.modelType = model_params['modelType']
        self.time_steps = model_params['time_steps']
        self.n_steps = model_params['n_steps']
        self.batch_size = model_params['batch_size']
        self.n_epochs = model_params['n_epochs']
        self.train_freq = model_params['train_freq']
        self.loadModelType = model_params['loadModelType']

        self.finetune = model_params['finetune']
        self.finetune_path = model_params['finetune_path']

        self.envType = env_params['envType']
        self.Vinf = env_params['Vinf']
        self.AOA = env_params['AOA']
        self.Ma = env_params['Ma']
        self.Re = env_params['Re']
        self.nPoints = env_params['nPoints']
        self.max_steps_ep = env_params['max_steps_ep']

        self.args = [self.max_steps_ep, self.nPoints, self.Vinf, self.AOA, self.Ma, self.Re]
        self.env = self.envType(self.args, self.paths, runName, train=self.train)
        self.env.reset()

        self.callbackLogPath = os.path.join(self.paths[0], 'callbackLog')
        callbackFinPath = os.path.join(self.paths[0], 'evalfinGeoms')
        if not os.path.exists(callbackFinPath):
            os.makedirs(callbackFinPath)
        if not os.path.exists(self.callbackLogPath):
            os.makedirs(self.callbackLogPath)
        self.evalEnv = self.envType(self.args, self.paths, runName, train=self.train)
        self.evalEnv.fin_dir = callbackFinPath
        self.evalEnv.reset()

        # Model specification
        self.env.reset()

    # Make directories
    def makeDirs(self, base_dir, runName):
        run_path = os.path.join(base_dir, f'{runName}')
        sb3log_path = os.path.join(run_path, 'sb3logs')
        fin_path = os.path.join(run_path, 'finGeoms')
        temp_path = os.path.join(run_path, 'tempFiles')
        paths = [run_path, sb3log_path, fin_path, temp_path]
        for p in paths:
            if not os.path.exists(p):
                os.makedirs(p)
        # NOTE: NEED TO COPY EXE TO RUN DIR?
        return paths

    def run(self):
        if self.modelType == 'ppo':
            policy_kwargs = dict(activation_fn=torch.nn.ReLU, 
                            net_arch=[dict(pi=[32, 32], vf=[32, 32])])
            model = PPO("MlpPolicy", 
                        self.env,
                        n_steps = self.n_steps,
                        batch_size = self.batch_size,
                        n_epochs = self.n_epochs,
                        policy_kwargs=policy_kwargs,
                        verbose=1, 
                        device='cuda'
            )
            self.evalModel = PPO
        elif self.modelType == 'ddpg':
            policy_kwargs = dict(net_arch=dict(pi=[32, 32], qf=[64, 64]))
            model = DDPG("MlpPolicy", 
                        self.env,
                        buffer_size = self.n_steps,
                        batch_size = self.batch_size,
                        train_freq = self.train_freq,
                        policy_kwargs=policy_kwargs,
                        verbose=1, 
                        device='cpu'
            )
            self.evalModel = DDPG
        elif self.modelType == 'td3':
            policy_kwargs = dict(net_arch=dict(pi=[32, 32], qf=[64, 64]))
            model = TD3("MlpPolicy", 
                    self.env,
                    buffer_size = self.n_steps,
                    batch_size = self.batch_size,
                    train_freq = self.train_freq,
                    policy_kwargs=policy_kwargs,
                    verbose=1, 
                    device='cuda'
            )
            self.evalModel = TD3
        # if self.loadModelType == 'save':
        save_model_path = os.path.join(self.paths[0], self.runName) + '.zip'
        # elif self.loadModelType == 'best':
        best_model_path = os.path.join(self.paths[0], 'best_model') + '.zip'
        if self.train:
            if self.finetune:
                trained_param_path = os.path.join(self.base_dir, f'{self.finetune_path}.zip')
                model = model.load(trained_param_path, env=self.env)
            # Set logger
            new_logger = configure(self.paths[1], ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)
            # Eval Callback (saves best model)
            callback = EvalCallback(self.evalEnv, best_model_save_path=self.paths[0], log_path=self.callbackLogPath, eval_freq=self.n_steps)
            # Train
            a = time.time()
            model.learn(total_timesteps=self.time_steps, callback=callback)
            b = time.time()
            print('TIME TAKEN: ', b-a)
            # Save final model
            model.save(save_model_path)
        else:
            test_path = os.path.join(self.paths[0], 'test')
            if not os.path.exists(test_path):
                os.mkdir(test_path)
            env = self.envType(self.args, self.paths, self.runName, self.train)
            obs = env.reset()
            loaded_model = self.evalModel.load(best_model_path, env, verbose=1, device='cpu')


            best_state = []
            best_rew_clcd = [-np.inf, -np.inf]
            for ep in range(self.evalEpochs):
                print('Episode #:', ep)
                done = False
                rewardArr = []
                CLCDArr = []
                obsArr = []
                rewardArr.append(0)
                CLCDArr.append(env.initialCLCD)
                obsArr.append(obs)
                for i in range(self.max_steps_ep):
                    env.render()
                    action, _ =  loaded_model.predict(obs)
                    # action, _ = model.predict(obs)
                    obs, reward, done, CLCD = env.step(action)
                    rewardArr.append(reward)
                    CLCDArr.append(CLCD)
                    obsArr.append(obs)
                env.close()

                # print(rewardArr)
                # print(CLCDArr)
                # print(obsArr)
                # plt.plot(rewardArr)
                # plt.show()
                
                index_max = max(range(len(CLCDArr)), key=CLCDArr.__getitem__)
                if CLCDArr[index_max] > best_rew_clcd[1]:
                    best_state = obsArr[index_max]
                    best_rew_clcd[0] = rewardArr[index_max]
                    best_rew_clcd[1] = CLCDArr[index_max]

            print('Best State, Reward, CLCD: ', best_state, best_rew_clcd[0], best_rew_clcd[1])
