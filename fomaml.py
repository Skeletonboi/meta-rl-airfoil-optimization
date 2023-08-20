import os
import sys
import torch
import copy
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
from stable_baselines3.common.logger import configure
# RolloutResults = collections.namedtuple(
#     'RolloutResults', ['gradients', 'parameters', 'metrics'])
class MAML():
# TBD single rollout learn & model params to be passed or gen?
    def __init__(self, env_params, model_params, policy_kwargs, maml_params):
        self.curr_dir = os.getcwd()

        self.train = model_params['train']
        self.evalEpochs = model_params['evalEpochs']
        self.modelType = model_params['modelType']
        self.time_steps = model_params['time_steps']
        self.n_steps = model_params['n_steps']
        self.batch_size = model_params['batch_size']
        self.n_epochs = model_params['n_epochs']
        self.train_freq = model_params['train_freq']
        self.verbose = model_params['verbose']
        self.device = model_params['device']
        # self.loadModelType = model_params['loadModelType']

        self.policy_kwargs = policy_kwargs

        self.envType = env_params['envType']
        self.nPoints = env_params['nPoints']
        self.max_steps_ep = env_params['max_steps_ep']

        self.num_iters = maml_params['num_iters']
        self.num_tasks = maml_params['num_tasks']
        self.task_batch_size = maml_params['task_batch_size']
        self.meta_lr = maml_params['meta_lr']
        self.meta_save_path = maml_params['meta_save_path']
        self.meta_save_name = maml_params['meta_save_name']
        self.finetune_meta = maml_params['finetune_meta']
        self.finetune_randinit = maml_params['finetune_randinit']
        self.reptile = maml_params['reptile']
        
        if self.finetune_meta:
            self.orig_env = self.makeEnv('naca_ppo_meta2_finetune_850k_0')
        elif self.finetune_randinit:
            self.orig_env = self.makeEnv('naca_ppo_meta_randinit')
        else:
            self.orig_env = self.makeEnv('naca_ppo_meta_base')
        self.orig_worker = self.modelType("MlpPolicy", 
                                        self.orig_env,
                                        n_steps = self.n_steps,
                                        batch_size = self.batch_size,
                                        n_epochs = self.n_epochs,
                                        policy_kwargs=self.policy_kwargs,
                                        verbose=self.verbose, 
                                        device=self.device)
        self.orig_model = copy.deepcopy(self.orig_worker.policy)
        self.optimizer = torch.optim.Adam(self.orig_model.parameters(), lr=self.meta_lr)

    def learn(self):
        for iter in range(self.num_iters):

            f = open("metalog.txt", "w")
            f.write(f"Iteration {iter} \n")
            f.close()
            
            orig_model_curr_state = self.orig_model.state_dict()
            iter_results = []
            for t in range(self.task_batch_size):
                env = np.random.choice(self.envList)
                results = self.trainSingleTask(env, orig_model_curr_state)
                iter_results.append(results)
            if not self.reptile:
                self.optimizer.zero_grad()
                for p in self.orig_model.parameters():
                    p.grad = torch.zeros_like(p).to(self.device)
                for res in iter_results:
                    for orig_p, grad in zip(self.orig_model.parameters(), res[0]):
                        orig_p.grad += grad / self.task_batch_size
                    # for orig_p, grad in zip(self.orig_model.parameters(), res.gradients):
                        # orig_p.grad += grad / self.task_batch_size
            else:
                self.optimizer.zero_grad()
                for i, orig_p in enumerate(self.orig_model.parameters()):
                    mean_p = sum(res[1][i] for res in iter_results)/self.task_batch_size
                    orig_p.grad = orig_p.data - mean_p
                # for i, orig_p in enumerate(self.orig_model.parameters()):
                    # mean_p = sum(res.parameters[i] for res in results) / self.task_batch_size
                    # orig_p.grad = orig_p.data - mean_p
            self.optimizer.step()

            if iter % 25 == 0:
                self.saveModel(self.orig_model.state_dict(), iter)
        self.saveModel(self.orig_model.state_dict(), self.num_iters)

    def trainSingleTask(self, env, orig_model_curr_state):
        # instantiate new model w/ input env
        model = self.modelType("MlpPolicy",
                            env = env,
                            n_steps = self.n_steps,
                            batch_size = self.batch_size,
                            n_epochs = self.n_epochs,
                            policy_kwargs=self.policy_kwargs,
                            verbose=self.verbose, 
                            device=self.device)
        # load model w/ current orig_model params
        model.policy.load_state_dict(orig_model_curr_state)
        # train on single task
        model.learn(total_timesteps=self.time_steps)
        # record output
        # metrics = [model.reward, model.success_rate, model.entropy_loss,
        #            model.pg_loss, model.value_loss, model.loss]
        metrics = 0
        gradients = [p.grad.data for p in model.policy.parameters()]
        parameters = [p.data for p in model.policy.parameters()]

        return (gradients, parameters, metrics) # tuple here may be an issue? idk yet
        # return RolloutResults(gradients=gradients, parameters=parameters, metrics=metrics)

    
    def createEnvs(self):
        envList = []
        for i in range(self.num_tasks):
            runName = f'naca_ppo_meta_{i}'
            env = self.makeEnv(runName)
            envList.append(env)
        self.envList = envList
        return

    def makeEnv(self, runName):
        envPaths = self.makeDirs(self.curr_dir, runName)
        # AOA [0,4] and Re [20k, 4m], scale Ma to Re but bound to [0.1, 0.5]
        AOA = np.random.uniform(0, 4)
        Re = np.random.uniform(2e4, 4e6)
        Ma = np.clip((Re*0.4)/(4e6 - 2e4), 0.1, 0.5)
        Vinf = 1
        args = [self.max_steps_ep, self.nPoints, Vinf, AOA, Ma, Re]
        env = self.envType(args, envPaths, runName, self.train) # paths[2] = finDir
        return env

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
    
    def saveModel(self, state_dict, iter):
        self.orig_worker.policy.load_state_dict(state_dict)
        self.orig_worker.save(os.path.join(self.meta_save_path, self.meta_save_name+f'_{iter}'))
    
    def evaluate(self):
        # Load trained meta-parameters from zip file
        trained_param_path = os.path.join(self.meta_save_path, 'meta4_state_dict.zip')
        # self.orig_worker.policy.load_state_dict(torch.load(param_path))
        self.orig_worker.load(trained_param_path)
        # Run an episode on orig_worker's random environment
        best_state = []
        best_rew_clcd = [-np.inf, -np.inf]
        for ep in range(self.evalEpochs):
            print('Episode #:', ep)
            done = False
            rewardArr = []
            CLCDArr = []
            obsArr = []
            obs = self.orig_env.reset()
            rewardArr.append(0)
            CLCDArr.append(self.orig_env.initialCLCD)
            obsArr.append(obs)
            for i in range(200):
                action, _ =  self.orig_worker.predict(obs)
                # action, _ = model.predict(obs)
                obs, reward, done, CLCD = self.orig_env.step(action)
                rewardArr.append(reward)
                CLCDArr.append(CLCD)
                obsArr.append(obs)
            self.orig_env.close()

            print(rewardArr)
            print(CLCDArr)
            print(obsArr)
            plt.plot(rewardArr)
            plt.plot(CLCDArr)
            plt.show()

                # index_max = max(range(len(CLCDArr)), key=CLCDArr.__getitem__)
            # if CLCDArr[index_max] > best_rew_clcd[1]:
            #     best_state = obsArr[index_max]
            #     best_rew_clcd[0] = rewardArr[index_max]
            #     best_rew_clcd[1] = CLCDArr[index_max]

    def finetuneModel(self, test_model_params, policy_kwargs, override_env_params):
        # NOTE: Manually override to True (since 'train' for meta-learning is False)
        self.orig_env.train = True
        for param, value in override_env_params.items():
            setattr(self.orig_env, param, value)
        # if not load_meta:
        #     # Change to a new directory for rand. init. run
        #     randinit_run_name = 'naca_ppo_meta_randinit'
        #     paths = self.makeDirs(self.curr_dir, randinit_run_name)
        #     self.orig_env.paths = paths
        #     self.orig_env.runName = randinit_run_name
        self.orig_worker = test_model_params['modelType']("MlpPolicy",
                                            env=self.orig_env,
                                            n_steps = test_model_params['n_steps'],
                                            batch_size = test_model_params['batch_size'],
                                            n_epochs = test_model_params['n_epochs'],
                                            policy_kwargs=policy_kwargs,
                                            verbose=test_model_params['verbose'], 
                                            device=test_model_params['device'])
        if self.finetune_meta:
            # Load trained meta-parameters from zip file
            trained_param_path = os.path.join(self.meta_save_path, 'meta4_state_dict.zip')
            self.orig_worker.load(trained_param_path)
            print('Successfully loaded meta params!')
        # Set logger
        new_logger = configure(self.orig_env.paths[1], ["stdout", "csv", "tensorboard"])
        self.orig_worker.set_logger(new_logger)
        # Learn
        self.orig_worker.learn(total_timesteps=test_model_params['time_steps'])
        # Save finetuned model
        finetuned_save_path = os.path.join(self.orig_env.paths[0], self.orig_env.runName) + '.zip'
        self.orig_worker.save(finetuned_save_path)
