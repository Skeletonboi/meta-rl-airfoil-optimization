import numpy as np
import copy
import ipdb
import os
import collections
import wandb

import torch

from multiprocessing import Pool

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common import logger, utils
from reach_task import ReachTargetCustom
from rlbench.backend.spawn_boundary import SpawnBoundary


import ray

MAML_ID = 0
REPTILE_ID = 1
REPTILIAN_MAML_ID = 2  # batched version of Reptile
ID_TO_NAME = {MAML_ID: "MAML", REPTILE_ID: "REPTILE",
              REPTILIAN_MAML_ID: "REPTILIAN_MAML"}
NAME_TO_ID = dict((v, k) for k, v in ID_TO_NAME.items())


AvgMetricStore = collections.namedtuple(
    'AvgMetricStore', ['reward', 'success_rate',
                       'entropy_loss', 'pg_loss', 'value_loss', 'loss'])
RolloutResults = collections.namedtuple(
    'RolloutResults', ['gradients', 'parameters', 'metrics'])


class MetricStore(object):
    def __init__(self):
        self.total_reward = 0.0
        self.total_success_rate = 0.0
        self.total_entropy_loss = 0.0
        self.total_pg_loss = 0.0
        self.total_value_loss = 0.0
        self.total_loss = 0.0

    def add(self, metrics):
        reward, success_rate, entropy_loss, pg_loss, value_loss, loss = metrics
        self.total_reward += reward
        self.total_success_rate += success_rate
        self.total_entropy_loss += entropy_loss
        self.total_pg_loss += pg_loss
        self.total_value_loss += value_loss
        self.total_loss += loss

    def avg(self, count):
        count = float(count)
        return AvgMetricStore(self.total_reward / count,
                              self.total_success_rate / count,
                              self.total_entropy_loss / count,
                              self.total_pg_loss / count,
                              self.total_value_loss / count,
                              self.total_loss / count)


@ray.remote
class MAML_Worker(object):
    def __init__(self, EnvClass, ModelClass, env_kwargs, model_kwargs):
        self.env = EnvClass(**env_kwargs)
        self.model = ModelClass(env=self.env, **model_kwargs)
        self.model.env.switch_task_wrapper = self.env.switch_task_wrapper
        self.base_init_kwargs = model_kwargs

    def perform_task_rollout(self, orig_model_state_dict, target,
                             base_adapt_kwargs, algo_type=None):

        self.model.policy.train()

        # pick a task
        self.model.env.switch_task_wrapper(
            self.model.env, ReachTargetCustom, target_position=target)
        print("Switched to new target:", target)

        # copy over current original weights
        if orig_model_state_dict is not None:
            self.model.policy.load_state_dict(orig_model_state_dict)

        if "n_steps" in base_adapt_kwargs:
            del base_adapt_kwargs['n_steps']
        
        # train new model on K trajectories
        self.model.learn(**base_adapt_kwargs)

        if algo_type == MAML_ID:
            # collect new gradients for a one iteration
            # (NOTE: not one trajectory like paper does, shouldn't make a difference)
            # learn() already calls loss.backward()
            self.model.learn(total_timesteps=1 *
                             self.base_init_kwargs['n_steps'])

        metrics = [self.model.reward, self.model.success_rate, self.model.entropy_loss,
                   self.model.pg_loss, self.model.value_loss, self.model.loss]

        gradients = [p.grad.data for p in self.model.policy.parameters()]
        parameters = [p.data for p in self.model.policy.parameters()]

        return RolloutResults(gradients=gradients, parameters=parameters, metrics=metrics)

    def sample_task(self):
        return self.model.env.reset()

    def get_model(self):
        return copy.deepcopy(self.model.policy), self.model.device

    def load_model(self, state_dict=None, model_path=None):
        if state_dict is not None:
            self.model.policy.load_state_dict(state_dict)
        else:
            assert(model_path is not None)
            self.model = self.model.load(model_path, env=self.env)
            self.model.env.switch_task_wrapper = self.env.switch_task_wrapper

    def save(self, state_dict, save_path):
        if state_dict is not None:
            self.load_model(state_dict=state_dict)
        self.model.save(save_path)

    def close(self):
        self.env.close()

    def run_eval_eposide(self, max_iters=200):
        done = False
        obs = self.env.reset()
        
        self.model.policy.eval()
        
        episode_rewards = []
        i = 0
        with torch.no_grad():
            while not done and i < max_iters:
                action, _states = self.model.predict(obs)
                obs, reward, done, desc = self.env.step(action)
                episode_rewards.append(reward)
                i += 1
                    
        final_done = desc["is_success"]

        return episode_rewards, final_done
    
    def evaluate(self, num_episodes=5):

        all_episode_rewards = []
        success = []

        for i in range(num_episodes):
            episode_rewards, final_done = self.run_eval_eposide()
            total_reward = sum(episode_rewards)
            all_episode_rewards.append(total_reward)
            success.append(final_done)

        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        success_rate = sum(success)/len(success)

        return mean_reward, std_reward, success_rate



class MAML(object):
    BASE_ID = 0

    def __init__(self, BaseAlgo: BaseAlgorithm, EnvClass, algo_type, num_tasks, task_batch_size,
                 alpha, beta, model_path, env_kwargs, base_init_kwargs, base_adapt_kwargs, eval_freq=1,targets=None):
        """
            BaseAlgo:
            task_envs: [GraspEnv, ...]

            Task-Agnostic because loss function defined by Advantage = Reward - Value function.

        """
        self.algo_type = algo_type
        self.num_tasks = num_tasks
        self.task_batch_size = task_batch_size
        self.eval_freq = eval_freq

        # learning hyperparameters
        self.alpha = alpha
        self.beta = beta

        self.base_init_kwargs = base_init_kwargs
        self.base_adapt_kwargs = base_adapt_kwargs

        self.model_policy_vec = [
            MAML_Worker.remote(EnvClass=EnvClass, ModelClass=BaseAlgo, env_kwargs=env_kwargs,
                               model_kwargs=base_init_kwargs)
            for i in range(task_batch_size)]

        # optional load existing model
        self.model_path = model_path
        if model_path != "":
            print("Loading Existing model: %s" % model_path)
            self.model_policy_vec[self.BASE_ID].load_model.remote(
                model_path=model_path)
        else:
            print("No Existing model. Randomly initializing weights")

        # randomly chosen set of static reach tasks
        if targets is None:
            self.targets = []
            for _ in range(num_tasks):
                [obs] = ray.get(
                    self.model_policy_vec[self.BASE_ID].sample_task.remote())

                target_position = obs[-3:]
                self.targets.append(target_position)
                print(target_position)
        else:
            self.targets = targets

    def learn(self, num_iters, save_kwargs):
        utils.configure_logger(
            self.base_init_kwargs["verbose"], save_kwargs["tensorboard_log"], "PPO")

        if self.algo_type == REPTILE_ID:
            # Reptile only samples one task at a time and performs SGD on that one task
            # then uses the updated weights to update current weights
            task_batch_size = 1
        else:
            # Our modification performs a batch update using multiple tasks, in essence
            # batch gradient update, linearity of gradients makes this possible
            task_batch_size = self.task_batch_size

        if save_kwargs["save_targets"]:
            target_path = os.path.join(save_kwargs["save_path"], "targets")
            np.save(target_path, self.targets)

        # initialize base model and optimizer
        orig_model, device = ray.get(
            self.model_policy_vec[self.BASE_ID].get_model.remote())
        optimizer = torch.optim.Adam(orig_model.parameters(), lr=self.beta)
        # lr_scheduler = torch.

        for iter in range(num_iters):
            # sample task_batch_size tasks from set of [0, num_task) tasks
            tasks = np.random.choice(
                a=self.num_tasks, size=task_batch_size, replace=False)

            metric_store = MetricStore()

            # run multiple task rollouts in parallel
            orig_model_state_dict = orig_model.state_dict()
            results = ray.get([
                self.model_policy_vec[i].perform_task_rollout.remote(
                    orig_model_state_dict=orig_model_state_dict,
                    target=self.targets[task],
                    base_adapt_kwargs=self.base_adapt_kwargs,
                    algo_type=self.algo_type)
                for i, task in enumerate(tasks)])

            # initialize gradients
            if self.algo_type == MAML_ID:
                optimizer.zero_grad()
                for p in orig_model.parameters():
                    p.grad = torch.zeros_like(p).to(device)

                # sum up gradients and store metrics
                for res in results:
                    for orig_p, grad in zip(orig_model.parameters(), res.gradients):
                        orig_p.grad += grad / task_batch_size

            else:  # REPTILE, REPTILIAN_MAML
                # sum up gradients and store metrics
                for i, orig_p in enumerate(orig_model.parameters()):
                    mean_p = sum(res.parameters[i]
                                 for res in results) / task_batch_size
                    # weights = weights_before + lr*(weights_after - weights_before)  <--- grad
                    # weights = weights_before + lr*grad
                    # optimizer: weights = weights_before - lr*grad  <--- gradient DESCENT
                    # optimizer: weights = weights_before + lr*(-grad)
                    # optimizer: weights = weights_before + lr*(weights_before - weights_after)
                    orig_p.grad = orig_p.data - mean_p

            for res in results:
                metric_store.add(res.metrics)

            # apply gradients
            optimizer.step()

            # track performance
            (avg_reward, avg_success_rate, avg_entropy_loss, avg_pg_loss,
                avg_val_loss, avg_loss) = metric_store.avg(task_batch_size)
            wandb.log(
                {
                    "mean_reward": avg_reward,
                    "success_rate": avg_success_rate,
                    "entropy_loss": avg_entropy_loss,
                    "policy_gradient_loss": avg_pg_loss,
                    "value_loss": avg_val_loss,
                    "loss": avg_loss
                }
            )

            # save weights every save_freq and at the end
            if (iter > 0 and iter % save_kwargs["save_freq"] == 0) or iter == num_iters-1:
                path = os.path.join(save_kwargs["save_path"], f"{iter}_iters")
                self.model_policy_vec[self.BASE_ID].save.remote(
                    orig_model.state_dict(), path)
                wandb.save(path+".zip")

    def eval_performance(self, model_type, save_kwargs, num_iters=100, targets=None, base_adapt_kwargs=None, model_path=''):
        """Used to compare speed in learning between randomly-initialized weights.
        Runs vanilla PPO using the base model on K fixed tasks, each independent.
        Mean reward is stored for each trial.

        To run, instatiate MAML using model_path='' or model_path='<existing_model>'
        and set task_batch_size = how many CPU cores available to run more tests in parallel.
        Then call eval_performance with some specified targets, or None if the test targets should be
        generated from scratch. Number of test targets should be <= task_batch_size
        just to avoid storing all weights multiple times.

        Args:
            targets ([type], optional): [description]. Defaults to None.
            restore_weights (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        # load test targets
        if targets is None:
            print("No Test Targets specified. Using default:")
            for v in self.targets:
                print(v)
            targets = self.targets
            num_tasks = self.num_tasks
        else:
            num_tasks = len(targets)
        assert(num_tasks <= self.task_batch_size)

        # load evaluation params for PPO training
        if base_adapt_kwargs is None:
            print("No PPO train args specified. Using default:")
            print(self.base_adapt_kwargs)
            base_adapt_kwargs = self.base_adapt_kwargs
        
        base_adapt_kwargs['total_timesteps'] = 1 * base_adapt_kwargs['n_steps']

        assert base_adapt_kwargs['total_timesteps'] == self.base_init_kwargs["n_steps"], \
            "We need to collect mean reward and loss at each timestep or epoch, so this must be 1*n_steps!"

        utils.configure_logger(
            self.base_init_kwargs["verbose"], save_kwargs["tensorboard_log"], "PPO")

        if save_kwargs["save_targets"]:
            target_path = os.path.join(save_kwargs["save_path"], "targets")
            np.save(target_path, self.targets)

        # load same initial model into all workers
        if model_path != "":
            [self.model_policy_vec[i].load_model.remote(model_path=model_path)
                for i in range(num_tasks)]
        else:
            # use base model and set all other worker models to be same initial weights
            orig_model, device = ray.get(
                self.model_policy_vec[self.BASE_ID].get_model.remote())
            orig_model_state_dict = orig_model.state_dict()

            other_workers = list(range(num_tasks))
            other_workers.pop(self.BASE_ID)
            [self.model_policy_vec[i].load_model.remote(state_dict=orig_model_state_dict)
                for i in other_workers]

        all_metrics = []
        avg_mean_reward_eval_all = []
        avg_std_reward_eval_all = []
        avg_success_rate_eval_all = []
        
        # for num_iters, observe how fast this set of initialized weights can learn each specific task
        for iter in range(num_iters):
            # for each batch of test tasks
            results = ray.get([
                self.model_policy_vec[i].perform_task_rollout.remote(
                    orig_model_state_dict=None,  # keep training existing model
                    target=target,
                    algo_type=None,
                    base_adapt_kwargs=base_adapt_kwargs)
                for i, target in enumerate(targets)])

            metric_store = MetricStore()
            for res in results:
                metric_store.add(res.metrics)

            # store metrics averaged over all test tasks
            all_metrics.append(metric_store.avg(num_tasks))

            # track performance
            (avg_reward, avg_success_rate, avg_entropy_loss, avg_pg_loss,
                avg_val_loss, avg_loss) = metric_store.avg(self.task_batch_size)
            wandb.log(
                {
                    "mean_reward": avg_reward,
                    "success_rate": avg_success_rate,
                    "entropy_loss": avg_entropy_loss,
                    "policy_gradient_loss": avg_pg_loss,
                    "value_loss": avg_val_loss,
                    "loss": avg_loss
                }
            )

            if iter % self.eval_freq == 0:
                results_eval = ray.get(
                    [worker.evaluate.remote() for worker in self.model_policy_vec]
                )
                all_mean_rewards = [result[0] for result in results_eval]
                all_std_rewards = [result[1] for result in results_eval]
                all_success_rate = [result[2] for result in results_eval]

                avg_mean_reward_eval = sum(all_mean_rewards)/ len(all_mean_rewards)
                avg_std_reward_eval = sum(all_std_rewards)/ len(all_std_rewards)
                avg_success_rate_eval = sum(all_success_rate)/ len(all_success_rate)

                avg_mean_reward_eval_all.append(avg_mean_reward_eval)
                avg_std_reward_eval_all.append(avg_std_reward_eval)
                avg_success_rate_eval_all.append(avg_success_rate_eval)

                log_obj = {
                    "mean_reward_eval": avg_mean_reward_eval,
                    "std_reward_eval": avg_std_reward_eval,
                    "success_rate_eval": avg_success_rate_eval,
                }
                
                print(log_obj)
                wandb.log(log_obj)
                

            # save weights every save_freq and at the end
            if (iter > 0 and iter % save_kwargs["save_freq"] == 0) or iter == num_iters-1:
                path = os.path.join(
                    save_kwargs["save_path"], f"{model_type}_{iter}_iters")
                self.model_policy_vec[self.BASE_ID].save.remote(None, path)
                wandb.save(path+".zip")
                print(path)

        # save the final metrics
        np.savez(f"eval_results_{model_type}", 
            avg_mean_reward_eval_all=avg_mean_reward_eval_all,
            avg_std_reward_eval_all=avg_std_reward_eval_all,
            avg_success_rate_eval_all=avg_success_rate_eval_all,
        )

        return all_metrics

    def close(self):
        [self.model_policy_vec[i].close.remote()
         for i in range(self.task_batch_size)]
