import numpy as np


def evaluate(model, env, num_episodes=100, max_iters=500):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = run_episode(model, env, max_iters)
        total_reward = sum(episode_rewards)
        all_episode_rewards.append(total_reward)

    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)

    return mean_reward, std_reward


def run_episode(model, env, max_iters, render=False):
    done = False
    obs = env.reset()
    episode_rewards = []
    i = 0
    while not done and i < max_iters:
        # RLBench env doesn't have render
        if render:
            env.render()

        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # info isn't returned by RLBench env
        obs, reward, done, _ = env.step(action)
        episode_rewards.append(reward)

        i += 1

    return episode_rewards
