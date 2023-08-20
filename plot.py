import pandas as pd
import matplotlib.pyplot as plt
def plotRT(df, name):
    d = df[['rollout/ep_rew_mean','time/total_timesteps','rollout/ep_len_mean','time/time_elapsed']].dropna()
    d['norm_ep_rew_mean'] = d['rollout/ep_rew_mean']/d['rollout/ep_len_mean']
    d.loc[d.norm_ep_rew_mean > 100, 'norm_ep_rew_mean'] = 50
    # d = d[d['time/total_timesteps'] < 16000]
    # plt.plot(d['time/total_timesteps'],d['rollout/ep_rew_mean'], label=name)
    plt.plot(d['time/total_timesteps'],d['norm_ep_rew_mean'], label=name)
    # plt.plot(df['train/policy_gradient_loss'].dropna())
    # plt.plot(df['train/value_loss'].dropna())
    # plt.plot(df['train/explained_variance'].dropna())
    plt.title('REPTILE+PPO v.s. FOMAML+PPO v.s. Rand. Init. @ Re=850k, AOA=0.0')
    plt.xlabel('Training Timesteps')
    plt.ylabel('Normalized Mean Episode Reward')
naca = True
if not naca:
    # doesn't exist ppo_run1 = pd.read_csv('./drl_meta_shape_optimization/sb3logs_ppo_run1/progress.csv') # Original 32k steps, 640 n_steps, 10 bs, 10 epochs, default net_arch - 18 hrs
    ppo_run2 = pd.read_csv('./drl_meta_shape_optimization - Copy/sb3logs_ppo_run2/progress.csv') # 16k steps, 320 n_steps, 10 bs, 10 epochs, 32-32 & 32-32 ReLU net_arch - 8.52 hrs
    # ppo_run3 = pd.read_csv('./drl_meta_shape_optimization/sb3logs_ppo_run3/progress.csv') # same as 2, w/ absYMax state update, and no direct CL reward - 12.8 hrs (modelname = ppo_run2)
    # ppo_run4 = pd.read_csv('./drl_meta_shape_optimization_main/sb3logs_ppo_run4/progress.csv') # 32k, 320 n_steps, 10 bs, 10 epochs, reduced net_arch, XFOILCL, absYMax 0.1 (just to compare w/ run2/3) - 8.8 hrs

    ddpg_run1 = pd.read_csv('./drl_meta_shape_optimization - Copy/sb3logs_ddpg_run1/progress.csv') # 16k, 640 buffer, 20 bs, train_freq 10 steps - 10.1 hrs
    # ddpg_run2 = pd.read_csv('./drl_meta_shape_optimization - Copy - Copy/sb3logs_ddpg_run2/progress.csv') # 16k, 640 buffer, 20 bs, train_freq 1 ep - 8.7 hrs
    # ddpg_run3 = ... # 32k, 640 buffer, 20 bs, train_freq 1 ep, absYMax = 0.2 - (crashed ... but it took so long and barely learnt over 7k steps... maybe normal?)
    # naca_ddpg_runA1 = pd.read_csv('./naca_ddpg_run_A1/sb3logs/progress.csv')

    fin_td3_run1 = pd.read_csv('./drl_meta_shape_optimization_main/fin_td3_run1/sb3logs/progress.csv') # 32k, 640 buffer, 20 bs, train_freq 1 ep, absYMax = 0.1 (cuda) - 23 hours
    # naca_td3_run1 = pd.read_csv('./naca_td3_run_A1/sb3logs/progress.csv') # 32k, 640 buffer, 20 bs, train_freq 1 ep, absYMax = 0.2 (cuda) - aborted halfway took WAY too long, meh reward
    plotRT(ppo_run2, 'PPO')
    # plotRT(ppo_run3, 'ppo3')
    # plotRT(ppo_run4, 'ppo4')
    plotRT(ddpg_run1, 'DDPG')
    # plotRT(ddpg_run2, 'DDPG')
    plotRT(fin_td3_run1, 'TD3')
if naca:
    ############## NACA PPO RUNS ###############
    naca_ppo_run1 = pd.read_csv('./naca_ppo_run1/sb3logs/progress.csv') # 16k timesteps
    naca_ppo_run2 = pd.read_csv('./naca_ppo_run2/sb3logs/progress.csv') # 32k timesteps
    naca_ppo_run3 = ... # 32k timesteps, increase panels to 225, 100 DAT points - cancelled
    naca_ppo_run4 = ... # 32k timesteps, ^ + reduce net_arch to 16,16 - cancelled
    # naca_ppo_run5 = pd.read_csv('./naca_ppo_run5/sb3logs/progress.csv') # 20k timesteps, 16,16 net_arch - crashed/cancelled
    naca_ppo_run6 = pd.read_csv('./naca_ppo_run6/sb3logs/progress.csv') # 20k timesteps, relative reward, ma 0.4, re 3.5m, 20 step eps - crashed
    # naca_ppo_run7 = pd.read_csv('./naca_ppo_run7/sb3logs/progress.csv') # ^ + ma 0.8, re 7m - crashed
    naca_ppo_run8 = pd.read_csv('./naca_ppo_run8/sb3logs/progress.csv') # 25k steps, ma 0.1, re 850k, end ep if crash 
    naca_ppo_run9 = pd.read_csv('./naca_ppo_run9/sb3logs/progress.csv') # 32k steps, ma 0.1, re 850k, end ep if crash
    naca_ppo_run10 = pd.read_csv('./naca_ppo_run10/sb3logs/progress.csv') # 32k steps, ma 0.4, re 3.5m, end ep if crash
    naca_ppo_run11 = pd.read_csv('./naca_ppo_run11/sb3logs/progress.csv') # 8k steps, ma 0.1, re 850k, end ep if crash, (5 steps per ep + 4x less steps)

    naca_ppo_run_A1 = pd.read_csv('./naca_ppo_run_A1/sb3logs/progress.csv') # 8k steps, ma 0.1, re 850k, 0 AOA
    naca_ppo_run_A2 = pd.read_csv('./naca_ppo_run_A2/sb3logs/progress.csv') # 8k steps, ma 0.1, re 850k, 2 AOA
    naca_ppo_run_A3 = pd.read_csv('./naca_ppo_run_A3/sb3logs/progress.csv') # 8k steps, ma 0.1, re 850k, 4 AOA
    naca_ppo_run_A4 = pd.read_csv('./naca_ppo_run_A4/sb3logs/progress.csv') # 8k steps, ma 0.1, re 650k, 2 AOA
    naca_ppo_run_A5 = pd.read_csv('./naca_ppo_run_A5/sb3logs/progress.csv') # 8k steps, ma 0.1, re 350k, 2 AOA
    naca_ppo_run_A6 = pd.read_csv('./naca_ppo_run_A6/sb3logs/progress.csv') # 8k steps, ma 0.1, re 150k, 2 AOA
    naca_ppo_run_A7 = pd.read_csv('./naca_ppo_run_A7/sb3logs/progress.csv') # 8k steps, ma 0.15, re 1.3m, 2 AOA
    naca_ppo_run_A8 = pd.read_csv('./naca_ppo_run_A8/sb3logs/progress.csv') # 8k steps, ma 0.3, re 2.6m, 2 AOA

    naca_ppo_150k_2_16k = pd.read_csv('./naca_ppo_150k_2_16k/sb3logs/progress.csv')
    naca_ppo_350k_2_16k = pd.read_csv('./naca_ppo_350k_2_16k/sb3logs/progress.csv')
    naca_ppo_850k_0_16k = pd.read_csv('./naca_ppo_850k_0_16k/sb3logs/progress.csv')
    naca_ppo_26m_2_16k = pd.read_csv('./naca_ppo_26m_2_16k/sb3logs/progress.csv')
    naca_ppo3_26m_2_16k = pd.read_csv('./naca_ppo3_26m_2/sb3logs/progress.csv')

    naca_ppo_850k_0_8k = pd.read_csv('./naca_ppo_850k_0_8k/sb3logs/progress.csv')
    naca_ppo2_850k_0_8k = pd.read_csv('./naca_ppo2_850k_0_8k/sb3logs/progress.csv')
    naca_ppo_26m_2_8k = pd.read_csv('./naca_ppo_26m_2_8k/sb3logs/progress.csv')
    naca_ppo2_26m_2_8k = pd.read_csv('./naca_ppo2_26m_2_8k/sb3logs/progress.csv')

    naca_ppo_meta1_850k_0 = pd.read_csv('./naca_ppo_meta1_850k_0/sb3logs/progress.csv')
    naca_ppo_meta1_850k_02 = pd.read_csv('./naca_ppo_meta1_850k_02/sb3logs/progress.csv')
    naca_ppo_meta1_26m_2 = pd.read_csv('./naca_ppo_meta1_26m_2/sb3logs/progress.csv')

    naca_ppo_meta3_26m_2 = pd.read_csv('./naca_ppo_meta3_26m_2/sb3logs/progress.csv')
    naca_ppo_meta3_850k_0 = pd.read_csv('./naca_ppo_meta3_850k_0/sb3logs/progress.csv')
    naca_ppo_meta3_150k_2 = pd.read_csv('./naca_ppo_meta3_150k_2/sb3logs/progress.csv')

    naca_ppo_meta4_850k_0 = pd.read_csv('./naca_ppo_meta4_850k_0/sb3logs/progress.csv')
    naca_ppo_meta4_26m_2 = pd.read_csv('./naca_ppo_meta4_26m_2/sb3logs/progress.csv')
    naca_ppo_meta4_26m_22 = pd.read_csv('./naca_ppo_meta4_26m_22/sb3logs/progress.csv')
    naca_ppo_meta4_26m_23 = pd.read_csv('./naca_ppo_meta4_26m_23/sb3logs/progress.csv')

    naca_ppo_meta5_850k_0 = pd.read_csv('./naca_ppo_meta5_850k_0/sb3logs/progress.csv')
    naca_ppo_meta5_26m_2 = pd.read_csv('./naca_ppo_meta5_26m_2/sb3logs/progress.csv')
    naca_ppo_meta5_26m_22 = pd.read_csv('./naca_ppo_meta5_26m_22/sb3logs/progress.csv')
    naca_ppo_meta5_26m_2mod = pd.read_csv('./naca_ppo_meta5_26m_2/sb3logs/progress_mod.csv')

    naca_ppo_rep1_850k_0 = pd.read_csv('./naca_ppo_rep1_850k_0/sb3logs/progress.csv')
    naca_ppo_rep1_26m_2 = pd.read_csv('./naca_ppo_rep1_26m_2/sb3logs/progress.csv')

# plotRT(fin_td3_run1, 'fin_td3_1')
# plotRT(naca_td3_run1, 'naca_td3_1')
# plotRT(ddpg_run1, 'fin_ddpg_1')
# plotRT(ddpg_run2, 'fin_ddpg_2')
# plotRT(naca_ddpg_runA1, 'ddpg_naca_1')

# plotRT(naca_ppo_run1)
# plotRT(naca_ppo_run2)
# plotRT(naca_ppo_run8)
# plotRT(naca_ppo_run9)
# plotRT(naca_ppo_run10)
# plotRT(naca_ppo_run11, 'A1_850k')

# plotRT(naca_ppo_run_A1, 'Re=850k, AOA=0')
# plotRT(naca_ppo_run_A3, 'Re=850k, AOA=4')

# plotRT(naca_ppo_run_A6, 'Re=150k')
# plotRT(naca_ppo_run_A5, 'Re=350k')
# plotRT(naca_ppo_run_A4, 'Re=650k')
# plotRT(naca_ppo_run_A2, 'Re=850k')
# plotRT(naca_ppo_run_A7, 'Re=1.3m')
# plotRT(naca_ppo_run_A8, 'Re=2.6m')

# plotRT(naca_ppo_150k_2_16k, 'rand_150k')
plotRT(naca_ppo_850k_0_16k, 'Random Initialization')
# plotRT(naca_ppo_26m_2_16k, 'rand_26m_unlucky')
# plotRT(naca_ppo3_26m_2_16k, 'Random Initialization')
# plotRT(naca_ppo_26m_2_8k, 'rand_8')
# plotRT(naca_ppo2_26m_2_8k, 'rand_8_2')
########## META 1 ############# 10 steps 5 bs, 20-20 tasks, 50 iters
# 150k?
# plotRT(naca_ppo_meta1_850k_0, 'FOMAML-v1')
# plotRT(naca_ppo_meta1_850k_02, 'FOMAML-v11')
# plotRT(naca_ppo_meta1_26m_2, 'FOMAML-v1')
########## META 3 ############# 10 steps, 5 bs, 40-20 tasks, 50 iters
# 150k?
# plotRT(naca_ppo_meta3_150k_2, 'meta3') # RAND?
# plotRT(naca_ppo_meta3_850k_0, 'FOMAML-v2')
# plotRT(naca_ppo_meta3_26m_2, 'FOMAML-v2')
########## META 4 ############# 40 steps, 5 bs, 50-10 tasks, 100 iters
# 150k?
# plotRT(naca_ppo_meta4_850k_0, 'FOMAML-v3')
# plotRT(naca_ppo_meta4_26m_2, 'meta4_1') # RAND
# plotRT(naca_ppo_meta4_26m_22, 'meta4_2') # RAND
# plotRT(naca_ppo_meta4_26m_23, 'FOMAML-v3')
######## META 5 ############# 5 steps, 5 bs, 50-10 tasks, 300 iters
# 150k?
plotRT(naca_ppo_meta5_850k_0, 'FOMAML-v4')
# plotRT(naca_ppo_meta5_26m_2, 'meta5') # early spike
# plotRT(naca_ppo_meta5_26m_22, 'meta5') # no early spike, crashed
# plotRT(naca_ppo_meta5_26m_2mod, 'FOMAML-v4')
########## REP 5 ############# 5 steps, 5 bs, 50-10 tasks, 300 iters
plotRT(naca_ppo_rep1_850k_0, 'REPTILE')
# plotRT(naca_ppo_rep1_26m_2, 'REPTILE')

plt.legend()
plt.show()

plt.clf()
# plotRT(naca_ppo_run11, 'A1_850k')
# plotRT(naca_ppo_meta_finetune_850k_0, 'meta_850k_0')
# plotRT(naca_ppo_meta2_finetune_850k_0, 'meta2_850k_0')
# plt.legend()
# plt.show()