# Deep Meta Reinforcement Learning for Cross-Domain Aerodynamic Shape Optimization
This repository contains all code used for my undergraduate thesis, which can be found [here](https://drive.google.com/file/d/1DQGCqw7G3B_1PqtpEcPGsDrG6DE1X-SL/view?usp=sharing).

The original purpose of this research is to challenge the generalization capability of deep
neural networks by attempting a novel application of meta-learning on deep reinforcement learning
policies used to perform aerodynamic shape optimization across both supersonic and subsonic conditions.

The meta-learning algorithm used is a first-order approximation of Model-Agnostic Meta-Learning (MAML), called FOMAML. 
This variant of MAML allows us to mitigate the computational complexity of computing second-order Jacobian matrices by using
a first-order approximation of the meta-learning gradient.

The deep reinforcement learning algorithm selected is Proximal Policy Optimization (PPO) due to better performance in the optimization environment,
but experiments and results have also been produced for DDPG, and TD3 algorithms.

