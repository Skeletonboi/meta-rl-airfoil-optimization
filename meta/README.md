# Results
You can see some results of trained agents using different parameters and reward-shaping techniques:
https://drive.google.com/file/d/1DyxaBclH8GbvDGLKfUSSwWXZ2QThympe/view?usp=sharing

# Requirements
1. Ubuntu 18.04 or above
2. python 3.6+

# Setup 
We recommand using Conda env for the following installation. 

Download and extract Copeliasim from [here](https://www.coppeliarobotics.com/downloads).

Next, follow installation instructions these repositories,
1. [PyRep](https://github.com/Team-Grasp/PyRep)
2. [RLBench](https://github.com/Team-Grasp/RLBench)
3. [Stable Baselines 3](https://github.com/Team-Grasp/stable-baselines3)


Next, clone this repository. 
```sh
$ git clone https://github.com/Team-Grasp/idl-project.git
```

# Common installation issues
1. Problem: 
    ```sh
    Traceback (most recent call last):
    File "main.py", line 108, in <module>
        model.learn(total_timesteps=total_timesteps, callback=callback) 
    File "/home/deval/anaconda3/envs/drl/lib/python3.7/site-packages/stable_baselines3/ppo/ppo.py", line 265, in learn
        reset_num_timesteps=reset_num_timesteps,
    File "/home/deval/anaconda3/envs/drl/lib/python3.7/site-packages/stable_baselines3/common/on_policy_algorithm.py", line 240, in learn
        logger.dump(step=self.num_timesteps)
    File "/home/deval/anaconda3/envs/drl/lib/python3.7/site-packages/stable_baselines3/common/logger.py", line 379, in dump
        Logger.CURRENT.dump(step)
    File "/home/deval/anaconda3/envs/drl/lib/python3.7/site-packages/stable_baselines3/common/logger.py", line 544, in dump
        _format.write(self.name_to_value, self.name_to_excluded, step)
    File "/home/deval/anaconda3/envs/drl/lib/python3.7/site-packages/stable_baselines3/common/logger.py", line 143, in write
        self.file.write("\n".join(lines) + "\n")
    ValueError: I/O operation on closed file.
    ```
    Solution: Add the following line to the logger.py file in the installation. Alternatively you can change the line and install the library locally. 

    Add the following line before line [142](https://github.com/DLR-RM/stable-baselines3/blob/e2b6f5460f362ecad3777d6fe2950f3199058d8f/stable_baselines3/common/logger.py#L142).
    ```py
    if not self.own_file: self.file = sys.stdout
    ```

2. Problem: 
    ```
    qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in "<python install dir>/lib/python3.6/site-packages/cv2/qt/plugins"
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

    Aborted (core dumped)
    ```
    Solution: Replace existing libqxc.so file from the Copeliasim directory.
    ```
    sudo cp <copeliasim install dir>/libqxcb.so <python install dir>/lib/python3.6/site-packages/cv2/qt/plugins/platforms/
    ```

# Run Random Episode 

```sh
$ cd idl-project
$ python3 main.py --eval --render
```

# Train model
Before running any training, create the following folders:
```
$ mkdir models && mkdir runs
```

Other hyperparameters such as max episode length and number of episodes to generate for one iteration can be set in main.py
Early-termination and special penalties for invalid actions can also be specified in main.py.
Next, train:
```sh
$ cd idl-project
$ python main.py --train --lr=3e-4
OR
$ python run_maml.py --train --seed <seed int> --algo_name <reptile or maml or reptilian_maml>
```

The above command will produce two folders: 
```
models/<some_timestamp>
runs/PPO_<id>
```

The first folder will have saved weights that can be loaded during evaluation. 
The second folder will contain the tensorboard metrics that can visualized with the following command:
```
$ tensorboard --logdir runs/PPO_<id>
```

# Evaluate Model performance
```sh
$ python run_maml.py --eval --seed <seed int> --algo_name <reptile or maml or reptilian_maml> --model_path <path to saved model>
```

# Run Trained model 
```sh
$ cd idl-project
$ python main.py --render --eval --model_path <path to saved model>
```
