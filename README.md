# Fourier Manipulation Reinforcement Learning
This is a reinforcement learning project for the Fourier GR1. Using the PPO algorithm provided by RSL RL we are teaching the robot manipulation in the simulation Isaac Sim and the reinforcement learning library Isaac Lab.

## General information about the project
The primary workspace for this project is located in the folder `gr1_manip/source/gr1_train/gr1_train/tasks/direct/gr1_train/`. 
- The file `gr1_train_env_cfg.py` is the config file for the environment. It contains important information such as the number of environments running in parallel, the frequency of the physics steps, the default joints of the robot.
- The file `gr1_train_env.py` is the actual python class file for the environment. It contains the behaviours that are run during the training, such as calculating rewards, doing actions atc.
- The file `agents/rsl_rl_ppo_cfg.py` is the config file for the algorithm and network themselves. It contains important hyperparameters for the algorithm such as the horizon and gamma value, I have written definitions in this file to explain each of the hyperparameters.

## Running instructions
Run all commands in the home repo `gr1_manip`

### Setting up
```bash
source setup_pip.sh
```
This command installs the project as a python package, it then also replaces the policy runner file `/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py` with `/workspace/fourier-sim/gr1_train/algos/on_policy_runner.py`. In our copy of the file I have added some code to store the mean rewards and create charts. If the runner doesn't work and it says the issue is in `on_policy_runner.py`, it is likely this file has been updated by Nvidia and you'll need to rewrite the graph making. I set up the array in `__init__`, then add to it in `log`, then making is in the `save` method.

### Running the Reinforcement learning
```bash
python3 scripts/rsl_rl/train.py --task=Gr1-Manip
```
This command runs the training using the RSL RL library. We have registered the RL task in `gr1_manip/source/gr1_train/gr1_train/tasks/direct/gr1_train/__init__.py` under the name `Gr1-Manip`. You can use the command line argument `--headless`, this will run the reinforcement learning without the GUI, it is often faster. You can use the command line argument `--num_envs=10`, however this can also just be changed in `gr1_train_env_cfg.py`. The number of iterations is specified in `agents/rsl_rl_ppo_cfg.py`.

This command will create the networks in `gr1_manip/logs`, it stores a copy of the network for every 50 iterations. However, these networks cannot be used in Isaac Sim directly, you must first run `play.py` (as specified below), this will create a folder `gr1_manip/logs/<date and id of network>/exported` which contains the `.pt` file you can use for the sim.

### Playing the Reinforcement learning
```bash
python3 scripts/rsl_rl/play.py --task=Gr1-Manip
```
This command runs the reinforcement learning environment using the current most recent network, it will not train the network and doesn't even run the `_get_rewards` behaviour of `gr1_train_env.py` (it should run all other MDP functions). You can again use the command line argument `--num-envs` or `--headless`.

This command is important when exporting a network to be used in Isaac Sim. The command will create a folder `gr1_manip/logs/<date and id of network>/exported` which contains the `.pt` file you can use for the sim.

## Docker instructions
In the alienware computer under the fourier user, you can run a docker for this project (and other Isaac Lab projects) that has important libraries like torch and has git setup and has the `fourier-sim` folder mounted (The docker requires the folder `fourier-sim` to be in the same place, can be altered by changing the docker compose in `IsaacLab`).

In the folder `~/IsaacLab`, to start the docker container run:
```bash
python3 docker/container.py start
```
You can enter the container in VS code by clicking `ctrl+shift+p` and then clicking *Attach to running container*. You can also enter in terminal with:
```bash
python3 docker/container.py enter
```
