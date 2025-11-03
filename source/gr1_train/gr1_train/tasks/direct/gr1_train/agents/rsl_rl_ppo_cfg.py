# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32 # Each environment collects 16 timesteps before a policy update
    # With N environments, the total rollout batch size is N * 16 
    # Lower values like 16 have more frequent updates, faster learning, but noisier gradients
    max_iterations = 400 # Total number of PPO iterations to perform
    save_interval = 50 # saves model checkpoints every 50 iterations
    experiment_name = "gr1-manip"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0, # Initial standard deviation for action sampling noise (high noise encourages early exploration)
        actor_obs_normalization=True, 
        # observations (such as joint states) are not normalised before being fed into networks
        # normalisation may bound network inputs to be in a similar numerical range [-1, 1]
        # this forces the network to learn it's own normalisation implicitly
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 64], # Both the policy network and value function have two hidden layers of 32 neurons
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg( # Specifically the PPO algorithm
        value_loss_coef=1.0, # Make sure actor and critic improve together
        use_clipped_value_loss=True, # Keeping learning safe
        clip_param=0.2, # Clipping the distribution change to 0.2
        # prevents policy from making destructive large updates
        entropy_coef=0.005, # encourage exploration by penalising overly deterministic policies
        num_learning_epochs=5, # gradient updates per PPO iteration
        num_mini_batches=4, # number of mini-batches to split the rollout data into during training
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99, # discount factor for future rewards
        lam=0.95, # advantage estimation parameter -> balance bias and variance in advantage estimates
        desired_kl=0.01, # KL divergence is how we define how probability distribtion A is different from B
        max_grad_norm=1.0, # Gradient clipping threshold
    )