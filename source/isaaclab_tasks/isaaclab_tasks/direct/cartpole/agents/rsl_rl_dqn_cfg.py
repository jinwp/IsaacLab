# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDqnAlgorithmCfg, RslRlDqnQNetworkCfg, RslRlOffPolicyRunnerCfg


@configclass
class CartpoleDQNRunnerCfg(RslRlOffPolicyRunnerCfg):
    # Single environment training tuned for stability and higher DQN performance.
    num_steps_per_env = 1
    max_iterations = 10000  # More experience (~100k steps) for DQN stability
    save_interval = 100  # Save every ~5% of training
    experiment_name = "cartpole_direct_dqn"
    policy = RslRlDqnQNetworkCfg(
        obs_normalization=False,
        hidden_dims=[128, 128],
        activation="relu",
        dueling=False,
    )
    algorithm = RslRlDqnAlgorithmCfg(
        replay_buffer_size=1000000,
        gamma=0.997,
        learning_rate=1.0e-4,
        batch_size=1024,
        min_buffer_size=20000,
        target_update_interval=3000,
        target_update_tau=None,
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay_steps=2000000,
        update_every=1,
        num_gradient_steps=4,
        max_grad_norm=10.0,
        double_q=True,
    )


@configclass
class CartpoleShowcaseDQNRunnerCfg(RslRlOffPolicyRunnerCfg):
    # Single-environment training tuned for stability.
    num_steps_per_env = 1
    max_iterations = 1600
    save_interval = 10
    experiment_name = "final trial: replay buffer fix: batch size=64: lr=1.0e-3"
    policy = RslRlDqnQNetworkCfg(
        obs_normalization=False,
        hidden_dims=[128, 128],
        activation="relu",
        dueling=False,
    )
    algorithm = RslRlDqnAlgorithmCfg(
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=1.0e-3,
        batch_size=64,
        min_buffer_size=1000,
        target_update_interval=500,
        target_update_tau=None,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10000,
        update_every=1,
        num_gradient_steps=1,
        max_grad_norm=10.0,
        double_q=False,
    )
