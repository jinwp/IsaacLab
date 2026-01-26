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
    max_iterations = 8000  # More experience (~100k steps) for DQN stability
    save_interval = 100  # Save every ~5% of training
    experiment_name = "cartpole_direct_dqn"
    policy = RslRlDqnQNetworkCfg(
        obs_normalization=True,
        hidden_dims=[256, 256],
        activation="relu",
        dueling=True,
    )
    algorithm = RslRlDqnAlgorithmCfg(
        replay_buffer_size=100000,
        gamma=0.99,
        learning_rate=1.0e-4,
        batch_size=128,
        min_buffer_size=2000,
        target_update_interval=500,
        target_update_tau=None,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=280000,
        update_every=1,
        num_gradient_steps=4,
        max_grad_norm=10.0,
        double_q=True,
    )
