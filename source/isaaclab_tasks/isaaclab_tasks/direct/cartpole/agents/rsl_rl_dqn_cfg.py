# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDqnAlgorithmCfg, RslRlDqnQNetworkCfg, RslRlOffPolicyRunnerCfg


@configclass
class CartpoleDQNRunnerCfg(RslRlOffPolicyRunnerCfg):
    # Tuned to match gym/EE210_DQN_Answer.ipynb update ratio for --num_envs 256.
    # num_steps_per_env=4 and num_gradient_steps=10 * 256 -> 2.5 updates per transition.
    num_steps_per_env = 4
    max_iterations = 300
    save_interval = 50
    experiment_name = "cartpole_direct_dqn"
    policy = RslRlDqnQNetworkCfg(
        obs_normalization=False,
        hidden_dims=[128, 128],
        activation="relu",
        dueling=False,
    )
    algorithm = RslRlDqnAlgorithmCfg(
        replay_buffer_size=150000,
        gamma=0.99,
        learning_rate=5.0e-4,
        batch_size=64,
        min_buffer_size=3000,
        target_update_interval=37500,
        target_update_tau=None,
        epsilon_start=0.08,
        epsilon_end=0.01,
        epsilon_decay_steps=420000,
        update_every=1,
        num_gradient_steps=2560,
        max_grad_norm=None,
        double_q=False,
    )
