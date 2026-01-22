# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDqnAlgorithmCfg, RslRlDqnQNetworkCfg, RslRlOffPolicyRunnerCfg


@configclass
class CartpoleDQNRunnerCfg(RslRlOffPolicyRunnerCfg):
    # Single environment training matching EE210_DQN_Answer.ipynb
    # Gym notebook: 1000 episodes, avg ~300 steps per episode
    # Total steps: 1000 episodes * 300 steps = 300,000 steps
    # With num_steps_per_env=4: 300,000 / 4 = 75,000 iterations
    num_steps_per_env = 4
    max_iterations = 75000  # ~1000 episodes worth of data
    save_interval = 7500  # Save every ~10% of training
    experiment_name = "cartpole_direct_dqn"
    policy = RslRlDqnQNetworkCfg(
        obs_normalization=False,
        hidden_dims=[128, 128],
        activation="relu",
        dueling=False,
    )
    algorithm = RslRlDqnAlgorithmCfg(
        replay_buffer_size=100000,  # Matches gym notebook buffer_limit=100000
        gamma=0.99,  # Matches gym notebook
        learning_rate=5.0e-4,  # Matches gym notebook (0.0005)
        batch_size=64,  # Matches gym notebook
        min_buffer_size=2000,  # Matches gym notebook UPDATE_START_SIZE=2000
        target_update_interval=50,  # Matches gym notebook TARGET_UPDATE_INTERVAL=50
        target_update_tau=None,  # Hard update like gym notebook
        epsilon_start=0.08,  # Matches gym notebook epsilon decay start
        epsilon_end=0.01,  # Matches gym notebook epsilon end
        epsilon_decay_steps=250000,  # Covers full training range
        update_every=4,  # Update every 4 steps like gym notebook (step_count % 4 == 0)
        num_gradient_steps=10,  # Matches gym notebook (for loop range(10) per update)
        max_grad_norm=None,
        double_q=False,
    )
