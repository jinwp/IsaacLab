# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import gymnasium as gym
import torch

from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnv, CartpoleEnvCfg


class CartpoleShowcaseEnv(CartpoleEnv):
    cfg: CartpoleEnvCfg

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Scale action magnitude with two bins based on pole angle.
        # Small angles still get a minimum force to allow corrective actions.
        max_angle = getattr(self.cfg, "max_pole_angle", math.pi / 2)
        low_angle_frac = getattr(self.cfg, "action_scale_low_angle_frac", 0.5)
        low_scale_frac = getattr(self.cfg, "action_scale_low_frac", 0.3)
        pole_angle = self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1)
        abs_angle = torch.abs(pole_angle)
        use_high_scale = abs_angle >= (max_angle * low_angle_frac)
        low_scale = self.cfg.action_scale * low_scale_frac
        high_scale = self.cfg.action_scale
        scaled_action_scale = torch.where(use_high_scale, high_scale, low_scale)

        # fundamental spaces
        # - Box
        if isinstance(self.single_action_space, gym.spaces.Box):
            target = scaled_action_scale * self.actions
        # - Discrete
        elif isinstance(self.single_action_space, gym.spaces.Discrete):
            target = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
            target = torch.where(self.actions == 1, -scaled_action_scale, target)
            target = torch.where(self.actions == 2, scaled_action_scale, target)
        # - MultiDiscrete
        elif isinstance(self.single_action_space, gym.spaces.MultiDiscrete):
            # value
            target = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
            target = torch.where(self.actions[:, [0]] == 1, scaled_action_scale / 2.0, target)
            target = torch.where(self.actions[:, [0]] == 2, scaled_action_scale, target)
            # direction
            target = torch.where(self.actions[:, [1]] == 0, -target, target)
        else:
            raise NotImplementedError(f"Action space {type(self.single_action_space)} not implemented")

        # set target
        self.cartpole.set_joint_effort_target(target, joint_ids=self._cart_dof_idx)
        # Per-step logging for TensorBoard (for debugging off-policy behavior).
        # We log only env-0 to keep the signal interpretable even when num_envs > 1.
        step_log = self.extras.setdefault("step_log", {})
        step_log["Step/Env0/effort"] = float(target[0, 0].item())

    def _get_observations(self) -> dict:
        # fundamental spaces
        # - Box
        if isinstance(self.single_observation_space["policy"], gym.spaces.Box):
            obs = torch.cat(
                (
                    self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                ),
                dim=-1,
            )
        # - Discrete
        elif isinstance(self.single_observation_space["policy"], gym.spaces.Discrete):
            data = (
                torch.cat(
                    (
                        self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                        self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                        self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                        self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                    ),
                    dim=-1,
                )
                >= 0
            )
            obs = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
            obs = torch.where(discretization_indices(data, [False, False, False, True]), 1, obs)
            obs = torch.where(discretization_indices(data, [False, False, True, False]), 2, obs)
            obs = torch.where(discretization_indices(data, [False, False, True, True]), 3, obs)
            obs = torch.where(discretization_indices(data, [False, True, False, False]), 4, obs)
            obs = torch.where(discretization_indices(data, [False, True, False, True]), 5, obs)
            obs = torch.where(discretization_indices(data, [False, True, True, False]), 6, obs)
            obs = torch.where(discretization_indices(data, [False, True, True, True]), 7, obs)
            obs = torch.where(discretization_indices(data, [True, False, False, False]), 8, obs)
            obs = torch.where(discretization_indices(data, [True, False, False, True]), 9, obs)
            obs = torch.where(discretization_indices(data, [True, False, True, False]), 10, obs)
            obs = torch.where(discretization_indices(data, [True, False, True, True]), 11, obs)
            obs = torch.where(discretization_indices(data, [True, True, False, False]), 12, obs)
            obs = torch.where(discretization_indices(data, [True, True, False, True]), 13, obs)
            obs = torch.where(discretization_indices(data, [True, True, True, False]), 14, obs)
            obs = torch.where(discretization_indices(data, [True, True, True, True]), 15, obs)
        # - MultiDiscrete
        elif isinstance(self.single_observation_space["policy"], gym.spaces.MultiDiscrete):
            zeros = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
            ones = torch.ones_like(zeros)
            obs = torch.cat(
                (
                    torch.where(
                        discretization_indices(self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1) >= 0, [True]),
                        ones,
                        zeros,
                    ).unsqueeze(dim=1),
                    torch.where(
                        discretization_indices(self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1) >= 0, [True]),
                        ones,
                        zeros,
                    ).unsqueeze(dim=1),
                    torch.where(
                        discretization_indices(self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1) >= 0, [True]),
                        ones,
                        zeros,
                    ).unsqueeze(dim=1),
                    torch.where(
                        discretization_indices(self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1) >= 0, [True]),
                        ones,
                        zeros,
                    ).unsqueeze(dim=1),
                ),
                dim=-1,
            )
        # composite spaces
        # - Tuple
        elif isinstance(self.single_observation_space["policy"], gym.spaces.Tuple):
            obs = (self.joint_pos, self.joint_vel)
        # - Dict
        elif isinstance(self.single_observation_space["policy"], gym.spaces.Dict):
            obs = {"joint-positions": self.joint_pos, "joint-velocities": self.joint_vel}
        else:
            raise NotImplementedError(
                f"Observation space {type(self.single_observation_space['policy'])} not implemented"
            )

        observations = {"policy": obs}
        return observations


def discretization_indices(x: torch.Tensor, condition: list[bool]) -> torch.Tensor:
    return torch.prod(x == torch.tensor(condition, device=x.device), axis=-1).to(torch.bool)
