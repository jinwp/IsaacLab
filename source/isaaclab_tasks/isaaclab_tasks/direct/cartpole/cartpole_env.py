# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]
    initial_cart_pos_range = [-0.5, 0.5]  # the range in which the cart position is sampled from on reset [m]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = 0.0
    rew_scale_pole_pos = 0.0
    rew_scale_cart_pos = 0.0
    rew_scale_cart_vel = 0.0
    rew_scale_pole_vel = 0.0

class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel
        self._discrete_action_values = None
        self._use_discrete_actions = getattr(self.cfg, "use_discrete_actions", False)
        if self._use_discrete_actions:
            self._discrete_action_values = torch.tensor(
                getattr(self.cfg, "discrete_action_values", (-1.0, 0.0, 1.0)),
                device=self.device,
                dtype=torch.float32,
            )
        # Episodic logging buffers for direct envs (to match manager-based logging style)
        self._reward_term_names = ("alive", "terminating", "pole_pos", "cart_pos", "cart_vel", "pole_vel")
        self._termination_term_names = ("cart_out_of_bounds", "time_out")
        self._episode_reward_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self._reward_term_names
        }
        self._term_dones = torch.zeros((self.num_envs, len(self._termination_term_names)), dtype=torch.bool, device=self.device)
        self._last_episode_dones = torch.zeros_like(self._term_dones)
        self.extras["log"] = {}

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["cartpole"] = self.cartpole
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._use_discrete_actions:
            action_ids = actions.to(dtype=torch.int64).view(-1)
            scaled_actions = self._discrete_action_values[action_ids].unsqueeze(-1)
            self.actions = self.action_scale * scaled_actions
        else:
            self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)
        # Per-step logging for TensorBoard (for debugging off-policy behavior).
        # We log only env-0 to keep the signal interpretable even when num_envs > 1.
        step_log = self.extras.setdefault("step_log", {})
        step_log["Step/Env0/effort"] = float(self.actions[0, 0].item())

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reward_terms = compute_reward_terms(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        for key in self._reward_term_names:
            self._episode_reward_sums[key] += reward_terms[key]
        return reward_terms["total"]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # Use max_pole_angle if defined (for discrete cartpole), otherwise default to π/2
        max_angle = getattr(self.cfg, "max_pole_angle", math.pi / 2)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > max_angle, dim=1)
        # Track termination terms for episodic logging
        self._term_dones[:, 0] = out_of_bounds
        self._term_dones[:, 1] = time_out
        done_rows = self._term_dones.any(dim=1).nonzero(as_tuple=True)[0]
        if done_rows.numel() > 0:
            self._last_episode_dones[done_rows] = self._term_dones[done_rows]
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        episode_lengths = self.episode_length_buf[env_ids].float().clone()
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        # joint_pos[:, self._cart_dof_idx] += sample_uniform(
        #     self.cfg.initial_cart_pos_range[0],
        #     self.cfg.initial_cart_pos_range[1],
        #     joint_pos[:, self._cart_dof_idx].shape,
        #     joint_pos.device,
        # )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Log episodic reward/termination terms for the envs being reset
        self.extras["log"] = {}
        episode_lengths_s = episode_lengths * self.step_dt
        episode_length_s = self.max_episode_length_s
        episodic_return = torch.zeros(1, device=self.device, dtype=torch.float32)
        for key in self._reward_term_names:
            episodic_sum_avg = torch.mean(self._episode_reward_sums[key][env_ids])
            self.extras["log"]["Episode_Reward/" + key] = episodic_sum_avg / episode_length_s
            episodic_return += episodic_sum_avg
            self._episode_reward_sums[key][env_ids] = 0.0
        self.extras["log"]["Episode_Return/total"] = episodic_return.item()
        self.extras["log"]["Episode_Length/steps"] = torch.mean(episode_lengths).item()
        self.extras["log"]["Episode_Length/s"] = torch.mean(episode_lengths_s).item()

        last_episode_done_stats = self._last_episode_dones[env_ids].float().mean(dim=0)
        for i, key in enumerate(self._termination_term_names):
            self.extras["log"]["Episode_Termination/" + key] = last_episode_done_stats[i].item()
        self._last_episode_dones[env_ids] = False


@configclass
class CartpoleDiscreteEnvCfg(CartpoleEnvCfg):
    # action_space = 2
    action_space = {7}
    use_discrete_actions = True
    discrete_action_values = (-1.0, -0.5, -0.2, 0, 0.2, 0.5, 1.0)
    # discrete_action_values = (-0.1, 0.1)
    # discrete_action_values = (-1.0, 0.0, 1.0)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )
    # Set angle tolerance to match Gym CartPole-v0 (±0.2095 rad = ±12°)
    # This only affects discrete cartpole, PPO continuous version remains unchanged
    max_pole_angle = math.pi / 2

    # Standard Gym Reward: +1 for every step alive, 0 otherwise
    rew_scale_alive = 1.0
    # rew_scale_terminated = 0.0
    # rew_scale_pole_pos = 0.0
    # rew_scale_cart_vel = 0.0
    # rew_scale_pole_vel = 0.0


@torch.jit.script
def compute_reward_terms(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_pos = rew_scale_cart_pos * torch.sum(torch.square(cart_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_pos + rew_cart_vel + rew_pole_vel
    return {
        "alive": rew_alive,
        "terminating": rew_termination,
        "pole_pos": rew_pole_pos,
        "cart_pos": rew_cart_pos,
        "cart_vel": rew_cart_vel,
        "pole_vel": rew_pole_vel,
        "total": total_reward,
    }
