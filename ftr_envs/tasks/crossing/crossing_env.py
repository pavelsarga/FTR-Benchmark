# -*- coding: utf-8 -*-
"""
====================================
@File Name ：crossing.py
@Time ： 2024/9/29 PM12:07
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
from typing import Sequence
import logging

import einops
import numpy as np
import torch
import torch.nn as nn

from omni.isaac.lab.envs import VecEnvObs
from omni.isaac.lab.sim import PhysxCfg

from ftr_envs.utils.torch import add_noise

from .ftr_env import FtrEnv, FtrEnvCfg, configclass

_log = logging.getLogger(__name__)

@torch.jit.script
def point_in_rotated_ellipse(x, y, h, k, a, b, theta):
    """
    where (h, k) is the center of the ellipse, a and b are the semi-major and semi-minor axes along the x and y axes before rotation, and theta is the rotation angle of the ellipse (in radians).
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    term1 = ((x - h) * cos_theta + (y - k) * sin_theta) ** 2 / a ** 2
    term2 = ((x - h) * sin_theta - (y - k) * cos_theta) ** 2 / b ** 2
    return term1 + term2 <= 1


@torch.jit.script
def out_of_range(pos, start, target, semi_major_slack: float = 3.0, semi_minor_slack: float = 2.0):
    center = (start + target) / 2
    op = (target - start)[:, :2]
    d_max = op[:, :2].norm(dim=-1, p=2)
    theta = torch.arctan(op[:, 1] / op[:, 0])
    return ~point_in_rotated_ellipse(
        pos[:, 0],
        pos[:, 1],
        center[:, 0],
        center[:, 1],
        d_max / 2 + semi_major_slack,
        d_max / 4 + semi_minor_slack,
        theta,
    )


@configclass
class CrossingEnvCfg(FtrEnvCfg):
    # env
    num_actions = 4
    num_observations = 966
    robot_type: str = "marv"  # "ftr" or "marv"
    num_states = 0
    shaping_coef = 27.279373033235267
    shaping_gamma = 0.999

    # Shock penalty — penalises linear acceleration magnitude using a deadzone formulation.
    # shock_coef < 0 → penalty; None → disabled.
    # shock_threshold: accelerations below this (m/s²) are ignored — covers normal locomotion.
    # shock_scale: excess above threshold at which shock_norm reaches 1.0 (m/s²).
    shock_coef: float | None = None
    shock_threshold: float = 11.0  # m/s² — deadzone; normal locomotion is ~5–10 m/s²
    shock_scale: float = 35.0      # m/s² of excess above threshold → shock_norm = 1.0

    # Ground clearance penalty — penalises robot body being too close to terrain.
    # clearance_coef < 0 → penalty; None → disabled.
    clearance_coef: float | None = None

    # Action bonus
    # Encourages the policy to take non-zero actions rather than freezing.
    # Set to None to disable.
    action_bonus_coef: float | None = None
    flipper_action_bonus_coef: float | None = None
    # Blend between actual velocity and intended velocity from policy output.s
    lin_action_ratio: float = 0.5

    # Per-step penalty applied every step (negative → constant reward penalty).
    step_penalty: float = 0.0

    # Terminal rewards / penalties
    goal_reached_reward: float = 1.0
    failed_reward: float = -1.0

    # Orientation penalties (set to None to disable)
    roll_coef: float | None = None
    roll_rate_coef: float | None = None
    pitch_coef: float | None = None
    pitch_rate_coef: float | None = None

    # Joint-velocity variance penalty (set to None to disable)
    joint_vel_variance_coef: float | None = None

    # Joint-angle variance penalty (set to None to disable)
    joint_angle_variance_coef: float | None = None

    joint_ang_from_flat_coef: float | None = None

    # Legacy flipper_training-style reward variants.
    # These replicate the exact formulas from PotentialGoalWithPenaltiesConfigurable so that
    # configs tuned on the native flipper_training simulator transfer without rescaling.
    # legacy_joint_vel_variance_coef  — var(|flipper_vel_cmds|),  same formula as joint_vel_variance_coef
    # legacy_joint_angle_variance_coef — var(|flipper_angles|),   vs CE's mean normalised L-R diff
    # legacy_track_vel_variance_coef  — var(|v, w cmds|),         not present in CE
    # legacy_roll_rate_coef / legacy_pitch_rate_coef — |ω| / π,   same formula as CE counterparts
    legacy_joint_vel_variance_coef: float | None = None
    legacy_joint_angle_variance_coef: float | None = None
    legacy_track_vel_variance_coef: float | None = None
    legacy_roll_rate_coef: float | None = None
    legacy_pitch_rate_coef: float | None = None
    flipper_style: bool = False

    # Timeout penalty — applied when an episode ends by timeout (truncation).
    # Separate from failed_reward (which covers rollover/out-of-range only).
    # Set to 0.0 to disable.
    timeout_penalty: float | None = 0.0

    fixed_forward_vel: float | None = None

    # Flipper-ground contact reward via applied joint torque.
    # When a flipper rests on the ground the actuator fights the contact constraint → high torque.
    # Reward = coef × geometric_mean(mean, min) of torque/effort_limit across all 4 flipper joints.
    # Normalized to [0, 1] by effort_limit. Set to None to disable.
    flipper_contact_coef: float | None = None
    flipper_contact_effort_limit: float = 1000.0  # Nm — normalizes torque signal to [0, 1]

    # Out-of-range ellipse slack (metres).
    # The ellipse is centred at the start→target midpoint.
    # semi_major_slack: extra reach beyond start/target along the path axis.
    # semi_minor_slack: lateral freedom beyond d_max/4 on either side.
    out_of_range_semi_major_slack: float = 2.0
    out_of_range_semi_minor_slack: float = 1.0

    # Shock-magnitude termination threshold (m/s²).
    # Episode is terminated (fail) when linear acceleration exceeds this value.
    # Set to None to disable.
    shock_fail_limit: float | None = 20.0

    # Raw per-robot acceleration logging (off by default).
    # When enabled, appends raw accel_mag values from healthy robots to a .npz file
    # every log_raw_accel_interval reward steps. Useful for offline histogram analysis.
    log_raw_accel: bool = False
    log_raw_accel_interval: int = 50   # flush every N steps; 0 = only on episode reset
    log_raw_accel_path: str | None = None  # set by trainer; None disables even if log_raw_accel=True


class CrossingEnv(FtrEnv):
    cfg: CrossingEnvCfg

    def __init__(self, cfg: CrossingEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.pitch_t = np.deg2rad(30)
        if self.cfg.terrain_name in ("cur_stairs_up", ):
            self.pitch_t = np.deg2rad(45)
        elif self.cfg.terrain_name in ("cur_mixed", ):
            # Reconstruct PhysxCfg preserving all values set by the training config.
            # Cap GPU heap/temp to safe values: cur_mixed with 256 envs needs ~256 MB
            # heap; 1 GB is ample. Configs that request 2 GB heap + 1 GB temp can exhaust
            # GPU memory before the scene is created, causing a hard crash.
            _px = self.cfg.sim.physx
            self.cfg.sim.physx = PhysxCfg(
                min_position_iteration_count=_px.min_position_iteration_count,
                max_velocity_iteration_count=_px.max_velocity_iteration_count,
                bounce_threshold_velocity=_px.bounce_threshold_velocity,
                gpu_heap_capacity=_px.gpu_heap_capacity,
                gpu_temp_buffer_capacity=_px.gpu_temp_buffer_capacity, 
                gpu_max_num_partitions=_px.gpu_max_num_partitions,
                gpu_max_rigid_contact_count=_px.gpu_max_rigid_contact_count,
                gpu_found_lost_pairs_capacity=_px.gpu_found_lost_pairs_capacity,
                gpu_found_lost_aggregate_pairs_capacity=_px.gpu_found_lost_aggregate_pairs_capacity,
                gpu_total_aggregate_pairs_capacity=_px.gpu_total_aggregate_pairs_capacity,
                gpu_collision_stack_size=_px.gpu_collision_stack_size,
            )
        elif self.cfg.terrain_name == "exp_stair33_up":
            self.pitch_t = np.deg2rad(40)
        super().__init__(cfg, render_mode, **kwargs)

        # reward stuff — must be after super().__init__() so self.scene/num_envs/device exist
        self.prev_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_lin_velocities = torch.zeros((self.num_envs, 3), device=self.device)
        self._success_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fail_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._explosion_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._timeout_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._raw_accel_buf: list = []
        self._raw_accel_steps: int = 0

    def _get_observations(self) -> VecEnvObs:
        height_map = self.calc_scanned_height_maps()
        hmap_shape = height_map.shape
        hmap_mean = einops.reduce(height_map, "n h w -> n", reduction="mean")
        hmap_mean = einops.repeat(hmap_mean, "n -> n h w", h=hmap_shape[1], w=hmap_shape[2])
        # hmap diagonal ≈ 2.483 m  sqrt((45×0.05)² + (21×0.05)²)
        hmap_diag = float((self.height_map_length[0]**2 + self.height_map_length[1]**2)**0.5)
        joint_limit = torch.deg2rad(torch.tensor(float(self.cfg.flipper_pos_max_deg))).item() if self.cfg.flipper_pos_max_deg is not None else None

        # goal vector in robot body frame, normalised by hmap diagonal
        goal_world = self.target_positions - self.positions          # [N,3] world frame
        yaw = self.orientations_3[:, 2]
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        goal_x = cos_y * goal_world[:, 0] + sin_y * goal_world[:, 1]
        goal_y = -sin_y * goal_world[:, 0] + cos_y * goal_world[:, 1]
        goal_body = torch.stack([goal_x, goal_y, goal_world[:, 2]], dim=-1) / hmap_diag  # [N,3]

        obs = torch.cat([
            (height_map - hmap_mean).view(self.num_envs, -1),                                               # 945  heightmap
            add_noise(self.orientations_3[:, :2] / np.pi, self.orientation_noise_std),                      # 2    roll, pitch  (÷π)
            add_noise(self.robot_lin_velocities, self.linear_vel_noise_std) / hmap_diag,                    # 3    lin vel      (÷diag)
            add_noise(self.robot_ang_velocities, self.angular_vel_noise_std) / np.pi,                       # 3    ang vel      (÷π)
            (add_noise(-self.flipper_positions if self.cfg.flipper_style else self.flipper_positions, self.flipper_pos_noise_std) + joint_limit) / (2*joint_limit) if joint_limit is not None else torch.zeros(self.num_envs, self.flipper_num, device=self.device),  # 4    joints [0,1] or 0 if locked
            goal_body,                                                                                       # 3    goal vector  (÷diag)
            self.last_action,                                                                                # 6    prev action  [v,w,fl×4]
        ], dim=-1)
        # total: 945 + 2 + 3 + 3 + 4 + 3 + 6 = 966

        # Detect robots with NaN/Inf in obs — flag them for termination next step.
        # (_get_observations runs after _get_dones, so termination is one step delayed.)
        bad_obs = torch.isnan(obs).any(dim=-1) | torch.isinf(obs).any(dim=-1)
        if bad_obs.any():
            _log.warning("NaN/Inf in observations for %d envs: %s", bad_obs.sum().item(), bad_obs.nonzero(as_tuple=False).squeeze(-1).tolist())
            self._obs_nan_mask |= bad_obs
        # Sanitize: NaN/Inf from physics-exploded robots must not enter the replay buffer,
        # as they corrupt gradients on the next optimisation step.
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            'policy': obs,
        }

    def _calculate_metrics_shutdown(self, i):
        N = 5

        if len(self.history_positions[i]) < N:
            return 0

        trajectory = torch.stack(list(self.history_positions[i])[-N:], dim=0)
        d_list = (trajectory[:, :2] - self.start_positions[i][:2]).norm(dim=1)

        forward_d = torch.diff(d_list)

        if torch.sum(forward_d[forward_d >= 0]) <= 1e-3:
            return -1

        return 0

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        terminal = self._explosion_mask | self._fail_mask
        healthy = ~terminal

        # ------------------------------------------------------------------
        # Helper: mean over healthy robots, or 0.0 if none are healthy.
        # ------------------------------------------------------------------
        def _hmean(t: torch.Tensor) -> float:
            return t[healthy].mean().item() if healthy.any() else 0.0

        def _hstats(t: torch.Tensor) -> tuple[float, float, float]:
            """Return (mean, max, min) over healthy robots."""
            h = t[healthy]
            if not healthy.any():
                return 0.0, 0.0, 0.0
            return h.mean().item(), h.max().item(), h.min().item()

        # ------------------------------------------------------------------
        # 1. Compute each reward component once — reuse for reward AND logging.
        # ------------------------------------------------------------------
        reward_info: dict[str, float] = {}

        # Potential-based shaping: coef * (gamma * phi(s') - phi(s)), phi = -dist
        curr_dist = (self.target_positions[:, :2] - self.positions[:, :2]).norm(dim=-1)
        prev_dist = (self.target_positions[:, :2] - self.prev_positions[:, :2]).norm(dim=-1)
        r_shaping = cfg.shaping_coef * (prev_dist - cfg.shaping_gamma * curr_dist)
        reward = r_shaping + cfg.step_penalty

        reward_info["rew/shaping"] = _hmean(r_shaping)
        reward_info["rew/step_penalty"] = cfg.step_penalty

        # Joint-velocity variance penalty
        if cfg.joint_vel_variance_coef is not None:
            r_jvv = cfg.joint_vel_variance_coef * self.actions[:, 2:].abs().var(dim=-1)
            reward -= r_jvv
            reward_info["rew/joint_vel_var_penalty"] = -_hmean(r_jvv)

        # Joints not horizontal penalty
        if cfg.joint_ang_from_flat_coef is not None:
            limit = torch.deg2rad(torch.tensor(float(cfg.flipper_pos_max_deg))).item()
            flipper_pos = torch.abs(self._robot.data.joint_pos[:, self._flipper_joint_ids])  # (N, 4)
            norm_dif = flipper_pos / limit  # [N, 4], in [0, 1]
            penalty = cfg.joint_ang_from_flat_coef * norm_dif.mean(dim=-1)  # [N]
            reward -= penalty
            reward_info["rew/joint_ang_from_flat_penalty"] = -_hmean(penalty)

        # Joint-angle variance penalty
        if cfg.joint_angle_variance_coef is not None:
            limit = torch.deg2rad(torch.tensor(float(cfg.flipper_pos_max_deg))).item()
            diffs = ((self.flipper_positions[:, 0] - self.flipper_positions[:, 1]).abs() / limit
                     + (self.flipper_positions[:, 2] - self.flipper_positions[:, 3]).abs() / limit) / 2
            r_jav = cfg.joint_angle_variance_coef * diffs
            reward -= r_jav
            reward_info["rew/joint_angle_var_penalty"] = -_hmean(r_jav)

        # Flipper-ground contact reward (torque-based)
        flipper_torques = self._robot.data.applied_torque[:, self._flipper_joint_ids].abs()  # (N, 4)
        if cfg.flipper_contact_coef is not None:
            contact_signal = flipper_torques / cfg.flipper_contact_effort_limit  # normalize to [0, 1]
            mean_signal = torch.clamp(contact_signal.mean(dim=-1)-0.5,min=0.0)
            min_signal = contact_signal.min(dim=-1).values
            alpha = 1
            r_contact = cfg.flipper_contact_coef * (alpha*mean_signal + (1-alpha)*min_signal)
            reward += r_contact
            reward_info["rew/flipper_contact"] = _hmean(r_contact)

        # Roll penalty
        if cfg.roll_coef is not None:
            roll_norm = 4 * (self.orientations_3[:, 0].abs() - np.deg2rad(15)) / torch.pi
            r_roll = cfg.roll_coef * torch.clamp(roll_norm, max=1, min=0)
            reward -= r_roll
            reward_info["rew/roll_penalty"] = -_hmean(r_roll)

        # Roll-rate penalty
        if cfg.roll_rate_coef is not None:
            r_roll_rate = cfg.roll_rate_coef * self.robot_ang_velocities[:, 0].abs() / np.pi
            reward -= r_roll_rate
            reward_info["rew/roll_rate_penalty"] = -_hmean(r_roll_rate)

        # Pitch penalty
        if cfg.pitch_coef is not None:
            pitch_norm = 4 * (self.orientations_3[:, 1].abs() - np.deg2rad(7.5)) / torch.pi
            r_pitch = cfg.pitch_coef * torch.clamp(pitch_norm, max=1, min=0)
            reward -= r_pitch
            reward_info["rew/pitch_penalty"] = -_hmean(r_pitch)

        # Pitch-rate penalty
        if cfg.pitch_rate_coef is not None:
            r_pitch_rate = cfg.pitch_rate_coef * self.robot_ang_velocities[:, 1].abs() / np.pi
            reward -= r_pitch_rate
            reward_info["rew/pitch_rate_penalty"] = -_hmean(r_pitch_rate)

        # Shock penalty
        dt = cfg.sim.dt * cfg.decimation
        accel_mag = (self.robot_lin_velocities - self.prev_lin_velocities).norm(dim=-1) / dt
        shock_norm = ((accel_mag - cfg.shock_threshold).clamp(min=0.0) / cfg.shock_scale).clamp(max=1.0)
        if cfg.shock_coef is not None:
            r_shock = cfg.shock_coef * shock_norm
            reward -= r_shock
            reward_info["rew/shock_penalty"] = -_hmean(r_shock)

        # Clearance penalty (compute ground height unconditionally for state logging)
        ground_height = self.current_frame_height_maps[
            :, self.height_map_size[0] // 2, self.height_map_size[1] // 2
        ]
        clearance = self.positions[:, 2] - self.track_wheel_radius - ground_height
        if cfg.clearance_coef is not None:
            r_clearance = cfg.clearance_coef * (1 / (1 + torch.exp(-((clearance - 0.2) / 0.02))))
            reward -= r_clearance
            reward_info["rew/clearance"] = -_hmean(r_clearance)

        # Action bonus
        if cfg.action_bonus_coef is not None:
            # v_norm = self.actions[:, 0
            v_norm = (torch.Tensor(self.actions[:, 0]).pow(3)*cfg.lin_action_ratio + torch.Tensor(self.robot_lin_velocities[:, 0] / self.track_vel_max).pow(3)*(1-cfg.lin_action_ratio))  # body-frame forward velocity
            
            r_action = cfg.action_bonus_coef * (
                torch.clamp(v_norm, max=1.0, min=-1.0)
            )
            reward += r_action
            reward_info["rew/action_bonus"] = _hmean(r_action)

        # Flipper actiopn bonus
        if cfg.flipper_action_bonus_coef is not None:
            flipper_norm = (self.actions[:, 2:]).abs().mean(dim=-1)
            r_f_action = cfg.flipper_action_bonus_coef * (
                torch.clamp(flipper_norm.pow(3), max=1.0)
            )
            reward += r_f_action
            reward_info["rew/flipper_action_bonus"] = _hmean(r_f_action)

        # Legacy flipper_training-style reward variants
        if cfg.legacy_joint_vel_variance_coef is not None:
            r = cfg.legacy_joint_vel_variance_coef * self.actions[:, 2:].abs().var(dim=-1)
            reward -= r
            reward_info["rew/legacy_joint_vel_var_penalty"] = -_hmean(r)

        if cfg.legacy_joint_angle_variance_coef is not None:
            r = cfg.legacy_joint_angle_variance_coef * self.flipper_positions.abs().var(dim=-1)
            reward -= r
            reward_info["rew/legacy_joint_angle_var_penalty"] = -_hmean(r)

        if cfg.legacy_track_vel_variance_coef is not None:
            r = cfg.legacy_track_vel_variance_coef * self.actions[:, :2].abs().var(dim=-1)
            reward -= r
            reward_info["rew/legacy_track_vel_var_penalty"] = -_hmean(r)

        if cfg.legacy_roll_rate_coef is not None:
            r = cfg.legacy_roll_rate_coef * self.robot_ang_velocities[:, 0].abs() / np.pi
            reward -= r
            reward_info["rew/legacy_roll_rate_penalty"] = -_hmean(r)

        if cfg.legacy_pitch_rate_coef is not None:
            r = cfg.legacy_pitch_rate_coef * self.robot_ang_velocities[:, 1].abs() / np.pi
            reward -= r
            reward_info["rew/legacy_pitch_rate_penalty"] = -_hmean(r)

        # ------------------------------------------------------------------
        # 2. Terminal masking & bonuses
        # ------------------------------------------------------------------
        reward[terminal] = 0.0
        reward[self._success_mask] += cfg.goal_reached_reward
        reward[self._fail_mask] += cfg.failed_reward
        if cfg.timeout_penalty:
            reward[self._timeout_mask] += cfg.timeout_penalty

        # terminal_bonus is the mean over ALL robots — diluted by batch size since most robots
        # are mid-episode. Use the separate rate/value logs below for interpretable monitoring.
        reward_info["rew/terminal_bonus"] = (
            self._success_mask.float() * cfg.goal_reached_reward
            + self._fail_mask.float() * cfg.failed_reward
            + self._timeout_mask.float() * (cfg.timeout_penalty or 0.0)
        ).mean().item()
        reward_info["rew/total_reward"] = reward.mean().item()

        # ------------------------------------------------------------------
        # 3. State monitoring (always logged, regardless of penalty enable).
        # ------------------------------------------------------------------

        # ── Shock group ───────────────────────────────────────────────────
        am_mean, am_max, am_min = _hstats(accel_mag)
        sn_mean, sn_max, sn_min = _hstats(shock_norm)
        reward_info["shock/accel_magnitude"] = am_mean
        reward_info["shock/accel_magnitude_max"] = am_max
        reward_info["shock/accel_magnitude_min"] = am_min
        reward_info["shock/shock_normalised"] = sn_mean
        reward_info["shock/shock_normalised_max"] = sn_max
        reward_info["shock/shock_normalised_min"] = sn_min

        if cfg.log_raw_accel and cfg.log_raw_accel_path is not None and healthy.any():
            self._raw_accel_buf.append(accel_mag[healthy].cpu().numpy())
            self._raw_accel_steps += 1
            if cfg.log_raw_accel_interval > 0 and self._raw_accel_steps % cfg.log_raw_accel_interval == 0:
                self._flush_raw_accel()

        # ── Torque group ──────────────────────────────────────────────────
        # Per-flipper torque and normalized contact signal
        flipper_names = ["FL", "FR", "RL", "RR"]
        if healthy.any():
            ft_h = flipper_torques[healthy]  # (H, 4)
            ft_mean = ft_h.mean(dim=0)
            ft_max = ft_h.max(dim=0).values
            ft_min = ft_h.min(dim=0).values
            for i, name in enumerate(flipper_names):
                reward_info[f"torque/flipper_torque_{name}"] = ft_mean[i].item()
                reward_info[f"torque/flipper_torque_{name}_max"] = ft_max[i].item()
                reward_info[f"torque/flipper_torque_{name}_min"] = ft_min[i].item()
            # Normalized contact signal — only if contact reward is configured
            if cfg.flipper_contact_coef is not None:
                cs_h = ft_h / cfg.flipper_contact_effort_limit
                cs_mean = cs_h.mean(dim=0)
                cs_max = cs_h.max(dim=0).values
                for i, name in enumerate(flipper_names):
                    reward_info[f"torque/flipper_contact_{name}"] = cs_mean[i].item()
                    reward_info[f"torque/flipper_contact_{name}_max"] = cs_max[i].item()

        # ── Clearance group ────────────────────────────────────────────────
        # (clearance already computed above in the reward section)
        cl_mean, cl_max, cl_min = _hstats(clearance)
        reward_info["clearance/height"] = cl_mean
        reward_info["clearance/height_max"] = cl_max
        reward_info["clearance/height_min"] = cl_min

        # ── Classic state group ────────────────────────────────────────────
        # Linear velocity magnitude
        lin_vel_mag = self.robot_lin_velocities.norm(dim=-1)
        lv_mean, lv_max, lv_min = _hstats(lin_vel_mag)
        reward_info["state/lin_velocity"] = lv_mean
        reward_info["state/lin_velocity_max"] = lv_max
        reward_info["state/lin_velocity_min"] = lv_min

        # Angular velocity magnitude
        ang_vel_mag = self.robot_ang_velocities.norm(dim=-1)
        av_mean, av_max, av_min = _hstats(ang_vel_mag)
        reward_info["state/ang_velocity"] = av_mean
        reward_info["state/ang_velocity_max"] = av_max
        reward_info["state/ang_velocity_min"] = av_min

        # Roll and pitch angles
        roll_mean, roll_max, roll_min = _hstats(self.orientations_3[:, 0])
        pitch_mean, pitch_max, pitch_min = _hstats(self.orientations_3[:, 1])
        reward_info["state/roll"] = roll_mean
        reward_info["state/roll_max"] = roll_max
        reward_info["state/roll_min"] = roll_min
        reward_info["state/pitch"] = pitch_mean
        reward_info["state/pitch_max"] = pitch_max
        reward_info["state/pitch_min"] = pitch_min

        # Linear velocity at success termination
        if self._success_mask.any():
            success_lin_vel = lin_vel_mag[self._success_mask].mean().item()
            reward_info["state/success_lin_velocity"] = success_lin_vel
        else:
            reward_info["state/success_lin_velocity"] = 0.0

        self.extras["reward_components"] = reward_info

        # Update previous state for next step
        self.prev_positions[:] = self.positions[:]
        self.prev_lin_velocities[:] = self.robot_lin_velocities[:]

        self.reward_buf[:] = reward
        return self.reward_buf

    def _flush_raw_accel(self) -> None:
        if not self._raw_accel_buf or self.cfg.log_raw_accel_path is None:
            return
        from pathlib import Path
        data = np.concatenate(self._raw_accel_buf)
        path = Path(self.cfg.log_raw_accel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = np.load(path)["accel"]
            data = np.concatenate([existing, data])
        np.savez_compressed(path, accel=data)
        self._raw_accel_buf.clear()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        super()._get_dones()

        # Explosion guard: detect physics-exploded robots via position/velocity thresholds
        # AND include robots already sanitized in _pre_physics_step (which resets them to
        # origin before PhysX runs, making them invisible to post-step checks).
        pos_norm = self.positions[:, :2].norm(dim=-1)
        lin_vel_norm_raw = self.robot_lin_velocities.norm(dim=-1)
        ang_vel_norm_raw = self.robot_ang_velocities.norm(dim=-1)
        post_pos_nan    = torch.isnan(pos_norm)
        post_pos_out    = pos_norm > self.explosion_pos_threshold
        post_lin_nan    = torch.isnan(lin_vel_norm_raw)
        post_lin_high   = lin_vel_norm_raw > 15.0
        post_ang_nan    = torch.isnan(ang_vel_norm_raw)
        post_ang_high   = ang_vel_norm_raw > 30.0
        post_orient_nan = torch.isnan(self.orientations_3).any(dim=-1)
        explosion_idx = (
            self._sanitized_mask
            | post_pos_nan | post_pos_out
            | post_lin_nan | post_lin_high
            | post_ang_nan | post_ang_high
            | post_orient_nan
        )

        # Target reached
        target_idx = (self.positions[:, :2] - self.target_positions[:, :2]).norm(dim=-1) <= 0.4
        # Rollover
        rollover_idx = torch.any(torch.abs(torch.rad2deg(self.orientations_3[:, :2])) >= 80, dim=-1)
        # Out of range
        out_range_idx = out_of_range(
            self.positions, self.start_positions, self.target_positions,
            self.cfg.out_of_range_semi_major_slack, self.cfg.out_of_range_semi_minor_slack,
        )
        # Timeout
        timeout_idx = self.episode_length_buf >= self.max_episode_length

        # Shock-magnitude termination
        dt = self.cfg.sim.dt * self.cfg.decimation
        accel_mag = (self.robot_lin_velocities - self.prev_lin_velocities).norm(dim=-1) / dt
        if self.cfg.shock_fail_limit is not None:
            shock_high_idx = accel_mag > self.cfg.shock_fail_limit
        else:
            shock_high_idx = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._success_mask = target_idx
        self._fail_mask = rollover_idx | out_range_idx | shock_high_idx
        self._explosion_mask = explosion_idx
        self._timeout_mask = timeout_idx

        self.extras["success"] = target_idx
        self.extras["failure"] = self._fail_mask
        self.extras["explosion"] = explosion_idx

        # Health stats — per-step scalars, logged to W&B.
        healthy = ~explosion_idx

        def _pct(mask: torch.Tensor) -> float:
            return mask.float().mean().item()

        # Explosion breakdown — always logged so high-explosion runs still surface causes.
        explosion_stats = {
            "explosion/pct_exploded":        _pct(explosion_idx),
            # Pre-physics causes (caught before PhysX step, robot teleported to origin)
            "explosion/pre_pos_nan":         _pct(self._explode_pre_pos_nan),
            "explosion/pre_lin_vel_nan":     _pct(self._explode_pre_lin_vel_nan),
            "explosion/pre_pos_out":         _pct(self._explode_pre_pos_out),
            "explosion/pre_lin_vel_high":    _pct(self._explode_pre_lin_vel_high),
            "explosion/pre_ang_vel_high":    _pct(self._explode_pre_ang_vel_high),
            "explosion/pre_obs_nan":         _pct(self._explode_pre_obs_nan),
            # Post-physics causes (detected after PhysX step)
            "explosion/post_pos_nan":        _pct(post_pos_nan),
            "explosion/post_pos_out":        _pct(post_pos_out),
            "explosion/post_lin_vel_nan":    _pct(post_lin_nan),
            "explosion/post_lin_vel_high":   _pct(post_lin_high),
            "explosion/post_ang_vel_nan":    _pct(post_ang_nan),
            "explosion/post_ang_vel_high":   _pct(post_ang_high),
            "explosion/post_orient_nan":     _pct(post_orient_nan),
            # Fail causes
            "failure/rollover":              _pct(rollover_idx),
            "failure/out_of_range":          _pct(out_range_idx),
            "failure/shock_high":            _pct(shock_high_idx),
        }
        self.extras["state_stats"] = explosion_stats

        if healthy.any():
            h_lin = lin_vel_norm_raw[healthy]
            h_ang = ang_vel_norm_raw[healthy]
            h_roll = self.orientations_3[healthy, 0].abs()
            h_pitch = self.orientations_3[healthy, 1].abs()
            h_dist = (self.target_positions[healthy, :2] - self.positions[healthy, :2]).norm(dim=-1)
            ground_h = self.current_frame_height_maps[
                healthy, self.height_map_size[0] // 2, self.height_map_size[1] // 2
            ]
            h_clearance = (self.positions[healthy, 2] - self.track_wheel_radius) - ground_h
            self.extras["state_stats"].update({
                "state/lin_vel_mean":    h_lin.mean().item(),
                "state/lin_vel_max":     h_lin.max().item(),
                "state/lin_vel_min":     h_lin.min().item(),
                "state/ang_vel_mean":    h_ang.mean().item(),
                "state/ang_vel_max":     h_ang.max().item(),
                "state/ang_vel_min":     h_ang.min().item(),
                "state/roll_deg_mean":   torch.rad2deg(h_roll).mean().item(),
                "state/roll_deg_max":    torch.rad2deg(h_roll).max().item(),
                "state/roll_deg_min":    torch.rad2deg(h_roll).min().item(),
                "state/pitch_deg_mean":  torch.rad2deg(h_pitch).mean().item(),
                "state/pitch_deg_max":   torch.rad2deg(h_pitch).max().item(),
                "state/pitch_deg_min":   torch.rad2deg(h_pitch).min().item(),
                "state/dist_to_goal":    h_dist.mean().item(),
                "state/clearance_mean":  h_clearance.mean().item(),
                "state/clearance_min":   h_clearance.min().item(),
                "state/clearance_max":   h_clearance.max().item(),
                "state/pct_healthy":     healthy.float().mean().item(),
            })

        self.reset_terminated += target_idx + rollover_idx + out_range_idx + explosion_idx
        self.reset_time_outs += timeout_idx
        return self.reset_terminated[:], self.reset_time_outs[:]

