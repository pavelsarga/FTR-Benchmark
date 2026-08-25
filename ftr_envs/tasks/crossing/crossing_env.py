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

import numpy as np
import torch
import torch.nn as nn

from omni.isaac.lab.envs import VecEnvObs
from omni.isaac.lab.sim import PhysxCfg

from rl_modules.rl_module import RLModule
from rl_modules.registry import RLMODULE_REGISTRY

from .ftr_env import FtrEnv, FtrEnvCfg, configclass

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

    # Selects the RLModule implementation from RLMODULE_REGISTRY (see rl_modules/registry.py).
    module_name: str = "marv_rl"

    # Per-key overrides merged over the active module's own YAML (rl_modules/<name>/
    # <name>_module.yaml) by RLModule.load_module_cfg. Those YAMLs hold each reproduction's
    # paper constants and its adaptation switches, and they live next to the module rather
    # than in configs/ — so without this a training config could not run, say, the
    # paper-faithful reward semantics and the MARV-adapted ones as two experiments without
    # editing a file inside the module. Unknown keys are rejected rather than ignored.
    module_cfg_overrides = {}

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

    # Rollover termination threshold (degrees). Episode fails when |roll| or |pitch|
    # reaches this angle.
    rollover_threshold_deg: float = 80.0

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
        elif self.cfg.terrain_name in ("cur_mixed", "custom_mixed"):
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
        self.rl_module: RLModule = RLMODULE_REGISTRY[self.cfg.module_name](self)

    def _get_observations(self) -> VecEnvObs:
        return self.rl_module.get_observations()

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
        # Helper: mean / (mean, max, min) over healthy robots, or 0.0 if none are healthy.
        # ------------------------------------------------------------------
        def _hmean(t: torch.Tensor) -> float:
            return t[healthy].mean().item() if healthy.any() else 0.0

        def _hstats(t: torch.Tensor) -> tuple[float, float, float]:
            h = t[healthy]
            if not healthy.any():
                return 0.0, 0.0, 0.0
            return h.mean().item(), h.max().item(), h.min().item()

        # ------------------------------------------------------------------
        # Universal per-step derived signals — shared across all RLModules, computed
        # once here so modules read them (env.shock_norm/env.clearance/env.flipper_torques)
        # instead of re-deriving them.
        # ------------------------------------------------------------------
        dt = cfg.sim.dt * cfg.decimation
        self.accel_mag = (self.robot_lin_velocities - self.prev_lin_velocities).norm(dim=-1) / dt
        self.shock_norm = ((self.accel_mag - cfg.shock_threshold).clamp(min=0.0) / cfg.shock_scale).clamp(max=1.0)
        ground_height = self.current_frame_height_maps[
            :, self.height_map_size[0] // 2, self.height_map_size[1] // 2
        ]
        self.clearance = self.positions[:, 2] - self.track_wheel_radius - ground_height
        self.flipper_torques = self._robot.data.applied_torque[:, self._flipper_joint_ids].abs()  # (N, 4)

        # ------------------------------------------------------------------
        # 1. Sum the module's individual reward components. Step penalty and terminal
        # masking/bonuses are the module's own responsibility, so "step_penalty" and
        # "terminal_bonus" already arrive as entries in `components`.
        # ------------------------------------------------------------------
        components = self.rl_module.get_reward_components()
        reward = torch.zeros(self.num_envs, device=self.device)
        reward_info: dict[str, float] = {}
        for name, comp in components.items():
            reward = reward + comp
            if name == "terminal_bonus":
                # Mean over ALL robots — diluted by batch size since most robots are
                # mid-episode. Use the separate rate/value logs below for interpretable
                # monitoring.
                reward_info[f"rew/{name}"] = comp.mean().item()
            else:
                reward_info[f"rew/{name}"] = _hmean(comp)

        reward_info["rew/total_reward"] = reward.mean().item()

        # Module-supplied diagnostics (optional). The Pan reproductions (atd3qn/icmd3qn)
        # publish their candidate flipper angles, angle deltas and per-end contact flags
        # here — those quantities encode every frame/sign assumption in
        # rl_modules/pan_shared.py, so a short debug run makes each of them directly
        # checkable instead of only observable through the success curve.
        for name, value in getattr(self.rl_module, "pan_diagnostics", {}).items():
            reward_info[f"pan/{name}"] = _hmean(value)

        # ------------------------------------------------------------------
        # 2. State monitoring (always logged, regardless of which components are enabled).
        # ------------------------------------------------------------------

        # ── Shock group ───────────────────────────────────────────────────
        am_mean, am_max, am_min = _hstats(self.accel_mag)
        sn_mean, sn_max, sn_min = _hstats(self.shock_norm)
        reward_info["shock/accel_magnitude"] = am_mean
        reward_info["shock/accel_magnitude_max"] = am_max
        reward_info["shock/accel_magnitude_min"] = am_min
        reward_info["shock/shock_normalised"] = sn_mean
        reward_info["shock/shock_normalised_max"] = sn_max
        reward_info["shock/shock_normalised_min"] = sn_min

        if cfg.log_raw_accel and cfg.log_raw_accel_path is not None and healthy.any():
            self._raw_accel_buf.append(self.accel_mag[healthy].cpu().numpy())
            self._raw_accel_steps += 1
            if cfg.log_raw_accel_interval > 0 and self._raw_accel_steps % cfg.log_raw_accel_interval == 0:
                self._flush_raw_accel()

        # ── Torque group ──────────────────────────────────────────────────
        # Per-flipper torque and normalized contact signal
        flipper_names = ["FL", "FR", "RL", "RR"]
        if healthy.any():
            ft_h = self.flipper_torques[healthy]  # (H, 4)
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
        cl_mean, cl_max, cl_min = _hstats(self.clearance)
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

        # ── Heading group ─────────────────────────────────────────────────
        # Duplicated from the extras["state_stats"] block further down on purpose: that
        # dict is only surfaced by the TRAINERS (pop_state_stats), while eval_ftr.py prints
        # reward_info. Anything needed to debug steering has to live here or it is invisible
        # in exactly the runs used to test steering.
        # Body frame throughout (robot_ang_velocities is root_ang_vel_b), so [:, 2] is the
        # yaw rate directly; only the goal bearing needs de-rotating.
        _gw = self.target_positions[:, :2] - self.positions[:, :2]
        _yaw = self.orientations_3[:, 2]
        _cy, _sy = torch.cos(_yaw), torch.sin(_yaw)
        _herr = torch.atan2(-_sy * _gw[:, 0] + _cy * _gw[:, 1],
                            _cy * _gw[:, 0] + _sy * _gw[:, 1])
        he_mean, he_max, he_min = _hstats(torch.rad2deg(_herr))
        reward_info["state/heading_err_deg"] = he_mean
        reward_info["state/heading_err_deg_max"] = he_max
        reward_info["state/heading_err_deg_min"] = he_min
        yr_mean, yr_max, yr_min = _hstats(self.robot_ang_velocities[:, 2])
        reward_info["state/yaw_rate"] = yr_mean
        reward_info["state/yaw_rate_max"] = yr_max
        reward_info["state/yaw_rate_min"] = yr_min

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
            reward_info["state/success_lin_velocity"] = lin_vel_mag[self._success_mask].mean().item()
        else:
            reward_info["state/success_lin_velocity"] = 0.0

        self.extras["reward_components"] = reward_info

        # Update previous state for next step
        self.prev_positions[:] = self.positions[:]
        self.prev_lin_velocities[:] = self.robot_lin_velocities[:]

        self.reward_buf[:] = reward
        return self.reward_buf
    
    def calc_scanned_height_maps(self, base_robot_frame=True):
        return self.rl_module.calc_scanned_height_maps(base_robot_frame)

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
        rollover_idx = torch.any(
            torch.abs(torch.rad2deg(self.orientations_3[:, :2])) >= self.cfg.rollover_threshold_deg, dim=-1
        )
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

            # --- Directional breakdown -------------------------------------------------
            # ang_vel_mean above is the 3-D norm, which conflates pitch rate (large and
            # expected while climbing) with yaw rate (uncommanded heading drift, and the
            # thing that actually costs goal progress). Split them, and measure heading
            # error and approach speed directly rather than leaving them to be back-computed
            # from a reward term. Both velocity buffers are BODY frame (ftr_env.py sets them
            # from root_lin_vel_b / root_ang_vel_b), so ang[:, 2] is the yaw rate and
            # lin[:, 1] is lateral slip, with no rotation needed. Body-frame w_z is only an
            # approximation of the world yaw rate once the robot is pitched, which is fine
            # at the few-degree mean pitch seen here and conservative at larger ones.
            h_ang_xyz = self.robot_ang_velocities[healthy]     # [roll_rate, pitch_rate, yaw_rate]
            h_lin_xyz = self.robot_lin_velocities[healthy]     # [forward, lateral, vertical]

            # Goal direction de-rotated into the body frame, same yaw-only convention as
            # ctrac_module.py's goal_xy / goal_velocity and ctrac_contact.py's contact points.
            g_w = self.target_positions[healthy, :2] - self.positions[healthy, :2]
            h_yaw = self.orientations_3[healthy, 2]
            cos_y, sin_y = torch.cos(h_yaw), torch.sin(h_yaw)
            g_bx = cos_y * g_w[:, 0] + sin_y * g_w[:, 1]
            g_by = -sin_y * g_w[:, 0] + cos_y * g_w[:, 1]
            # 0 = nose pointing straight at the goal; sign follows the body y axis (left +).
            heading_err = torch.atan2(g_by, g_bx)
            g_b = torch.stack([g_bx, g_by], dim=-1)
            dir_to_goal = g_b / g_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            v_toward = (h_lin_xyz[:, :2] * dir_to_goal).sum(dim=-1)   # m/s, signed

            # Fraction of the speed the robot actually has that is net approach. Only defined
            # for robots that are moving — including stationary ones would divide ~0 by ~0 and
            # swamp the mean with noise, so they are excluded rather than counted as 0.
            moving = h_lin > 0.02
            approach_frac = (v_toward[moving] / h_lin[moving]).mean().item() if moving.any() else 0.0
            self.extras["state_stats"].update({
                "state/lin_vel_mean":    h_lin.mean().item(),
                "state/lin_vel_max":     h_lin.max().item(),
                "state/lin_vel_min":     h_lin.min().item(),
                "state/ang_vel_mean":    h_ang.mean().item(),
                "state/ang_vel_max":     h_ang.max().item(),
                "state/ang_vel_min":     h_ang.min().item(),
                "state/roll_rate_mean":  h_ang_xyz[:, 0].abs().mean().item(),
                "state/pitch_rate_mean": h_ang_xyz[:, 1].abs().mean().item(),
                "state/yaw_rate_mean":   h_ang_xyz[:, 2].abs().mean().item(),
                "state/yaw_rate_max":    h_ang_xyz[:, 2].abs().max().item(),
                "state/vel_fwd_mean":    h_lin_xyz[:, 0].mean().item(),
                "state/vel_lat_abs_mean": h_lin_xyz[:, 1].abs().mean().item(),
                "state/v_toward_mean":   v_toward.mean().item(),
                "state/approach_frac":   approach_frac,
                "state/heading_err_deg_mean": torch.rad2deg(heading_err.abs()).mean().item(),
                "state/heading_err_deg_max":  torch.rad2deg(heading_err.abs()).max().item(),
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

