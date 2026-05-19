# -*- coding: utf-8 -*-
"""
====================================
@File Name ：ftr_env.py
@Time ： 2024/9/29 下午12:11
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import logging
import os
from functools import cached_property
from itertools import cycle
from typing import Any, Sequence
from collections import deque

import carb
import einops
import numpy as np
import omni.isaac.lab.sim as sim_utils
import torch
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.world import World
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, VecEnvObs, VecEnvStepReturn
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from ftr_envs.assets.articulation.ftr import FtrWheelArticulation
from ftr_envs.assets.ftr import FTR_CFG, FTR_SIM_CFG
from ftr_envs.assets.terrain.terrain import Terrain
from ftr_envs.utils.torch import add_noise, rand_range

_log = logging.getLogger(__name__)


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, torch.Tensor):
        return data.numpy()

    return np.array(data)


def to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)

    if isinstance(data, torch.Tensor):
        return data

    return torch.tensor(data)


@configclass
class FtrEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5
    episode_length_s = 30
    action_scale = 100.0
    num_actions = 1
    num_observations = 955
    num_states = 0

    # simulation
    sim = FTR_SIM_CFG

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=0.0, replicate_physics=True)
    terrain_name = "cur_mixed"

    # robot
    robot: ArticulationCfg = FTR_CFG
    robot_type: str = "ftr"  # "ftr" or "marv"
    initial_flipper_range = (0, 0)
    spawn_yaw_range: float = 0.0              # ± yaw perturbation at spawn (degrees)
    flipper_pos_max_deg: float | None = 90.0  # None = lock flippers horizontal (pos=0)
    track_vel_max: float = 0.7     # max |v| (m/s)
    track_vel_scale: float = 1.0   # multiplicative scale applied to track velocities before sending to robot
    track_ang_vel_max: float = 1.0   # max |w| (rad/s)
    fixed_forward_vel: float | None = None  # if set, overrides policy linear velocity with this constant (m/s)

    # Friction — settable via env_cfg_overrides in YAML; defaults match robot_config below
    # so old configs that don't set these fields continue to work unchanged.
    flipper_material_friction: float = 5.0    # rigid flipper arm (steel chassis)
    wheel_material_friction: float = 10.0    # rubber tracks; effective = wheel × terrain_dynamic
    terrain_static_friction: float = 0.9      # global scene physics_material default
    terrain_dynamic_friction: float = 0.7     # global scene physics_material default

    robot_config = {
        "sync_flipper_control": False,

        "flipper_material_friction": 5,
        "wheel_material_friction": 10,

        "chassis_wheel_render_mass": 2.98,
        "flipper_wheel_render_mass": 1,
        "flipper_pos_max": flipper_pos_max_deg,
    }
    robot_render_config = {
        "flipper": {
            "only_render_front_flipper": False,
            "drive_wheel_radius": 0.1165,
            "auxiliary_wheel_radius": 0.0780,
        },
        "track": {
            "render_radius": 0.1165,
        }
    }
    noise = {
        "hmap_noise_std": 0.1,
        "flipper_drive_noise_std": 0.01,
        "baselink_drive_noise_std": 0.01,
        "flipper_pos_noise_std": 0.01,
        "angular_vel_noise_std": 0.2,
        "linear_vel_noise_std": 0.1,
        "orientation_noise_std": 0.01,
    }


class FtrEnv(DirectRLEnv):
    cfg: FtrEnvCfg

    def __init__(self, cfg: FtrEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        # Apply top-level friction fields into the nested robot_config dict and physics_material
        # so that env_cfg_overrides in YAML can control friction without touching robot_config.
        self.cfg.robot_config = dict(self.cfg.robot_config)  # instance copy — don't mutate class default
        self.cfg.robot_config["flipper_material_friction"] = self.cfg.flipper_material_friction
        self.cfg.robot_config["wheel_material_friction"] = self.cfg.wheel_material_friction
        self.cfg.sim.physics_material.static_friction = self.cfg.terrain_static_friction
        self.cfg.sim.physics_material.dynamic_friction = self.cfg.terrain_dynamic_friction
        self.terrain_cfg = Terrain(cfg.terrain_name)

        self.sync_flipper_control = self.cfg.robot_config["sync_flipper_control"]
        self.only_front_flipper = self.cfg.robot_render_config["flipper"]["only_render_front_flipper"]
        self.flipper_num = 4
        if self.sync_flipper_control:
            self.flipper_num = int(self.flipper_num / 2)
        if self.only_front_flipper:
            self.flipper_num = int(self.flipper_num / 2)
        if self.cfg.flipper_style:
            # Native mode: 4 independent track vels + 4 flipper angles (matches flipper_training engine)
            self.cfg.num_actions = 2 * self.flipper_num
            # 4096 hmap + 15 state (orient/vel/joints/goal) + 2*flipper_num prev_action
            self.cfg.num_observations = 4096 + 15 + 2 * self.flipper_num
        else:
            self.cfg.num_actions = self.flipper_num + 2   # +2 for track v, w
            self.cfg.num_observations += (-4 + self.flipper_num)
        self.track_wheel_radius = self.cfg.robot_render_config["track"]["render_radius"]

        super().__init__(cfg, render_mode, **kwargs)
        self.world = World.instance()

        # Flipper joint indices — used for torque-based contact detection.
        self._flipper_joint_ids, _ = self._robot.find_joints(self._robot.flipper_joint_names)

        self.hmap_noise_std = self.cfg.noise["hmap_noise_std"]
        self.flipper_drive_noise_std = self.cfg.noise["flipper_drive_noise_std"]
        self.baselink_drive_noise_std = self.cfg.noise["baselink_drive_noise_std"]
        self.orientation_noise_std = self.cfg.noise["orientation_noise_std"]
        self.flipper_pos_noise_std = self.cfg.noise["flipper_pos_noise_std"]
        self.angular_vel_noise_std = self.cfg.noise["angular_vel_noise_std"]
        self.linear_vel_noise_std = self.cfg.noise["linear_vel_noise_std"]
        self.track_vel_max = self.cfg.track_vel_max
        self.track_vel_scale = self.cfg.track_vel_scale
        self.track_ang_vel_max = self.cfg.track_ang_vel_max

        self.flipper_dt = 5

        self.extractor = torch.nn.AvgPool2d(3)

        self.initial_flipper_range = self.cfg.initial_flipper_range
        self.spawn_yaw_range = np.deg2rad(self.cfg.spawn_yaw_range)

        # Explosion threshold: half-diagonal of the terrain bounding box.
        # A robot outside the terrain is always exploded; using the terrain extents avoids
        # hardcoding a fixed distance that may not match the actual map size.
        lower = self.terrain_cfg.map.lower[:2]
        upper = self.terrain_cfg.map.upper[:2]
        self.explosion_pos_threshold = float(((upper - lower) ** 2).sum() ** 0.5)
        self._sanitized_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._obs_nan_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Per-cause breakdown masks stored by _pre_physics_step, consumed by _get_dones for logging.
        _z = lambda: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._explode_pre_pos_nan     = _z()
        self._explode_pre_lin_vel_nan = _z()
        self._explode_pre_pos_out     = _z()
        self._explode_pre_lin_vel_high = _z()
        self._explode_pre_ang_vel_high = _z()
        self._explode_pre_obs_nan     = _z()
        self.flipper_target_pos = torch.zeros(self.num_envs, self.flipper_num, device=self.device)
        self._prepare_reset_info()

        self.start_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.start_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.flipper_positions = torch.zeros((self.num_envs, self.flipper_num), device=self.device)
        self.orientations = torch.zeros((self.num_envs, 4), device=self.device)
        self.orientations_3 = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_lin_velocities = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_ang_velocities = torch.zeros((self.num_envs, 3), device=self.device)
        N = 5
        self.history_positions = [deque(maxlen=N) for _ in range(self.num_envs)]

        if self.cfg.flipper_style:
            # 3.25 m square → get_obs returns 65×65; center-cropped to 64×64 in calc_scanned_height_maps
            self.height_map_length = (3.25, 3.25)
            self.height_map_size = (65, 65)
        else:
            self.height_map_length = (2.25, 1.05)
            self.height_map_size = (45, 21)
        self.current_frame_height_maps = torch.zeros((self.num_envs, *self.height_map_size), device=self.device)

        last_action_dim = 2 * self.flipper_num if self.cfg.flipper_style else self.flipper_num + 2
        self.last_action = torch.zeros(self.num_envs, last_action_dim, device=self.device)
        self.forward_vel_commands = torch.zeros(self.num_envs, device=self.device)

    def _apply_action(self):
        if self.cfg.flipper_style:
            # Native track control: actions = [v_FL, v_FR, v_BL, v_BR, a_FL, a_FR, a_BL, a_BR]
            # Left tracks: indices 0 (FL), 2 (BL); right tracks: indices 1 (FR), 3 (BR)
            v_left  = ((self.actions[:, 0] + self.actions[:, 2]) / 2).clamp(-self.track_vel_max, self.track_vel_max) * self.track_vel_scale
            v_right = ((self.actions[:, 1] + self.actions[:, 3]) / 2).clamp(-self.track_vel_max, self.track_vel_max) * self.track_vel_scale
            noise = add_noise(torch.stack([v_right, v_left], dim=-1), std=self.baselink_drive_noise_std)
            v_right_n, v_left_n = noise[:, 0], noise[:, 1]
            # Right joints (RL*/RR*): positive SDF velocity = forward.
            # Left joints (LF*/LR*): joint axes are mirrored → negate to go forward.
            v_lr = torch.stack([v_right_n, -v_left_n], dim=-1)
            self._robot.set_right_and_left_velocities(v_lr)
        else:
            if self.cfg.fixed_forward_vel is not None:
                track_v = torch.full((self.num_envs, 1), self.cfg.fixed_forward_vel, device=self.device)
            else:
                track_v = self.actions[:, 0:1].clamp(-self.track_vel_max, self.track_vel_max)
            track_w = self.actions[:, 1:2].clamp(-self.track_ang_vel_max, self.track_ang_vel_max)
            real_track_cmd = add_noise(torch.cat([track_v, track_w], dim=-1), std=self.baselink_drive_noise_std)
            self._robot.set_v_w(real_track_cmd)

        real_flipper_cmd = add_noise(
            self._calc_comp_flipper_pos(self.flipper_target_pos),
            std=self.flipper_pos_noise_std
        )
        self._robot.set_all_flipper_position_targets(
            real_flipper_cmd,
            clip_value=np.deg2rad(self.cfg.robot_config["flipper_pos_max"])
        )

    def _setup_scene(self):
        if self.cfg.robot_type == "marv":
            from ftr_envs.assets.marv import MARV_CFG
            from ftr_envs.assets.articulation.marv import MarvWheelArticulation
            robot_cfg = MARV_CFG
            RobotClass = MarvWheelArticulation
        else:
            robot_cfg = self.cfg.robot
            RobotClass = FtrWheelArticulation
        self._robot = RobotClass(robot_cfg, device=self.device)
        self._robot.set_robot_env(self.cfg.robot_config, self.cfg.robot_render_config)
        self._robot.load_all_wheel_radius()
        self.scene.articulations["robot"] = self._robot

        stage = self.scene.stage
        self.terrain_cfg.apply(stage)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.terrain_cfg.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        # Zero root state (pose + velocities) for this batch.
        self._robot.write_root_state_to_sim(torch.zeros(n, 13, device=self.device), env_ids=env_ids)
        # Also zero all joint positions and velocities to flush PhysX internal impulse caches.
        # Without this, an exploded articulation re-produces NaN on the very next substep even
        # after a root-state reset, because PhysX's GPU solver retains cached constraint forces.
        num_joints = self._robot.num_joints
        self._robot.write_joint_state_to_sim(
            torch.zeros(n, num_joints, device=self.device),
            torch.zeros(n, num_joints, device=self.device),
            env_ids=env_ids,
        )

        self.last_action[env_ids] = 0.0

        reset_infos = [self._reset_info_generate() for _ in range(n)]
        poses = torch.stack([i["pose"] for i in reset_infos]).to(self.device)
        if self.spawn_yaw_range > 0.0:
            yaw = rand_range([-self.spawn_yaw_range, self.spawn_yaw_range], (n,), device=self.device)
            # Isaac Lab quat convention: [w, x, y, z] at pose indices [3, 4, 5, 6]
            # yaw-only quaternion around world Z: [cos(θ/2), 0, 0, sin(θ/2)]
            qw = torch.cos(yaw / 2)
            qz = torch.sin(yaw / 2)
            # pre-multiply: q_new = q_yaw * q_base  (applies yaw in world frame)
            bw, bx, by, bz = poses[:, 3].clone(), poses[:, 4].clone(), poses[:, 5].clone(), poses[:, 6].clone()
            poses[:, 3] = qw * bw - qz * bz  # w
            poses[:, 4] = qw * bx - qz * by  # x
            poses[:, 5] = qw * by + qz * bx  # y
            poses[:, 6] = qw * bz + qz * bw  # z
            poses[:, 3:7] = torch.nn.functional.normalize(poses[:, 3:7], dim=-1)
        self._robot.write_root_pose_to_sim(poses, env_ids=env_ids)
        self.flipper_positions[env_ids, :] = torch.deg2rad(rand_range(
            self.initial_flipper_range,
            (n, self.flipper_num),
            device=self.device
        ))
        self._robot.set_all_flipper_positions(self._calc_comp_flipper_pos(self.flipper_positions[env_ids, :]), indices=env_ids)
        self.forward_vel_commands[env_ids] = 0.5
        self.start_positions[env_ids] = torch.stack([i["start_point"] for i in reset_infos]).to(self.device)
        self.orientations[env_ids] = torch.stack([i["start_orient"] for i in reset_infos]).to(self.device)
        self.target_positions[env_ids] = torch.stack([i["target_point"] for i in reset_infos]).to(self.device)

        # clear history data
        for i in env_ids:
            self.history_positions[i].clear()

        # update position and velocity history
        self.prev_positions[env_ids] = self.start_positions[env_ids]
        if hasattr(self, "prev_lin_velocities"):
            self.prev_lin_velocities[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions[:] = actions
        self.last_action[:] = actions  # store raw action before noise 

        # Detect envs whose position OR velocity is NaN or has exploded.
        # These must be teleported to the origin *before* the physics step so PhysX
        # never tries to simulate an articulation at extreme states, which causes
        # illegal GPU memory accesses (GpuArticulationView) that crash the entire process.
        # Velocity checks catch robots *about to* explode — before positions go extreme.
        pos_norm = torch.nan_to_num(self.positions.norm(dim=-1))
        lin_vel_norm = torch.nan_to_num(self.robot_lin_velocities.norm(dim=-1))
        ang_vel_norm = torch.nan_to_num(self.robot_ang_velocities.norm(dim=-1))
        self._explode_pre_pos_nan      = torch.isnan(self.positions).any(dim=-1)
        self._explode_pre_lin_vel_nan  = torch.isnan(self.robot_lin_velocities).any(dim=-1)
        self._explode_pre_pos_out      = pos_norm > self.explosion_pos_threshold
        self._explode_pre_lin_vel_high = lin_vel_norm > 15.0
        self._explode_pre_ang_vel_high = ang_vel_norm > 30.0
        self._explode_pre_obs_nan      = self._obs_nan_mask.clone()
        bad = (
            self._explode_pre_pos_nan
            | self._explode_pre_lin_vel_nan
            | self._explode_pre_pos_out
            | self._explode_pre_lin_vel_high
            | self._explode_pre_ang_vel_high
        )
        # Also carry forward any NaN observations detected last step
        bad = bad | self._obs_nan_mask
        self._obs_nan_mask.zero_()
        self._sanitized_mask = bad
        if bad.any():
            self.actions[bad] = 0.0
            bad_ids = bad.nonzero(as_tuple=False).squeeze(-1)
            n = len(bad_ids)
            _log.warning(
                "Sanitized %d/%d envs (pos_max=%.1f lin_vel_max=%.1f ang_vel_max=%.1f): env_ids=%s",
                n, self.num_envs,
                pos_norm[bad_ids].max().item(),
                lin_vel_norm[bad_ids].max().item(),
                ang_vel_norm[bad_ids].max().item(),
                bad_ids.tolist(),
            )
            # root_state: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
            # Use identity quaternion (qw=1) and z=0.5 so the robot is above the
            # ground plane and PhysX does not generate huge contact forces during
            # the decimation substeps before _get_dones can issue a proper reset.
            safe_state = torch.zeros(n, 13, device=self.device)
            safe_state[:, 2] = 0.6  # z: 0.6 m above terrain origin
            safe_state[:, 6] = 1.0  # qw = 1 (identity rotation)
            # Isaac Lab's root_state_w may be an inference tensor (set by PhysX readback
            # inside torch.inference_mode()).  TorchRL's rollout runs under @torch.no_grad()
            # which is NOT inference_mode, so inplace writes on inference tensors are blocked.
            # Wrapping in inference_mode() re-enables the inplace-write permission.
            with torch.inference_mode():
                self._robot.write_root_state_to_sim(safe_state, env_ids=bad_ids)
                num_joints = self._robot.num_joints
                self._robot.write_joint_state_to_sim(
                    torch.zeros(n, num_joints, device=self.device),
                    torch.zeros(n, num_joints, device=self.device),
                    env_ids=bad_ids,
                )
        if self.cfg.flipper_pos_max_deg is not None:
            flipper_offset = 4 if self.cfg.flipper_style else 2
            # flipper_style uses native convention (front+=down, rear+=up) which is
            # inverted relative to FTR's user convention → negate to align.
            flipper_sign = -1 if self.cfg.flipper_style else 1
            flipper_delta = flipper_sign * self.actions[:, flipper_offset:] * self.flipper_dt
            limit = np.deg2rad(self.cfg.flipper_pos_max_deg)
            self.flipper_target_pos = torch.clip(
                torch.deg2rad(flipper_delta) + self.flipper_positions,
                -limit,
                limit,
            )
        else:
            self.flipper_target_pos.zero_()

    def _post_physics_step(self):
        self.positions[:] = self._robot.data.root_pos_w
        self.orientations[:] = self._robot.data.root_quat_w
        self.robot_lin_velocities[:] = self._robot.data.root_lin_vel_b
        self.robot_ang_velocities[:] = self._robot.data.root_ang_vel_b
        self.orientations_3[:] = torch.stack(
            list(torch.from_numpy(quat_to_euler_angles(i)).to(self.device) for i in self.orientations.cpu())
        )
        self.flipper_positions[:] = self.get_flipper_pos()
        self.calc_current_frame_height_maps()

        # update history data
        for i in range(self.num_envs):
            self.history_positions[i].append(self.positions[i].clone())

    def get_flipper_pos(self):
        flipper_pos = self._robot.get_all_flipper_positions()
        if self.sync_flipper_control and self.only_front_flipper:
            flipper_pos = flipper_pos[:, [0]]
        elif self.sync_flipper_control and not self.only_front_flipper:
            flipper_pos = flipper_pos[:, [0, 2]]
        elif not self.sync_flipper_control and self.only_front_flipper:
            flipper_pos = flipper_pos[:, [0, 1]]

        return flipper_pos

    def _calc_comp_flipper_pos(self, flipper_pos):
        if self.sync_flipper_control and self.only_front_flipper:
            comp_flipper_pos = torch.cat([
                torch.repeat_interleave(flipper_pos, 2, dim=-1),
                torch.ones(self.num_envs, 2, device=flipper_pos.device) * np.deg2rad(120)
            ], dim=-1)
        elif self.sync_flipper_control and not self.only_front_flipper:
            comp_flipper_pos = torch.repeat_interleave(flipper_pos, 2, dim=-1)
        elif not self.sync_flipper_control and self.only_front_flipper:
            comp_flipper_pos = torch.cat([
                flipper_pos,
                torch.ones(self.num_envs, 2, device=flipper_pos.device) * np.deg2rad(120)
            ], dim=-1)
        else:
            comp_flipper_pos = flipper_pos
        return comp_flipper_pos

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._post_physics_step()
        self.reset_terminated = torch.zeros_like(self.reset_terminated)
        self.reset_time_outs = torch.zeros_like(self.reset_time_outs)
        self.reward_buf = torch.zeros(self.num_envs, device=self.device)

        # subclass imp
        ...

        return self.reset_terminated[:], self.reset_time_outs[:]

    def _prepare_reset_info(self):
        self._reset_info = self.terrain_cfg.birth

        # 对数据进行格式统一化
        for info in self._reset_info:
            if len(info["start_orient"]) == 3:
                info["start_orient"] = euler_angles_to_quat(to_numpy(info["start_orient"]))

            for key, value in info.items():
                info[key] = to_tensor(value).float()

            info['pose'] = torch.cat([info['start_point'], info['start_orient']])
        _data = cycle(self._reset_info)
        self._reset_info_generate = lambda: next(_data)

    def calc_current_frame_height_maps(self):
        lower = self.terrain_cfg.map.lower
        upper = self.terrain_cfg.map.upper
        for i in range(self.num_envs):
            pos = self.positions[i].cpu()
            if not (lower[0] < pos[0] < upper[0]) or not (lower[1] < pos[1] < upper[1]):
                carb.log_error(f"The position of the robot seems to be abnormal. {pos=}")
                continue

            angle = torch.rad2deg(self.orientations_3[i]).cpu().numpy()[2]
            local_map = self.terrain_cfg.map.get_obs(pos, angle, self.height_map_length)
            if local_map is None:
                continue

            if local_map.shape != self.height_map_size:
                carb.log_error("Your map doesn't seem big enough.")
                continue

            local_map = torch.from_numpy(local_map).to(self.device).clone()
            # Flip row axis so row 0 = front (+x), row 44 = rear (−x).
            # Lateral stays: col 0 = left (−y), col 20 = right (+y).
            self.current_frame_height_maps[i, :, :] = local_map.flip(0)

    def calc_scanned_height_maps(self, base_robot_frame=True):
        h, w = self.height_map_size
        shaped_map = torch.reshape(self.current_frame_height_maps, (-1, 1, h, w))
        if base_robot_frame:
            shaped_map = shaped_map - einops.repeat(
                self.positions[:, 2] - self.track_wheel_radius, 'n -> n c rh rw', c=1, rh=h, rw=w
            )
        height_maps = shaped_map.squeeze(1)
        if self.cfg.flipper_style:
            # center-crop 65×65 → 64×64: drop last row and last col
            height_maps = height_maps[:, :64, :64]
        return add_noise(height_maps, std=self.hmap_noise_std)

    @cached_property
    def max_episode_length(self):
        return int(self.cfg.episode_length_s / (self.physics_dt * self.cfg.decimation))

    @property
    def current_time(self):
        return self.world.current_time
