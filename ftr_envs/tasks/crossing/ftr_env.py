# -*- coding: utf-8 -*-
"""
====================================
@File Name ：ftr_env.py
@Time ： 2024/9/29 PM12:11
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

# Modules whose rewards read real per-flipper PhysX contact forces
# (rl_modules/ctrac/ctrac_contact.py). MARV-only — see the ContactSensor block in
# _setup_scene for why.
_CONTACT_SENSOR_MODULES = ("ctrac", "atd3qn", "icmd3qn")


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

    # MARV only: per-direction flipper angle limits (degrees). When all four are set,
    # replaces the symmetric ±flipper_pos_max_deg clamp with asymmetric per-pair limits.
    # Convention (matches flipper_training / MARV URDF):
    #   front up   = negative angle  → front_up_deg is the magnitude of the negative bound
    #   front down = positive angle  → front_down_deg is the positive bound
    #   rear  up   = positive angle  → back_up_deg is the positive bound
    #   rear  down = negative angle  → back_down_deg is the magnitude of the negative bound
    # None on any field disables the asymmetric mode and falls back to flipper_pos_max_deg.
    marv_flipper_front_up_deg: float | None = None
    marv_flipper_front_down_deg: float | None = None
    marv_flipper_back_up_deg: float | None = None
    marv_flipper_back_down_deg: float | None = None
    track_vel_max: float = 0.7     # max |v| (m/s)
    track_vel_scale: float = 1.0   # multiplicative scale applied to track velocities before sending to robot
    track_ang_vel_max: float = 1.0   # max |w| (rad/s)
    fixed_forward_vel: float | None = None  # if set, overrides policy linear velocity with this constant (m/s)

    # ctrac only: attaches a real ContactSensor for ground-truth per-flipper contact
    # (ctrac_contact.py). Default True (needed for training's reward/C-VAE targets). Set
    # False via env_cfg_overrides to skip it — e.g. for local eval/visualization on a
    # machine where the sensor's own event-driven initialization races the main
    # simulation's physics-view creation and loses (confirmed: reproduces even at
    # num_envs=1 on one specific local GPU/driver combination while the identical code
    # runs fine on the training cluster — an IsaacLab-internal timing issue, not
    # something fixable from here). Safe to disable at eval time: the actor's action only
    # depends on the C-VAE's own *estimated* contact from proprioceptive history, not
    # ground truth — ctrac_contact.py already degrades gracefully with no sensor
    # (contact_prob always 0, points fall back to the lowest wheel), which only skews the
    # logged reward's stabilization term and training targets, not the policy's behavior.
    ctrac_contact_sensor_enabled: bool = True

    # MarvWheelArticulation normally groups wheel joints into true left/right side
    # (af6aa92 fixed the equivalent FTR LF/LR/RL/RR mislabeling bug). Policies trained on
    # FTR *before* that fix learned to steer under the old (buggy) front/rear wheel-group
    # split rather than true left/right — set this to True when evaluating such an old
    # checkpoint on robot_type "marv" so it gets the controls it was actually trained with.
    # No effect on robot_type "ftr" (FtrWheelArticulation always uses the fixed grouping).
    legacy_ftr_turning: bool = False

    # Experiment: scales the *collision-only* radius of wheel1 on every MARV flipper
    # (the kinematic radius used for velocity-target scaling is unaffected), to test
    # whether wheel1's larger contact geometry (vs FTR's smaller equivalent) makes it
    # snag terrain mesh features more often. 1.0 = unchanged. No effect on robot_type "ftr".
    wheel1_collision_radius_scale: float = 1.0

    # MARV only: disable the flipper arm body's own collision geometry (3 boxes + 2 end
    # cylinders imported from the URDF). The arm body (mass≈0) has collision shapes that
    # exactly overlap the driven child-link wheel cylinders (wheel1 at the pivot, wheel5 at
    # the far end), creating two contact bodies at the same position simultaneously —
    # a known source of contact instability in PhysX. When True, only the 5 driven wheel
    # cylinders contact terrain. No effect on robot_type "ftr".
    disable_flipper_arm_collision: bool = True

    # Settable via env_cfg_overrides in YAML; default matches robot_config below. 2-DOF
    # front/rear flipper pairs (9-action discrete space) instead of 4 independent flippers.
    sync_flipper_control: bool = False

    # How the policy's flipper action (per-flipper value in [-1, 1]) is interpreted:
    #   "velocity" (default) — the action is a flipper angular velocity command,
    #       integrated onto the current angle every step (scaled by flipper_dt deg/step)
    #       and clamped to the flipper angle limits. This is the historical behaviour.
    #   "increment" — the action selects a fixed-size angle step per control step in the
    #       commanded direction (sign of the action), added to the current angle and
    #       clamped to the limits. The step magnitude is flipper_angle_increment_deg.
    #       Matches AT-D3QN's discrete ±π/12 CW/hold/CCW design (Pan et al. 2023).
    #   "position" — the action is an absolute target, linearly mapped onto the flipper
    #       angle limits (action -1 → lower/up bound, +1 → upper/down bound). No
    #       integration/decimation; the commanded angle is applied directly. Works with
    #       both the symmetric ±flipper_pos_max_deg limits and the asymmetric per-pair
    #       MARV limits (marv_flipper_*_deg).
    flipper_control_mode: str = "velocity"

    # Per-step flipper angle step (degrees) used only when flipper_control_mode == "increment".
    # Default 15.0° = π/12, the increment AT-D3QN (Pan et al. 2023) was designed around.
    flipper_angle_increment_deg: float = 15.0

    # Friction — settable via env_cfg_overrides in YAML; defaults match robot_config below
    # so old configs that don't set these fields continue to work unchanged.
    flipper_material_friction: float = 5.0    # rigid flipper arm (steel chassis)
    wheel_material_friction: float = 10.0    # rubber tracks; effective = wheel × terrain_dynamic
    body_material_friction: float = 0.1       # MARV only: base_link/hull, slick by default — see robot_config below
    terrain_static_friction: float = 0.9      # global scene physics_material default
    terrain_dynamic_friction: float = 0.7     # global scene physics_material default

    # Settable via env_cfg_overrides in YAML; defaults match robot_config/robot_render_config/
    # noise below. These previously had no top-level field/bridge (same class of bug
    # sync_flipper_control had) — a config setting them via env_cfg_overrides would have
    # silently done nothing, since env_cfg_overrides only setattr's top-level attributes.
    chassis_wheel_render_mass: float = 2.98
    flipper_wheel_render_mass: float = 1.0
    only_render_front_flipper: bool = False
    drive_wheel_radius: float = 0.1165
    auxiliary_wheel_radius: float = 0.0780
    render_radius: float = 0.1165
    hmap_noise_std: float = 0.03
    hmap_patch_noise_std: float = 0.07
    flipper_drive_noise_std: float = 0.01
    baselink_drive_noise_std: float = 0.01
    flipper_pos_noise_std: float = 0.01
    angular_vel_noise_std: float = 0.2
    linear_vel_noise_std: float = 0.1
    orientation_noise_std: float = 0.01

    robot_config = {
        "sync_flipper_control": False,

        "flipper_material_friction": 5,
        "wheel_material_friction": 10,
        "body_material_friction": 0.1,

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
        "hmap_noise_std": 0.03,
        "hmap_patch_noise_std": 0.07,
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
        self.cfg.robot_config["sync_flipper_control"] = self.cfg.sync_flipper_control
        self.cfg.robot_config["flipper_material_friction"] = self.cfg.flipper_material_friction
        self.cfg.robot_config["wheel_material_friction"] = self.cfg.wheel_material_friction
        self.cfg.robot_config["body_material_friction"] = self.cfg.body_material_friction
        self.cfg.robot_config["flipper_pos_max"] = self.cfg.flipper_pos_max_deg
        self.cfg.robot_config["legacy_ftr_turning"] = self.cfg.legacy_ftr_turning
        self.cfg.robot_config["wheel1_collision_radius_scale"] = self.cfg.wheel1_collision_radius_scale
        self.cfg.robot_config["disable_flipper_arm_collision"] = self.cfg.disable_flipper_arm_collision
        self.cfg.robot_config["chassis_wheel_render_mass"] = self.cfg.chassis_wheel_render_mass
        self.cfg.robot_config["flipper_wheel_render_mass"] = self.cfg.flipper_wheel_render_mass
        self.cfg.sim.physics_material.static_friction = self.cfg.terrain_static_friction
        self.cfg.sim.physics_material.dynamic_friction = self.cfg.terrain_dynamic_friction

        # Instance copies (including nested sub-dicts) — don't mutate the class defaults.
        self.cfg.robot_render_config = {
            "flipper": dict(self.cfg.robot_render_config["flipper"]),
            "track": dict(self.cfg.robot_render_config["track"]),
        }
        self.cfg.robot_render_config["flipper"]["only_render_front_flipper"] = self.cfg.only_render_front_flipper
        self.cfg.robot_render_config["flipper"]["drive_wheel_radius"] = self.cfg.drive_wheel_radius
        self.cfg.robot_render_config["flipper"]["auxiliary_wheel_radius"] = self.cfg.auxiliary_wheel_radius
        self.cfg.robot_render_config["track"]["render_radius"] = self.cfg.render_radius

        self.cfg.noise = dict(self.cfg.noise)
        self.cfg.noise["hmap_noise_std"] = self.cfg.hmap_noise_std
        self.cfg.noise["hmap_patch_noise_std"] = self.cfg.hmap_patch_noise_std
        self.cfg.noise["flipper_drive_noise_std"] = self.cfg.flipper_drive_noise_std
        self.cfg.noise["baselink_drive_noise_std"] = self.cfg.baselink_drive_noise_std
        self.cfg.noise["flipper_pos_noise_std"] = self.cfg.flipper_pos_noise_std
        self.cfg.noise["angular_vel_noise_std"] = self.cfg.angular_vel_noise_std
        self.cfg.noise["linear_vel_noise_std"] = self.cfg.linear_vel_noise_std
        self.cfg.noise["orientation_noise_std"] = self.cfg.orientation_noise_std

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
        self.hmap_patch_noise_std = self.cfg.noise["hmap_patch_noise_std"]
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
                # _pre_physics_step stores the policy's raw action in last_action, but
                # this override discards it, so a
                # module that feeds last_action back as an observation (marv_rl_module.py's
                # 6-D "prev action [v, w, fl x 4]" tail — the only consumer of the v/w slots;
                # train_creps.py reads last_action[:, 2:] only) would be told the robot
                # executed a velocity it never executed. Write what was actually applied.
                if self.last_action.shape[1] > 0:
                    self.last_action[:, 0] = self.cfg.fixed_forward_vel
            else:
                track_v = self.actions[:, 0:1].clamp(-self.track_vel_max, self.track_vel_max)
            track_w = self.actions[:, 1:2].clamp(-self.track_ang_vel_max, self.track_ang_vel_max)
            real_track_cmd = add_noise(torch.cat([track_v, track_w], dim=-1), std=self.baselink_drive_noise_std)
            self._robot.set_v_w(real_track_cmd)

        real_flipper_cmd = add_noise(
            self._calc_comp_flipper_pos(self.flipper_target_pos),
            std=self.flipper_pos_noise_std
        )
        marv_asym_active = (
            self.cfg.robot_type == "marv"
            and all(v is not None for v in (
                self.cfg.marv_flipper_front_up_deg,
                self.cfg.marv_flipper_front_down_deg,
                self.cfg.marv_flipper_back_up_deg,
                self.cfg.marv_flipper_back_down_deg,
            ))
        )
        self._robot.set_all_flipper_position_targets(
            real_flipper_cmd,
            clip_value=None if marv_asym_active else np.deg2rad(self.cfg.robot_config["flipper_pos_max"]),
        )

    def _setup_scene(self):
        if self.cfg.robot_type == "marv":
            import copy

            from ftr_envs.assets.marv import MARV_CFG
            from ftr_envs.assets.articulation.marv import MarvWheelArticulation
            robot_cfg = copy.deepcopy(MARV_CFG)
            RobotClass = MarvWheelArticulation
            # train_ftr.py applies CLI/YAML physics overrides (robot_max_linear_velocity,
            # solver_*_iterations, max_depenetration_velocity, damping, ...) directly onto
            # self.cfg.robot (FTR_CFG) before env construction. MARV_CFG is a separate
            # ArticulationCfg instance those overrides never touch, so copy the
            # already-overridden values across here rather than silently running MARV
            # with marv.py's hardcoded defaults.
            src_rigid = self.cfg.robot.spawn.rigid_props
            src_artic = self.cfg.robot.spawn.articulation_props
            dst_rigid = robot_cfg.spawn.rigid_props
            dst_artic = robot_cfg.spawn.articulation_props
            dst_rigid.max_linear_velocity = src_rigid.max_linear_velocity
            dst_rigid.max_angular_velocity = src_rigid.max_angular_velocity
            dst_rigid.max_depenetration_velocity = src_rigid.max_depenetration_velocity
            dst_rigid.linear_damping = src_rigid.linear_damping
            dst_rigid.angular_damping = src_rigid.angular_damping
            dst_artic.solver_position_iteration_count = src_artic.solver_position_iteration_count
            dst_artic.solver_velocity_iteration_count = src_artic.solver_velocity_iteration_count
        else:
            robot_cfg = self.cfg.robot
            RobotClass = FtrWheelArticulation

        if getattr(self.cfg, "module_name", None) == "ctrac" and self.cfg.robot_type != "marv":
            # Hard error for ctrac only: contact points are load-bearing there (C-VAE
            # regression target + privileged critic input). atd3qn/icmd3qn merely prefer
            # real contacts for R_contact and fall back to the joint-torque proxy.
            raise NotImplementedError(
                "ctrac module requires robot_type: marv (its ContactSensorCfg wiring "
                "targets MarvWheelArticulation's wheel-body prim layout only)."
            )

        # PhysX only reports contacts for bodies that have the contact-reporter API
        # attached, which IsaacLab only adds at spawn time when this flag is set. Without
        # it, ContactSensor._initialize_impl() raises "could not find any bodies with
        # contact reporter API" even though the prims/body names match the sensor's regex
        # fine (confirmed against a real cluster run — the sensor's own find_bodies()
        # failure further below was a downstream symptom of this, not a separate bug:
        # body_physx_view stays None because _initialize_impl() never completed). Lives
        # directly on the spawn cfg (RigidObjectSpawnerCfg, e.g. UsdFileCfg) — a SIBLING of
        # rigid_props/articulation_props, not nested inside either. Must be set before
        # RobotClass(...) actually spawns the prims.
        _wants_contacts = (
            getattr(self.cfg, "module_name", None) in _CONTACT_SENSOR_MODULES
            and self.cfg.robot_type == "marv"
            and self.cfg.ctrac_contact_sensor_enabled
        )
        if _wants_contacts:
            robot_cfg.spawn.activate_contact_sensors = True

        self._robot = RobotClass(robot_cfg, device=self.device)
        self._robot.set_robot_env(self.cfg.robot_config, self.cfg.robot_render_config)
        self._robot.load_all_wheel_radius()
        self.scene.articulations["robot"] = self._robot

        # C-TRAC (rl_modules/ctrac) needs real per-wheel PhysX contact forces for its
        # ground-truth contact extraction (ctrac_contact.py) — IsaacLab sensors must be
        # declared in the scene before scene creation, so this can't be done from inside
        # CTRACModule itself (which is only constructed after _setup_scene has already
        # run — see CrossingEnv.__init__). Only wired for robot_type == "marv" (every
        # module this project adds targets marv; FTR's wheel-body prim layout differs —
        # compare MarvWheelArticulation.set_robot_env's sibling-of-container wheel bodies
        # vs FtrWheelArticulation's nested flipper_list/<side>_wheel/<code>{i} layout).
        # robot_type is already validated above (before activate_contact_sensors was set).
        # atd3qn/icmd3qn use the same extractor for their R_contact term, which on MARV asks
        # "is each flipper pair on the ground" (see rl_modules/pan_shared.py) — a question
        # only real contact forces answer well; they degrade to the joint-torque proxy if the
        # sensor is missing.
        if _wants_contacts:
            from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg

            wheel_names_regex = "(" + "|".join(
                f"{side}_flipper_wheel[1-5]" for side in ("front_left", "front_right", "rear_left", "rear_right")
            ) + ")"
            contact_sensor_cfg = ContactSensorCfg(
                prim_path=f"{robot_cfg.prim_path}/{wheel_names_regex}",
                update_period=0.0,
                track_pose=False,
                track_air_time=False,
                force_threshold=1.0,
                # No filter_prim_paths_expr: net_forces_w already reports each wheel body's
                # total contact force, which in practice is terrain contact (wheels don't
                # meaningfully contact each other) — filtering specifically to the terrain
                # prim would need its exact path, which isn't uniformly known here across
                # every terrain asset this project supports. Documented simplification, see
                # ctrac_contact.py's module docstring.
            )
            try:
                self._ctrac_contact_sensor = ContactSensor(contact_sensor_cfg)
                self.scene.sensors["ctrac_contact"] = self._ctrac_contact_sensor
            except Exception as e:
                # Defensive: the sensor's own event-driven _initialize_impl() can fail
                # asynchronously (races the main simulation's physics-view creation on
                # some local GPU/driver combinations — see ctrac_contact_sensor_enabled's
                # docstring) rather than raising synchronously here, so this except may
                # not even be what catches it in practice. Either way, ctrac_contact.py
                # already handles env._ctrac_contact_sensor being absent gracefully.
                _log.warning(f"ctrac ContactSensor construction failed ({e}); continuing without it "
                             "— contact_prob will read 0, points fall back to the lowest wheel.")
                self._ctrac_contact_sensor = None

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
        low, high = self.flipper_angle_bounds()
        if low is not None:
            flipper_offset = 4 if self.cfg.flipper_style else 2
            flipper_cmd = self.actions[:, flipper_offset:]

            if self.cfg.flipper_control_mode == "position":
                # Positional control: the policy outputs an absolute flipper target in
                # [-1, 1], linearly mapped onto [low, high] (action -1 → low, +1 → high).
                # No integration/decimation — the commanded angle is applied directly.
                unit = (flipper_cmd.clamp(-1.0, 1.0) + 1.0) * 0.5
                self.flipper_target_pos = low + unit * (high - low)
            else:
                # Velocity and increment control both integrate a per-step delta onto the
                # current flipper angle, then clamp to [low, high]. flipper_style uses native
                # convention (front+=down, rear+=up), inverted vs FTR's user convention → negate.
                flipper_sign = -1 if self.cfg.flipper_style else 1
                if self.cfg.flipper_control_mode == "increment":
                    # Fixed-size angle step per control step in the commanded direction
                    # (sign of the action, so magnitude is ignored — a discrete CW/hold/CCW
                    # command). Matches AT-D3QN's ±π/12 flipper increments.
                    delta = flipper_sign * torch.sign(flipper_cmd) * np.deg2rad(self.cfg.flipper_angle_increment_deg)
                    next_pos = delta + self.flipper_positions
                else:
                    # Velocity control (default): the action is an angular velocity command,
                    # scaled by flipper_dt (deg/step).
                    flipper_delta = flipper_sign * flipper_cmd * self.flipper_dt
                    next_pos = torch.deg2rad(flipper_delta) + self.flipper_positions
                self.flipper_target_pos = torch.clamp(next_pos, low, high)
        else:
            self.flipper_target_pos.zero_()

    def flipper_angle_bounds(self):
        """Per-flipper angle limits (radians) in flipper_target_pos convention, as a
        ``(low, high)`` pair of ``(flipper_num,)`` tensors — or ``(None, None)`` when the
        flippers are locked horizontal.

        The same bounds clamp the integrated angle (velocity/increment mode) and define the
        [-1, 1] → [low, high] target map (position mode). Exposed as a method because the
        reward/observation modules need it too: normalising a flipper angle by the symmetric
        ``flipper_pos_max_deg`` is wrong whenever the asymmetric MARV limits are active
        (front -90/+80, rear -80/+90 in the D3QN configs), which made the normalised feature
        neither full-range nor centred on the flat pose. See rl_modules/pan_shared.py.
        """
        if self.cfg.flipper_pos_max_deg is None and self.cfg.marv_flipper_front_up_deg is None:
            return None, None

        marv_limits = (
            self.cfg.marv_flipper_front_up_deg,
            self.cfg.marv_flipper_front_down_deg,
            self.cfg.marv_flipper_back_up_deg,
            self.cfg.marv_flipper_back_down_deg,
        )
        dtype = self.flipper_target_pos.dtype
        if self.cfg.robot_type == "marv" and all(v is not None for v in marv_limits):
            # Asymmetric per-pair limits for MARV: up = negative angle, down = positive
            # angle for front; up = positive angle, down = negative angle for rear.
            # The target's width/ordering depends on sync_flipper_control/only_front_flipper
            # (see get_flipper_pos()) — mirror that same branching here so low/high always
            # match the target's actual per-flipper layout.
            front_up, front_down, back_up, back_down = (np.deg2rad(v) for v in marv_limits)
            if self.sync_flipper_control and self.only_front_flipper:
                low_vals, high_vals = [-front_up], [front_down]
            elif self.sync_flipper_control and not self.only_front_flipper:
                low_vals, high_vals = [-front_up, -back_down], [front_down, back_up]
            elif not self.sync_flipper_control and self.only_front_flipper:
                low_vals, high_vals = [-front_up, -front_up], [front_down, front_down]
            else:
                low_vals  = [-front_up,  -front_up,  -back_down, -back_down]
                high_vals = [ front_down,  front_down,  back_up,    back_up]
            low  = torch.tensor(low_vals,  device=self.device, dtype=dtype)
            high = torch.tensor(high_vals, device=self.device, dtype=dtype)
        else:
            # Symmetric limits: ±flipper_pos_max_deg for every flipper.
            limit = np.deg2rad(self.cfg.flipper_pos_max_deg)
            low  = torch.full((self.flipper_num,), -limit, device=self.device, dtype=dtype)
            high = torch.full((self.flipper_num,),  limit, device=self.device, dtype=dtype)
        return low, high

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

        # normalize data format
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
            # Lateral stays as get_obs produced it: col 0 = −y, col 20 = +y. Under REP-103
            # (x forward, y left, z up) that means **col 0 is the robot's RIGHT** and col 20
            # its LEFT — this comment previously said the opposite, and the deployment side
            # has to mirror the lateral axis because of it (grid_map puts col 0 at +y; see
            # marv_rl_training/training/ftr_heightmap_window.py). Verified by probing
            # MapHelper.get_obs with an asymmetric map, not inferred from this comment.
            self.current_frame_height_maps[i, :, :] = local_map.flip(0)

    @cached_property
    def max_episode_length(self):
        return int(self.cfg.episode_length_s / (self.physics_dt * self.cfg.decimation))

    @property
    def current_time(self):
        return self.world.current_time
