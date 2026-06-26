# -*- coding: utf-8 -*-
"""
====================================
@File Name ：articulation.py
@Time ： 2024/9/29 下午3:47
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import os
from functools import cached_property

import numpy as np
import torch
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.assets.articulation.articulation import Articulation

from ftr_envs.utils.prim import (
    find_matching_prim_paths,
    get_prim_at_path,
    get_prim_radius,
    set_joint_damping,
    set_joint_max_vel,
    set_joint_stiffness,
    set_material_color,
    set_material_friction,
    set_prim_invisible,
    set_render_radius,
)


L = 0.3600

class FtrWheelArticulation(Articulation):
    # Pivot-to-wheel5 distance (m), wheel1 sits at the pivot (distance 0) for every
    # flipper on both FTR and MARV — see load_wheel_pivot_distances(). Measured directly
    # from ftr_v1.usd (front_left_flipper pivot -> FL5 world position).
    _AXLE_DISTANCE = 0.34

    def __init__(self, cfg: ArticulationCfg, device):
        super().__init__(cfg)
        self._device = device

        self.flipper_joint_names = self.cfg.actuators["flipper_joint"].joint_names_expr
        # self.baselink_wheel_joint_names = self.cfg.actuators["baselink_wheel"].joint_names_expr
        self.flipper_wheel_joint_names = self.cfg.actuators["flipper_wheel"].joint_names_expr

    def get_all_flipper_positions(self, indices=None, degree=False):
        positions = self.data.joint_pos[:, self.flipper_dof_idx_list]
        a = torch.tensor([-1, -1, 1, 1], device=positions.device)
        return torch.rad2deg(positions * a) if degree else positions * a

    def set_all_flipper_positions(self, positions: torch.Tensor, indices=None, degree=False, clip_value=None):
        a = torch.tensor([-1, -1, 1, 1], device=positions.device)
        pos = torch.deg2rad(positions * a) if degree else positions * a
        if clip_value is not None:
            pos = torch.clamp(pos, -clip_value, clip_value)
        self.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=indices, joint_ids=self.flipper_dof_idx_list)

    def set_all_flipper_position_targets(self, positions: torch.Tensor, indices=None, degree=False, clip_value=None):
        if clip_value is not None:
            positions = torch.clamp(positions, -clip_value, clip_value)
        a = torch.tensor([-1, -1, 1, 1], device=positions.device)
        pos = torch.deg2rad(positions * a) if degree else positions * a
        pos = torch.clip(pos, -2 * torch.pi, 2 * torch.pi)
        self.set_joint_position_target(pos, env_ids=indices, joint_ids=self.flipper_dof_idx_list)

    def set_v_w(self, v_w: torch.Tensor, indices=None):
        v = v_w[:, 0]
        w = v_w[:, 1]

        vels = torch.zeros(v_w.shape, device=v_w.device)
        vels[:, 0] = (2 * v + w * L) / 2
        vels[:, 1] = (2 * v - w * L) / 2

        return self.set_right_and_left_velocities(vels, indices=indices)

    def _flipper_rotation_correction(self, wheel_flipper_dof, indices=None):
        """Forward (chassis-x) ground-contact velocity each wheel picks up purely from its
        own flipper arm actively rotating about its pivot — independent of the wheel's own
        spin or the chassis's v/w.

        Every wheel is rigidly mounted to its flipper arm at a fixed distance r from the
        pivot, not on a free-floating belt, so when the arm rotates with angular velocity
        theta_dot, the wheel's center sweeps a small arc and picks up a real velocity
        component of r * sin(theta) * theta_dot along the chassis x-axis (theta = the raw,
        unsigned joint angle — zero at theta=0 since the wheel only moves vertically when
        the arm is flat, growing as the arm tilts). set_right_and_left_velocities below only
        ever solved for the chassis-commanded surface speed, so an actively-rotating flipper
        was forcing wheel-ground slip equal to this term. Subtracting it from the commanded
        surface speed before converting to a wheel spin target removes that slip.

        NOTE: the sign here assumes the same world-frame rotation handedness verified for
        the wheel spin joints (uniformly +Y across all four flippers) carries over to the
        flipper pivot joints. If this correction makes things worse rather than better,
        flip the sign.
        """
        dof_idx = torch.as_tensor(wheel_flipper_dof, device=self.device, dtype=torch.long)
        joint_pos = self.data.joint_pos if indices is None else self.data.joint_pos[indices]
        joint_vel = self.data.joint_vel if indices is None else self.data.joint_vel[indices]
        theta = joint_pos[:, dof_idx]
        theta_dot = joint_vel[:, dof_idx]
        return self.wheel_pivot_distance * torch.sin(theta) * theta_dot

    def set_right_and_left_velocities(self, vels, indices=None):
        set_joint_func = self.set_joint_velocity_target
        # set_joint_func(
        #     -vels[:, 0].unsqueeze(dim=-1).repeat(1, len(self.r_indices)) / self.baselink_radius,
        #     env_ids=indices,
        #     joint_ids=self.r_indices,
        # )
        # set_joint_func(
        #     -vels[:, 1].unsqueeze(dim=-1).repeat(1, len(self.l_indices)) / self.baselink_radius,
        #     env_ids=indices,
        #     joint_ids=self.l_indices,
        # )
        fr_correction = self._flipper_rotation_correction(self.fr_wheel_flipper_dof, indices=indices)
        fl_correction = self._flipper_rotation_correction(self.fl_wheel_flipper_dof, indices=indices)
        set_joint_func(
            (vels[:, 0].unsqueeze(dim=-1).repeat(1, len(self.fr_indices)) - fr_correction) / self.flipper_radius,
            env_ids=indices,
            joint_ids=self.fr_indices,
        )
        set_joint_func(
            (vels[:, 1].unsqueeze(dim=-1).repeat(1, len(self.fl_indices)) - fl_correction) / self.flipper_radius,
            env_ids=indices,
            joint_ids=self.fl_indices,
        )

    def find_idx(self):
        self.flipper_dof_idx_list = [self.find_joints(i)[0][0] for i in self.flipper_joint_names]

        # self.r_indices = [self.find_joints(i)[0][0] for i in self.baselink_wheel_joint_names if i.startswith("R")]
        # self.l_indices = [self.find_joints(i)[0][0] for i in self.baselink_wheel_joint_names if i.startswith("L")]

        # Joint-name prefixes don't match left/right side despite appearances: verified against
        # each wheel joint's body0 parent link transform in ftr_v1.usd — LR* is physically
        # attached to front_right_flipper_link and RL* to rear_left_flipper_link (front-right and
        # rear-left are swapped relative to what the "L"/"R" first letter implies). True mapping:
        # LF* = front-left, LR* = front-right, RL* = rear-left, RR* = rear-right.
        self.fl_front_indices = [self.find_joints(i)[0][0] for i in self.flipper_wheel_joint_names if i.startswith("LF")]
        self.fr_front_indices = [self.find_joints(i)[0][0] for i in self.flipper_wheel_joint_names if i.startswith("LR")]
        self.fl_rear_indices  = [self.find_joints(i)[0][0] for i in self.flipper_wheel_joint_names if i.startswith("RL")]
        self.fr_rear_indices  = [self.find_joints(i)[0][0] for i in self.flipper_wheel_joint_names if i.startswith("RR")]

        self.fl_indices = self.fl_front_indices + self.fl_rear_indices
        self.fr_indices = self.fr_front_indices + self.fr_rear_indices

        # Per-wheel parent-flipper DOF index (into flipper_dof_idx_list, ordered
        # [front_left, front_right, rear_left, rear_right]), aligned with fl_indices/
        # fr_indices — used by _flipper_rotation_correction to read each wheel's own
        # flipper's raw joint angle/velocity.
        front_left_dof, front_right_dof, rear_left_dof, rear_right_dof = self.flipper_dof_idx_list
        self.fl_wheel_flipper_dof = (
            [front_left_dof] * len(self.fl_front_indices) + [rear_left_dof] * len(self.fl_rear_indices)
        )
        self.fr_wheel_flipper_dof = (
            [front_right_dof] * len(self.fr_front_indices) + [rear_right_dof] * len(self.fr_rear_indices)
        )

    def load_all_wheel_radius(self):
        prim_path = self.robot_prim_path
        # self.baselink_radius = torch.tensor(
        #     [get_prim_radius(f"{prim_path}/wheel_list/wheel_left/L{i}/Render") for i in range(1, 9)],
        #     device=self.device,
        # )
        flipper_radius = [
            get_prim_radius(f"{prim_path}/flipper_list/front_left_wheel/FL{i}/FlipperRender") for i in range(1, 6)
        ]
        self.flipper_radius = torch.tensor(
            flipper_radius + flipper_radius[::-1],
            device=self.device,
        )
        self.load_wheel_pivot_distances()

    def load_wheel_pivot_distances(self):
        # Mirrors load_all_wheel_radius's front/reversed-rear layout exactly, so this stays
        # aligned with fl_indices/fr_indices — see _flipper_rotation_correction.
        distances = list(np.linspace(0.0, self._AXLE_DISTANCE, 5))
        self.wheel_pivot_distance = torch.tensor(
            distances + distances[::-1],
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self, env_ids):
        self.find_idx()

    @cached_property
    def robot_prim_path(self):
        return (self.cfg.prim_path + "/" + "pumbaa_wheel").replace('.*', '0')

    def set_robot_color(self, color):
        prim_path = self.robot_prim_path
        set_material_color(f"{prim_path}/Looks/flipper_material", color)
        set_material_color(f"{prim_path}/Looks/wheel_material", color)

    def set_robot_env(self, robot_config, render_config):
        prim_path = self.robot_prim_path

        drive_wheel_radius = render_config["flipper"]["drive_wheel_radius"]
        auxiliary_wheel_radius = render_config["flipper"]["auxiliary_wheel_radius"]
        flipper_render_names = [
            [f"front_left_wheel/FL{i}" for i in range(1, 6)],
            [f"front_right_wheel/FR{i}" for i in range(1, 6)],
            [f"rear_left_wheel/RL{i}" for i in range(5, 0, -1)],
            [f"rear_right_wheel/RR{i}" for i in range(5, 0, -1)],
        ]
        track_render_names = [
            *[f"wheel_left/L{i}" for i in range(1, 9)],
            *[f"wheel_right/R{i}" for i in range(1, 9)],
        ]
        for names in flipper_render_names:
            for name, radius in zip(names, np.linspace(drive_wheel_radius, auxiliary_wheel_radius, 5)):
                set_render_radius(f"{prim_path}/flipper_list/{name}/FlipperRender", radius)
        for name in track_render_names:
            set_render_radius(f"{prim_path}/wheel_list/{name}/Render", render_config["track"]["render_radius"])

        set_prim_invisible(f"{prim_path}/front_left_flipper_link")
        set_prim_invisible(f"{prim_path}/front_right_flipper_link")
        set_prim_invisible(f"{prim_path}/rear_left_flipper_link")
        set_prim_invisible(f"{prim_path}/rear_right_flipper_link")
        if render_config["flipper"]["only_render_front_flipper"]:
            set_prim_invisible(f"{prim_path}/flipper_list/rear_left_wheel")
            set_prim_invisible(f"{prim_path}/flipper_list/rear_right_wheel")

        flipper_joint_cfg = self.cfg.actuators["flipper_joint"]
        for joint_name in flipper_joint_cfg.joint_names_expr:
            joint_path = f"{prim_path}/chassis_link/{joint_name}"
            set_joint_max_vel(joint_path, flipper_joint_cfg.velocity_limit)
            set_joint_stiffness(joint_path,  flipper_joint_cfg.stiffness)
            set_joint_damping(joint_path, flipper_joint_cfg.damping)

        for chassis_base in find_matching_prim_paths(f"{prim_path}/wheel_list/wheel_*/[LR]*"):
            chassis_render = f"{chassis_base}/Render"
            chassis_joint = f"{chassis_base}/{os.path.basename(chassis_base)}RevoluteJoint"
            get_prim_at_path(chassis_render).GetAttribute("physics:mass").Set(
                robot_config.get("chassis_wheel_render_mass", 3)
            )
            set_joint_stiffness(chassis_joint, robot_config.get("chassis_wheel_stiffness", 1))
            set_joint_damping(chassis_joint, robot_config.get("chassis_wheel_damping", 100))

        for flipper_base in find_matching_prim_paths(f"{prim_path}/flipper_list/[fr]*/[FR]*"):
            flipper_render = f"{flipper_base}/FlipperRender"
            get_prim_at_path(flipper_render).GetAttribute("physics:mass").Set(
                robot_config.get("flipper_wheel_render_mass", 3)
            )

        for flipper_wheel_joint in find_matching_prim_paths(f"{prim_path}/flipper_list/[fr]*/[FR]*/.*"):
            if "FlipperRender" in flipper_wheel_joint:
                continue
            set_joint_stiffness(flipper_wheel_joint, self.cfg.actuators["flipper_wheel"].stiffness)
            set_joint_damping(flipper_wheel_joint, self.cfg.actuators["flipper_wheel"].damping)

        set_material_friction(
            f"{prim_path}/Looks/flipper_material",
            robot_config.get("flipper_material_friction", 1),
        )
        set_material_friction(
            f"{prim_path}/Looks/wheel_material",
            robot_config.get("wheel_material_friction", 1),
        )

    @property
    def positions(self):
        return self.data.root_pos_w

    @property
    def orientations(self):
        return self.data.root_quat_w

    @property
    def lin_velocities(self):
        return self.data.root_lin_vel_b

    @property
    def ang_velocities(self):
        return self.data.root_ang_vel_b

    @property
    def projected_gravity(self):
        return self.data.projected_gravity_b
