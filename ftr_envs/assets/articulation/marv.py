from functools import cached_property

import numpy as np
import torch

from ftr_envs.assets.articulation.ftr import FtrWheelArticulation
from ftr_envs.utils.prim import set_joint_damping, set_joint_max_vel, set_joint_stiffness, set_material_friction


class MarvWheelArticulation(FtrWheelArticulation):
    """FtrWheelArticulation adapted for the MARV robot USD (generated from MARV URDF).

    Key differences vs FTR:
    - Flipper joint names end in ``_j`` not ``_joint``.
    - Virtual track wheel joints follow the pattern ``{side}_flipper_wheel{n}_j``.
    - Root prim is ``base_link`` (no ``pumbaa_wheel`` sub-prim).
    - Wheel radii are computed analytically from known MARV geometry.
    """

    # MARV wheel geometry (matches common.yaml)
    _BIG_RADIUS = 0.1165
    _SMALL_RADIUS = 0.0780

    def find_idx(self):
        self.flipper_dof_idx_list = [self.find_joints(i)[0][0] for i in self.flipper_joint_names]

        self.fr_indices = [
            self.find_joints(i)[0][0]
            for i in self.flipper_wheel_joint_names
            if "_right_" in i
        ]
        self.fl_indices = [
            self.find_joints(i)[0][0]
            for i in self.flipper_wheel_joint_names
            if "_left_" in i
        ]

    def load_all_wheel_radius(self):
        # Number of wheels per flipper: total wheel joints / 4 flippers
        n = len(self.cfg.actuators["flipper_wheel"].joint_names_expr) // 4
        radii = np.linspace(self._BIG_RADIUS, self._SMALL_RADIUS, n)
        # MARV wheel 1 is always the big-radius end, wheel n is the small-radius end,
        # for both front and rear flippers — so use the same radii sequence for both.
        self.flipper_radius = torch.tensor(
            list(radii) + list(radii),
            dtype=torch.float32,
            device=self.device,
        )

    @cached_property
    def robot_prim_path(self):
        return self.cfg.prim_path.replace(".*", "0")

    def set_robot_env(self, robot_config, render_config):
        container = self.robot_prim_path  # /World/envs/env_0/marv_description

        # Flipper joints live under base_link in the MARV USD structure
        flipper_joint_cfg = self.cfg.actuators["flipper_joint"]
        for joint_name in flipper_joint_cfg.joint_names_expr:
            joint_path = f"{container}/base_link/{joint_name}"
            set_joint_max_vel(joint_path, flipper_joint_cfg.velocity_limit)
            set_joint_stiffness(joint_path, flipper_joint_cfg.stiffness)
            set_joint_damping(joint_path, flipper_joint_cfg.damping)

        # Wheel joints live under their parent flipper link, e.g.
        # front_left_flipper_wheel1_j is under front_left_flipper
        flipper_wheel_cfg = self.cfg.actuators["flipper_wheel"]
        flipper_friction = robot_config.get("flipper_material_friction", 1)
        for joint_name in flipper_wheel_cfg.joint_names_expr:
            parent_link = "_".join(joint_name.split("_")[:3])  # e.g. "front_left_flipper"
            joint_path = f"{container}/{parent_link}/{joint_name}"
            set_joint_stiffness(joint_path, flipper_wheel_cfg.stiffness)
            set_joint_damping(joint_path, flipper_wheel_cfg.damping)
            wheel_link = joint_name.replace("_j", "")
            try:
                set_material_friction(f"{container}/{parent_link}/{wheel_link}", flipper_friction)
            except Exception:
                pass  # MARV wheel prims from URDF import don't carry PhysicsMaterial attrs
