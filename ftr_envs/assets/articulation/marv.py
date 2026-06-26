from functools import cached_property

import numpy as np
import torch

from ftr_envs.assets.articulation.ftr import FtrWheelArticulation
from ftr_envs.utils.prim import (
    bind_physics_material,
    disable_collision_geometry,
    ensure_physics_material,
    set_collision_offsets,
    set_cylinder_collision_radius,
    set_joint_damping,
    set_joint_max_vel,
    set_joint_stiffness,
)


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
    # Pivot-to-wheel5 distance (m) = rover_flipperLength - bigRadius - smallRadius
    # = 0.498 - 0.1165 - 0.078, matching the measured pivot->FL5 world distance. wheel1
    # sits at the pivot (distance 0) for every flipper — see load_wheel_pivot_distances().
    _AXLE_DISTANCE = 0.3035

    def find_idx(self):
        self.flipper_dof_idx_list = [self.find_joints(i)[0][0] for i in self.flipper_joint_names]
        front_left_dof, front_right_dof, rear_left_dof, rear_right_dof = self.flipper_dof_idx_list
        dof_by_flipper = {
            "front_left": front_left_dof,
            "front_right": front_right_dof,
            "rear_left": rear_left_dof,
            "rear_right": rear_right_dof,
        }

        def _parent_flipper_dof(wheel_joint_name):
            for key, dof in dof_by_flipper.items():
                if key in wheel_joint_name:
                    return dof
            raise ValueError(f"could not resolve parent flipper for wheel joint {wheel_joint_name!r}")

        if getattr(self, "legacy_ftr_turning", False):
            # Mimic the pre-af6aa92 FTR bug (ftr_envs/assets/articulation/ftr.py):
            # FtrWheelArticulation.find_idx used to group wheel joints by
            # startswith("R")/startswith("L") on the (mislabeled) LF/LR/RL/RR prefixes,
            # which actually split the wheels into FRONT (→ "fl_indices", got the
            # set_v_w "v - w*L/2" term) vs REAR (→ "fr_indices", got "v + w*L/2") groups
            # instead of true left/right sides. Policies trained on FTR before that fix
            # learned to steer under this front/rear split — replicate it here so they
            # behave on MARV the way they did on (pre-fix) FTR. Enable via
            # env_cfg_overrides.legacy_ftr_turning: true for old policy checkpoints.
            self.fr_indices = [
                self.find_joints(i)[0][0]
                for i in self.flipper_wheel_joint_names
                if "rear_" in i
            ]
            self.fl_indices = [
                self.find_joints(i)[0][0]
                for i in self.flipper_wheel_joint_names
                if "front_" in i
            ]
            self.fr_wheel_flipper_dof = [
                _parent_flipper_dof(i) for i in self.flipper_wheel_joint_names if "rear_" in i
            ]
            self.fl_wheel_flipper_dof = [
                _parent_flipper_dof(i) for i in self.flipper_wheel_joint_names if "front_" in i
            ]
        else:
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
            self.fr_wheel_flipper_dof = [
                _parent_flipper_dof(i) for i in self.flipper_wheel_joint_names if "_right_" in i
            ]
            self.fl_wheel_flipper_dof = [
                _parent_flipper_dof(i) for i in self.flipper_wheel_joint_names if "_left_" in i
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
        self.load_wheel_pivot_distances()

    def load_wheel_pivot_distances(self):
        # Mirrors load_all_wheel_radius's (non-reversed) front/rear layout exactly, so
        # this stays aligned with fl_indices/fr_indices — see _flipper_rotation_correction.
        n = len(self.cfg.actuators["flipper_wheel"].joint_names_expr) // 4
        distances = np.linspace(0.0, self._AXLE_DISTANCE, n)
        self.wheel_pivot_distance = torch.tensor(
            list(distances) + list(distances),
            dtype=torch.float32,
            device=self.device,
        )

    @cached_property
    def robot_prim_path(self):
        return self.cfg.prim_path.replace(".*", "0")

    def set_robot_env(self, robot_config, render_config):
        # Stash for find_idx(), which runs later (at reset time) and only receives
        # no arguments — see the legacy_ftr_turning comment in find_idx() above.
        self.legacy_ftr_turning = bool(robot_config.get("legacy_ftr_turning", False))

        container = self.robot_prim_path  # /World/envs/env_0/marv_description

        # Flipper joints live under base_link in the MARV USD structure
        flipper_joint_cfg = self.cfg.actuators["flipper_joint"]
        for joint_name in flipper_joint_cfg.joint_names_expr:
            joint_path = f"{container}/base_link/{joint_name}"
            set_joint_max_vel(joint_path, flipper_joint_cfg.velocity_limit)
            set_joint_stiffness(joint_path, flipper_joint_cfg.stiffness)
            set_joint_damping(joint_path, flipper_joint_cfg.damping)

        # The MARV URDF import has no PhysicsMaterial anywhere on the stage (unlike FTR's
        # baked Looks/wheel_material + Looks/flipper_material), so friction has to be
        # created from scratch and bound to the collision geometry directly.
        wheel_material = ensure_physics_material(
            f"{container}/Looks/wheel_material", robot_config.get("wheel_material_friction", 1)
        )
        flipper_material = ensure_physics_material(
            f"{container}/Looks/flipper_material", robot_config.get("flipper_material_friction", 1)
        )
        for flipper_link in ("front_left_flipper", "front_right_flipper", "rear_left_flipper", "rear_right_flipper"):
            bind_physics_material(f"{container}/{flipper_link}/collisions", flipper_material)

        # Wheel joints live under their parent flipper link, e.g.
        # front_left_flipper_wheel1_j is under front_left_flipper. The wheel *bodies*
        # (front_left_flipper_wheel1, ...) are siblings of marv_description, not nested
        # under their parent flipper link.
        flipper_wheel_cfg = self.cfg.actuators["flipper_wheel"]
        for joint_name in flipper_wheel_cfg.joint_names_expr:
            parent_link = "_".join(joint_name.split("_")[:3])  # e.g. "front_left_flipper"
            joint_path = f"{container}/{parent_link}/{joint_name}"
            set_joint_stiffness(joint_path, flipper_wheel_cfg.stiffness)
            set_joint_damping(joint_path, flipper_wheel_cfg.damping)
            wheel_link = joint_name.replace("_j", "")
            bind_physics_material(f"{container}/{wheel_link}/collisions", wheel_material)

        # The MARV URDF flipper link itself carries collision geometry (3 boxes + 2
        # cylinders at the wheel-end positions) that physically overlaps with the driven
        # fake-wheel child-link cylinders (wheel1 at the pivot, wheel5 at the far end).
        # Having a near-massless rigid body (flipper_mass=1e-05 kg) and a real mass body
        # (0.73 kg wheel) at the same position both contacting terrain simultaneously
        # produces two contact manifolds at the same point, creating PhysX contact
        # instability (duplicated impulses, solver oscillation). Disabling the flipper
        # arm body's own collision geometry leaves only the 5 driven wheel cylinders for
        # ground contact — they already span the full arm length so the coverage is
        # equivalent. Default: enabled (True) since preliminary testing shows this is
        # the most likely cause of the contact ringing and sluggish motion; opt-out via
        # disable_flipper_arm_collision: false in robot_config.
        if robot_config.get("disable_flipper_arm_collision", True):
            for flipper_link in ("front_left_flipper", "front_right_flipper", "rear_left_flipper", "rear_right_flipper"):
                n = disable_collision_geometry(f"{container}/{flipper_link}")
                if n == 0:
                    import warnings
                    warnings.warn(f"disable_flipper_arm_collision: no collision shapes found at {container}/{flipper_link}")

        # wheel1 (the big-radius, leading wheel on every flipper, closest to the pivot
        # and bearing the most ground-contact load) intermittently gets its joint
        # velocity kicked to large negative values under load (verified via direct
        # joint_vel logging — only ever wheel1, never wheels 2-5; not present during
        # free-fall before ground contact; unaffected by solver position/velocity
        # iteration count). PhysX auto-derives contact offset from collision shape
        # size, so wheel1 (the largest wheel) gets the largest default speculative-
        # contact margin — tightening it to an explicit small value roughly halves
        # the glitch rate (measured: ~22/60 steps at the default margin → ~11/60 at
        # contact_offset=0.001; going lower (0.0001) is rejected outright by PhysX,
        # going higher (0.005) regresses back toward the default-margin rate).
        for flipper in ("front_left_flipper", "front_right_flipper", "rear_left_flipper", "rear_right_flipper"):
            set_collision_offsets(f"{container}/{flipper}_wheel1/collisions", contact_offset=0.001, rest_offset=0.0)

        # Experiment: MARV's wheel1 (radius 0.1165 m) is ~30% larger than FTR's
        # equivalent (0.09 m) — a bigger collision cylinder sweeps through more terrain
        # mesh triangles per step, which could make it snag more often. This shrinks only
        # the *collision* shape, not the kinematic radius used by load_all_wheel_radius
        # for velocity-target scaling, so commanded driving speed is unaffected — isolates
        # whether contact-geometry size alone changes the wheel1 glitch rate. Opt-in via
        # env_cfg_overrides.wheel1_collision_radius_scale (default 1.0 = unchanged).
        wheel1_radius_scale = robot_config.get("wheel1_collision_radius_scale", 1.0)
        if wheel1_radius_scale != 1.0:
            for flipper in ("front_left_flipper", "front_right_flipper", "rear_left_flipper", "rear_right_flipper"):
                set_cylinder_collision_radius(
                    f"{container}/{flipper}_wheel1/collisions", self._BIG_RADIUS * wheel1_radius_scale
                )
