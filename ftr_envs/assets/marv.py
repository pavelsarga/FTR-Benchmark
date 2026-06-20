from pathlib import Path

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

from ftr_envs.assets.ftr import FTR_SIM_CFG  # noqa: F401 — re-exported for convenience

_NUM_WHEELS = 5  # must match num_wheels passed to marv_to_usd.py

MARV_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/marv_description",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(__file__).parent / "usd" / "marv" / "marv.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.05,
            angular_damping=0.05,
            max_linear_velocity=10.0,
            max_angular_velocity=720.0,
            max_depenetration_velocity=0.15,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "flipper_wheel": ImplicitActuatorCfg(
            joint_names_expr=[
                *[f"front_left_flipper_wheel{i}_j" for i in range(1, _NUM_WHEELS + 1)],
                *[f"front_right_flipper_wheel{i}_j" for i in range(1, _NUM_WHEELS + 1)],
                *[f"rear_left_flipper_wheel{i}_j" for i in range(1, _NUM_WHEELS + 1)],
                *[f"rear_right_flipper_wheel{i}_j" for i in range(1, _NUM_WHEELS + 1)],
            ],
            stiffness=1,
            damping=100,
        ),
        "flipper_joint": ImplicitActuatorCfg(
            joint_names_expr=[
                "front_left_flipper_j",
                "front_right_flipper_j",
                "rear_left_flipper_j",
                "rear_right_flipper_j",
            ],
            stiffness=3e4,
            damping=1000,
            effort_limit=1000,
            velocity_limit=90,
            armature=100,
        ),
    },
)
