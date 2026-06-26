# -*- coding: utf-8 -*-
"""
====================================
@File Name ：ftr.py
@Time ： 2024/9/29 下午2:13
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
from pathlib import Path

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg


FTR_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/pumbaa_wheel",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(__file__).parent / "usd" / "ftr" / "ftr_v1.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.05,         # passive energy dissipation — damps pre-explosion velocities
            angular_damping=0.05,        # passive rotational damping — prevents tumbling runaway
            max_linear_velocity=10.0,    # m/s — 2× max expected forward vel; catches explosions
            max_angular_velocity=720.0,  # deg/s (~12.6 rad/s) — allows terrain traversal, catches tumbling
            max_depenetration_velocity=0.15,  # gentle depenetration — prevents launch impulses from flipper-terrain overlap
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,  # 32 → 16: still high, sufficient for tracked robot
            solver_velocity_iteration_count=4,   # 1 → 4: critical — dissipates energy from contact penetrations
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        # "baselink_wheel": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         *[f"L{i+1}RevoluteJoint" for i in range(8)],
        #         *[f"R{i+1}RevoluteJoint" for i in range(8)],
        #     ],
        #     stiffness=1,
        #     damping=100,
        # ),
        "flipper_wheel": ImplicitActuatorCfg(
            joint_names_expr=[
                *[f"LF{i+1}RevoluteJoint" for i in range(5)],
                *[f"LR{i+1}RevoluteJoint" for i in range(5)],
                *[f"RL{i+1}RevoluteJoint" for i in range(5)],
                *[f"RR{i+1}RevoluteJoint" for i in range(5)],
            ],
            # stiffness=0: this is a velocity-only drive (set_joint_velocity_target is the
            # only thing ever called on these joints, never set_joint_position_target). A
            # nonzero stiffness here creates a phantom restoring torque toward the stale
            # position target (0, frozen since reset) that grows with accumulated wheel
            # rotation on these continuous joints — confirmed via teleop A/B test (on MARV)
            # to reduce contact chatter and let the robot actually cross an obstacle it was
            # stuck on at stiffness=1. Same bug, same fix applied here for FTR.
            stiffness=0,
            damping=100,
        ),
        "flipper_joint": ImplicitActuatorCfg(
            joint_names_expr=[
                "front_left_flipper_joint",
                "front_right_flipper_joint",
                "rear_left_flipper_joint",
                "rear_right_flipper_joint",
            ],
            stiffness=3e4,
            damping=1000,
            effort_limit=1000,
            velocity_limit=180,
            armature=100,

            # stiffness=1000,
            # damping=0,
        ),

    },
)

FTR_SIM_CFG = SimulationCfg(
    dt=1 / 400,
    render_interval=5,
    disable_contact_processing=False,
    physx=PhysxCfg(
        min_position_iteration_count=16,
        max_velocity_iteration_count=4,              # 0 → 4: dissipates energy from penetrations (primary explosion fix)
        bounce_threshold_velocity=0.2,               # contacts below 0.2 m/s don't bounce
        gpu_max_rigid_contact_count=2**24,           # 2× default — prevents buffer overflow with many tracked-robot contacts
        gpu_found_lost_pairs_capacity=2**22,         # 2× default
        gpu_found_lost_aggregate_pairs_capacity=2**26,  # 2× default
        gpu_total_aggregate_pairs_capacity=2**24,    # increased for contact-heavy scenes
        gpu_collision_stack_size=2**27,              # 2× default — tracked robots generate deep collision stacks
    ),
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.9,
        dynamic_friction=0.7,
        restitution=0.0,
    ),
)