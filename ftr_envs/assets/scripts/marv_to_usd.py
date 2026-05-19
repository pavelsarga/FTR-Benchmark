"""Convert MARV URDF to USD for use in the FTR-benchmark Isaac Sim environment.

Run with the isaaclab conda Python (SimulationApp must be available):
    /home/robot/conda/envs/isaaclab/bin/python \
        src/FTR-benchmark/ftr_envs/assets/scripts/marv_to_usd.py

Or via the isaaclab runner:
    /home/robot/Libraries/IsaacLab/isaaclab.sh -p \
        src/FTR-benchmark/ftr_envs/assets/scripts/marv_to_usd.py

The script generates:
    src/FTR-benchmark/ftr_envs/assets/usd/marv/marv.usd

NUM_WHEELS must match the _NUM_WHEELS constant in ftr_envs/assets/marv.py (default 5).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# ── paths ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent          # …/assets/scripts
_WORKSPACE = _SCRIPT_DIR.parents[4]                   # …/robot_rodeo_gym_ws
_MARV_ROOT = _WORKSPACE / "src/robot_rodeo_gym_ros2/robots/marv"
_XACRO = _MARV_ROOT / "urdf/marv.xacro"
_COMMON_YAML = _MARV_ROOT / "config/description/common.yaml"
_OUT_USD = _SCRIPT_DIR.parent / "usd" / "marv" / "marv.usd"  # …/assets/usd/marv/marv.usd
_NUM_WHEELS = 5

# ROS2 setup to source before running xacro (resolves $(find ...) package paths)
_ROS2_SETUP = _WORKSPACE / "install/setup.bash"

_XACRO_OVERRIDES = {
    "num_wheels": str(_NUM_WHEELS),
    # disable sensors — not needed for the physics simulation
    "has_ouster_lidar": "0",
    "has_omnicam_vras": "0",
    "has_kinova": "False",
    "has_boson_thermocam": "False",
    "has_mote_deployer": "False",
    "has_cliff_sensors": "False",
    "has_mobilicom": "False",
}


def _build_xacro_args(common_yaml: Path, overrides: dict) -> list:
    """Start from all common.yaml values and layer in overrides."""
    with open(common_yaml) as f:
        base = yaml.safe_load(f)
    merged = {k: str(v) for k, v in base.items()}
    merged.update(overrides)
    return [f"{k}:={v}" for k, v in merged.items()]


def _generate_urdf(xacro_path: Path, xacro_args: list) -> str:
    with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as f:
        urdf_path = f.name

    args_str = " ".join(xacro_args)
    # Source the workspace to make $(find ...) resolve, then call xacro via
    # python3.12 so the entry-point lookup issue with the Jazzy binary is bypassed.
    bash_cmd = (
        f"source {_ROS2_SETUP} && "
        f"python3.12 -c \""
        f"import sys; sys.path.insert(0, '/opt/ros/jazzy/lib/python3.12/site-packages'); "
        f"import xacro; "
        f"sys.argv = ['xacro', '{xacro_path}', {', '.join(repr(a) for a in xacro_args)}, '-o', '{urdf_path}']; "
        f"xacro.main()\""
    )
    print(f"Generating URDF from xacro …")
    subprocess.run(["bash", "-c", bash_cmd], check=True)
    print(f"URDF written to {urdf_path}")
    return urdf_path


def _convert_to_usd(urdf_path: str, usd_path: Path) -> None:
    # SimulationApp must be started before any omni.* imports
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    from omni.importer.urdf import _urdf

    usd_path.parent.mkdir(parents=True, exist_ok=True)

    import_cfg = _urdf.ImportConfig()
    import_cfg.set_merge_fixed_joints(False)
    import_cfg.set_convex_decomp(False)
    import_cfg.set_fix_base(False)
    import_cfg.set_make_default_prim(True)
    import_cfg.set_self_collision(False)
    import_cfg.set_create_physics_scene(False)
    import_cfg.set_import_inertia_tensor(True)
    import_cfg.set_default_drive_strength(1047.2)
    import_cfg.set_default_position_drive_damping(52.4)
    import_cfg.set_default_drive_type(_urdf.UrdfJointTargetType.JOINT_DRIVE_NONE)

    urdf_interface = _urdf.acquire_urdf_interface()
    robot = urdf_interface.parse_urdf(
        str(Path(urdf_path).parent),  # directory containing URDF
        Path(urdf_path).name,         # URDF filename
        import_cfg,
    )
    # import_robot saves the USD at stage= path and returns the articulation root prim path
    prim_path = urdf_interface.import_robot(
        str(usd_path.parent),   # assetRoot (output directory)
        usd_path.stem,          # assetName without extension
        robot,
        import_cfg,
        stage=str(usd_path),    # save the stage directly to this file
    )
    _urdf.release_urdf_interface(urdf_interface)
    app.close()

    if usd_path.exists():
        print(f"USD saved to: {usd_path} (articulation root: {prim_path})")
    else:
        raise RuntimeError("URDF import failed — USD file not created")


if __name__ == "__main__":
    print(f"Workspace : {_WORKSPACE}")
    print(f"Xacro     : {_XACRO}")
    print(f"Output USD: {_OUT_USD}")
    assert _XACRO.exists(), f"Xacro not found: {_XACRO}"
    assert _COMMON_YAML.exists(), f"common.yaml not found: {_COMMON_YAML}"
    assert _ROS2_SETUP.exists(), f"ROS2 setup not found: {_ROS2_SETUP}"

    xacro_args = _build_xacro_args(_COMMON_YAML, _XACRO_OVERRIDES)
    urdf_path = _generate_urdf(_XACRO, xacro_args)
    _convert_to_usd(urdf_path, _OUT_USD)
