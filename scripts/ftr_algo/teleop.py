"""FTR MARV robot teleoperation via Xbox gamepad in Isaac Lab.

Reads joystick input directly from /dev/input/js0 (same as ROS2 joy package),
bypassing the carb gamepad API which doesn't work with standard Linux HID drivers.

Controls (Xbox layout, matching marv_teleop buttons_mapping.yaml):
  Left stick Y     : linear velocity  (forward / backward, max 0.95 m/s)
  Left stick X     : angular velocity (turn left / right,   max 1.0 rad/s)
  Right stick Y    : flipper velocity magnitude (~40 deg/s at full)
  LB  (hold)       : front-left  flipper active
  RB  (hold)       : front-right flipper active
  LT  (> 50%)      : rear-left   flipper active
  RT  (> 50%)      : rear-right  flipper active

Launch (conda):
  cd src/FTR-benchmark
  conda run -n isaaclab python scripts/ftr_algo/teleop.py --num_envs 1 --device cuda:0

Launch (apptainer):
  apptainer exec --nv containers/isaaclab_optuna.sif \\
      python scripts/ftr_algo/teleop.py --num_envs 1 --device cuda:0

Flat terrain for basic testing:
  conda run -n isaaclab python scripts/ftr_algo/teleop.py --num_envs 1 --terrain ground

MARV robot:
  conda run -n isaaclab python scripts/ftr_algo/teleop.py --num_envs 1 --robot_type marv
"""

# Isaac Sim AppLauncher MUST be initialised before any omni/carb imports.
import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="FTR MARV Xbox gamepad teleoperation in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments to simulate (use 1 for teleop).")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O (slower, for debugging).")
parser.add_argument("--terrain", type=str, default="cur_mixed",
                    help="Terrain name (default: cur_mixed). Use 'ground' for flat testing.")
parser.add_argument("--js", type=str, default="/dev/input/js0",
                    help="Joystick device path (default: /dev/input/js0).")
parser.add_argument("--robot_type", type=str, default="ftr", choices=["ftr", "marv"],
                    help="Robot model to simulate (default: ftr).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------- post-launch imports ----------
import struct
import sys
import threading
from pathlib import Path

import gymnasium as gym
import numpy as np
import omni.isaac.lab_tasks  # noqa: F401 — registers built-in Isaac Lab tasks
import torch
from omni.isaac.lab_tasks.utils import parse_env_cfg

# ftr_envs is not an installed package; add the repo root to sys.path.
_FTR_ROOT = str(Path(__file__).parents[2])
if _FTR_ROOT not in sys.path:
    sys.path.insert(0, _FTR_ROOT)

import ftr_envs.tasks           # noqa: F401 — registers Ftr-Crossing-Direct-v0
import ftr_envs.utils.omega_conf  # noqa: F401 — OmegaConf resolvers


# ---------- joystick reader ----------

class _LinuxJoystick:
    """Reads Linux joystick events from /dev/input/jsX in a background thread.

    Event format (8 bytes): uint32 time | int16 value | uint8 type | uint8 number
    type 1 = button  (value: 0/1)
    type 2 = axis    (value: -32767..32767, normalised to [-1, 1])
    The init-event flag (0x80) is stripped so synthetic init events are handled
    the same way as real input events.
    """

    _FMT = "IhBB"
    _SIZE = struct.calcsize(_FMT)
    _BUTTON = 1
    _AXIS = 2

    def __init__(self, device: str = "/dev/input/js0"):
        try:
            self._fd = open(device, "rb")
        except PermissionError:
            raise RuntimeError(
                f"Permission denied opening {device}. "
                f"Run: sudo chmod a+r {device}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"{device} not found. Is the controller connected?"
            )

        self._axes: dict[int, float] = {}
        self._buttons: dict[int, bool] = {}
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True, name="js_reader")
        self._thread.start()
        print(f"[FtrGamepad] Reading joystick from {device}", flush=True)

    def _loop(self) -> None:
        while self._running:
            data = self._fd.read(self._SIZE)
            if not data or len(data) < self._SIZE:
                break
            _time, value, ev_type, number = struct.unpack(self._FMT, data)
            ev_type &= ~0x80  # strip JS_EVENT_INIT
            with self._lock:
                if ev_type == self._AXIS:
                    self._axes[number] = value / 32767.0
                elif ev_type == self._BUTTON:
                    self._buttons[number] = bool(value)

    def axis(self, n: int, default: float = 0.0) -> float:
        with self._lock:
            return self._axes.get(n, default)

    def button(self, n: int, default: bool = False) -> bool:
        with self._lock:
            return self._buttons.get(n, default)

    def close(self) -> None:
        self._running = False
        try:
            self._fd.close()
        except Exception:
            pass


# ---------- gamepad handler ----------

class FtrGamepad:
    """Xbox controller wrapper for the FTR 6-DOF action space.

    Reads from /dev/input/js0 via _LinuxJoystick.

    Xbox axis/button numbering (Linux xpad / xboxdrv driver,
    same as marv_teleop buttons_mapping.yaml):
      Axis 0  : left stick X  (left=-1, right=+1)
      Axis 1  : left stick Y  (up=-1,   down=+1)   ← inverted vs intuition
      Axis 2  : LT            (not pressed=-1, fully pressed=+1)
      Axis 3  : right stick X
      Axis 4  : right stick Y (up=-1,   down=+1)   ← inverted
      Axis 5  : RT            (not pressed=-1, fully pressed=+1)
      Button 4: LB
      Button 5: RB
    """

    def __init__(
        self,
        v_sensitivity: float = 0.95,
        w_sensitivity: float = 1.0,
        flipper_sensitivity: float = 0.8,
        dead_zone: float = 0.05,
        device: str = "/dev/input/js0",
    ):
        self.v_sensitivity = v_sensitivity
        self.w_sensitivity = w_sensitivity
        self.flipper_sensitivity = flipper_sensitivity
        self.dead_zone = dead_zone
        self._js = _LinuxJoystick(device)

    def _apply_dz(self, val: float) -> float:
        if abs(val) < self.dead_zone:
            return 0.0
        sign = 1.0 if val > 0.0 else -1.0
        return sign * (abs(val) - self.dead_zone) / (1.0 - self.dead_zone)

    def advance(self) -> np.ndarray:
        """Return the 6-element FTR action [v, w, fl, fr, rl, rr]."""
        # Y axes are inverted on Linux (up = -1); X axis is not — left is positive.
        v  = self._apply_dz(-self._js.axis(1)) * self.v_sensitivity
        w  = self._apply_dz( self._js.axis(0)) * self.w_sensitivity
        fv = self._apply_dz(-self._js.axis(4)) * self.flipper_sensitivity

        lb = self._js.button(4)
        rb = self._js.button(5)
        # Triggers: -1 = not pressed, +1 = fully pressed → active when > 0
        lt = self._js.axis(2) > 0.0
        rt = self._js.axis(5) > 0.0

        return np.array([
            v, w,
            fv if lb else 0.0,
            fv if rb else 0.0,
            fv if lt else 0.0,
            fv if rt else 0.0,
        ], dtype=np.float32)

    def reset(self) -> None:
        pass  # stateless — joystick reader always reflects hardware state

    def close(self) -> None:
        self._js.close()

    def __str__(self) -> str:
        return f"FtrGamepad  device='{self._js._fd.name}'"


# ---------- helpers ----------

def print_controls() -> None:
    print()
    print("=" * 56)
    print("  FTR MARV Teleop — Xbox Gamepad")
    print("=" * 56)
    print("  DRIVING")
    print("    Left  stick Y  : linear  velocity  (fwd/bwd)")
    print("    Left  stick X  : angular velocity  (turn L/R)")
    print()
    print("  FLIPPERS  (right stick Y = speed; hold to select)")
    print("    LB  (hold)     : front-left  flipper")
    print("    RB  (hold)     : front-right flipper")
    print("    LT  (> 50%)    : rear-left   flipper")
    print("    RT  (> 50%)    : rear-right  flipper")
    print("    Right stick UP : positive delta (extends up)")
    print()
    print("  Status printed every ~1 s  |  Close viewport to quit")
    print("=" * 56)
    print()


# ---------- main ----------

TASK = "Ftr-Crossing-Direct-v0"
STATUS_STEPS = 10   # print every ~0.25 s so we can see axis values quickly


def main() -> None:
    # --- environment ---
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.terrain_name = args_cli.terrain
    env_cfg.robot_type = args_cli.robot_type
    env_cfg.initial_flipper_range = (0, 0)   # start all flippers horizontal
    env_cfg.forward_vel_range = (0.0, 0.0)   # no env-imposed forward velocity

    env = gym.make(TASK, cfg=env_cfg)

    # --- joystick ---
    try:
        gamepad = FtrGamepad(
            v_sensitivity=0.95,
            w_sensitivity=1.5,
            flipper_sensitivity=0.4,
            dead_zone=0.05,
            device=args_cli.js,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", flush=True)
        env.close()
        return

    print(gamepad)
    print_controls()

    obs, _ = env.reset()
    step = 0

    while simulation_app.is_running():
        cmd = gamepad.advance()
        actions = torch.tensor(cmd, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
        obs, _rew, terminated, truncated, _info = env.step(actions)

        if terminated.any() or truncated.any():
            print("[INFO] Episode ended — resetting.", flush=True)
            obs, _ = env.reset()

        step += 1
        if step % STATUS_STEPS == 0:
            vel = env.unwrapped.robot_lin_velocities[0, 0].item()
            flips_deg = [
                round(np.rad2deg(f), 1)
                for f in env.unwrapped.flipper_positions[0].tolist()
            ]
            js = gamepad._js
            rew_info = env.unwrapped.extras.get("reward_components", {})
            accel_mag = rew_info.get("shock/accel_magnitude", float("nan"))
            shock_norm = rew_info.get("shock/shock_normalised", float("nan"))
            import sys as _sys
            print(
                f"[{step:6d}]  cmd=[v={cmd[0]:+.2f} w={cmd[1]:+.2f}]  "
                f"v_actual={vel:+.2f} m/s  flippers={flips_deg} deg  "
                f"shock={accel_mag:.1f} m/s²  shock_norm={shock_norm:.3f}  "
                f"ax0-5=[{js.axis(0):+.2f} {js.axis(1):+.2f} {js.axis(2):+.2f} "
                f"{js.axis(3):+.2f} {js.axis(4):+.2f} {js.axis(5):+.2f}]  "
                f"btn=[{js.button(4)} {js.button(5)}]",
                file=_sys.stderr, flush=True,
            )

    gamepad.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
