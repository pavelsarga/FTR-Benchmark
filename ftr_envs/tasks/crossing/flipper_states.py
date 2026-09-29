"""Discrete flipper states for flipper_control_mode == "state".

The HFC-IL states (rl_modules/hfcil/hfcil_transitions.py order, BALLERINA dropped) without the
transition graph: any state may follow any other, but a change is only accepted once the
cooldown since the previous change has run out. Pure torch so it can be tested without Isaac.
"""

import torch

# (name, front_deg, rear_deg) in the MARV ROS/URDF convention: front negative = up,
# rear positive = up. Left and right flipper of a pair share the angle.
DEFAULT_FLIPPER_STATES_DEG = (
    ("N", -60.0, 60.0),
    ("AF", -45.0, 0.0),
    ("AS", 20.0, -15.0),
    ("AR", 30.0, -80.0),
    ("DF", 60.0, 0.0),
    ("DS", 0.0, 2.0),
    ("DR", -10.0, 45.0),
)


def state_names(states_deg) -> list[str]:
    return [s[0] for s in states_deg]


def state_pose_targets(states_deg, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """(K, 4) [FL, FR, RL, RR] targets in FtrEnv's logical angle (rad), clamped to [low, high].

    The logical angle is the raw joint angle x [-1, -1, 1, 1] (see FtrEnvCfg's
    marv_flipper_*_deg note), so a ROS-convention front angle changes sign and a rear one does not.
    """
    front = torch.tensor([-float(s[1]) for s in states_deg])
    rear = torch.tensor([float(s[2]) for s in states_deg])
    poses = torch.deg2rad(torch.stack([front, front, rear, rear], dim=-1)).to(low)
    return torch.maximum(torch.minimum(poses, high), low)


def step_flipper_state(scores: torch.Tensor, state: torch.Tensor, cooldown: torch.Tensor, cooldown_steps: int):
    """Advance the per-env state by one control step.

    scores (N, K) -> proposed state argmax. cooldown counts control steps until a change is
    allowed again; it is decremented first, so with cooldown_steps = 2 a change at step t
    blocks t+1 and allows t+2 (200 ms at 10 Hz).

    Returns (state, cooldown, switched, blocked); the inputs are not modified.
    """
    proposed = scores.argmax(dim=-1)
    cooldown = (cooldown - 1).clamp(min=0)
    wants = proposed != state
    switched = wants & (cooldown == 0)
    blocked = wants & ~switched
    state = torch.where(switched, proposed, state)
    cooldown = torch.where(switched, torch.full_like(cooldown, cooldown_steps), cooldown)
    return state, cooldown, switched, blocked
