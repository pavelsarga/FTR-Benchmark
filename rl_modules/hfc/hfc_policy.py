import math

import torch
import torch.nn as nn

from rl_modules.hfc.hfc_observation import HFCEncoder

# Pose order matches Azayev & Zimmermann 2022 Fig. 1 / the reference repo's state_list
# (paper_repos/augmented_robot_trackers/src/control/marv_flipper_controller.py).
POSE_NAMES = [
    "NEUTRAL",
    "ASCENDING_FRONT",
    "UP_STAIRS",
    "ASCENDING_REAR",
    "DESCENDING_FRONT",
    "DOWN_STAIRS",
    "DESCENDING_REAR",
]


class HFCPolicy(nn.Module):
    """
    Hybrid Flipper Controller (HFC), adapted from Azayev & Zimmermann 2022. Wraps
    HFCEncoder's Q(S'_t, a_t) over the paper's 7 named poses with epsilon-greedy discrete
    selection, then decodes the selected pose into the flipper command needed for
    FtrEnv's `flipper_control_mode: position` (env_cfg_overrides — required for this
    module) — NOT a CW/hold/CCW delta like ATD3QNPolicy/ICMD3QNPolicy. Position mode maps
    a raw [-1, 1] command linearly onto [low, high] (the asymmetric MARV limits) with no
    integration/decimation, so each pose's target angle is reached directly in one step.

    Each pose's (front, rear) target angle is precomputed once at construction as a
    fraction of the configured asymmetric MARV flipper limits (front_up_deg/front_down_deg/
    back_up_deg/back_down_deg — these must match env_cfg_overrides.marv_flipper_*_deg),
    then inverted through the same low+unit*(high-low) mapping FtrEnv's "position" control
    uses (ftr_env.py:_pre_physics_step) to get the raw command this policy must emit.

    The fractions themselves come from the reference repo's FLIPPERS_<STATE> templates
    (paper_repos/augmented_robot_trackers/src/control/configs/marv_flipper_controller_config.yaml),
    normalised against that repo's own most-extreme raw value per direction (front-up ref
    = 2.0 rad from NEUTRAL, front-down ref = 0.35 rad from DESCENDING_FRONT, rear-up ref =
    1.5 rad from NEUTRAL, rear-down ref = 0.7 rad from DESCENDING_FRONT) since the repo's
    raw radian values assume a different physical joint range than this simulator's
    configured limits and can't be copied verbatim.

    The paper's roll-stabilization and escape-maneuver terms (Eqs. 7-8, requiring
    independent left/right flippers) are intentionally omitted — this project uses
    sync_flipper_control: true (synced front/rear pairs), which the reference repo's own
    FLIPPERS_<STATE> templates already assume (front-left == front-right, rear-left ==
    rear-right in every template).
    """

    NUM_ACTIONS = 7

    # [front_frac, rear_frac] per pose (see class docstring for the normalisation method).
    # front_frac: negative = up (scaled by front_up_deg), positive = down (front_down_deg).
    # rear_frac: positive = up (back_up_deg), negative = down (back_down_deg).
    POSE_FRACTIONS = [
        (-1.0, 1.0),        # NEUTRAL
        (-0.2, 0.0),        # ASCENDING_FRONT
        (0.2857, -0.1429),  # UP_STAIRS
        (0.2857, -0.8571),  # ASCENDING_REAR
        (1.0, -1.0),        # DESCENDING_FRONT
        (0.0, 0.0333),      # DOWN_STAIRS
        (-0.15, 0.2667),    # DESCENDING_REAR
    ]

    def __init__(
        self,
        track_vel: float = 0.5,
        epsilon: float = 0.0,
        front_up_deg: float = 90.0,
        front_down_deg: float = 80.0,
        back_up_deg: float = 90.0,
        back_down_deg: float = 80.0,
        **encoder_kwargs,
    ):
        super().__init__()
        self.q_network = HFCEncoder(**encoder_kwargs)
        self.track_vel = track_vel
        self.epsilon = epsilon

        front_up_rad, front_down_rad = math.radians(front_up_deg), math.radians(front_down_deg)
        back_up_rad, back_down_rad = math.radians(back_up_deg), math.radians(back_down_deg)
        # Matches FtrEnv._pre_physics_step's sync_flipper_control low/high (front: [-up, down],
        # rear: [-down, up]) so the inverse mapping below lands on the exact target angle.
        front_low, front_high = -front_up_rad, front_down_rad
        rear_low, rear_high = -back_down_rad, back_up_rad

        front_cmds, rear_cmds = [], []
        for front_frac, rear_frac in self.POSE_FRACTIONS:
            front_target = front_frac * (front_up_rad if front_frac < 0 else front_down_rad)
            rear_target = rear_frac * (back_up_rad if rear_frac > 0 else back_down_rad)
            # Inverse of FtrEnv's `target = low + unit * (high - low)`, unit = (cmd + 1) / 2.
            front_cmds.append(2.0 * (front_target - front_low) / (front_high - front_low) - 1.0)
            rear_cmds.append(2.0 * (rear_target - rear_low) / (rear_high - rear_low) - 1.0)
        self.register_buffer("pose_commands", torch.tensor([front_cmds, rear_cmds]).T.clamp(-1.0, 1.0))  # (7, 2)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, action_idx): `action` (N, 4) drives the env, `action_idx`
        (N,) int64 (selected pose index) is what the DQN loss trains on."""
        q = self.q_network(obs)  # (..., 7)

        greedy_idx = q.argmax(dim=-1)
        if self.training and self.epsilon > 0:
            random_idx = torch.randint(0, self.NUM_ACTIONS, greedy_idx.shape, device=obs.device)
            explore = torch.rand(greedy_idx.shape, device=obs.device) < self.epsilon
            idx = torch.where(explore, random_idx, greedy_idx)
        else:
            idx = greedy_idx

        flipper_command = self.pose_commands[idx]  # (..., 2): [front, rear] in [-1, 1]

        lead_shape = flipper_command.shape[:-1]
        track_v = torch.full((*lead_shape, 1), self.track_vel, device=obs.device, dtype=flipper_command.dtype)
        track_w = torch.zeros((*lead_shape, 1), device=obs.device, dtype=flipper_command.dtype)

        action = torch.cat([track_v, track_w, flipper_command], dim=-1)  # (..., 4)
        return action, idx
