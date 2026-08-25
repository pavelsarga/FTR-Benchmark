import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from omni.isaac.lab.envs import VecEnvObs

from rl_modules.rl_module import RLModule

_log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "creps_module.yaml"


class CREPSModule(RLModule):
    """Observation + reward for Pecka, Salansky, Zimmermann & Svoboda 2016, "Autonomous
    Flipper Control with Safety Constraints" (Constrained REPS / CREPS, IROS). Unlike
    every other module in this project (all per-step PPO/D3QN, gradient-trained neural
    networks), CREPS is an episodic, gradient-free black-box policy search over a
    deterministic 6-parameter LINEAR lower-level policy (CREPSLowerLevelPolicy,
    creps_policy.py) — see train_creps.py for the outer-loop algorithm
    (creps_algorithm.py's dual solve + weighted maximum-likelihood refit).

    get_reward_components() intentionally returns ONLY raw per-step forward-progress
    distance, with NO terminal-bonus/zero-on-terminal masking like every other module
    uses. The paper's R_sw is simply "total distance traveled in a fixed 30s episode" —
    summing this component over an episode (via train_creps.py's own per-env done-latch,
    not CrossingEnv's terminal masking) gives exactly that. The paper's safety indicator
    S_sw (rollover / hard impact / explosion) and mechanical-constraint indicator C_sw
    (commanded flipper angle exceeding configured limits) are NOT part of this reward at
    all — they're read directly by train_creps.py's rollout loop from
    env._fail_mask/env._explosion_mask and env.last_action, matching the paper's
    explicit design of keeping safety/reward as independently optimized measures (see
    the paper's Section I: "separating safety and rewards into independently optimized
    measures simplifies the learning process").

    Requires sync_flipper_control: true (front/rear flipper pairs, matching the paper's
    2-D flipper action space) and flipper_control_mode: position (the paper's action is
    an absolute flipper-angle target). Requires env_cfg_overrides.fixed_forward_vel to be
    set to a nonzero constant (the paper's "robot automatically driven forward by a
    constant speed") — CREPSLowerLevelPolicy always emits v=w=0 itself since forward
    speed and steering are not part of the paper's learned policy.
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.cfg = OmegaConf.load(_CONFIG_PATH)

    def get_observations(self) -> VecEnvObs:
        env = self.env
        pitch = env.orientations_3[:, 1]  # (N,), raw radians

        # height_ahead: current_frame_height_maps is (N, 45, 21), cell_size=0.05m; after
        # ftr_env.py's calc_current_frame_height_maps() .flip(0), row index decreases
        # toward the front of the robot. Center = (height_map_size[0]//2, [1]//2) =
        # env.positions (base_link), NOT the front of the physical body. row
        # cfg.height_ahead_row (default 9, creps_module.yaml) is the front flipper
        # tip's flat-pose reach (0.6375m) ahead of base_link -- sampling right at the
        # tip, i.e. what the flippers would actually contact first -- see
        # creps_module.yaml's comment for the derivation/history.
        row = self.cfg.height_ahead_row
        col = env.height_map_size[1] // 2
        ground_ahead = env.current_frame_height_maps[:, row, col]
        # Relative to the robot's own ground-clearance reference (same convention
        # crossing_env.py uses for `clearance`), not an absolute world-Z height — more
        # physically meaningful to a linear controller reacting to obstacle rise.
        robot_ground_ref = env.positions[:, 2] - env.track_wheel_radius
        height_ahead = ground_ahead - robot_ground_ref

        obs = torch.stack([pitch, height_ahead], dim=-1)  # (N, 2)

        bad_obs = torch.isnan(obs).any(dim=-1) | torch.isinf(obs).any(dim=-1)
        if bad_obs.any():
            _log.warning(
                "NaN/Inf in observations for %d envs: %s",
                bad_obs.sum().item(), bad_obs.nonzero(as_tuple=False).squeeze(-1).tolist(),
            )
            env._obs_nan_mask |= bad_obs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": obs}

    def get_reward_components(self) -> dict[str, torch.Tensor]:
        env = self.env
        travel_dir = torch.nn.functional.normalize(
            env.target_positions[:, :2] - env.start_positions[:, :2], dim=-1
        )
        step_delta = env.positions[:, :2] - env.prev_positions[:, :2]
        # Signed forward displacement this step, in METERS (not normalized by total
        # distance, unlike e.g. mitriakov's traversal_reward) — summed over an episode
        # by train_creps.py's run_creps_episode(), this literally equals total distance
        # traveled, matching the paper's R_sw exactly.
        return {"distance_reward": (step_delta * travel_dir).sum(dim=-1)}
