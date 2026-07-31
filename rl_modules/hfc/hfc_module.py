import logging
from pathlib import Path

import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from omni.isaac.lab.envs import VecEnvObs

from ftr_envs.utils.torch import add_noise


from rl_modules.rl_module import RLModule

_log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "hfc_module.yaml"


class HFCModule(RLModule):
    """Observations + reward for the Hybrid Flipper Controller (HFC), adapted from
    Azayev & Zimmermann 2022, "Autonomous State-Based Flipper Control for Articulated
    Tracked Robots in Urban Environments". Requires sync_flipper_control: false (4
    independent flippers) — Eq. 7's roll stabilization and Eq. 8's escape maneuvers both
    act asymmetrically on left/right or front/rear flippers (see hfc_policy.py).

    The paper trains its state-transition classifier via imitation learning from human
    demonstrations (Eq. 2), which this project has no data for. Per the user's explicit
    decision, this module instead trains the *entire* controller (state classifier +
    roll-stabilization gains + escape-maneuver magnitudes, all in hfc_policy.py) via PPO
    (train_ftr.py), using marv_rl's own reward function as the training signal —
    get_reward_components() below is therefore an intentional line-for-line copy of
    MarvRLModule.get_reward_components() (rl_modules/marv_rl/marv_rl_module.py), not an
    import: every RLModule in this project owns its reward independently (no cross-module
    imports), and this module is no exception even though the formula is identical.
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.cfg = OmegaConf.load(_CONFIG_PATH)

    def calc_scanned_height_maps(self, base_robot_frame=True):
        """H: N=15 terrain-height averages, matching atd3qn/icmd3qn/the old hfc_module."""
        env = self.env
        h, w = 45, 21
        shaped_map = torch.reshape(env.current_frame_height_maps, (-1, 1, h, w))
        if base_robot_frame:
            shaped_map = shaped_map - einops.repeat(
                env.positions[:, 2] - env.track_wheel_radius, 'n -> n c rh rw', c=1, rh=h, rw=w
            )
        height_maps = shaped_map.squeeze(1)
        height_maps = einops.reduce(height_maps, "n (h k) w -> n h", reduction="mean", k=3)
        return add_noise(height_maps, std=env.hmap_noise_std)

    def get_observations(self) -> VecEnvObs:
        env = self.env
        height_map = self.calc_scanned_height_maps()  # (N, 15)
        flippers_raw = add_noise(env.flipper_positions, env.flipper_pos_noise_std)  # (N, 4): [FL,FR,RL,RR], raw rad
        roll_raw = env.orientations_3[:, 0].unsqueeze(-1)  # (N, 1), raw rad
        roll_rate_raw = env.robot_ang_velocities[:, 0].unsqueeze(-1)  # (N, 1), raw rad/s
        pitch_raw = env.orientations_3[:, 1].unsqueeze(-1)  # (N, 1), raw rad
        fwd_vel_raw = env.robot_lin_velocities[:, 0].unsqueeze(-1)  # (N, 1), raw m/s, body frame
        reset_flag = (env.episode_length_buf == 0).float().unsqueeze(-1)  # (N, 1)

        obs = torch.cat(
            [height_map.view(env.num_envs, -1), flippers_raw, roll_raw, roll_rate_raw, pitch_raw, fwd_vel_raw, reset_flag],
            dim=-1,
        )  # 15+4+1+1+1+1+1=24

        bad_obs = torch.isnan(obs).any(dim=-1) | torch.isinf(obs).any(dim=-1)
        if bad_obs.any():
            _log.warning("NaN/Inf in observations for %d envs: %s", bad_obs.sum().item(), bad_obs.nonzero(as_tuple=False).squeeze(-1).tolist())
            env._obs_nan_mask |= bad_obs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {'policy': obs}

    def get_reward_components(self) -> dict[str, torch.Tensor]:
        """Intentional line-for-line copy of MarvRLModule.get_reward_components() — see
        this class's docstring for why. Keep in sync with rl_modules/marv_rl/marv_rl_module.py
        if that formula changes and you want HFC's training signal to track it.
        """
        env = self.env
        cfg = env.cfg
        components: dict[str, torch.Tensor] = {}

        # Potential-based shaping: coef * (gamma * phi(s') - phi(s)), phi = -dist
        curr_dist = (env.target_positions[:, :2] - env.positions[:, :2]).norm(dim=-1)
        prev_dist = (env.target_positions[:, :2] - env.prev_positions[:, :2]).norm(dim=-1)
        components["shaping"] = cfg.shaping_coef * (prev_dist - cfg.shaping_gamma * curr_dist)

        # Joint-velocity variance penalty
        if cfg.joint_vel_variance_coef is not None:
            components["joint_vel_var_penalty"] = -cfg.joint_vel_variance_coef * env.actions[:, 2:].abs().var(dim=-1)

        # Joints not horizontal penalty
        if cfg.joint_ang_from_flat_coef is not None:
            limit = torch.deg2rad(torch.tensor(float(cfg.flipper_pos_max_deg))).item()
            flipper_pos = torch.abs(env._robot.data.joint_pos[:, env._flipper_joint_ids])  # (N, 4)
            norm_dif = flipper_pos / limit  # [N, 4], in [0, 1]
            components["joint_ang_from_flat_penalty"] = -cfg.joint_ang_from_flat_coef * norm_dif.mean(dim=-1)

        # Joint-angle variance penalty
        if cfg.joint_angle_variance_coef is not None:
            limit = torch.deg2rad(torch.tensor(float(cfg.flipper_pos_max_deg))).item()
            diffs = ((env.flipper_positions[:, 0] - env.flipper_positions[:, 1]).abs() / limit
                     + (env.flipper_positions[:, 2] - env.flipper_positions[:, 3]).abs() / limit) / 2
            components["joint_angle_var_penalty"] = -cfg.joint_angle_variance_coef * diffs

        # Flipper-ground contact reward (torque-based)
        if cfg.flipper_contact_coef is not None:
            contact_signal = env.flipper_torques / cfg.flipper_contact_effort_limit  # normalize to [0, 1]
            mean_signal = torch.clamp(contact_signal.mean(dim=-1) - 0.5, min=0.0)
            min_signal = contact_signal.min(dim=-1).values
            alpha = 1
            components["flipper_contact"] = cfg.flipper_contact_coef * (alpha * mean_signal + (1 - alpha) * min_signal)

        # Roll penalty
        if cfg.roll_coef is not None:
            roll_norm = 4 * (env.orientations_3[:, 0].abs() - np.deg2rad(15)) / torch.pi
            components["roll_penalty"] = -cfg.roll_coef * torch.clamp(roll_norm, max=1, min=0)

        # Roll-rate penalty
        if cfg.roll_rate_coef is not None:
            components["roll_rate_penalty"] = -cfg.roll_rate_coef * env.robot_ang_velocities[:, 0].abs() / np.pi

        # Pitch penalty
        if cfg.pitch_coef is not None:
            pitch_norm = 4 * (env.orientations_3[:, 1].abs() - np.deg2rad(7.5)) / torch.pi
            components["pitch_penalty"] = -cfg.pitch_coef * torch.clamp(pitch_norm, max=1, min=0)

        # Pitch-rate penalty
        if cfg.pitch_rate_coef is not None:
            components["pitch_rate_penalty"] = -cfg.pitch_rate_coef * env.robot_ang_velocities[:, 1].abs() / np.pi

        # Shock penalty
        if cfg.shock_coef is not None:
            components["shock_penalty"] = -cfg.shock_coef * env.shock_norm

        # Clearance penalty
        if cfg.clearance_coef is not None:
            components["clearance"] = -cfg.clearance_coef * (1 / (1 + torch.exp(-((env.clearance - 0.2) / 0.02))))

        # Action bonus
        if cfg.action_bonus_coef is not None:
            v_norm = (torch.Tensor(env.actions[:, 0]).pow(3) * cfg.lin_action_ratio
                      + torch.Tensor(env.robot_lin_velocities[:, 0] / env.track_vel_max).pow(3) * (1 - cfg.lin_action_ratio))
            components["action_bonus"] = cfg.action_bonus_coef * torch.clamp(v_norm, max=1.0, min=-1.0)

        # Flipper action bonus
        if cfg.flipper_action_bonus_coef is not None:
            flipper_norm = (env.actions[:, 2:]).abs().mean(dim=-1)
            components["flipper_action_bonus"] = cfg.flipper_action_bonus_coef * torch.clamp(flipper_norm.pow(3), max=1.0)

        # Legacy flipper_training-style reward variants
        if cfg.legacy_joint_vel_variance_coef is not None:
            components["legacy_joint_vel_var_penalty"] = -cfg.legacy_joint_vel_variance_coef * env.actions[:, 2:].abs().var(dim=-1)

        if cfg.legacy_joint_angle_variance_coef is not None:
            components["legacy_joint_angle_var_penalty"] = -cfg.legacy_joint_angle_variance_coef * env.flipper_positions.abs().var(dim=-1)

        if cfg.legacy_track_vel_variance_coef is not None:
            components["legacy_track_vel_var_penalty"] = -cfg.legacy_track_vel_variance_coef * env.actions[:, :2].abs().var(dim=-1)

        if cfg.legacy_roll_rate_coef is not None:
            components["legacy_roll_rate_penalty"] = -cfg.legacy_roll_rate_coef * env.robot_ang_velocities[:, 0].abs() / np.pi

        if cfg.legacy_pitch_rate_coef is not None:
            components["legacy_pitch_rate_penalty"] = -cfg.legacy_pitch_rate_coef * env.robot_ang_velocities[:, 1].abs() / np.pi

        # ------------------------------------------------------------------
        # Step penalty & terminal masking/bonus — zero every component on failure/
        # explosion, then add the terminal bonus (goal/fail/timeout — never zeroed).
        # ------------------------------------------------------------------
        terminal = env._explosion_mask | env._fail_mask
        components["step_penalty"] = torch.full((env.num_envs,), cfg.step_penalty, device=env.device)
        for name, comp in components.items():
            components[name] = torch.where(terminal, torch.zeros_like(comp), comp)

        components["terminal_bonus"] = (
            env._success_mask.float() * cfg.goal_reached_reward
            + env._fail_mask.float() * cfg.failed_reward
            + env._timeout_mask.float() * (cfg.timeout_penalty or 0.0)
        )

        return components
