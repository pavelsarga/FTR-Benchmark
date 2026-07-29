import logging
from pathlib import Path

import einops
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
    Tracked Robots in Urban Environments". Requires sync_flipper_control: true (front/rear
    flipper pairs) — the reference repo's own FLIPPERS_<STATE> templates that HFCPolicy's
    pose targets are derived from are themselves synced per axle.

    The paper trains its state classifier via imitation learning from human demonstrations
    (Eq. 2); this module instead provides a reward so the pose-selection Q-network
    (HFCEncoder, 7 poses) can be trained via D3QN — the paper defines no RL reward at all,
    so get_reward_components() below is this project's own adaptation, not from the paper.
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.cfg = OmegaConf.load(_CONFIG_PATH)
        self.orientations_3_history = torch.zeros(
            env.num_envs, self.cfg.orientation_history_k, 3, device=env.device
        )

    def _update_orientation_history(self):
        """Roll in env.orientations_3 as the newest entry; re-seed envs that just reset."""
        env = self.env
        self.orientations_3_history = torch.cat(
            [self.orientations_3_history[:, 1:], env.orientations_3.unsqueeze(1)], dim=1
        )
        fresh = env.episode_length_buf == 0
        if fresh.any():
            k = self.cfg.orientation_history_k
            self.orientations_3_history[fresh] = env.orientations_3[fresh].unsqueeze(1).expand(-1, k, -1)

    def calc_scanned_height_maps(self, base_robot_frame=True):
        """H: N=15 terrain-height averages, matching atd3qn/icmd3qn."""
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
        flippers_raw = add_noise(env.flipper_positions, env.flipper_pos_noise_std)  # (N, 2): [front, rear], raw rad
        pitch_raw = env.orientations_3[:, 1].unsqueeze(-1)  # (N, 1), raw rad

        obs = torch.cat([height_map.view(env.num_envs, -1), flippers_raw, pitch_raw], dim=-1)  # 15+2+1=18

        bad_obs = torch.isnan(obs).any(dim=-1) | torch.isinf(obs).any(dim=-1)
        if bad_obs.any():
            _log.warning("NaN/Inf in observations for %d envs: %s", bad_obs.sum().item(), bad_obs.nonzero(as_tuple=False).squeeze(-1).tolist())
            env._obs_nan_mask |= bad_obs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {'policy': obs}

    def _pitch_reward(self) -> torch.Tensor:
        """Penalises fast/large chassis pitch changes — same formula as ATD3QNModule."""
        env = self.env
        cfg = self.cfg
        if self.orientations_3_history.size(1) <= 1:
            return torch.zeros(env.num_envs, device=env.device)

        pitch_delta = (self.orientations_3_history[:, -1, 1] - self.orientations_3_history[:, -2, 1]).abs()
        step_deltas = self.orientations_3_history[:, 1:, 1] - self.orientations_3_history[:, :-1, 1]
        mean_pitch_delta = step_deltas.abs().mean(dim=1)

        violation = (pitch_delta > cfg.pitch_delta_threshold) | (mean_pitch_delta > 1 / cfg.mean_pitch_delta_threshold)
        return torch.where(violation, -torch.ones_like(mean_pitch_delta), -cfg.mean_pitch_delta_threshold * mean_pitch_delta)

    def _movement_reward(self) -> torch.Tensor:
        """Penalises the magnitude of the commanded per-step flipper delta. HFCPolicy has
        no persistent notion of 'current pose' on the env side to compare against (see
        hfc_policy.py's docstring on why this module is stateless), so this is a proxy for
        the paper's own 'fewer state changes' smoothness criterion (Section VI.A): large
        commands mean the policy is actively driving the flippers toward a different
        target, small commands mean it's already settled near the current one.
        """
        env = self.env
        flipper_command = env.actions[:, 2:]  # (N, 2): [-1, 1], the last commanded delta
        return -flipper_command.abs().mean(dim=-1)

    def get_reward_components(self) -> dict[str, torch.Tensor]:
        env = self.env
        cfg = env.cfg
        mcfg = self.cfg
        components: dict[str, torch.Tensor] = {}

        self._update_orientation_history()

        components["pitch_penalty"] = mcfg.kappa_pitch * self._pitch_reward()
        components["movement_penalty"] = mcfg.kappa_movement * self._movement_reward()

        terminal = env._explosion_mask | env._fail_mask
        for name, comp in components.items():
            components[name] = torch.where(terminal, torch.zeros_like(comp), comp)

        components["terminal_bonus"] = (
            env._success_mask.float() * cfg.goal_reached_reward
            + env._fail_mask.float() * cfg.failed_reward
            + env._timeout_mask.float() * (cfg.timeout_penalty if cfg.timeout_penalty is not None else 0.0)
        )

        return components
