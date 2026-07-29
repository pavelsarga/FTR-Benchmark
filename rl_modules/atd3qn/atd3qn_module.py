import logging
from pathlib import Path

import einops
import torch
from omegaconf import OmegaConf
from omni.isaac.lab.envs import VecEnvObs

from ftr_envs.utils.torch import add_noise


from rl_modules.rl_module import RLModule

_log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "atd3qn_module.yaml"


class ATD3QNModule(RLModule):
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
        env = self.env
        h, w = 45, 21
        shaped_map = torch.reshape(env.current_frame_height_maps, (-1, 1, h, w))
        if base_robot_frame:
            shaped_map = shaped_map - einops.repeat(
                env.positions[:, 2] - env.track_wheel_radius, 'n -> n c rh rw', c=1, rh=h, rw=w
            )
        height_maps = shaped_map.squeeze(1)
        height_maps = einops.reduce(height_maps, "n (h k) w -> n h", reduction="mean", k=3)  # -> (N, 15)
        return add_noise(height_maps, std=env.hmap_noise_std)
    
    def get_observations(self) -> VecEnvObs:
        env = self.env
        height_map = self.calc_scanned_height_maps()
        joint_limit = torch.deg2rad(torch.tensor(float(env.cfg.flipper_pos_max_deg))).item() if env.cfg.flipper_pos_max_deg is not None else None

        obs = torch.cat([
            (height_map).view(env.num_envs, -1),
            (add_noise(env.flipper_positions, env.flipper_pos_noise_std) + joint_limit) / (2*joint_limit) if joint_limit is not None else torch.zeros(env.num_envs, env.flipper_num, device=env.device),  # 2    joints [0,1] or 0 if locked (sync_flipper_control halves flipper_num to 2)
        ], dim=-1)
        # total: 15 + 2 = 17

        # Detect robots with NaN/Inf in obs — flag them for termination next step.
        # (get_observations runs after _get_dones, so termination is one step delayed.)
        bad_obs = torch.isnan(obs).any(dim=-1) | torch.isinf(obs).any(dim=-1)
        if bad_obs.any():
            _log.warning("NaN/Inf in observations for %d envs: %s", bad_obs.sum().item(), bad_obs.nonzero(as_tuple=False).squeeze(-1).tolist())
            env._obs_nan_mask |= bad_obs
        # Sanitize: NaN/Inf from physics-exploded robots must not enter the replay buffer,
        # as they corrupt gradients on the next optimisation step.
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            'policy': obs,
        }

    def _pitch_reward(self) -> torch.Tensor:
        """Penalises fast/large chassis pitch changes, using self.cfg's (module-yaml)
        thresholds."""
        env = self.env
        cfg = self.cfg
        if self.orientations_3_history.size(1) <= 1:
            return torch.zeros(env.num_envs, device=env.device)

        # history[:, -1] is the current step (just appended), history[:, -2] the previous
        # one — reset envs are reseeded with the fresh value in _update_orientation_history,
        # so this is correctly 0 right after a reset.
        pitch_delta = (self.orientations_3_history[:, -1, 1] - self.orientations_3_history[:, -2, 1]).abs()
        step_deltas = self.orientations_3_history[:, 1:, 1] - self.orientations_3_history[:, :-1, 1]
        mean_pitch_delta = step_deltas.abs().mean(dim=1)

        violation = (pitch_delta > cfg.pitch_delta_threshold) | (mean_pitch_delta > 1 / cfg.mean_pitch_delta_threshold)
        return torch.where(violation, -torch.ones_like(mean_pitch_delta), -cfg.mean_pitch_delta_threshold * mean_pitch_delta)

    def _pan_geom_reward(self) -> torch.Tensor:
        """Penalises the front flipper angle deviating from a terrain-derived candidate
        angle, using self.cfg's (module-yaml) geometry/threshold."""
        env = self.env
        cfg = self.cfg
        hmap = self.calc_scanned_height_maps()
        x, z = env.positions[:, 0], env.positions[:, 2]
        pitch = self.orientations_3_history[:, -1, 1]
        x_wheel = torch.cos(pitch) * cfg.robot_wheel_base_length / 2 + x
        z_wheel = torch.sin(pitch) * cfg.robot_wheel_base_length / 2 + z

        max_z, max_idx = hmap[:, :5].max(dim=1)
        max_x = (5 - max_idx) * 0.15 + x_wheel
        angle = torch.atan2((max_z + cfg.robot_wheel_diam) - z_wheel, max_x - x_wheel)

        flipper_angle = env.flipper_positions[:, 0:2].mean(dim=1)
        violation = (flipper_angle - angle).abs() > 1 / cfg.flipper_angle_threshold + torch.pi / 36  # +/- pi/36, per the paper
        return torch.where(violation, -torch.ones_like(angle), -cfg.flipper_angle_threshold * angle)

    def get_reward_components(self) -> dict[str, torch.Tensor]:
        """Individual reward terms and terminal masking/bonus — this module owns its
        full reward. All logging/state-monitoring lives in CrossingEnv._get_rewards —
        not here.
        """
        env = self.env
        cfg = env.cfg
        components: dict[str, torch.Tensor] = {}

        self._update_orientation_history()

        components["pitch_penalty"] = self._pitch_reward()
        components["pan_geom_reward"] = self._pan_geom_reward()

        # Terminal masking/bonus — zero every component on failure/explosion, then add
        # the terminal bonus (goal/fail/timeout — never zeroed).
        terminal = env._explosion_mask | env._fail_mask
        for name, comp in components.items():
            components[name] = torch.where(terminal, torch.zeros_like(comp), comp)

        components["terminal_bonus"] = (
            env._success_mask.float() * cfg.goal_reached_reward
            + env._fail_mask.float() * cfg.failed_reward
            + env._timeout_mask.float() * (cfg.timeout_penalty if cfg.timeout_penalty is not None else 0.0)
        )

        return components
