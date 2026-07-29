import logging
from pathlib import Path

import einops
import torch
from omegaconf import OmegaConf
from omni.isaac.lab.envs import VecEnvObs

from ftr_envs.utils.torch import add_noise


from rl_modules.rl_module import RLModule

_log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "icmd3qn_module.yaml"


class ICMD3QNModule(RLModule):
    """Observations + extrinsic reward for Pan et al. 2023's ICM-D3QN. Requires
    sync_flipper_control: true (front/rear flipper pairs, matching the paper's 3x3
    discrete action space, Eq. 3) — this halves FtrEnv.flipper_num from 4 to 2, which
    get_observations() relies on for its 18-dim state vector.
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
        """H (Eq. 1): N=15 terrain-height averages. Reduces both the 45->15 row groups and
        the full 21-column width down to (N, 15) in one step."""
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
        joint_limit = torch.deg2rad(torch.tensor(float(env.cfg.flipper_pos_max_deg))).item() if env.cfg.flipper_pos_max_deg is not None else None
        pitch_limit = torch.pi / 3  # theta_R domain bound (Eq. 2)

        if joint_limit is not None:
            flippers_norm = add_noise(env.flipper_positions, env.flipper_pos_noise_std) / joint_limit  # [front, rear]
        else:
            flippers_norm = torch.zeros(env.num_envs, env.flipper_num, device=env.device)
        pitch_norm = (env.orientations_3[:, 1] / pitch_limit).unsqueeze(-1)

        obs = torch.cat([height_map.view(env.num_envs, -1), flippers_norm, pitch_norm], dim=-1)  # 15+2+1=18

        bad_obs = torch.isnan(obs).any(dim=-1) | torch.isinf(obs).any(dim=-1)
        if bad_obs.any():
            _log.warning("NaN/Inf in observations for %d envs: %s", bad_obs.sum().item(), bad_obs.nonzero(as_tuple=False).squeeze(-1).tolist())
            env._obs_nan_mask |= bad_obs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {'policy': obs}

    def _flipper_candidate_angles(self, height_map: torch.Tensor) -> torch.Tensor:
        """Candidate front flipper angle theta_f1* (Fig. 4): max-angle vector from the
        front hinge to a height-map bin ahead of it. Discretised approximation of Eq. 4's
        point-cloud geometry; assumes bin index increases with +x (flip
        `hmap_x_increasing` in the yaml if shaping looks reversed)."""
        cfg, env = self.cfg, self.env
        n = height_map.shape[-1]
        bin_width = (env.height_map_length[0] / env.height_map_size[0]) * 3
        x = (torch.arange(n, device=height_map.device, dtype=height_map.dtype) - (n - 1) / 2) * bin_width
        if not cfg.hmap_x_increasing:
            x = -x
        hinge, half = cfg.robot_wheel_base_length / 2.0, n // 2

        angles = torch.atan2(height_map[:, half:n] + cfg.robot_wheel_diam / 2.0, x[half:n] - hinge)
        return angles.max(dim=-1).values

    def _flipper_reward(self, theta_star: torch.Tensor) -> torch.Tensor:
        """R_flipper (Eq. 4), front flipper only — the paper does not define this term
        for the rear flipper."""
        lam = self.cfg.lambda_flipper
        theta = self.env.flipper_positions[:, 0]  # front
        delta = torch.minimum((theta - theta_star - torch.pi / 36).abs(), (theta - theta_star + torch.pi / 36).abs())
        return torch.where(delta > 1.0 / lam, -torch.ones_like(delta), -lam * delta)

    def _pitch_reward(self) -> torch.Tensor:
        """R_pitch (Eq. 6): smoothness reward on the chassis pitch angle theta_R."""
        env = self.env
        pitch_hist = self.orientations_3_history[:, :, 1]
        if pitch_hist.shape[1] <= 1:
            return torch.zeros(env.num_envs, device=env.device)

        delta_abs = pitch_hist[:, -1].abs() - pitch_hist[:, -2].abs()  # Delta|theta_R(t)|
        mean_delta_k = (pitch_hist[:, 1:] - pitch_hist[:, :-1]).abs().mean(dim=1)  # Delta theta_R^k(t)

        lam = self.cfg.lambda_pitch
        danger = ((pitch_hist[:, -1].abs() > torch.pi / 4) & (delta_abs > 0)) | (mean_delta_k > 1.0 / lam)
        return torch.where(danger, -torch.ones_like(mean_delta_k), -lam * mean_delta_k)

    def _contact_reward(self) -> torch.Tensor:
        """R_contact (Eq. 7-8) approximation: penalises the chassis lifting off the
        ground (via ground clearance) instead of the paper's literal per-point contact
        classification, which this simulator doesn't expose."""
        env = self.env
        ungrounded = env.clearance.abs() > self.cfg.contact_clearance_threshold
        return torch.where(ungrounded, -torch.ones_like(env.clearance), torch.zeros_like(env.clearance))

    def get_reward_components(self) -> dict[str, torch.Tensor]:
        """R^e_t (Eq. 10). The intrinsic curiosity reward R^i_t (Eq. 12-13) is NOT
        included here — it needs a network forward pass over full transitions and is
        owned by the trainer (ICMD3QNCuriosityModule in icmd3qn_icm.py)."""
        env = self.env
        cfg = env.cfg
        mcfg = self.cfg
        components: dict[str, torch.Tensor] = {}

        self._update_orientation_history()

        height_map = self.calc_scanned_height_maps()
        theta_star = self._flipper_candidate_angles(height_map)
        components["flipper_penalty"] = mcfg.kappa_flipper * self._flipper_reward(theta_star)
        components["pitch_penalty"] = mcfg.kappa_pitch * self._pitch_reward()
        components["contact_penalty"] = mcfg.kappa_contact * self._contact_reward()

        # Terminal masking/bonus (Eq. 9) — zero every component on failure/explosion, add
        # the terminal bonus (goal/fail/timeout, never zeroed).
        terminal = env._explosion_mask | env._fail_mask
        for name, comp in components.items():
            components[name] = torch.where(terminal, torch.zeros_like(comp), comp)

        components["terminal_bonus"] = (
            env._success_mask.float() * cfg.goal_reached_reward
            + env._fail_mask.float() * cfg.failed_reward
            + env._timeout_mask.float() * (cfg.timeout_penalty if cfg.timeout_penalty is not None else 0.0)
        )

        return components
