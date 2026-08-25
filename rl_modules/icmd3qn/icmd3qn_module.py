import logging
from pathlib import Path

import torch
from omni.isaac.lab.envs import VecEnvObs

from ftr_envs.utils.torch import add_noise

from rl_modules.pan_shared import PanRewardMixin
from rl_modules.rl_module import RLModule

_log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "icmd3qn_module.yaml"


class ICMD3QNModule(PanRewardMixin, RLModule):
    """Observations + extrinsic reward for Pan et al. 2023's ICM-D3QN. Requires
    sync_flipper_control: true (front/rear flipper pairs, matching the paper's 3x3
    discrete action space, Eq. 3) — this halves FtrEnv.flipper_num from 4 to 2, which
    get_observations() relies on for its 18-dim state vector.

    All terrain geometry and the individual reward terms live in rl_modules/pan_shared.py,
    shared with ATD3QNModule; that file also documents the MARV-morphology adaptations
    (no main tracks, rear flipper is drivetrain) and the YAML keys that gate them.
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.cfg = self.load_module_cfg(_CONFIG_PATH)
        self.init_pan_common()

    def calc_scanned_height_maps(self, base_robot_frame=True):
        return self.scanned_height_map(base_robot_frame)

    def get_observations(self) -> VecEnvObs:
        env = self.env
        height_map = self.calc_scanned_height_maps()  # (N, 15)
        pitch_limit = torch.pi / 3  # theta_R domain bound (Eq. 2)

        low, high = env.flipper_angle_bounds()
        if low is not None:
            # Normalise each flipper by the limits actually enforced in _pre_physics_step
            # rather than by the symmetric flipper_pos_max_deg. MARV's per-end limits are
            # asymmetric (front -90/+80, rear -80/+90 in the D3QN configs), so the previous
            # symmetric normaliser gave a feature that never spanned its nominal range and
            # was not centred on the flat pose.
            span = (high - low).clamp_min(1e-6)
            raw = add_noise(env.flipper_positions, env.flipper_pos_noise_std)
            flippers_norm = 2.0 * (raw - low) / span - 1.0  # [-1, 1]
        else:
            flippers_norm = torch.zeros(env.num_envs, env.flipper_num, device=env.device)
        pitch_norm = (env.orientations_3[:, 1] / pitch_limit).unsqueeze(-1)

        obs = torch.cat([height_map.view(env.num_envs, -1), flippers_norm, pitch_norm], dim=-1)  # 15+2+1=18

        bad_obs = torch.isnan(obs).any(dim=-1) | torch.isinf(obs).any(dim=-1)
        if bad_obs.any():
            _log.warning(
                "NaN/Inf in observations for %d envs: %s",
                bad_obs.sum().item(),
                bad_obs.nonzero(as_tuple=False).squeeze(-1).tolist(),
            )
            env._obs_nan_mask |= bad_obs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {'policy': obs}

    def get_reward_components(self) -> dict[str, torch.Tensor]:
        """R^e_t (Eq. 10). The intrinsic curiosity reward R^i_t (Eq. 12-13) is NOT
        included here — it needs a network forward pass over full transitions and is
        owned by the trainer (ICMD3QNCuriosityModule in icmd3qn_icm.py)."""
        mcfg = self.cfg
        components: dict[str, torch.Tensor] = {}

        self.update_orientation_history()

        height_map = self.calc_scanned_height_maps()
        theta_star = self.candidate_flipper_angles(height_map)
        components["flipper_penalty"] = mcfg.kappa_flipper * self.flipper_reward(theta_star)
        components["pitch_penalty"] = mcfg.kappa_pitch * self.pitch_reward()
        components["contact_penalty"] = mcfg.kappa_contact * self.contact_reward()

        return self.apply_terminal(components)
