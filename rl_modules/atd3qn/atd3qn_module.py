import logging
from pathlib import Path

import torch
from omni.isaac.lab.envs import VecEnvObs

from ftr_envs.utils.torch import add_noise

from rl_modules.pan_shared import PanRewardMixin
from rl_modules.rl_module import RLModule

_log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "atd3qn_module.yaml"


class ATD3QNModule(PanRewardMixin, RLModule):
    """Observations + reward for Pan et al. 2023's AT-D3QN. Requires
    sync_flipper_control: true (front/rear flipper pairs, matching the paper's 3x3
    discrete action space, Eq. 3), which halves FtrEnv.flipper_num from 4 to 2.

    Shares all terrain geometry and reward terms with ICMD3QNModule via
    rl_modules/pan_shared.py — the two papers define R_flipper (Eq. 4), R_pitch (Eq. 6)
    and R_end (Eq. 7 / Eq. 9) identically; only ICM-D3QN adds R_contact and the curiosity
    module on top. That file documents the MARV-morphology adaptations and their YAML
    gates.
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

        low, high = env.flipper_angle_bounds()
        if low is not None:
            # [0, 1] per flipper, normalised by the limits actually enforced in
            # _pre_physics_step (see the equivalent comment in icmd3qn_module).
            span = (high - low).clamp_min(1e-6)
            raw = add_noise(env.flipper_positions, env.flipper_pos_noise_std)
            flippers_norm = (raw - low) / span
        else:
            flippers_norm = torch.zeros(env.num_envs, env.flipper_num, device=env.device)

        # theta_R (Eq. 2). Normalised to [0, 1] over the paper's stated domain [-pi/3, pi/3],
        # matching the [0, 1] convention this module already uses for the flipper angles.
        # ICMD3QNModule scales its state block to [-1, 1] instead; neither paper specifies a
        # normalisation, and the two modules already differed on the flipper features, so this
        # keeps each module internally consistent rather than introducing a third convention.
        pitch_limit = torch.pi / 3
        pitch_norm = ((env.orientations_3[:, 1] / pitch_limit) + 1.0).unsqueeze(-1) * 0.5

        obs = torch.cat([height_map.view(env.num_envs, -1), flippers_norm, pitch_norm], dim=-1)  # 15+2+1=18

        # Detect robots with NaN/Inf in obs — flag them for termination next step.
        # (get_observations runs after _get_dones, so termination is one step delayed.)
        bad_obs = torch.isnan(obs).any(dim=-1) | torch.isinf(obs).any(dim=-1)
        if bad_obs.any():
            _log.warning(
                "NaN/Inf in observations for %d envs: %s",
                bad_obs.sum().item(),
                bad_obs.nonzero(as_tuple=False).squeeze(-1).tolist(),
            )
            env._obs_nan_mask |= bad_obs
        # Sanitize: NaN/Inf from physics-exploded robots must not enter the replay buffer,
        # as they corrupt gradients on the next optimisation step.
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {'policy': obs}

    def get_reward_components(self) -> dict[str, torch.Tensor]:
        """R_flipper (Eq. 4) + R_pitch (Eq. 6) + R_end (Eq. 7), each weighted by its kappa.

        The previous implementation's `pan_geom_reward` was not Eq. 4: it returned
        `-lambda_1 * theta*` (the terrain candidate angle itself) instead of
        `-lambda_1 * |theta_f1 - theta*|`, so it was unbounded and went POSITIVE whenever
        theta* was negative — which it almost always was, because that same function mixed a
        heightmap already made relative to the robot with an absolute world z. In
        logs/train_marv_atd3qn_11203270 it sat at about +0.06 to +0.09 per step, roughly
        +25 over a 300-step episode against a terminal bonus of +/-2, and training success
        fell monotonically from 0.33 to 0.19 as the agent optimised it. Both problems are
        fixed in pan_shared.candidate_flipper_angles / flipper_reward.
        """
        mcfg = self.cfg
        components: dict[str, torch.Tensor] = {}

        self.update_orientation_history()

        height_map = self.calc_scanned_height_maps()
        theta_star = self.candidate_flipper_angles(height_map)
        components["flipper_penalty"] = mcfg.kappa_flipper * self.flipper_reward(theta_star)
        components["pitch_penalty"] = mcfg.kappa_pitch * self.pitch_reward()

        # AT-D3QN's own paper has no R_contact — it is introduced by the same authors'
        # ICM-D3QN paper as the fix for "the front and rear flippers supporting the chassis
        # off the ground". Enabled by default here (kappa_3 = 0.005, ICM-D3QN Table 2) because
        # it is the only term constraining ride height, and without it AT-D3QN's clearance
        # climbs monotonically over training — see atd3qn_module.yaml for the measurements and
        # for what to watch. kappa_contact: 0.0 restores the paper's reward exactly.
        if float(mcfg.get("kappa_contact", 0.0)) != 0.0:
            components["contact_penalty"] = mcfg.kappa_contact * self.contact_reward()

        return self.apply_terminal(components)
