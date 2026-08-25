import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from omni.isaac.lab.envs import VecEnvObs

from rl_modules.rl_module import RLModule

_log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "mitriakov_module.yaml"


class MitriakovModule(RLModule):
    """Observations + reward for Mitriakov et al. 2021, "Reinforcement Learning Based,
    Staircase Negotiation Learning: Simulation and Transfer to Reality for Articulated
    Tracked Robots" (IEEE RA Magazine). Unlike every other module in this project
    (marv_rl/atd3qn/icmd3qn/hfc), the observation has NO terrain heightmap at all — the
    paper's state vector (Eq. 2) is purely the robot's geometric relationship to the next/
    previous staircase step edge, plus velocity and flipper angles. Trained with PPO via
    this project's existing train_ftr.py (no dedicated trainer script needed) plus a
    dedicated policy (MitriakovPolicyConfig, mitriakov_policy.py — self-contained rather
    than the generic MLPPolicyConfig, see that file's docstring for why), matching the
    paper's own choice of PPO. See configs/marv_config_mitriakov.yaml.

    Requires sync_flipper_control: true (front/rear flipper pairs, matching the paper's
    ψ^front/ψ^rear) and flipper_control_mode: position (the paper's action is an absolute
    flipper-angle target, not an increment/velocity — see hfc's use of the same mode).

    The paper trains SEPARATE ascent/descent policies with different reward types
    (Table 2's *-COG vs *-Ang task variants) and different flipper angle limits. This
    project's "mitriakov_stairs" terrain instead mixes both ascending and descending
    tiles across envs in one training run (stairs_ascent_descent, alternating by tile
    repeat) — so get_reward_components() selects Eq. 5's cog_penalty vs Eq. 6's
    pitch_velocity_penalty PER ENV via _ascending_mask() (target_z vs start_z), not a
    fixed config choice. The asymmetric flipper angle limits are also selected per env:
    env_cfg_overrides.marv_flipper_*_deg sets a single, looser outer envelope (same for
    every env), while MitriakovPolicyConfig's mitriakov_limit_front_deg/
    mitriakov_limit_rear_deg (mitriakov_policy.py's MitriakovFlipperBounds) applies a
    tighter per-env sub-range using this same is_ascending signal, matching each env's
    own detected direction.

    Step-edge geometry (big_/small_ascent_/descent_step_edge_offsets_m/heights_m in
    mitriakov_module.yaml) is analytic, derived from the "mitriakov_stairs" terrain this
    config points at (see that yaml's comments for the exact derivation). That terrain has
    TWO parallel rows/lanes — Table 3's "big" (5-step) and "small" (3-step) staircases —
    so besides _ascending_mask(), _row_mask() (keyed off spawn Y, which row places envs
    at) selects the matching one of 4 geometry tables per env in _progress_and_edges().
    This mirrors how the paper itself trained in simulation using known Gazebo staircase
    geometry directly, not real perception (real camera/marker-based perception was only
    used for their real-robot deployment, not simulation training).

    get_reward_components() also adds an optional clearance_reward term (formula ported
    from marv_rl_module.py's clearance_penalty, sign flipped, gated by
    env_cfg_overrides.clearance_coef, off by default), beyond the paper's own Eq. 5/6
    penalties — see that term's inline comment for why it's needed on MARV specifically
    (no continuous main track, unlike the paper's Absolem/Jaguar) and why it rewards
    rather than penalises clearance here.
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.cfg = OmegaConf.load(_CONFIG_PATH)

    def _ascending_mask(self) -> torch.Tensor:
        """True for envs currently on an ascending tile (goal higher than spawn), False
        for descending. With repeats > 1 in the terrain config, stairs_ascent_descent
        alternates the two by tile repeat, so both are mixed across envs in one training
        run — this is one of the two per-env signals both the observation geometry (see
        _progress_and_edges) and the reward direction (see get_reward_components) key off
        (the other being _row_mask())."""
        env = self.env
        return env.target_positions[:, 2] > env.start_positions[:, 2]

    def _row_mask(self) -> torch.Tensor:
        """True for envs on row 1 ("small" staircase, Table 3), False for row 0 ("big").
        CourseGrid places rows at different Y offsets (grid.py's iter_tiles: y_center =
        (row_index - (n_rows-1)/2) * tile.depth), so with "mitriakov_stairs"'s 2 rows,
        row 0 sits at Y=-1.5 and row 1 at Y=+1.5 — sign of the env's own (fixed-at-spawn)
        Y position tells us which row's geometry table to use. Uses start_positions
        (fixed for the episode) rather than the live position, same reasoning as
        _ascending_mask() using start/target rather than anything that could drift."""
        return self.env.start_positions[:, 1] > 0

    def _progress_and_edges(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-env signed progress along the travel axis (start -> target) and height
        (relative to spawn), plus the next/previous step-edge offset+height and the
        first/last edge of this env's own staircase — resolved per env from FOUR possible
        geometry tables (big/small x ascent/descent, mitriakov_module.yaml) via
        _row_mask()/_ascending_mask(), since a single fixed table doesn't apply to every
        env once both staircase sizes and both directions are mixed across envs.
        """
        env = self.env
        cfg = self.cfg
        dtype, device = env.positions.dtype, env.device

        travel_dir = torch.sign(env.target_positions[:, 0] - env.start_positions[:, 0])
        travel_dir = torch.where(travel_dir == 0, torch.ones_like(travel_dir), travel_dir)
        progress_x = (env.positions[:, 0] - env.start_positions[:, 0]) * travel_dir  # (N,)
        height_rel = env.positions[:, 2] - env.start_positions[:, 2]  # (N,)

        n = env.positions.shape[0]
        ascending = self._ascending_mask().unsqueeze(-1)  # (N, 1)

        def direction_tables(prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
            asc_e = torch.tensor(cfg[f"{prefix}_ascent_step_edge_offsets_m"], device=device, dtype=dtype)
            desc_e = torch.tensor(cfg[f"{prefix}_descent_step_edge_offsets_m"], device=device, dtype=dtype)
            asc_h = torch.tensor(cfg[f"{prefix}_ascent_step_edge_heights_m"], device=device, dtype=dtype)
            desc_h = torch.tensor(cfg[f"{prefix}_descent_step_edge_heights_m"], device=device, dtype=dtype)
            edges = torch.where(ascending, asc_e.unsqueeze(0).expand(n, -1), desc_e.unsqueeze(0).expand(n, -1))
            heights = torch.where(ascending, asc_h.unsqueeze(0).expand(n, -1), desc_h.unsqueeze(0).expand(n, -1))
            return edges, heights

        def resolve(edges: torch.Tensor, heights: torch.Tensor):
            next_idx = torch.clamp(torch.searchsorted(edges, progress_x.unsqueeze(-1)).squeeze(-1), max=edges.shape[-1] - 1)
            prev_idx = torch.clamp(next_idx - 1, min=0)
            next_edge = edges.gather(-1, next_idx.unsqueeze(-1)).squeeze(-1)
            next_height = heights.gather(-1, next_idx.unsqueeze(-1)).squeeze(-1)
            prev_edge = edges.gather(-1, prev_idx.unsqueeze(-1)).squeeze(-1)
            prev_height = heights.gather(-1, prev_idx.unsqueeze(-1)).squeeze(-1)
            return next_edge, next_height, prev_edge, prev_height, edges[:, 0], edges[:, -1]

        big_edges, big_heights = direction_tables("big")
        small_edges, small_heights = direction_tables("small")
        big_result = resolve(big_edges, big_heights)
        small_result = resolve(small_edges, small_heights)

        is_small = self._row_mask()
        next_edge, next_height, prev_edge, prev_height, first_edge, last_edge = (
            torch.where(is_small, s, b) for s, b in zip(small_result, big_result)
        )
        return progress_x, height_rel, next_edge, next_height, prev_edge, prev_height, first_edge, last_edge

    def get_observations(self) -> VecEnvObs:
        env = self.env
        progress_x, height_rel, next_edge, next_height, prev_edge, prev_height, _, _ = self._progress_and_edges()

        p_x_front = next_edge - progress_x     # next edge ahead of the robot
        p_y_front = next_height - height_rel
        p_x_rear = progress_x - prev_edge       # previous edge behind the robot
        p_y_rear = height_rel - prev_height

        v_s = env.robot_lin_velocities[:, 0]                     # forward body-frame velocity
        psi_front, psi_rear = env.flipper_positions[:, 0], env.flipper_positions[:, 1]  # raw radians
        is_ascending = self._ascending_mask().float()  # project-specific 8th feature — see MitriakovObservation docstring

        obs = torch.stack([p_x_front, p_y_front, p_x_rear, p_y_rear, v_s, psi_front, psi_rear, is_ascending], dim=-1)  # (N, 8)

        bad_obs = torch.isnan(obs).any(dim=-1) | torch.isinf(obs).any(dim=-1)
        if bad_obs.any():
            _log.warning("NaN/Inf in observations for %d envs: %s", bad_obs.sum().item(), bad_obs.nonzero(as_tuple=False).squeeze(-1).tolist())
            env._obs_nan_mask |= bad_obs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {'policy': obs}

    def _on_staircase(self) -> torch.Tensor:
        """True while the robot's along-travel progress is within the staircase's own
        span (Eq. 5/6: penalties apply only "when a robot's COG projection is located
        between stair step edges" / "between the staircase nosing limits")."""
        progress_x, _, _, _, _, _, first_edge, last_edge = self._progress_and_edges()
        return (progress_x >= first_edge) & (progress_x <= last_edge)

    def _cog_penalty(self) -> torch.Tensor:
        """R_t^D (Eq. 4-5): COG-deviation penalty (the paper's ascent reward — selected
        per env in get_reward_components() by _ascending_mask(), not a fixed choice).
        D_t = sqrt(d^2 + h^2), the
        distance between the geometric-center ground projection O and the COG's (x, y)-axis
        projections (Fig. 2b). Without the paper's optional arm (this project's 3-DoF
        variant, matching Absolem-3DoF), the COG can't be independently repositioned — its
        deviation from O is driven entirely by chassis tilt, so D_t is approximated as the
        tilt-induced horizontal shift of a fixed-mass rigid chassis: half the wheelbase
        times sin(pitch)/sin(roll).
        """
        env = self.env
        cfg = self.cfg
        half_base = cfg.robot_wheel_base_length / 2.0
        pitch, roll = env.orientations_3[:, 1], env.orientations_3[:, 0]
        d = half_base * torch.sin(pitch)
        h = half_base * torch.sin(roll)
        D_t = torch.sqrt(d ** 2 + h ** 2)
        on_stairs = self._on_staircase()
        return torch.where(on_stairs, -cfg.K_D * D_t, torch.zeros_like(D_t))

    def _pitch_velocity_penalty(self) -> torch.Tensor:
        """R_t^W (Eq. 6): platform-shaking penalty, proportional to pitch angular
        velocity W_t (the paper's descent reward — selected per env in
        get_reward_components() by _ascending_mask(), not a fixed choice)."""
        env = self.env
        cfg = self.cfg
        W_t = env.robot_ang_velocities[:, 1].abs()
        on_stairs = self._on_staircase()
        return torch.where(on_stairs, -cfg.K_W * W_t, torch.zeros_like(W_t))

    def get_reward_components(self) -> dict[str, torch.Tensor]:
        env = self.env
        cfg = env.cfg
        components: dict[str, torch.Tensor] = {}

        # R^tr(tau) (Eq. 3): positive reward proportional to progress made this step,
        # as a fraction of the total start-to-finish distance.
        curr_dist = (env.target_positions[:, :2] - env.positions[:, :2]).norm(dim=-1)
        prev_dist = (env.target_positions[:, :2] - env.prev_positions[:, :2]).norm(dim=-1)
        d_max = (env.target_positions[:, :2] - env.start_positions[:, :2]).norm(dim=-1).clamp(min=1e-3)
        components["traversal_reward"] = (prev_dist - curr_dist) / d_max

        # Eq. 4: R_t = x_t/D_max + r_t, where r_t is one of the two negative penalties
        # below. The paper trains separate ascent/descent policies (Table 2's *-COG vs
        # *-Ang task variants) since each robot only ever sees one direction; this
        # project's terrain mixes both tile types across envs in one training run (see
        # mitriakov_module.yaml), so r_t is selected PER ENV by _ascending_mask() instead
        # of a single config-wide choice.
        ascending = self._ascending_mask()
        cog = self._cog_penalty()
        pitch_vel = self._pitch_velocity_penalty()
        components["direction_penalty"] = torch.where(ascending, cog, pitch_vel)

        # Clearance reward — sign flipped from marv_rl_module.py's identical-formula term
        # (which penalises high clearance) after empirical evidence it didn't help:
        # env.clearance is a universal per-step signal (hull height above the terrain
        # directly beneath it) computed once by CrossingEnv regardless of module. Neither
        # of the paper's own penalties (Eq. 5/6) constrains this: both only look at
        # pitch/roll-derived COG deviation or pitch velocity, which stay well-defined with
        # or without a continuously-tracked main chassis. Clearance does not — the paper's
        # robots keep the hull's underside close to the surface throughout via the main
        # track itself, so they never needed a reward term for it. MARV has no such track
        # (see MitriakovFlipperBounds' docstring): the hull can only clear a stair tread
        # by the flippers actively lifting/bridging it, so REWARDING clearance (rather
        # than penalising it, as marv_rl does for its own, different reason — discouraging
        # hovering on open/flat ground) pushes toward that lifted, non-scraping posture
        # instead of one where the robot keeps its belly low and grinds against the step
        # edge. Opt-in via env_cfg_overrides.clearance_coef (None by default = disabled,
        # matching every other cfg.*_coef guard in this project).
        if cfg.clearance_coef is not None:
            components["clearance_reward"] = cfg.clearance_coef * (1 / (1 + torch.exp(-((env.clearance - 0.2) / 0.02))))

        # Terminal masking/bonus — zero every component on failure/explosion, then add
        # the terminal bonus (goal/fail/timeout — never zeroed). failed_reward should be
        # set to -1 in the training config to match the paper's tip-over penalty exactly
        # (Eq. 5/6's "-1, if tip over"); tip-over itself is CrossingEnv's existing
        # rollover_threshold_deg fail condition, not recomputed here.
        terminal = env._explosion_mask | env._fail_mask
        for name, comp in components.items():
            components[name] = torch.where(terminal, torch.zeros_like(comp), comp)

        components["terminal_bonus"] = (
            env._success_mask.float() * cfg.goal_reached_reward
            + env._fail_mask.float() * cfg.failed_reward
            + env._timeout_mask.float() * (cfg.timeout_penalty if cfg.timeout_penalty is not None else 0.0)
        )

        return components
