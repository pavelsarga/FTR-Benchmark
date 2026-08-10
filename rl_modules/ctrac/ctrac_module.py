"""Observations + reward for C-TRAC (Pan et al. 2025, "C-TRAC: Terrain-Adaptive Control
for Articulated Tracked Robots via Contact-Aware Reinforcement Learning"). See
ctrac_contact.py for the ground-truth contact extraction this module's stabilization
reward and (via ctrac_policy.py's C-VAE) supervised training targets both depend on.

Heightmap ranges: the paper's local/privileged heightmaps span robot-frame x in
[0.4,1.0]/[-1.0,1.4] m. This project's existing per-step heightmap
(env.current_frame_height_maps, the same 45x21 @ 0.05 m/cell grid every other module reads
via calc_scanned_height_maps) is robot-centered and only covers +-1.125 m fore/aft — narrower
than the paper's +1.4 m privileged range. _crop_and_pad below crops within whatever the grid
actually covers and edge-replicates past that boundary, so the output shapes always match
the paper's cell counts (12x20 / 48x20) — a documented, deliberate approximation of the
paper's literal metric range (this project's existing heightmap infra sets the achievable
range, not the paper's numbers), not a numeric-fidelity claim.

Requires sync_flipper_control: false (4 independent flippers — Eq. 3's action is per-
flipper) and terrain: custom_mixed (existing terrain asset, per the user's decision not to
build the paper's per-episode procedural stair/stepfield/terrace/ramp generator).
"""
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from omni.isaac.lab.envs import VecEnvObs

from ftr_envs.utils.torch import add_noise

from rl_modules.rl_module import RLModule
from rl_modules.ctrac.ctrac_contact import CTRACContactExtractor
from rl_modules.ctrac.ctrac_observation import PRIVILEGED_HMAP_COLS, PRIVILEGED_HMAP_ROWS

_log = logging.getLogger(__name__)
_CONFIG_PATH = Path(__file__).parent / "ctrac_module.yaml"

_HM_CELL = 0.05  # m/cell — matches env.height_map_length / env.height_map_size exactly (both axes)
_LOCAL_ROWS, _LOCAL_COLS = 12, 20
_LOCAL_X_LO, _LOCAL_X_HI = 0.4, 1.0     # m ahead of robot (Eq. 1's h^l_t)
_LOCAL_Y_LO, _LOCAL_Y_HI = -0.5, 0.5
_PRIV_X_LO, _PRIV_X_HI = -1.0, 1.4      # Eq. 2's h^f_t
_PRIV_Y_LO, _PRIV_Y_HI = -0.5, 0.5


def _crop_and_pad(raw: torch.Tensor, x_lo: float, x_hi: float, y_lo: float, y_hi: float,
                   target_rows: int, target_cols: int) -> torch.Tensor:
    """Crop a robot-centered (N,H,W) heightmap grid (cell=_HM_CELL, grid center = directly
    under the robot, same convention crossing_env.py's ground_height lookup uses) to the
    requested robot-frame x/y range, edge-replicate-padding past the grid's own coverage.
    Always returns exactly (N, target_rows, target_cols).
    """
    n, h, w = raw.shape
    center_row, center_col = h // 2, w // 2
    row_lo = center_row + round(x_lo / _HM_CELL)
    row_hi = row_lo + target_rows
    col_lo = center_col + round(y_lo / _HM_CELL)
    col_hi = col_lo + target_cols

    clamp_row_lo, clamp_row_hi = max(row_lo, 0), min(row_hi, h)
    clamp_col_lo, clamp_col_hi = max(col_lo, 0), min(col_hi, w)
    cropped = raw[:, clamp_row_lo:clamp_row_hi, clamp_col_lo:clamp_col_hi]

    pad_left = clamp_col_lo - col_lo
    pad_top = clamp_row_lo - row_lo
    pad_right = target_cols - cropped.shape[2] - pad_left
    pad_bottom = target_rows - cropped.shape[1] - pad_top
    if pad_left or pad_right or pad_top or pad_bottom:
        cropped = F.pad(
            cropped.unsqueeze(1),
            (max(pad_left, 0), max(pad_right, 0), max(pad_top, 0), max(pad_bottom, 0)),
            mode="replicate",
        ).squeeze(1)
    if cropped.shape[1:] != (target_rows, target_cols):
        cropped = F.interpolate(cropped.unsqueeze(1), size=(target_rows, target_cols), mode="nearest").squeeze(1)
    return cropped


def _min_edge_signed_distance(polygon: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """polygon: (N,4,2) vertices in a consistent winding order (here [FL,RL,RR,FR], a
    non-self-intersecting CCW rectangle winding — NOT env.flipper_positions' [FL,FR,RL,RR]
    order, which would cross diagonally, and NOT [FL,FR,RR,RL] either, which is a valid
    non-crossing rectangle but wound CW under this function's cross-product convention).
    point: (N,2). Returns (N,) the minimum signed perpendicular distance from point to each
    polygon edge line — positive when the point is on the interior side of every edge (i.e.
    "inside" the support polygon), for a CCW-wound polygon under the standard 2D cross
    product sign convention. Verified against a synthetic unit-square case: CoG at the
    square's center returns +0.5 (half the square's side) with this vertex order, vs -0.5
    with [FL,FR,RR,RL] — confirming the CW ordering previously used here inverted the sign,
    so a robot standing stably on all 4 flippers was scored as if its CoG were outside its
    support base (rc pinned near -1 almost every step, ~-0.8 to -0.9 observed in real 13M-
    step training run rew.csv — not a real instability signal, a sign bug).
    """
    v0 = polygon
    v1 = torch.roll(polygon, shifts=-1, dims=1)
    edge = v1 - v0  # (N,4,2)
    to_point = point.unsqueeze(1) - v0  # (N,4,2)
    cross = edge[..., 0] * to_point[..., 1] - edge[..., 1] * to_point[..., 0]  # (N,4)
    edge_len = edge.norm(dim=-1).clamp_min(1e-6)
    signed_dist = cross / edge_len
    return signed_dist.min(dim=-1).values


class CTRACModule(RLModule):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.cfg = OmegaConf.load(_CONFIG_PATH)
        self._contact_extractor = CTRACContactExtractor(env, force_threshold=self.cfg.contact_force_threshold)
        k = int(self.cfg.swing_window_k)
        self._roll_hist = torch.zeros(env.num_envs, k + 1, device=env.device)
        self._pitch_hist = torch.zeros(env.num_envs, k + 1, device=env.device)

    def calc_scanned_height_maps(self, base_robot_frame=True):
        env = self.env
        raw = env.current_frame_height_maps.clone()
        if base_robot_frame:
            raw = raw - (env.positions[:, 2] - env.track_wheel_radius).view(-1, 1, 1)
        return raw  # (N, H, W) — full existing project grid; callers crop via _crop_and_pad

    def get_observations(self) -> VecEnvObs:
        env = self.env
        raw_hmap = self.calc_scanned_height_maps()

        local_hmap = _crop_and_pad(raw_hmap, _LOCAL_X_LO, _LOCAL_X_HI, _LOCAL_Y_LO, _LOCAL_Y_HI, _LOCAL_ROWS, _LOCAL_COLS)
        priv_hmap = _crop_and_pad(raw_hmap, _PRIV_X_LO, _PRIV_X_HI, _PRIV_Y_LO, _PRIV_Y_HI, PRIVILEGED_HMAP_ROWS, PRIVILEGED_HMAP_COLS)

        fwd_vel_raw = env.robot_lin_velocities[:, 0:1]  # (N,1), raw m/s
        # env.flipper_positions native order [FL,FR,RL,RR] -> paper order [FL,RL,RR,FR]
        flippers_raw = env.flipper_positions[:, [0, 2, 3, 1]]  # (N,4), raw rad
        roll = env.orientations_3[:, 0:1]
        pitch = env.orientations_3[:, 1:2]
        yaw = env.orientations_3[:, 2:3]

        goal_world = env.target_positions - env.positions
        yaw_only = env.orientations_3[:, 2]
        cos_y, sin_y = torch.cos(yaw_only), torch.sin(yaw_only)
        goal_x = cos_y * goal_world[:, 0] + sin_y * goal_world[:, 1]
        goal_y = -sin_y * goal_world[:, 0] + cos_y * goal_world[:, 1]
        goal_xy = torch.stack([goal_x, goal_y], dim=-1)  # (N,2), body frame

        reset_flag = (env.episode_length_buf == 0).float().unsqueeze(-1)

        # Domain randomization noise (Table I): orientation N(0,0.1^2) rad, lin vel
        # N(0,0.05^2) m/s, flipper N(0,0.1^2) rad, terrain perception N(0,0.08^2) m —
        # applied only to the partial (actor-visible) slice, never the privileged one
        # (ground truth by definition).
        fwd_vel_n = add_noise(fwd_vel_raw, 0.05)
        flippers_n = add_noise(flippers_raw, 0.1)
        roll_n = add_noise(roll, 0.1)
        pitch_n = add_noise(pitch, 0.1)
        yaw_n = add_noise(yaw, 0.1)
        local_hmap_n = add_noise(local_hmap, 0.08)

        partial = torch.cat([
            fwd_vel_n, flippers_n, roll_n, pitch_n, yaw_n, goal_xy,
            local_hmap_n.reshape(env.num_envs, -1), reset_flag,
        ], dim=-1)  # 1+4+1+1+1+2+240+1 = 251

        contact_points, contact_prob = self._contact_extractor.compute()  # ground truth, no noise
        privileged = torch.cat([
            priv_hmap.reshape(env.num_envs, -1),
            contact_points.reshape(env.num_envs, -1),
            contact_prob,
        ], dim=-1)  # 960+12+4 = 976

        obs = torch.cat([partial, privileged], dim=-1)

        bad_obs = torch.isnan(obs).any(dim=-1) | torch.isinf(obs).any(dim=-1)
        if bad_obs.any():
            _log.warning("NaN/Inf in observations for %d envs: %s", bad_obs.sum().item(), bad_obs.nonzero(as_tuple=False).squeeze(-1).tolist())
            env._obs_nan_mask |= bad_obs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": obs}

    def _update_swing_hist(self) -> None:
        env = self.env
        roll = env.orientations_3[:, 0]
        pitch = env.orientations_3[:, 1]
        fresh = env.episode_length_buf == 0
        self._roll_hist = torch.cat([self._roll_hist[:, 1:], roll.unsqueeze(-1)], dim=-1)
        self._pitch_hist = torch.cat([self._pitch_hist[:, 1:], pitch.unsqueeze(-1)], dim=-1)
        if fresh.any():
            self._roll_hist[fresh] = roll[fresh].unsqueeze(-1)
            self._pitch_hist[fresh] = pitch[fresh].unsqueeze(-1)

    @staticmethod
    def _swing_penalty(hist: torch.Tensor, bmax_rad: float) -> torch.Tensor:
        """rs_i (Eq. 6). hist: (N, k+1) raw signed angle history, oldest..newest. Causal
        per-step adaptation: Delta|b_{i,t}| (the paper's t/t+1 forward-looking condition)
        is approximated here using the two most recent recorded steps instead."""
        delta = hist[:, 1:] - hist[:, :-1]  # raw signed diffs, (N,k)
        avg_swing = delta.abs().mean(dim=-1)
        curr_abs, prev_abs = hist[:, -1].abs(), hist[:, -2].abs()
        worsening = (curr_abs > bmax_rad) & ((curr_abs - prev_abs) > 0)
        return torch.where(worsening, torch.full_like(avg_swing, -1.0), -avg_swing)

    def _stabilization_penalty(self, contact_points: torch.Tensor, contact_prob: torch.Tensor) -> torch.Tensor:
        """rc (Eq. 7) — static-stability-margin approximation of the paper's literal
        energy-height NESM (see this module's docstring for why: no CoG/inertia data
        exposed at this level). contact_points/prob: this step's ground truth, [FL,FR,RL,RR]
        order (ctrac_contact.py's FLIPPER_NAMES), in the ROBOT frame."""
        cfg = self.cfg
        fl, fr, rl, rr = (contact_points[:, i, :2] for i in range(4))
        polygon = torch.stack([fl, rl, rr, fr], dim=1)  # CCW rectangle winding — see _min_edge_signed_distance
        # CoG approximated by the robot base (no separate CoG offset data). ctrac_contact.py
        # now returns robot-frame points, so the base is the origin by construction — keeping
        # env.positions here (correct only while the points were world-frame) would add the
        # robot's world coordinate to a robot-frame polygon, putting the CoG far outside it.
        cog_xy = torch.zeros_like(polygon[:, 0, :])

        margin = _min_edge_signed_distance(polygon, cog_xy)  # (N,)
        norm_margin = torch.sigmoid(margin / cfg.nesm_char_length)  # Norm(Emin_nesm) in [0,1]
        rc = norm_margin - 1.0

        undefined = contact_prob.sum(dim=-1) < 3  # Enesm undefined — fewer than 3 wheels actually touching
        return torch.where(undefined, torch.full_like(rc, -1.0), rc)

    def get_reward_components(self) -> dict[str, torch.Tensor]:
        env = self.env
        cfg = env.cfg
        mcfg = self.cfg
        components: dict[str, torch.Tensor] = {}

        # Progress (Eq. 4): rv = progress_frac - 1, progress_frac = fraction of the
        # start->target distance covered so far, in [0,1]. This project's concrete
        # operationalization of the paper's ambiguous shared-symbol p_t^x.
        total_dist = (env.target_positions[:, :2] - env.start_positions[:, :2]).norm(dim=-1).clamp_min(1e-3)
        dist_to_goal = (env.target_positions[:, :2] - env.positions[:, :2]).norm(dim=-1)
        progress_frac = (1.0 - dist_to_goal / total_dist).clamp(0.0, 1.0)
        components["progress"] = progress_frac - 1.0

        # Posture swing (Eq. 5-6)
        self._update_swing_hist()
        bmax_pitch_rad = torch.deg2rad(torch.tensor(float(mcfg.swing_bmax_pitch_deg))).item()
        bmax_roll_rad = torch.deg2rad(torch.tensor(float(mcfg.swing_bmax_roll_deg))).item()
        rs_pitch = self._swing_penalty(self._pitch_hist, bmax_pitch_rad)
        rs_roll = self._swing_penalty(self._roll_hist, bmax_roll_rad)
        components["posture_swing"] = mcfg.swing_alpha_pitch * rs_pitch + mcfg.swing_alpha_roll * rs_roll

        # Stabilization (Eq. 7) — ground-truth contact points, not the C-VAE's estimate
        # (matching the paper: c_t is privileged state used directly for reward; only the
        # *estimated* c-tilde_t reaches the actor, via ctrac_policy.py).
        contact_points, contact_prob = self._contact_extractor.compute()
        components["stabilization"] = self._stabilization_penalty(contact_points, contact_prob)

        # ------------------------------------------------------------------
        # Step penalty & terminal masking/bonus — same convention every other module uses.
        # ------------------------------------------------------------------
        terminal = env._explosion_mask | env._fail_mask
        components["step_penalty"] = torch.full((env.num_envs,), cfg.step_penalty, device=env.device)
        for name, comp in components.items():
            components[name] = torch.where(terminal, torch.zeros_like(comp), comp)

        # Termination (Eq. 8): literal 150 / -150 / -225, via this module's own
        # env_cfg_overrides values (goal_reached_reward/failed_reward/timeout_penalty).
        components["terminal_bonus"] = (
            env._success_mask.float() * cfg.goal_reached_reward
            + env._fail_mask.float() * cfg.failed_reward
            + env._timeout_mask.float() * (cfg.timeout_penalty or 0.0)
        )

        return components
