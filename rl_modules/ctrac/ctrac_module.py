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

    ROW INDEX DECREASES TOWARD THE FRONT. ftr_env.calc_current_frame_height_maps() stores
    `local_map.flip(0)`, so row r sits at x = (center_row - r) * _HM_CELL — row 0 is the
    frontmost strip, not the rearmost. creps_module.py's height_ahead_row relies on the same
    convention ("row index decreases toward the front of the robot").

    This was originally written as `row_lo = center_row + round(x_lo / _HM_CELL)`, i.e.
    assuming row index grows forward. That silently sampled the MIRROR of every requested
    range: the paper's local map at x in [+0.4, +1.0] m ahead came out as rows 30..42, which
    is x in [-0.375, -0.975] m — behind the robot. The policy was blind to whatever it was
    about to drive onto and instead saw what it had already crossed, on both the actor's
    local map and the critic's privileged map.
    """
    n, h, w = raw.shape
    center_row, center_col = h // 2, w // 2
    # Front-to-back output ordering (row 0 of the crop = furthest ahead), matching the
    # source grid's own orientation so the two stay consistent.
    row_lo = center_row - round(x_hi / _HM_CELL)
    row_hi = center_row - round(x_lo / _HM_CELL)
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
        kf = int(self.cfg.flipper_activity_window)
        self._flipper_hist = torch.zeros(env.num_envs, kf + 1, env.flipper_num, device=env.device)

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

    def _update_flipper_hist(self) -> None:
        """Trailing window of raw flipper angles, oldest..newest, (N, kf+1, flipper_num).

        Same shape of bookkeeping as _update_swing_hist. On a fresh episode every slot is
        filled with the current angles so the window opens at zero net displacement rather
        than reading a carry-over from the previous episode's final posture.
        """
        env = self.env
        pos = env.flipper_positions
        fresh = env.episode_length_buf == 0
        self._flipper_hist = torch.cat([self._flipper_hist[:, 1:], pos.unsqueeze(1)], dim=1)
        if fresh.any():
            self._flipper_hist[fresh] = pos[fresh].unsqueeze(1)

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
        components["progress"] = mcfg.progress_weight * (progress_frac - 1.0)

        # Goal-approach velocity — this project's addition, not in the paper.
        #
        # "progress" above rewards CLOSENESS, not the RATE of approach: moving 0.03 m on an
        # ~8 m lane changes it by ~0.004, which is invisible next to the other per-step terms.
        # Nothing else in Eq. 4-8 pays for speed either, and the measured consequence is that
        # the policy gets steadily SLOWER as it trains — lin_vel_mean fell 0.327 -> 0.184 m/s
        # on run 11365624 and 0.334 -> 0.258 m/s on 11369835, while dist_to_goal plateaued
        # around 4.9 m. At 0.26 m/s a 30 s episode covers ~7.8 m, so the robot is right at the
        # edge of being unable to finish at all, and the lanes still stuck at 0.000 success are
        # exactly the ones needing committed forward drive to mount an obstacle.
        #
        # Signed projection of world-frame linear velocity onto the goal direction, so only NET
        # approach counts: driving fast in a circle, or oscillating toward and away, nets ~0
        # rather than farming reward the way a raw |v| term would. Backing up is penalised but
        # bounded at -1, leaving room for the reposition-then-climb manoeuvre.
        # env.robot_lin_velocities is root_lin_vel_b — BODY frame (IsaacLab's _b suffix) — so
        # the goal direction has to be de-rotated into the body frame before projecting, the
        # same yaw-only rotation get_observations() applies to goal_world above. Projecting a
        # body-frame velocity onto a world-frame direction would silently make this term a
        # function of heading rather than of approach speed.
        goal_w = env.target_positions[:, :2] - env.positions[:, :2]
        yaw = env.orientations_3[:, 2]
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        goal_bx = cos_y * goal_w[:, 0] + sin_y * goal_w[:, 1]
        goal_by = -sin_y * goal_w[:, 0] + cos_y * goal_w[:, 1]
        goal_b = torch.stack([goal_bx, goal_by], dim=-1)
        dir_to_goal = goal_b / goal_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_toward = (env.robot_lin_velocities[:, :2] * dir_to_goal).sum(dim=-1)  # m/s, signed
        components["goal_velocity"] = mcfg.goal_velocity_weight * (
            v_toward / mcfg.goal_velocity_ref
        ).clamp(-1.0, 1.0)

        # Flipper actuation while blocked — this project's addition, not in the paper.
        #
        # Measured on run 11400127 at 48.7M frames: the policy commands v_mean 0.43-0.56 and
        # achieves |v| 0.175-0.251, of which the VERTICAL component (0.213) exceeds the forward
        # one (0.132) — the robot judders against an obstacle instead of traversing. Meanwhile
        # the flipper actions sit at 0.076-0.200 on [-1, 1] (i.e. "hold posture", since
        # flipper_control_mode is velocity) while flipper torque maxes out at exactly 1000.0,
        # the MARV_CFG effort_limit, and clearance/height_min is -0.069 m: belly-down, pushing
        # at the torque ceiling, flippers never repositioned. dist_to_goal has been pinned at
        # ~4.7 m since 7M frames as a result.
        #
        # Rewards MOVEMENT (net angle change), not deflection from flat. An earlier version of
        # this term paid for |angle| — a static quantity — which meant a blocked robot holding
        # any bent posture collected the bonus indefinitely without actuating anything; on run
        # 11426609 that term rose to 0.077/step while clearance/height fell 0.036 -> 0.021 and
        # eval success sat flat at 0.36-0.39 for 25M frames.
        #
        # Movement is measured as NET displacement across a trailing window
        # (flipper_activity_window steps), |theta_t - theta_{t-kf}|, not as the summed
        # per-step |delta theta|. That distinction is the whole anti-farming argument: in
        # velocity control mode the action IS a rotation rate (flipper_dt = 5 deg/step), so a
        # summed-|delta| reward is maximised by chattering at +-1, which is both easier to
        # find than climbing and useless — it produces no posture. Net displacement over a
        # window cancels chatter to ~0 while paying full value for sustained, committed
        # actuation in one direction. Set flipper_activity_window: 1 to recover the raw
        # per-step rate (and its farmability) if that is ever wanted deliberately.
        #
        # Gated on being blocked so it pays only when the robot is stuck: on open ground
        # holding a flat posture is correct and re-articulating for its own sake should not
        # be paid for.
        self._update_flipper_hist()
        v_cmd = env.last_action[:, 0]
        blocked = (v_cmd > mcfg.blocked_v_cmd_min) & (v_toward < mcfg.blocked_v_toward_max)
        move_ref = torch.deg2rad(torch.tensor(float(mcfg.flipper_activity_move_ref_deg), device=env.device))
        net_move = (self._flipper_hist[:, -1] - self._flipper_hist[:, 0]).abs()  # (N, flipper_num) rad
        per_flipper = (net_move / move_ref).clamp(0.0, 1.0)
        # Exponent applied PER FLIPPER, before the mean, and for the same reason the
        # deflection version used one: a linear term has a constant marginal rate, so a 1-deg
        # drift over the window pays at exactly the rate a full-range sweep does and there is
        # no deadzone. Cubing leaves the saturated value (1.0) — and therefore the
        # weight-vs-goal_velocity_weight ceiling argument in ctrac_module.yaml — untouched
        # while making near-stationary flippers pay essentially nothing. Cubing the mean
        # instead would score the standard climbing manoeuvre (front pair sweeping, rear pair
        # held) at (0.5)**3 = 0.125 rather than (1+1+0+0)/4 = 0.5, penalising correct use of
        # two of the four flippers.
        movement = per_flipper.pow(float(mcfg.flipper_activity_exponent)).mean(dim=-1)
        components["flipper_activity"] = (
            mcfg.flipper_activity_weight * blocked.float() * movement
        )

        # Posture swing (Eq. 5-6)
        self._update_swing_hist()
        bmax_pitch_rad = torch.deg2rad(torch.tensor(float(mcfg.swing_bmax_pitch_deg))).item()
        bmax_roll_rad = torch.deg2rad(torch.tensor(float(mcfg.swing_bmax_roll_deg))).item()
        rs_pitch = self._swing_penalty(self._pitch_hist, bmax_pitch_rad)
        rs_roll = self._swing_penalty(self._roll_hist, bmax_roll_rad)
        components["posture_swing"] = mcfg.posture_swing_weight * (
            mcfg.swing_alpha_pitch * rs_pitch + mcfg.swing_alpha_roll * rs_roll
        )

        # Stabilization (Eq. 7) — ground-truth contact points, not the C-VAE's estimate
        # (matching the paper: c_t is privileged state used directly for reward; only the
        # *estimated* c-tilde_t reaches the actor, via ctrac_policy.py).
        # Recomputed here rather than reusing get_observations()'s value: IsaacLab's
        # DirectRLEnv.step calls _get_rewards() (line 340) BEFORE _get_observations()
        # (line 356), so a cache filled by get_observations would hand this the PREVIOUS
        # step's contacts. The duplicate ContactSensor read is the cheap, correct option.
        contact_points, contact_prob = self._contact_extractor.compute()
        components["stabilization"] = mcfg.stabilization_weight * self._stabilization_penalty(contact_points, contact_prob)

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
