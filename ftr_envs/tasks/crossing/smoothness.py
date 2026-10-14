"""Motion-smoothness costs of the robot body, sampled at the physics rate.

Smoothness is measured on derivatives of the motion, never on levels, so it needs no reference
values: a steady 30 deg pitch on stairs costs nothing, a pitch change costs. Two families:

  temporal  low-passed body pitch / roll rate -> angular acceleration, forward speed -> jerk.
            These grow with speed in front of a terrain change (an edge hit twice as fast
            gives ~4x the angular acceleration), which is what couples speed to terrain.
  spatial   body height z and pitch resampled every `ds` metres of 3-D path length,
            low-passed in s, second difference -> d2z/ds2, d2pitch/ds2: the shape of the
            body path, independent of speed (a drop taken with the flippers up is a corner).

Each channel's cost is the integral of the squared derivative (dt for temporal, ds for
spatial), accumulated per control step. The integral is additive: the per-step costs summed
over an episode are exactly the episode metric, so a dense reward on them optimises the eval
metric directly.

Pure torch, no Isaac: CrossingEnv feeds `substep()` from _apply_action once per physics step.
Validated offline on Gazebo rides in misc/metric_research (lam 0.1-0.2 m and fc 2.5-10 Hz keep
the per-obstacle ranking stable).
"""
from __future__ import annotations

import math

import torch

CHANNELS = ("pitch_acc", "roll_acc", "v_jerk", "z_ss", "pitch_ss")


def butter2(fc: float, fs: float) -> tuple[float, float, float, float, float]:
    """Second-order Butterworth low-pass by the bilinear transform: (b0, b1, b2, a1, a2)."""
    k = math.tan(math.pi * min(fc, 0.45 * fs) / fs)
    norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
    b0 = k * k * norm
    return b0, 2.0 * b0, b0, 2.0 * (k * k - 1.0) * norm, (1.0 - math.sqrt(2.0) * k + k * k) * norm


def pitch_roll(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = quat_wxyz.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    return pitch, roll


class SmoothnessTracker:
    """Per-env filter state and cost accumulators. All tensors are (N, ...) on `device`.

    Temporal signals [pitch rate, roll rate, forward speed] go through two cascaded 2nd-order
    Butterworth sections (4th order, causal) at `fc_hz`. Spatial signals [z, pitch] are
    linearly interpolated onto a uniform grid of `ds` metres of 3-D path length and low-passed
    there by two cascaded exponential smoothers with cutoff wavelength `lam_m`.
    """

    # grid points one substep may cross; at 200 Hz and ds 2 cm that is up to 8 m/s
    MAX_CROSS = 2

    def __init__(self, num_envs: int, device, physics_dt: float, fc_hz: float = 5.0,
                 lam_m: float = 0.15, ds_m: float = 0.02, max_substep_cost: float = 1e4):
        self.n, self.device, self.dt, self.ds = num_envs, device, physics_dt, ds_m
        self.coef = butter2(fc_hz, 1.0 / physics_dt)
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * ds_m / lam_m)
        # grid samples before spatial cost is counted: the smoother is seeded as if the body
        # had been standing still at its reset pose, so on a slope its first ~lam of output
        # is a fake kink from "flat" into the slope. 2 lam leaves (1 - alpha)^n < 1e-3.
        self.warmup = max(2, math.ceil(2.0 * lam_m / ds_m))
        self.max_cost = max_substep_cost
        z = lambda *s: torch.zeros((num_envs, *s), device=device)
        # temporal: 3 signals x 2 sections x (x1, x2, y1, y2); filtered history for derivatives
        self.biquad = z(2, 4, 3)
        self.tf1, self.tf2 = z(3), z(3)          # filtered signal at k-1, k-2
        # spatial: 2 signals (z, pitch)
        self.s_acc = z()                         # path length since the last grid point
        self.prev_pos, self.prev_sp = z(3), z(2)
        self.ema = z(2, 2)                       # two smoother stages
        self.g1, self.g2 = z(2), z(2)            # filtered grid samples at j-1, j-2
        self.n_grid = torch.zeros(num_envs, dtype=torch.long, device=device)
        # accumulators
        self.step_cost = z(len(CHANNELS))
        self.episode_cost = z(len(CHANNELS))
        self.episode_path = z()
        self.episode_time = z()

    # ------------------------------------------------------------------ reset
    def reset(self, idx, pos: torch.Tensor, quat_wxyz: torch.Tensor, ang_vel_b: torch.Tensor,
              lin_vel_b: torch.Tensor) -> None:
        """Re-arm the filters of envs `idx` at their current state (no start-up transient)."""
        pitch, roll = pitch_roll(quat_wxyz)
        sig = torch.stack([ang_vel_b[:, 1], ang_vel_b[:, 0], lin_vel_b[:, 0]], dim=-1)
        self.biquad[idx] = sig[:, None, None, :].expand(-1, 2, 4, -1)
        self.tf1[idx] = sig
        self.tf2[idx] = sig
        sp = torch.stack([pos[:, 2], pitch], dim=-1)
        self.s_acc[idx] = 0.0
        self.prev_pos[idx] = pos
        self.prev_sp[idx] = sp
        self.ema[idx] = sp[:, None, :].expand(-1, 2, -1)
        self.g1[idx] = sp
        self.g2[idx] = sp
        self.n_grid[idx] = 0
        self.step_cost[idx] = 0.0
        self.episode_cost[idx] = 0.0
        self.episode_path[idx] = 0.0
        self.episode_time[idx] = 0.0

    # ------------------------------------------------------------------ physics substep
    def _biquad(self, x: torch.Tensor) -> torch.Tensor:
        b0, b1, b2, a1, a2 = self.coef
        for sec in range(2):
            st = self.biquad[:, sec]             # (N, 4, 3): x1, x2, y1, y2
            y = b0 * x + b1 * st[:, 0] + b2 * st[:, 1] - a1 * st[:, 2] - a2 * st[:, 3]
            self.biquad[:, sec] = torch.stack([x, st[:, 0], y, st[:, 2]], dim=1)
            x = y
        return x

    def substep(self, pos: torch.Tensor, quat_wxyz: torch.Tensor, ang_vel_b: torch.Tensor,
                lin_vel_b: torch.Tensor) -> None:
        dt = self.dt
        pitch, roll = pitch_roll(quat_wxyz)
        cost = torch.zeros_like(self.step_cost)

        # -- temporal
        f = self._biquad(torch.stack([ang_vel_b[:, 1], ang_vel_b[:, 0], lin_vel_b[:, 0]], dim=-1))
        ang_acc = (f[:, :2] - self.tf1[:, :2]) / dt
        jerk = (f[:, 2] - 2.0 * self.tf1[:, 2] + self.tf2[:, 2]) / (dt * dt)
        self.tf2, self.tf1 = self.tf1, f
        cost[:, 0] = ang_acc[:, 0].square() * dt
        cost[:, 1] = ang_acc[:, 1].square() * dt
        cost[:, 2] = jerk.square() * dt

        # -- spatial: every grid point crossed during this substep (at most a few)
        sp = torch.stack([pos[:, 2], pitch], dim=-1)
        step = (pos - self.prev_pos).norm(dim=-1)
        total = self.s_acc + step
        # A jump of more than MAX_CROSS grid steps in one substep (teleport, explosion) is not a
        # motion: it contributes no cost and restarts the spatial history from here. The loop
        # bound is a constant, not n_cross.max(), so the substep never syncs GPU -> CPU.
        jumped = total > self.MAX_CROSS * self.ds
        n_cross = torch.where(jumped, 0, torch.floor(total / self.ds).long())
        for j in range(1, self.MAX_CROSS + 1):
            m = n_cross >= j
            frac = ((j * self.ds - self.s_acc) / step.clamp(min=1e-9)).clamp(0.0, 1.0)
            y = self.prev_sp + frac[:, None] * (sp - self.prev_sp)
            e0 = self.ema[:, 0] + self.alpha * (y - self.ema[:, 0])
            e1 = self.ema[:, 1] + self.alpha * (e0 - self.ema[:, 1])
            d2 = (e1 - 2.0 * self.g1 + self.g2) / (self.ds * self.ds)
            ok = m & (self.n_grid >= self.warmup)
            cost[:, 3] += torch.where(ok, d2[:, 0].square() * self.ds, 0.0)
            cost[:, 4] += torch.where(ok, d2[:, 1].square() * self.ds, 0.0)
            mm = m[:, None]
            self.ema[:, 0] = torch.where(mm, e0, self.ema[:, 0])
            self.ema[:, 1] = torch.where(mm, e1, self.ema[:, 1])
            self.g2 = torch.where(mm, self.g1, self.g2)
            self.g1 = torch.where(mm, e1, self.g1)
            self.n_grid = self.n_grid + m.long()
        self.s_acc = torch.where(jumped, 0.0, total - n_cross.float() * self.ds)
        self.n_grid = torch.where(jumped, 0, self.n_grid)
        self.ema = torch.where(jumped[:, None, None], sp[:, None, :].expand(-1, 2, -1), self.ema)
        self.g1 = torch.where(jumped[:, None], sp, self.g1)
        self.g2 = torch.where(jumped[:, None], sp, self.g2)
        self.prev_pos, self.prev_sp = pos.clone(), sp

        cost = torch.nan_to_num(cost, nan=0.0, posinf=self.max_cost).clamp(max=self.max_cost)
        self.step_cost += cost
        self.episode_cost += cost
        self.episode_path += torch.where(jumped, 0.0, step)
        self.episode_time += dt

    # ------------------------------------------------------------------ control step
    def pop_step(self) -> torch.Tensor:
        """(N, C) costs accumulated since the last call, then cleared."""
        out = self.step_cost.clone()
        self.step_cost.zero_()
        return out
