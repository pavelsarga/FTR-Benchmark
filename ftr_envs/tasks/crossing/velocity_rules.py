"""V-Gov, the rule-based forward-speed governor: v_default reduced by weighted, configurable criteria.

One implementation for Isaac (velocity_control.RuleVelocity) and Gazebo / the robot
(robot_rodeo_gym's velocity_rule_node, which loads this file by path): torch, math and yaml only,
no ftr_envs or Isaac imports. Batched over N envs; Gazebo runs N = 1.

Every criterion turns a scalar metric per env into a ratio r in [0, 1]:
    linear   r = clip((m - lo) / (hi - lo)); with hold_s, a peak is kept for hold_s before r follows again
    bang     r = 1 once m > trip, held until m < release has lasted hold_s (hysteresis + hold)
with an optional EMA of the metric (tau, s). Its factor is 1 - weight * r, so the weight is the
largest fraction of the speed it can take away. One tick:

    v* = v_default * prod(factors), not below v_min
                  predictive_combine: strongest -- the map criteria give only their smallest factor
                  instead of their product (a log is a step *and* rough: one reduction, not two)
    stall guard   progress along the heading over the last window_s < progress_m while commanding
                  >= v_min -> v* = boost level (the rules are overridden), held until the robot has
                  made release_m of progress or max_boost_s has passed. A boost that runs out without
                  progress goes straight to the next level; once the top level runs out, the guard
                  waits a full window again. Getting through resets the level. The rules must not
                  pin the robot on an edge, and rocking in place is not progress. With
                  max_roll_deg, a robot rolled past it gets the first level only: unstick, but no
                  hard push on a tilted pallet or a diagonal edge; likewise max_pitch_down_deg for a
                  robot pitched nose-down past it (creeping over a lip is deliberate, not a stall).
    caps          a criterion with `cap` and r = 1 limits v* to cap, also below v_min (special cases)
    slew          v moves towards v* by at most accel*dt up / decel*dt down

Criteria (the config key is the criterion's name):
    predictive, on the heightmap -- dropped with predictive: false (mapless)
      step_up / step_down  height change over `baseline` along the median profile of the track
                           footprint, anywhere from `behind` m behind the centre to `lookahead` m
                           ahead. The map is world-fixed, so an edge keeps the reduction until the
                           centre is `behind` past it.
      roughness            per scale s (cells): the map box-averaged over s x s, then its second
                           difference at lag s (0 on any plane) -- a band-pass: s = 1 sees
                           cell-to-cell variation, s = 4 patches of ~20 cm with the cell noise
                           averaged out. Its `quantile` over the region under the robot, then the
                           max over scales of weight * value.
      map_mismatch         median height of the map above the robot's own contact plane under the
                           footprint (tall grass: the canopy is mapped, not the ground). From
                           r >= distrust_map_at the map is distrusted and step / roughness give 0.
    reactive
      tilt                 hypot(roll_k * roll, k * pitch), degrees; k = pitch_down_k nose-down (+), pitch_k nose-up
      shake                hypot of the body roll and pitch rates, rad/s
      flippers             largest measured flipper speed, rad/s
      turning              |commanded yaw rate|, rad/s

Heightmap: [N, 45, 21] at 0.05 m, row 0 = front (+1.10 m), column 0 = right (-0.50 m), heights
relative to (base z - track wheel radius) -- velocity_il_features / FtrEnv's convention.
rpy = (roll, pitch, yaw) in rad, roll > 0 = left side up, pitch > 0 = nose down (ROS).
"""
import math
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
import yaml

HM_ROWS, HM_COLS, HM_RES = 45, 21, 0.05
CENTER_ROW, CENTER_COL = HM_ROWS // 2, HM_COLS // 2

TOP_KEYS = {"v_default", "v_min", "predictive", "predictive_combine", "distrust_map_at", "footprint", "slew", "stall",
            "criteria"}
CRITERION_KEYS = {"weight", "mode", "lo", "hi", "trip", "release", "hold_s", "tau", "cap", "enabled"}   # not on/off: YAML booleans
STALL_KEYS = {"window_s", "progress_m", "release_m", "max_boost_s", "boost", "max_roll_deg", "max_pitch_down_deg"}
DISTRUSTED = ("step_up", "step_down", "roughness")


class Inputs(NamedTuple):
    hm: torch.Tensor | None     # [N, 45, 21] or None (mapless)
    v_meas: torch.Tensor        # [N] m/s
    flip_vel: torch.Tensor      # [N, 4] rad/s
    rpy: torch.Tensor           # [N, 3] rad
    ang_vel: torch.Tensor       # [N, 3] rad/s, body frame
    w_cmd: torch.Tensor         # [N] rad/s
    pos: torch.Tensor | None = None     # [N, 2] world xy, m (the stall guard's progress)


def load_rule_config(path, overrides: dict | None = None) -> dict:
    """YAML config; `extends: <file>` (relative to this one) is loaded first and deep-merged under it."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("extends", None)
    if base is not None:
        cfg = _merge(load_rule_config(path.parent / base), cfg)
    return _merge(cfg, overrides or {})


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _check(where, cfg, allowed):
    unknown = set(cfg) - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} (allowed {sorted(allowed)})")


# ---- window geometry ----------------------------------------------------------------------------

def _rows(x0, x1, margin=0):
    """Window rows covering robot-frame x in [x0, x1] m (front row first), with `margin` extra rows each side."""
    r0, r1 = CENTER_ROW - round(x1 / HM_RES) - margin, CENTER_ROW - round(x0 / HM_RES) + 1 + margin
    if not 0 <= r0 < r1 <= HM_ROWS:
        raise ValueError(f"x range [{x0}, {x1}] (+{margin} cells) leaves the {HM_ROWS}-row window")
    return slice(r0, r1)


def _cols(half_width, margin=0):
    k = round(half_width / HM_RES) + margin
    if k > CENTER_COL:
        raise ValueError(f"half width {half_width} (+{margin} cells) leaves the {HM_COLS}-column window")
    return slice(CENTER_COL - k, CENTER_COL + k + 1)


# ---- metrics: each builder precomputes its geometry and returns metric(Inputs) -> [N] -----------

def _step(c, fp, device, sign):
    rows, cols = _rows(-c["behind"], c["lookahead"]), _cols(fp["half_width"])
    lag = max(1, round(c["baseline"] / HM_RES))

    def metric(x):
        p = x.hm[:, rows, cols].median(dim=-1).values          # [N, R], front first
        return (sign * (p[:, :-lag] - p[:, lag:])).amax(dim=-1)  # > 0: higher ahead (up) / lower (down)
    return metric


def _band_cells(s, x0, x1, hw, device):
    """Flat indices of the lag-s second differences of the s-box-averaged ('valid') window whose
    centres lie in the region x in [x0, x1], |y| <= hw: (differences along x, along y)."""
    hb, wb, off = HM_ROWS - s + 1, HM_COLS - s + 1, (s - 1) / 2   # box cell b is centred on window cell b + off

    def cells(n_rows, n_cols, dr, dc):
        x = (CENTER_ROW - (torch.arange(n_rows) + dr + off)) * HM_RES
        y = (torch.arange(n_cols) + dc + off - CENTER_COL) * HM_RES
        inside = ((x >= x0 - 1e-6) & (x <= x1 + 1e-6))[:, None] & (y.abs() <= hw + 1e-6)[None, :]
        if not inside.any():
            raise ValueError(f"roughness scale {s}: no cell of x [{x0}, {x1}], half width {hw} fits the window")
        return inside.flatten().nonzero().squeeze(1).to(device)
    return cells(hb - 2 * s, wb, s, 0), cells(hb, wb - 2 * s, 0, s)


def _roughness(c, fp, device):
    x0, x1 = c.get("x", fp["x"])
    hw = c.get("half_width", fp["half_width"])
    q = float(c.get("quantile", 0.5))
    bands = []
    for s, w in c["scales"]:
        ix, iy = _band_cells(int(s), x0, x1, hw, device)
        k = 1 + round(q * (len(ix) + len(iy) - 1))             # the q-quantile as an order statistic
        bands.append((int(s), float(w), ix, iy, k))

    def metric(x):
        out = None
        for s, w, ix, iy, k in bands:
            a = x.hm if s == 1 else F.avg_pool2d(x.hm[:, None], s, stride=1)[:, 0]
            d_x = a[:, 2 * s:] - 2 * a[:, s:-s] + a[:, :-2 * s]
            d_y = a[:, :, 2 * s:] - 2 * a[:, :, s:-s] + a[:, :, :-2 * s]
            d = torch.cat([d_x.flatten(1)[:, ix], d_y.flatten(1)[:, iy]], dim=1).abs()
            m = w * d.kthvalue(k, dim=1).values
            out = m if out is None else torch.maximum(out, m)
        return out
    return metric


def _map_mismatch(c, fp, device):
    rows, cols = _rows(*fp["x"]), _cols(fp["half_width"])
    xs = ((CENTER_ROW - torch.arange(HM_ROWS, device=device))[rows] * HM_RES)[:, None]   # [R, 1]
    ys = ((torch.arange(HM_COLS, device=device) - CENTER_COL)[cols] * HM_RES)[None, :]   # [1, C]

    def metric(x):
        roll, pitch = x.rpy[:, 0, None, None], x.rpy[:, 1, None, None]
        plane = ys * torch.tan(roll) - xs * torch.tan(pitch)    # contact plane under each cell
        return (x.hm[:, rows, cols] - plane).flatten(1).median(dim=1).values.clamp_min(0.0)
    return metric


def _tilt(c, fp, device):
    """Roll and pitch in one angle; pitch_down_k (default pitch_k) weighs nose-down pitch (+), pitch_k nose-up."""
    kr, kp = float(c.get("roll_k", 1.0)), float(c.get("pitch_k", 1.0))
    kd = float(c.get("pitch_down_k", kp))

    def metric(x):
        pitch = x.rpy[:, 1]
        return torch.rad2deg(torch.hypot(kr * x.rpy[:, 0], torch.where(pitch > 0, kd, kp) * pitch))
    return metric


KINDS = {   # name: (metric builder, its own config keys, predictive)
    "step_up": (lambda c, fp, d: _step(c, fp, d, 1.0), {"lookahead", "behind", "baseline"}, True),
    "step_down": (lambda c, fp, d: _step(c, fp, d, -1.0), {"lookahead", "behind", "baseline"}, True),
    "roughness": (_roughness, {"scales", "quantile", "x", "half_width"}, True),
    "map_mismatch": (_map_mismatch, set(), True),
    "tilt": (_tilt, {"roll_k", "pitch_k", "pitch_down_k"}, False),
    "shake": (lambda c, fp, d: lambda x: torch.hypot(x.ang_vel[:, 0], x.ang_vel[:, 1]), set(), False),
    "flippers": (lambda c, fp, d: lambda x: x.flip_vel.abs().amax(dim=-1), set(), False),
    "turning": (lambda c, fp, d: lambda x: x.w_cmd.abs(), set(), False),
}


class Criterion:
    """metric -> ratio in [0, 1] (linear or bang with hysteresis and hold), per-env state for EMA / latch."""

    def __init__(self, name, c, metric, predictive, num_envs, device):
        self.name, self.metric, self.predictive = name, metric, predictive
        self.weight = float(c["weight"])
        self.mode = c["mode"]
        if self.mode == "linear":
            self.lo, self.hi = float(c["lo"]), float(c["hi"])
            if self.hi <= self.lo:
                raise ValueError(f"criterion {name}: hi must exceed lo")
        elif self.mode == "bang":
            self.trip = float(c["trip"])
            self.release = float(c.get("release", self.trip))
        else:
            raise ValueError(f"criterion {name}: mode {self.mode!r}, expected linear or bang")
        self.hold_s = float(c.get("hold_s", 0.0))
        self.tau = float(c.get("tau", 0.0))
        self.cap = c.get("cap")
        self.ema = torch.zeros(num_envs, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.latch = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.below_s = torch.zeros(num_envs, device=device)
        self.peak = torch.zeros(num_envs, device=device)      # linear hold: held ratio and its age
        self.peak_s = torch.zeros(num_envs, device=device)

    def reset(self, env_ids):
        self.fresh[env_ids] = True
        self.latch[env_ids] = False
        self.below_s[env_ids] = 0.0
        self.peak[env_ids] = 0.0
        self.peak_s[env_ids] = 0.0

    def ratio(self, m, dt):
        if self.tau > 0:
            self.ema = torch.where(self.fresh, m, self.ema + (1.0 - math.exp(-dt / self.tau)) * (m - self.ema))
            self.fresh.zero_()
            m = self.ema
        if self.mode == "linear":
            r = ((m - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0)
            if self.hold_s > 0:
                new_peak = r >= self.peak
                self.peak_s = torch.where(new_peak, torch.zeros_like(self.peak_s), self.peak_s + dt)
                self.peak = torch.where(new_peak | (self.peak_s >= self.hold_s), r, self.peak)
                r = self.peak
            return r
        self.below_s = torch.where(m < self.release, self.below_s + dt, torch.zeros_like(self.below_s))
        self.latch = (m > self.trip) | (self.latch & (self.below_s < self.hold_s))
        return self.latch.float()


class StallGuard:
    """Progress-based stall guard: a ring of SAMPLES poses spanning window_s gives the progress along
    the heading of the pose window_s ago. Stalled = less than progress_m of it while commanding; the
    response is an override at the current boost level until release_m of progress since the boost
    began; after max_boost_s without it, the next level at once, or -- past the top level -- a fresh
    window. All envs share dt, so one sample clock serves the batch."""

    SAMPLES = 10

    def __init__(self, c, num_envs, device):
        _check("stall", c, STALL_KEYS)
        self.window_s, self.progress_m = float(c["window_s"]), float(c["progress_m"])
        self.release_m, self.max_boost_s = float(c["release_m"]), float(c["max_boost_s"])
        self.levels = torch.tensor([float(b) for b in c["boost"]], device=device)
        self.max_roll = math.radians(float(c["max_roll_deg"])) if c.get("max_roll_deg") is not None else None
        self.max_down = math.radians(float(c["max_pitch_down_deg"])) if c.get("max_pitch_down_deg") is not None else None
        self.period = self.window_s / self.SAMPLES
        self.ring = torch.zeros(num_envs, self.SAMPLES, 3, device=device)     # x, y, yaw
        self.head, self.clock = 0, 0.0
        self.armed_s = torch.zeros(num_envs, device=device)                   # since the window was (re)filled
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.level = torch.zeros(num_envs, dtype=torch.long, device=device)   # 1-based while boosting
        self.boosting = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.boost_s = torch.zeros(num_envs, device=device)
        self.start = torch.zeros(num_envs, 3, device=device)                  # pose where the boost began
        self.progress = torch.zeros(num_envs, device=device)

    def reset(self, env_ids):
        self.fresh[env_ids] = True
        self.level[env_ids] = 0
        self.boosting[env_ids] = False

    @staticmethod
    def _along(pose, xy):
        """Progress of xy from `pose` along pose's heading."""
        d = xy - pose[:, :2]
        return d[:, 0] * torch.cos(pose[:, 2]) + d[:, 1] * torch.sin(pose[:, 2])

    def __call__(self, xy, yaw, commanding, dt, roll=None, pitch=None):
        """Override speed [N] while boosting, NaN elsewhere."""
        pose = torch.cat([xy, yaw[:, None]], dim=-1)
        if self.fresh.any():
            self.ring[self.fresh] = pose[self.fresh, None, :]
            self.armed_s[self.fresh] = 0.0
            self.fresh.zero_()
        self.armed_s += dt
        self.clock += dt
        if self.clock >= self.period:
            self.clock %= self.period
            self.ring[:, self.head] = pose
            self.head = (self.head + 1) % self.SAMPLES
        self.progress = self._along(self.ring[:, self.head], xy)               # the oldest sample, ~window_s ago

        self.boost_s += dt
        released = self.boosting & (self._along(self.start, xy) >= self.release_m)
        expired = self.boosting & ~released & (self.boost_s >= self.max_boost_s)
        escalate = expired & (self.level < len(self.levels))                   # push harder, at once
        ended = released | (expired & ~escalate)
        self.level = torch.where(escalate, self.level + 1, self.level.masked_fill(released, 0))
        self.start = torch.where(escalate[:, None], pose, self.start)
        self.boost_s = self.boost_s.masked_fill(escalate, 0.0)
        self.boosting &= ~ended
        self.fresh |= ended                                                    # a new window from here on
        self.armed_s = self.armed_s.masked_fill(ended, 0.0)

        stalled = (~self.boosting & ~ended & commanding & (self.armed_s >= self.window_s)
                   & (self.progress < self.progress_m))
        self.level = torch.where(stalled, (self.level + 1).clamp_max(len(self.levels)), self.level)
        self.start = torch.where(stalled[:, None], pose, self.start)
        self.boost_s = self.boost_s.masked_fill(stalled, 0.0)
        self.boosting |= stalled
        level = self.level
        if self.max_roll is not None:
            level = torch.where(roll.abs() > self.max_roll, level.clamp_max(1), level)
        if self.max_down is not None:
            level = torch.where(pitch > self.max_down, level.clamp_max(1), level)
        return torch.where(self.boosting, self.levels[(level - 1).clamp_min(0)], torch.full_like(self.progress, math.nan))


class RuleSpeed:
    def __init__(self, cfg: dict, num_envs: int = 1, device="cpu"):
        _check("velocity rule config", cfg, TOP_KEYS)
        self.n, self.device = num_envs, device
        self.v_default, self.v_min = float(cfg["v_default"]), float(cfg["v_min"])
        self.predictive = bool(cfg.get("predictive", True))
        self.combine = cfg.get("predictive_combine", "product")
        if self.combine not in ("product", "strongest"):
            raise ValueError(f"predictive_combine {self.combine!r}: expected product or strongest")
        fp = cfg.get("footprint", {"x": [-0.35, 0.35], "half_width": 0.28})
        _check("footprint", fp, {"x", "half_width"})
        _check("slew", cfg.get("slew") or {}, {"accel", "decel"})
        self.criteria = []
        for name, c in cfg["criteria"].items():
            if name not in KINDS:
                raise ValueError(f"unknown velocity rule criterion {name!r} (known {sorted(KINDS)})")
            build, own_keys, predictive = KINDS[name]
            _check(f"criterion {name}", c, CRITERION_KEYS | own_keys)
            if c.get("enabled", True) and (self.predictive or not predictive):
                self.criteria.append(Criterion(name, c, build(c, fp, device), predictive, num_envs, device))
        names = {c.name for c in self.criteria}
        self.uses_map = bool(names & {n for n, k in KINDS.items() if k[2]})
        self.distrust_at = cfg.get("distrust_map_at") if "map_mismatch" in names else None
        self.distrusted = [n for n in DISTRUSTED if n in names]
        self.capped = [c for c in self.criteria if c.cap is not None]
        slew = cfg.get("slew") or {}
        self.accel, self.decel = float(slew.get("accel", math.inf)), float(slew.get("decel", math.inf))
        self.stall = StallGuard(cfg["stall"], num_envs, device) if cfg.get("stall") else None
        self.v_prev = torch.zeros(num_envs, device=device)
        self.debug = {}           # name -> (metric [N], ratio [N]) of the last tick, plus "target"

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        for c in self.criteria:
            c.reset(env_ids)
        self.v_prev[env_ids] = 0.0
        if self.stall is not None:
            self.stall.reset(env_ids)

    # inference_mode(False): the per-env state (criteria, stall guard, v_prev) is rebound here, and a tensor
    # rebound under an outer torch.inference_mode -- train_diffusion's mid-training eval -- becomes an
    # inference tensor that the next in-place update outside it (the training rollout, reset) rejects.
    @torch.inference_mode(False)
    @torch.no_grad()
    def __call__(self, x: Inputs, dt: float) -> torch.Tensor:
        if self.uses_map and x.hm is None:
            raise ValueError("predictive criteria need a heightmap (predictive: false for mapless)")
        metrics = {c.name: c.metric(x) for c in self.criteria}
        ratios = {c.name: c.ratio(metrics[c.name], dt) for c in self.criteria}
        if self.distrust_at is not None:
            bad = ratios["map_mismatch"] >= self.distrust_at
            for n in self.distrusted:
                ratios[n] = ratios[n].masked_fill(bad, 0.0)

        v = torch.full((self.n,), self.v_default, device=self.device)
        strongest = torch.ones(self.n, device=self.device)
        for c in self.criteria:
            f = 1.0 - c.weight * ratios[c.name]
            if c.predictive and self.combine == "strongest":
                strongest = torch.minimum(strongest, f)
            else:
                v = v * f
        v = (v * strongest).clamp_min(self.v_min)
        if self.stall is not None:
            if x.pos is None:
                raise ValueError("the stall guard needs Inputs.pos (world xy)")
            boost = self.stall(x.pos, x.rpy[:, 2], self.v_prev >= self.v_min - 1e-6, dt, roll=x.rpy[:, 0],
                               pitch=x.rpy[:, 1])
            v = torch.where(torch.isnan(boost), v, boost)
        for c in self.capped:
            v = torch.where(ratios[c.name] >= 1.0, v.clamp_max(float(c.cap)), v)

        self.debug = {n: (metrics[n], ratios[n]) for n in metrics}
        self.debug["target"] = v
        if self.stall is not None:
            self.debug["stall"] = (self.stall.level * self.stall.boosting).float()
            self.debug["progress"] = self.stall.progress
        v = self.v_prev + (v - self.v_prev).clamp(-self.decel * dt, self.accel * dt)
        self.v_prev = v
        return v

    def factors(self, env=0) -> dict:
        """{criterion: speed factor} of env `env` at the last tick -- for the UI and logs."""
        return {c.name: 1.0 - c.weight * float(self.debug[c.name][1][env]) for c in self.criteria}
