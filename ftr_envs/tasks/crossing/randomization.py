"""Per-robot domain randomization for FtrEnv: robot embodiment, actuation, contact, timing.

Enabled by ``FtrEnvCfg.domain_randomization`` (a dict; None = off, the default, and then
nothing in the env changes). Every robot gets its own parameter draw at each reset. What is
randomized and how, each verified on the GPU pipeline (docs/terrain_generation_research.md §9):

| key                   | unit        | applied through                                              |
|-----------------------|-------------|--------------------------------------------------------------|
| flipper_up_deg        | deg         | per-robot clamp of the flipper target (software limit), front and rear pair drawn separately |
| flipper_down_deg      | deg         | same                                                         |
| flipper_max_vel_deg_s | deg/s       | PhysX joint max velocity of the 4 flipper dofs               |
| encoder_offset_deg    | deg (+-)    | physical target = commanded + b, reading = physical - b      |
| track_vel_max         | m/s         | per-robot clamp of the forward command                       |
| track_lr_gain         | fraction +- | right/left track gain mismatch (1 +- g)                      |
| latency_s             | s           | per-robot action delay, rounded to control steps            |
| mass_scale            | factor      | all link masses and inertias (uniform-density scaling)       |
| com_offset_x_m / com_offset_z_m | m (+-) | base_link centre of mass                              |
| friction_scale        | factor      | robot shape materials, per group (wheel / flipper / body), static and dynamic |

Value forms per key: ``[lo, hi]`` (uniform), ``{range: [lo, hi], log: true}`` (log-uniform), or
for automatic domain randomization ``{nominal: x, limits: [LO, HI]}`` (the range starts at the
nominal point and ADR moves each boundary between nominal and its limit). ``friction_scale``
may be one spec for every group or ``{wheel: spec, flipper: spec, body: spec}``.

The position action keeps a FIXED angle mapping: with randomized limits the [-1, 1] -> angle
map spans the widest limits any draw can produce (``envelope_bounds``), and each robot clamps
to its own drawn limits — the same action always means the same angle, and a robot simply
cannot go past its limit, as on the real one (software limits).

ADR (``adr: {...}``, OpenAI 2019): with probability ``boundary_prob`` a reset pins ONE ADR
parameter of that robot to one of its current boundaries; the episode's success is filed
under (parameter, side). After ``buffer`` results the boundary moves outward by ``step`` x
(limit span) if the success rate is >= ``expand_thr``, inward if <= ``shrink_thr``.

Not randomized: flipper drive effort/gains. The flipper drive is a stiff implicit PD (3e4 / 1000)
that only settles inside a narrow band (the articulation solve with 20 wheel drives, cf. the
flipper lock in docs/terrain_v2_design.md §9). Measured on 8 robots, target 60 deg: clipping
the force alone limit-cycles below ~300 N m (4-8 deg) and at 370-520 N m with a high speed cap;
scaling stiffness + damping with it is worse (10-28 deg at 600 N m, 180 deg/s). Weaker / slower
flippers are modelled through ``flipper_max_vel_deg_s`` (stable 30-180 deg/s) and ``latency_s``.

All PhysX setters need CPU tensors (they reject CUDA ones) and take full-size buffers plus an
index list, so the nominal buffers are read once in ``setup()`` and edited in place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_PER_ROBOT_KEYS = (
    "flipper_up_deg", "flipper_down_deg", "flipper_max_vel_deg_s", "encoder_offset_deg",
    "track_vel_max", "track_lr_gain", "latency_s", "mass_scale", "com_offset_x_m", "com_offset_z_m",
)
_FRICTION_GROUPS = ("wheel", "flipper", "body")
_SYMMETRIC = {"encoder_offset_deg", "track_lr_gain", "com_offset_x_m", "com_offset_z_m"}  # spec = +- amplitude allowed
_ADR_DEFAULTS = {"enabled": False, "boundary_prob": 0.2, "buffer": 32, "expand_thr": 0.7, "shrink_thr": 0.4, "step": 0.05}


@dataclass
class ParamSpec:
    lo: float
    hi: float
    log: bool = False
    nominal: float | None = None
    limit_lo: float | None = None
    limit_hi: float | None = None

    @property
    def adr(self) -> bool:
        return self.limit_lo is not None


def to_plain(x):
    """OmegaConf DictConfig / ListConfig (what env_cfg_overrides arrive as) -> plain dict / list."""
    if hasattr(x, "items") and not isinstance(x, dict):
        x = dict(x.items())
    if isinstance(x, dict):
        return {str(k): to_plain(v) for k, v in x.items()}
    if not isinstance(x, (str, bytes)) and hasattr(x, "__iter__") and hasattr(x, "__len__"):
        return [to_plain(v) for v in x]
    return x


def parse_spec(name: str, raw) -> ParamSpec:
    """[lo, hi] | scalar amplitude (symmetric keys) | {range, log} | {nominal, limits[, range, log]}."""
    if isinstance(raw, (int, float)):
        if name.split(".")[0] not in _SYMMETRIC:
            raise ValueError(f"domain_randomization.{name}: a bare number is only allowed for +- keys {sorted(_SYMMETRIC)}")
        return ParamSpec(-float(raw), float(raw))
    if isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            raise ValueError(f"domain_randomization.{name}: expected [lo, hi], got {raw}")
        return ParamSpec(float(raw[0]), float(raw[1]))
    if isinstance(raw, dict):
        unknown = set(raw) - {"range", "log", "nominal", "limits"}
        if unknown:
            raise ValueError(f"domain_randomization.{name}: unknown fields {sorted(unknown)}")
        log = bool(raw.get("log", False))
        if "limits" in raw:
            if "nominal" not in raw:
                raise ValueError(f"domain_randomization.{name}: ADR form needs 'nominal' and 'limits'")
            nom = float(raw["nominal"])
            lim_lo, lim_hi = (float(v) for v in raw["limits"])
            lo, hi = (float(v) for v in raw.get("range", (nom, nom)))
            if not (lim_lo <= lo <= nom <= hi <= lim_hi):
                raise ValueError(f"domain_randomization.{name}: need limits[0] <= range[0] <= nominal <= range[1] <= limits[1]")
            return ParamSpec(lo, hi, log, nom, lim_lo, lim_hi)
        if "range" not in raw:
            raise ValueError(f"domain_randomization.{name}: dict form needs 'range' (or 'nominal' + 'limits' for ADR)")
        lo, hi = (float(v) for v in raw["range"])
        return ParamSpec(lo, hi, log)
    raise ValueError(f"domain_randomization.{name}: cannot parse {raw!r}")


def _sample(spec: ParamSpec, n: int, device) -> torch.Tensor:
    u = torch.rand(n, device=device)
    if spec.log and spec.lo > 0 and spec.hi > 0:
        return torch.exp(math.log(spec.lo) + u * (math.log(spec.hi) - math.log(spec.lo)))
    return spec.lo + u * (spec.hi - spec.lo)


class DomainRandomizer:
    def __init__(self, env, raw: dict):
        raw = to_plain(raw)
        self.env = env
        self.N = env.num_envs
        self.device = env.device
        adr_raw = dict(raw.pop("adr", {}) or {})
        unknown = set(adr_raw) - set(_ADR_DEFAULTS)
        if unknown:
            raise ValueError(f"domain_randomization.adr: unknown fields {sorted(unknown)}")
        self.adr_cfg = {**_ADR_DEFAULTS, **adr_raw}
        fric = raw.pop("friction_scale", None)
        unknown = set(raw) - set(_PER_ROBOT_KEYS)
        if unknown:
            raise ValueError(f"domain_randomization: unknown keys {sorted(unknown)}; known: "
                             f"{sorted(_PER_ROBOT_KEYS) + ['friction_scale', 'adr']}")
        self.specs: dict[str, ParamSpec] = {k: parse_spec(k, v) for k, v in raw.items()}
        if fric is not None:
            groups = fric if isinstance(fric, dict) and set(fric) <= set(_FRICTION_GROUPS) else {g: fric for g in _FRICTION_GROUPS}
            for g, v in groups.items():
                self.specs[f"friction_scale.{g}"] = parse_spec(f"friction_scale.{g}", v)
        if env.flipper_num not in (2, 4) and ({"flipper_up_deg", "flipper_down_deg", "encoder_offset_deg"} & set(self.specs)):
            raise ValueError("domain_randomization flipper limits/offsets need 4 independent or 2 synced flippers")

        dev = self.device
        # per-robot values, initialised to "no change"; nominal flipper limits filled in setup()
        self.flip_low = None
        self.flip_high = None
        self.enc_offset = torch.zeros(self.N, env.flipper_num, device=dev)
        self.track_vmax = torch.full((self.N,), float(env.track_vel_max), device=dev)
        self.lr_gain = torch.zeros(self.N, device=dev)
        self.latency_steps = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.values = {}  # key -> (N,) tensor of the last draw, for logging / the embodiment vector
        lat = self.specs.get("latency_s")
        self.max_delay = 0 if lat is None else int(math.ceil(max(lat.hi, lat.limit_hi or lat.hi) / env.step_dt))
        self._act_buf = None
        # ADR bookkeeping
        self.probe = torch.full((self.N,), -1, dtype=torch.long, device=dev)  # index into _adr_slots, -1 = none
        self.paused = False  # mid-training eval: no boundary probes, nothing recorded
        self._adr_slots = [(k, side) for k, s in self.specs.items() if s.adr for side in (0, 1)]
        self._adr_results: dict[tuple[str, int], list[float]] = {slot: [] for slot in self._adr_slots}
        if self.adr_cfg["enabled"] and not self._adr_slots:
            raise ValueError("domain_randomization.adr.enabled but no parameter uses the {nominal, limits} form")

    # ------------------------------------------------------------------ setup
    def setup(self):
        """Read the nominal PhysX buffers once (CPU) and classify the robot's shapes."""
        env = self.env
        robot = env._robot
        view = robot.root_physx_view
        self.view = view
        # flipper dof indices in flipper_joint_names order (robot.flipper_dof_idx_list only exists after
        # the first robot reset, FtrEnv computes the same list in __init__)
        self.fd = [int(i) for i in env._flipper_joint_ids]
        self._all_idx = torch.arange(self.N, dtype=torch.int32)
        self.max_vel0 = view.get_dof_max_velocities().clone().cpu()
        self.mass0 = view.get_masses().clone().cpu()
        self.inertia0 = view.get_inertias().clone().cpu()
        self.com0 = view.get_coms().clone().cpu()
        self.mat0 = view.get_material_properties().clone().cpu()
        self.max_vel = self.max_vel0.clone()
        self.mass = self.mass0.clone()
        self.inertia = self.inertia0.clone()
        self.com = self.com0.clone()
        self.mat = self.mat0.clone()
        names = list(robot.body_names)
        self.base_body = names.index("base_link") if "base_link" in names else 0
        # shape groups from their nominal static friction (materials are bound per group by
        # MarvWheelArticulation.set_robot_env, so the nominal value identifies the group)
        nominal = {
            "wheel": float(env.cfg.wheel_material_friction),
            "flipper": float(env.cfg.flipper_material_friction),
            "body": float(env.cfg.body_material_friction),
        }
        mu0 = self.mat0[0, :, 0]
        self.shape_groups = {}
        taken = torch.zeros_like(mu0, dtype=torch.bool)
        for g in _FRICTION_GROUPS:
            m = (mu0 - nominal[g]).abs() < 1e-4
            m &= ~taken  # equal nominal values: the first group listed claims the shapes
            taken |= m
            self.shape_groups[g] = m
        low, high = env._nominal_flipper_bounds()
        self.nominal_low, self.nominal_high = low.clone(), high.clone()
        self.flip_low = low.unsqueeze(0).repeat(self.N, 1).clone()
        self.flip_high = high.unsqueeze(0).repeat(self.N, 1).clone()
        groups = {g: int(m.sum()) for g, m in self.shape_groups.items()}
        print(f"[DomainRandomizer] {len(self.specs)} parameters: {sorted(self.specs)}; shape groups {groups}; "
              f"ADR {'on' if self.adr_cfg['enabled'] else 'off'} ({len(self._adr_slots)} boundaries); "
              f"action envelope low={[round(math.degrees(v), 1) for v in self.envelope_bounds()[0].tolist()]} "
              f"high={[round(math.degrees(v), 1) for v in self.envelope_bounds()[1].tolist()]} deg", flush=True)

    # ------------------------------------------------------------------ flipper bounds
    def _pair_layout(self):
        """Which flipper slots are front (True) / rear (False) in the env's flipper layout."""
        return [True, True, False, False] if self.env.flipper_num == 4 else [True, False]

    def _bound_extreme(self, key: str, nominal_deg: torch.Tensor) -> torch.Tensor:
        spec = self.specs.get(key)
        if spec is None:
            return nominal_deg
        top = spec.limit_hi if spec.adr else spec.hi
        return torch.full_like(nominal_deg, top)

    def envelope_bounds(self):
        """(low, high) of the fixed action map: the widest limits any draw can produce."""
        low, high = self.nominal_low.clone(), self.nominal_high.clone()
        front = torch.tensor(self._pair_layout(), device=low.device)
        up_max = math.radians(self._bound_extreme("flipper_up_deg", torch.zeros(1)).item()) if "flipper_up_deg" in self.specs else None
        dn_max = math.radians(self._bound_extreme("flipper_down_deg", torch.zeros(1)).item()) if "flipper_down_deg" in self.specs else None
        # convention (FtrEnv.flipper_angle_bounds): front low = -up, high = down; rear low = -down, high = up
        if up_max is not None:
            low = torch.where(front, torch.minimum(low, torch.full_like(low, -up_max)), low)
            high = torch.where(~front, torch.maximum(high, torch.full_like(high, up_max)), high)
        if dn_max is not None:
            high = torch.where(front, torch.maximum(high, torch.full_like(high, dn_max)), high)
            low = torch.where(~front, torch.minimum(low, torch.full_like(low, -dn_max)), low)
        return low, high

    def clamp_flipper(self, target: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(target, self.flip_high), self.flip_low)

    # ------------------------------------------------------------------ actions
    def delay(self, actions: torch.Tensor) -> torch.Tensor:
        """Per-robot action delay of latency_steps control steps (0 = this step's action)."""
        if self.max_delay == 0:
            return actions
        if self._act_buf is None or self._act_buf.shape[-1] != actions.shape[-1]:
            self._act_buf = torch.zeros(self.N, self.max_delay + 1, actions.shape[-1], device=actions.device)
        self._act_buf = torch.roll(self._act_buf, shifts=1, dims=1)
        self._act_buf[:, 0] = actions
        return self._act_buf[torch.arange(self.N, device=actions.device), self.latency_steps]

    def track_command(self, v: torch.Tensor, w: torch.Tensor, gauge: float):
        """Per-robot forward cap, then the L/R gain mismatch folded back into (v, w)."""
        v = torch.maximum(torch.minimum(v, self.track_vmax), -self.track_vmax)
        if "track_lr_gain" not in self.specs:
            return v, w
        r = (v + w * gauge / 2) * (1.0 + self.lr_gain)
        l_ = (v - w * gauge / 2) * (1.0 - self.lr_gain)
        return (r + l_) / 2, (r - l_) / gauge

    # ------------------------------------------------------------------ resample
    def _draw(self, key: str, n: int, probe_side: torch.Tensor | None) -> torch.Tensor:
        spec = self.specs[key]
        x = _sample(spec, n, self.device)
        if probe_side is not None:  # ADR probes: -1 none, 0 low boundary, 1 high boundary
            x = torch.where(probe_side == 0, torch.full_like(x, spec.lo), x)
            x = torch.where(probe_side == 1, torch.full_like(x, spec.hi), x)
        return x

    def resample(self, env_ids):
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(list(env_ids), dtype=torch.long)
        ids = ids.to(self.device, torch.long)
        n = len(ids)
        if n == 0:
            return
        # ADR probe assignment
        self.probe[ids] = -1
        if self.adr_cfg["enabled"] and not self.paused:
            is_probe = torch.rand(n, device=self.device) < self.adr_cfg["boundary_prob"]
            slot = torch.randint(len(self._adr_slots), (n,), device=self.device)
            self.probe[ids] = torch.where(is_probe, slot, torch.full_like(slot, -1))
        probe = self.probe[ids]

        def side_for(key):
            if not self.adr_cfg["enabled"] or not self.specs[key].adr:
                return None
            sides = torch.full((n,), -1, dtype=torch.long, device=self.device)
            for si, (k, side) in enumerate(self._adr_slots):
                if k == key:
                    sides = torch.where(probe == si, torch.full_like(sides, side), sides)
            return sides

        vals = {k: self._draw(k, n, side_for(k)) for k in self.specs}
        for k, v in vals.items():
            self.values.setdefault(k, torch.zeros(self.N, device=self.device))[ids] = v
        cpu_ids = ids.cpu().to(torch.int32)
        icpu = cpu_ids.long()
        front = torch.tensor(self._pair_layout(), device=self.device)

        # flipper limits: front and rear pair drawn separately (second, independent draw for the rear)
        for key, is_up in (("flipper_up_deg", True), ("flipper_down_deg", False)):
            if key not in self.specs:
                continue
            f_val = torch.deg2rad(vals[key])
            r_val = torch.deg2rad(self._draw(key, n, side_for(key)))
            per_slot = torch.where(front, f_val[:, None], r_val[:, None])
            if is_up:   # front low = -up, rear high = up
                self.flip_low[ids] = torch.where(front, -per_slot, self.flip_low[ids])
                self.flip_high[ids] = torch.where(~front, per_slot, self.flip_high[ids])
            else:       # front high = down, rear low = -down
                self.flip_high[ids] = torch.where(front, per_slot, self.flip_high[ids])
                self.flip_low[ids] = torch.where(~front, -per_slot, self.flip_low[ids])
        if "encoder_offset_deg" in vals:
            self.enc_offset[ids] = torch.deg2rad(self._draw("encoder_offset_deg", n * self.env.flipper_num, None)).view(n, -1)
        if "track_vel_max" in vals:
            self.track_vmax[ids] = vals["track_vel_max"]
        if "track_lr_gain" in vals:
            self.lr_gain[ids] = vals["track_lr_gain"]
        if "latency_s" in vals:
            self.latency_steps[ids] = torch.round(vals["latency_s"] / self.env.step_dt).long().clamp(0, self.max_delay)
            if self._act_buf is not None:
                self._act_buf[ids] = 0.0

        # PhysX-side parameters (CPU tensors, full buffer + indices)
        if "flipper_max_vel_deg_s" in vals:
            v = torch.deg2rad(vals["flipper_max_vel_deg_s"]).cpu()
            for d in self.fd:
                self.max_vel[icpu, d] = v
            self.view.set_dof_max_velocities(self.max_vel, cpu_ids)
        if "mass_scale" in vals:
            s = vals["mass_scale"].cpu()
            self.mass[icpu] = self.mass0[icpu] * s[:, None]
            self.inertia[icpu] = self.inertia0[icpu] * s[:, None, None]
            self.view.set_masses(self.mass, cpu_ids)
            self.view.set_inertias(self.inertia, cpu_ids)
        if "com_offset_x_m" in vals or "com_offset_z_m" in vals:
            self.com[icpu] = self.com0[icpu]
            if "com_offset_x_m" in vals:
                self.com[icpu, self.base_body, 0] += vals["com_offset_x_m"].cpu()
            if "com_offset_z_m" in vals:
                self.com[icpu, self.base_body, 2] += vals["com_offset_z_m"].cpu()
            self.view.set_coms(self.com, cpu_ids)
        fric_keys = [g for g in _FRICTION_GROUPS if f"friction_scale.{g}" in vals]
        if fric_keys:
            self.mat[icpu] = self.mat0[icpu]
            for g in fric_keys:
                mask = self.shape_groups[g]
                s = vals[f"friction_scale.{g}"].cpu()[:, None]
                sub = self.mat[icpu]
                sub[:, mask, 0] = self.mat0[icpu][:, mask, 0] * s
                sub[:, mask, 1] = self.mat0[icpu][:, mask, 1] * s
                self.mat[icpu] = sub
            self.view.set_material_properties(self.mat, cpu_ids)

    # ------------------------------------------------------------------ ADR
    def record_outcomes(self, env_ids, success: torch.Tensor):
        """File the finished episodes of ADR-probe robots under their (parameter, side)."""
        if not self.adr_cfg["enabled"] or self.paused:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        probe = self.probe[ids]
        sel = probe >= 0
        if not sel.any():
            return
        for slot, ok in zip(probe[sel].tolist(), success[sel].float().tolist()):
            key = self._adr_slots[slot]
            self._adr_results[key].append(ok)
            if len(self._adr_results[key]) >= self.adr_cfg["buffer"]:
                self._adr_update(key)

    def _adr_update(self, slot):
        key, side = slot
        spec = self.specs[key]
        rate = sum(self._adr_results[slot]) / len(self._adr_results[slot])
        self._adr_results[slot] = []
        step = self.adr_cfg["step"] * (spec.limit_hi - spec.limit_lo)
        old = (spec.lo, spec.hi)
        if rate >= self.adr_cfg["expand_thr"]:
            if side == 0:
                spec.lo = max(spec.limit_lo, spec.lo - step)
            else:
                spec.hi = min(spec.limit_hi, spec.hi + step)
        elif rate <= self.adr_cfg["shrink_thr"]:
            if side == 0:
                spec.lo = min(spec.nominal, spec.lo + step)
            else:
                spec.hi = max(spec.nominal, spec.hi - step)
        if (spec.lo, spec.hi) != old:
            print(f"[ADR] {key} {'low' if side == 0 else 'high'} boundary: success {rate:.2f} -> "
                  f"range [{old[0]:.3g}, {old[1]:.3g}] -> [{spec.lo:.3g}, {spec.hi:.3g}]", flush=True)

    # ------------------------------------------------------------------ state / logging
    def state_dict(self) -> dict:
        return {"ranges": {k: (s.lo, s.hi) for k, s in self.specs.items()},
                "adr_results": {f"{k}|{side}": list(v) for (k, side), v in self._adr_results.items()}}

    def load_state_dict(self, state: dict):
        for k, (lo, hi) in state.get("ranges", {}).items():
            if k in self.specs:
                self.specs[k].lo, self.specs[k].hi = float(lo), float(hi)
        for key, v in state.get("adr_results", {}).items():
            k, side = key.rsplit("|", 1)
            if (k, int(side)) in self._adr_results:
                self._adr_results[(k, int(side))] = list(v)

    def stats(self) -> dict:
        out = {}
        for k, s in self.specs.items():
            out[f"dr/{k}_lo"] = s.lo
            out[f"dr/{k}_hi"] = s.hi
        return out

    def embodiment_vector(self) -> torch.Tensor:
        """Configuration-known parameters (limits, max speeds), each scaled to [0, 1] over its
        widest range — for the E3a ablation (observing the embodiment). (N, K)."""
        cols = []
        for k in ("flipper_up_deg", "flipper_down_deg", "flipper_max_vel_deg_s", "track_vel_max"):
            if k not in self.specs:
                continue
            s = self.specs[k]
            lo, hi = (s.limit_lo, s.limit_hi) if s.adr else (s.lo, s.hi)
            cols.append((self.values[k] - lo) / max(hi - lo, 1e-6))
        return torch.stack(cols, dim=-1) if cols else torch.zeros(self.N, 0, device=self.device)
