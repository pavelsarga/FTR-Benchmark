"""Adaptive spawn sampler: which tile the next episode starts on.

Replaces FtrEnv's round-robin ``cycle()`` over ``birth.json`` when ``FtrEnvCfg.spawn_sampler`` is
set (None = off, the default). Design and rationale: docs/terrain_generation_research.md §5.2.

A *tile* is one obstacle instance driven in one direction: birth entries that share a target
point (rounded to 0.1 m), or an explicit ``tile_id`` field written by the generator. The
lateral-jitter variants of a tile are handed out round-robin within it.

Per tile, from the episodes that start and end on it:

- **visit budget** — at most ``visit_budget`` episode starts per tile instance, the
  anti-memorization cap; the tile is then *exhausted* (robots already on it finish).
- **mastered** — Wilson lower bound of the success rate above ``mastery_threshold`` (after
  ``mastery_min_episodes``): retired early.
- **new tiles first** — while a live tile has never been started, the draw is uniform over those.
- **dead** — after ``dead_after`` finished episodes, Wilson upper bound below
  ``dead_threshold``: retired early (it is reported, so the generator can learn from it).
- among live tiles: weight = ``eps`` + ``lp_coef`` x learning progress (|fast - slow| EMA of
  success) + ``var_coef`` x p(1 - p) (learnability) + ``staleness_coef`` x staleness;
  (unvisited tiles get ``new_tile_weight`` once the easy start is on). An optional easy
  start multiplies by exp(-easy_start_k x difficulty) fading out over ``easy_start_episodes``
  (needs a ``difficulty`` field in the birth entries).
- **course exhausted** — once ``exhaust_fraction`` of the tiles are exhausted / mastered / dead,
  ``exhausted`` turns True; the trainer then rotates to the next course. Until it does, spawns
  continue on the least-visited non-dead tiles.

``paused`` (mid-training eval) hands entries out round-robin and records nothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DEFAULTS = {
    "visit_budget": 24,
    "mastery_threshold": 0.85,
    "mastery_min_episodes": 8,
    "dead_after": 16,
    "dead_threshold": 0.10,       # Wilson-hi of 0/16 at z 1.28 is 0.093
    "confidence_z": 1.28,         # one-sided ~90 %
    "exhaust_fraction": 0.75,
    "eps": 0.05,
    "lp_coef": 1.0,
    "var_coef": 1.0,
    "staleness_coef": 0.1,
    "lp_fast": 0.3,
    "lp_slow": 0.05,
    "new_tile_weight": 1.0,
    "easy_start_episodes": 0,
    "easy_start_k": 3.0,
    "seed": 0,
}


def wilson_bounds(s: np.ndarray, n: np.ndarray, z: float):
    """Wilson score interval (lower, upper) of a success rate; (0, 1) where n == 0."""
    n_ = np.maximum(n, 1)
    p = s / n_
    denom = 1 + z * z / n_
    centre = (p + z * z / (2 * n_)) / denom
    half = z * np.sqrt(p * (1 - p) / n_ + z * z / (4 * n_ * n_)) / denom
    lo = np.where(n > 0, centre - half, 0.0)
    hi = np.where(n > 0, centre + half, 1.0)
    return lo, hi


def _tile_key(entry) -> str:
    if "tile_id" in entry:
        return f"id:{int(entry['tile_id'])}"
    t = entry["target_point"]
    t = t.tolist() if hasattr(t, "tolist") else list(t)
    return f"t:{round(float(t[0]), 1)}:{round(float(t[1]), 1)}"


class TileSpawnSampler:
    def __init__(self, entries: list, spec: dict | None = None):
        spec = {str(k): v for k, v in (spec or {}).items()}  # also accepts an OmegaConf DictConfig
        unknown = set(spec) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"spawn_sampler: unknown keys {sorted(unknown)}; known: {sorted(DEFAULTS)}")
        self.cfg = {**DEFAULTS, **spec}
        self.rng = np.random.default_rng(int(self.cfg["seed"]))
        keys = [_tile_key(e) for e in entries]
        self.tile_keys = list(dict.fromkeys(keys))  # stable order
        index = {k: i for i, k in enumerate(self.tile_keys)}
        self.entry_tile = np.array([index[k] for k in keys], dtype=np.int64)
        T = len(self.tile_keys)
        self.entries_of_tile = [np.flatnonzero(self.entry_tile == t) for t in range(T)]
        self._rr = np.zeros(T, dtype=np.int64)  # round-robin pointer within a tile
        diff = [e.get("difficulty") for e in entries]
        self.difficulty = None
        if all(d is not None for d in diff):
            d = np.zeros(T)
            for i, t in enumerate(self.entry_tile):
                d[t] = float(diff[i])
            self.difficulty = d
        self.T = T
        self.started = np.zeros(T, dtype=np.int64)
        self.done = np.zeros(T, dtype=np.int64)
        self.success = np.zeros(T, dtype=np.int64)
        self.ema_fast = np.zeros(T)
        self.ema_slow = np.zeros(T)
        self.last_visit = np.zeros(T, dtype=np.int64)
        self.status = np.zeros(T, dtype=np.int8)  # 0 live, 1 exhausted, 2 mastered, 3 dead
        self.episodes = 0  # total episodes started
        self.paused = False
        self._pause_ptr = 0

    # ------------------------------------------------------------------ status
    STATUS = ("live", "exhausted", "mastered", "dead")

    def _update_status(self, t: int):
        c = self.cfg
        if self.status[t] in (2, 3):
            return
        lo, hi = wilson_bounds(self.success[t:t + 1], self.done[t:t + 1], c["confidence_z"])
        if self.done[t] >= c["mastery_min_episodes"] and lo[0] > c["mastery_threshold"]:
            self.status[t] = 2
        elif self.done[t] >= c["dead_after"] and hi[0] < c["dead_threshold"]:
            self.status[t] = 3
        elif self.started[t] >= c["visit_budget"]:
            self.status[t] = 1

    @property
    def retired_fraction(self) -> float:
        return float((self.status > 0).mean()) if self.T else 1.0

    @property
    def exhausted(self) -> bool:
        return self.retired_fraction >= self.cfg["exhaust_fraction"]

    # ------------------------------------------------------------------ weights / draw
    def weights(self) -> np.ndarray:
        c = self.cfg
        p = (self.success + 1) / (self.done + 2)
        lp = np.abs(self.ema_fast - self.ema_slow)
        stale = np.minimum(1.0, (self.episodes - self.last_visit) / max(self.T, 1))
        w = c["eps"] + c["lp_coef"] * lp + c["var_coef"] * p * (1 - p) + c["staleness_coef"] * stale
        w = np.where(self.started == 0, c["new_tile_weight"], w)
        if self.difficulty is not None and c["easy_start_episodes"] > 0:
            fade = max(0.0, 1.0 - self.episodes / c["easy_start_episodes"])
            w = w * np.exp(-c["easy_start_k"] * self.difficulty * fade)
        return np.where(self.status == 0, w, 0.0)

    def next_entry(self) -> tuple[int, int]:
        """(birth entry index, tile index or -1 when paused)."""
        if self.paused:
            i = self._pause_ptr % len(self.entry_tile)
            self._pause_ptr += 1
            return int(i), -1
        fresh = np.flatnonzero((self.started == 0) & (self.status == 0))
        if len(fresh) and (self.difficulty is None or self.cfg["easy_start_episodes"] <= 0):
            t = int(self.rng.choice(fresh))
            return self._start(t)
        w = self.weights()
        if w.sum() <= 0:
            # every tile retired but the course has not been rotated yet: least-visited non-dead
            cand = np.flatnonzero(self.status != 3)
            if len(cand) == 0:
                cand = np.arange(self.T)
            t = int(cand[np.argmin(self.started[cand])])
        else:
            t = int(self.rng.choice(self.T, p=w / w.sum()))
        return self._start(t)

    def _start(self, t: int) -> tuple[int, int]:
        ents = self.entries_of_tile[t]
        e = int(ents[self._rr[t] % len(ents)])
        self._rr[t] += 1
        self.started[t] += 1
        self.episodes += 1
        self.last_visit[t] = self.episodes
        self._update_status(t)
        return e, t

    # ------------------------------------------------------------------ outcomes
    def record(self, tiles, successes):
        c = self.cfg
        for t, ok in zip(tiles, successes):
            t = int(t)
            if t < 0:
                continue
            ok = float(bool(ok))
            self.done[t] += 1
            self.success[t] += int(ok)
            self.ema_fast[t] += c["lp_fast"] * (ok - self.ema_fast[t])
            self.ema_slow[t] += c["lp_slow"] * (ok - self.ema_slow[t])
            self._update_status(t)

    # ------------------------------------------------------------------ reporting / state
    def stats(self) -> dict:
        counts = {name: int((self.status == i).sum()) for i, name in enumerate(self.STATUS)}
        done = self.done.sum()
        return {
            "spawn/tiles": self.T,
            **{f"spawn/tiles_{k}": v for k, v in counts.items()},
            "spawn/retired_fraction": self.retired_fraction,
            "spawn/exhausted": float(self.exhausted),
            "spawn/episodes_started": int(self.episodes),
            "spawn/success_rate": float(self.success.sum() / done) if done else 0.0,
            "spawn/visits_max": int(self.started.max()) if self.T else 0,
        }

    def tile_table(self) -> list[dict]:
        lo, hi = wilson_bounds(self.success, self.done, self.cfg["confidence_z"])
        return [{"tile": k, "started": int(self.started[i]), "done": int(self.done[i]), "success": int(self.success[i]),
                 "status": self.STATUS[self.status[i]], "lp": float(abs(self.ema_fast[i] - self.ema_slow[i])),
                 "success_lo": float(lo[i]), "success_hi": float(hi[i])} for i, k in enumerate(self.tile_keys)]

    def write_snapshot(self, path, extra: dict | None = None):
        """Per-tile statistics for the course generator (JSON)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"stats": self.stats(), "tiles": self.tile_table(), **(extra or {})}, indent=1))
        tmp.replace(path)  # atomic: the generator may read it at any time

    def state_dict(self) -> dict:
        return {"tile_keys": self.tile_keys, "started": self.started.tolist(), "done": self.done.tolist(),
                "success": self.success.tolist(), "ema_fast": self.ema_fast.tolist(), "ema_slow": self.ema_slow.tolist(),
                "last_visit": self.last_visit.tolist(), "status": self.status.tolist(), "episodes": int(self.episodes)}

    def load_state_dict(self, state: dict) -> int:
        """Restore per-tile statistics for tiles with the same key (same course); returns how many matched.
        A different course matches nothing, which is exactly a fresh course."""
        index = {k: i for i, k in enumerate(self.tile_keys)}
        n = 0
        for j, k in enumerate(state.get("tile_keys", [])):
            i = index.get(k)
            if i is None:
                continue
            for name in ("started", "done", "success", "ema_fast", "ema_slow", "last_visit", "status"):
                getattr(self, name)[i] = state[name][j]
            n += 1
        if n:
            self.episodes = int(state.get("episodes", self.episodes))
        return n


def describe(spec: dict | None) -> str:
    c = {**DEFAULTS, **(spec or {})}
    return (f"visit budget {c['visit_budget']}, mastered at Wilson-lo > {c['mastery_threshold']} (>= {c['mastery_min_episodes']} ep), "
            f"dead at Wilson-hi < {c['dead_threshold']} (>= {c['dead_after']} ep), course exhausted at "
            f"{math.floor(c['exhaust_fraction'] * 100)} % retired")
