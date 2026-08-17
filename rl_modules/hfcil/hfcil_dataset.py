"""Demonstration dataset format + validation for HFC-IL.

A recording is a directory of `shard_*.npz` (or one `.npz`), each holding, per timestep and
in acquisition order:

    obs        (T, 24) float32  HFCObservation's exact layout, see FIELD_LAYOUT below
    state      (T,)    int64    the demonstrator's state index AT that timestep (0..7)
    episode    (T,)    int64    contiguous run id; a transition is only valid within one

`state[t]` is the state the machine is IN at t, i.e. Control.cpp's `statemachine_state`
AFTER that tick's button handling — exactly what mode 8 publishes on `~/flippers_mode`.
Training pairs are therefore (obs[t], state[t]) -> state[t+1], which is what the operator
decided given what they saw. See build_pairs().

Why the episode column is not optional: concatenating two recordings creates a boundary
where state[t] and state[t+1] come from different runs. That pair is not a demonstrated
transition, and is frequently an illegal one (e.g. AR at the end of one run, DF at the
start of the next), which the validator would then report as corruption.

The 24-D observation must be reproduced from the real robot in the SAME units and frame as
the simulator's HFCObservation, or the classifier will not transfer. Notably every field is
RAW (radians, m/s, metres) — HFCObservation sets supports_vecnorm = False precisely so
these stay physical.
"""
import logging
from pathlib import Path

import numpy as np

from rl_modules.hfcil.hfcil_transitions import (
    LEGAL_SUCCESSORS,
    NUM_STATES,
    STATE_SHORT_NAMES,
    is_legal,
)

_log = logging.getLogger(__name__)

OBS_DIM = 24

# Mirrors hfc_policy.py's index constants (HM_DIM=15, then the scalars). Kept as text for
# the recorder's benefit — the authoritative indices live in hfc_policy.
FIELD_LAYOUT = """
  [0:15]  terrain     15 heightmap band means, metres, RELATIVE to the track contact plane
                      (sim: the 45x21 @ 0.05 m/cell robot-centric grid, mean-pooled over
                      3 rows x all 21 columns -> 15 bands of 0.15 m depth spanning
                      x in [-1.125, +1.125] m, y in [-0.525, +0.525] m, minus
                      robot_z - track_wheel_radius)
  [15:19] flippers    FL, FR, RL, RR joint angles, RAW radians
  [19]    roll        RAW radians
  [20]    roll_rate   RAW rad/s
  [21]    pitch       RAW radians
  [22]    fwd_vel     body-frame forward velocity, RAW m/s
  [23]    reset_flag  1.0 on the first step of an episode, else 0.0
"""


def build_pairs(obs: np.ndarray, state: np.ndarray, episode: np.ndarray):
    """-> (obs_t, state_t, next_state_t) with cross-episode boundaries dropped."""
    same = episode[:-1] == episode[1:]
    return obs[:-1][same], state[:-1][same], state[1:][same]


def validate(obs: np.ndarray, state: np.ndarray, episode: np.ndarray, strict: bool = True) -> dict:
    """Check a recording against the demonstrator's own rules. Returns a stats dict; raises
    on corruption when strict.

    An illegal (s -> s') pair means the recording does not describe the state machine in
    hfcil_transitions — a desynchronised label stream, a dropped message, or two runs
    concatenated without bumping the episode id. It is never something to train through,
    so this raises instead of dropping the offending pairs: silently discarding them would
    hide a systematic recording fault behind a slightly smaller dataset.
    """
    problems = []
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        problems.append(f"obs must be (T, {OBS_DIM}), got {obs.shape}")
    if not (len(obs) == len(state) == len(episode)):
        problems.append(f"length mismatch: obs {len(obs)}, state {len(state)}, episode {len(episode)}")
    if state.min() < 0 or state.max() >= NUM_STATES:
        problems.append(f"state out of range [0, {NUM_STATES}): [{state.min()}, {state.max()}]")
    if not np.isfinite(obs).all():
        problems.append(f"{(~np.isfinite(obs)).sum()} non-finite values in obs")
    if problems and strict:
        raise ValueError("Invalid HFC-IL recording:\n  " + "\n  ".join(problems))

    o, s, s_next = build_pairs(obs, state, episode)
    illegal = np.array([not is_legal(int(a), int(b)) for a, b in zip(s, s_next)])
    if illegal.any():
        idx = np.flatnonzero(illegal)[:5]
        detail = ", ".join(f"{STATE_SHORT_NAMES[s[i]]}->{STATE_SHORT_NAMES[s_next[i]]}@{i}" for i in idx)
        msg = (f"{illegal.sum()} illegal transitions ({100*illegal.mean():.2f}% of pairs), e.g. {detail}. "
               "The label stream does not match Control.cpp's state machine — check for dropped "
               "/flippers_mode messages, mis-synchronised timestamps, or runs concatenated without "
               "a new episode id.")
        if strict:
            raise ValueError(msg)
        problems.append(msg)

    changes = s != s_next
    per_edge: dict[tuple[int, int], int] = {}
    for a, b in zip(s[changes], s_next[changes]):
        per_edge[(int(a), int(b))] = per_edge.get((int(a), int(b)), 0) + 1

    return {
        "timesteps": int(len(obs)),
        "episodes": int(len(np.unique(episode))),
        "pairs": int(len(s)),
        "transitions": int(changes.sum()),
        "stay_fraction": float(1.0 - changes.mean()) if len(s) else float("nan"),
        "state_counts": {STATE_SHORT_NAMES[i]: int((state == i).sum()) for i in range(NUM_STATES)},
        "edge_counts": {f"{STATE_SHORT_NAMES[a]}->{STATE_SHORT_NAMES[b]}": n for (a, b), n in sorted(per_edge.items())},
        "problems": problems,
    }


def coverage_report(stats: dict, target_per_edge: int = 200) -> str:
    """Which demonstrated edges are still under-sampled. Only *branching* states carry a
    real decision — a state with one legal successor is a timing decision, not a choice —
    so those are reported separately and need far fewer examples.
    """
    lines, edges = [], stats["edge_counts"]
    branching, timing = [], []
    for s, succ in LEGAL_SUCCESSORS.items():
        real = [x for x in succ if x != s]
        for d in real:
            key = f"{STATE_SHORT_NAMES[s]}->{STATE_SHORT_NAMES[d]}"
            (branching if len(real) > 1 else timing).append((key, edges.get(key, 0)))

    lines.append(f"{'edge':<12}{'count':>8}   (target {target_per_edge} for choices, {target_per_edge // 4} for timing)")
    lines.append("-- branching (which successor?) " + "-" * 24)
    for k, n in branching:
        lines.append(f"{k:<12}{n:>8}   {'OK' if n >= target_per_edge else 'NEEDS MORE'}")
    lines.append("-- single-successor (when to switch?) " + "-" * 18)
    for k, n in timing:
        t = target_per_edge // 4
        lines.append(f"{k:<12}{n:>8}   {'OK' if n >= t else 'NEEDS MORE'}")
    return "\n".join(lines)


def load(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a shard directory or single .npz, concatenating with episode ids kept unique
    across shards (see the module docstring on why that matters)."""
    p = Path(path)
    files = sorted(p.glob("shard_*.npz")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"No shard_*.npz found in {p}")
    obs_l, st_l, ep_l, offset = [], [], [], 0
    for f in files:
        d = np.load(f)
        ep = d["episode"].astype(np.int64)
        obs_l.append(d["obs"].astype(np.float32))
        st_l.append(d["state"].astype(np.int64))
        ep_l.append(ep + offset)
        offset = int(ep_l[-1].max()) + 1
    return np.concatenate(obs_l), np.concatenate(st_l), np.concatenate(ep_l)
