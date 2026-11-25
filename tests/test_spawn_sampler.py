"""TileSpawnSampler unit tests — pure numpy, no Isaac.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest src/FTR-Benchmark/tests/test_spawn_sampler.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# load the module by path: importing the ftr_envs.tasks package would pull in Isaac Sim
_spec = importlib.util.spec_from_file_location(
    "spawn_sampler", Path(__file__).resolve().parents[1] / "ftr_envs" / "tasks" / "crossing" / "spawn_sampler.py"
)
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)


def entries(n_tiles=10, jitter=3, difficulty=None):
    out = []
    for t in range(n_tiles):
        for j in range(jitter):
            e = {"start_point": [t * 10.0 + 4, j * 0.15, 0.6], "target_point": [t * 10.0 - 4, 0.0, 0.6]}
            if difficulty is not None:
                e["difficulty"] = difficulty[t]
            out.append(e)
    return out


def drive(s, n_episodes, p_success, rng):
    """Start + finish n episodes; tile t succeeds with probability p_success[t]."""
    for _ in range(n_episodes):
        _, t = s.next_entry()
        s.record([t], [rng.random() < p_success[t]])


def test_grouping_and_round_robin():
    s = ss.TileSpawnSampler(entries(4, 3))
    assert s.T == 4 and all(len(e) == 3 for e in s.entries_of_tile)
    seen = {}
    for _ in range(12):
        e, t = s.next_entry()
        seen.setdefault(t, []).append(e)
    for t, es in seen.items():  # jitter variants cycle within a tile
        assert len(set(es[:3])) == min(3, len(es))


def test_every_tile_visited_early():
    s = ss.TileSpawnSampler(entries(20, 1))
    tiles = {s.next_entry()[1] for _ in range(20)}
    assert len(tiles) == 20  # unvisited tiles are drawn first


def test_visit_budget_caps_starts():
    s = ss.TileSpawnSampler(entries(5, 1), {"visit_budget": 6, "exhaust_fraction": 1.0})
    rng = np.random.default_rng(0)
    drive(s, 30, [0.5] * 5, rng)
    assert s.started.max() <= 6
    assert s.exhausted


def test_mastery_and_dead_retire_early():
    s = ss.TileSpawnSampler(entries(3, 1), {"visit_budget": 1000, "exhaust_fraction": 1.0})
    rng = np.random.default_rng(1)
    drive(s, 200, [1.0, 0.0, 0.5], rng)
    assert ss.TileSpawnSampler.STATUS[s.status[0]] == "mastered"
    assert ss.TileSpawnSampler.STATUS[s.status[1]] == "dead"
    assert ss.TileSpawnSampler.STATUS[s.status[2]] == "live"
    assert s.started[0] < 20 and s.started[1] <= 20  # retired after a handful, not after the budget


def test_frontier_gets_more_episodes():
    # trivially easy / impossible tiles vs 50 % tiles: the learnability term favours the 50 % ones
    p = [1.0] * 5 + [0.0] * 5 + [0.5] * 5
    s = ss.TileSpawnSampler(entries(15, 1), {"visit_budget": 10_000, "exhaust_fraction": 1.0,
                                              "mastery_min_episodes": 10_000, "dead_after": 10_000})
    drive(s, 3000, p, np.random.default_rng(2))
    assert s.started[10:].mean() > 2 * s.started[:10].mean()


def test_exhausted_flag_and_fallback():
    s = ss.TileSpawnSampler(entries(4, 1), {"visit_budget": 2, "exhaust_fraction": 0.75})
    for _ in range(7):  # 4 first visits, then 3 second visits -> 3 of 4 tiles at budget
        s.next_entry()
    assert s.exhausted
    for _ in range(10):  # all retired: keeps handing out spawns (least visited first)
        e, t = s.next_entry()
        assert 0 <= t < 4


def test_paused_records_nothing():
    s = ss.TileSpawnSampler(entries(4, 2))
    s.paused = True
    got = [s.next_entry() for _ in range(8)]
    assert [t for _, t in got] == [-1] * 8 and sorted(e for e, _ in got) == list(range(8))
    s.record([-1] * 8, [True] * 8)
    assert s.started.sum() == 0 and s.done.sum() == 0


def test_easy_start_prefers_easy_tiles():
    d = np.linspace(0, 1, 10)
    s = ss.TileSpawnSampler(entries(10, 1, difficulty=d), {"easy_start_episodes": 10_000, "easy_start_k": 5.0,
                                                           "new_tile_weight": 0.2, "visit_budget": 10_000})
    drive(s, 400, [0.5] * 10, np.random.default_rng(3))
    assert s.started[:3].sum() > 3 * s.started[-3:].sum()


def test_state_roundtrip_same_course_and_new_course(tmp_path):
    s = ss.TileSpawnSampler(entries(5, 1))
    drive(s, 40, [0.5] * 5, np.random.default_rng(4))
    st = s.state_dict()
    s2 = ss.TileSpawnSampler(entries(5, 1))
    assert s2.load_state_dict(st) == 5 and (s2.started == s.started).all()
    other = [{"start_point": [0, 99, 0], "target_point": [0, 100.0 + i, 0]} for i in range(5)]
    s3 = ss.TileSpawnSampler(other)
    assert s3.load_state_dict(st) == 0 and s3.started.sum() == 0
    s.write_snapshot(tmp_path / "snap.json", extra={"course": "x"})
    snap = json.loads((tmp_path / "snap.json").read_text())
    assert snap["course"] == "x" and len(snap["tiles"]) == 5 and "spawn/tiles_live" in snap["stats"]


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        ss.TileSpawnSampler(entries(2, 1), {"visit_budgett": 3})
