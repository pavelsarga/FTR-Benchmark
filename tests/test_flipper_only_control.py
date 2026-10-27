"""elevation_map_emulator + velocity_control: pure torch, no Isaac.

    env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/robot/conda/envs/isaaclab/bin/python \
        -m pytest src/FTR-Benchmark/tests/test_flipper_only_control.py -q

The modules are loaded by path, and this file lives outside the ftr_envs package, because the
ftr_envs.tasks.crossing package __init__ imports Isaac.
"""
import importlib.util
import math
import os

import pytest
import torch


CROSSING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ftr_envs", "tasks", "crossing")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CROSSING, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


emu_mod = _load("elevation_map_emulator")
vc = _load("velocity_control")


def _with_margin(core, m):
    return torch.nn.functional.pad(core.unsqueeze(1), [m] * 4, mode="replicate").squeeze(1)


def test_emulator_shape_and_identity():
    hm = torch.randn(3, 45, 21)
    ident = emu_mod.ElevationMapEmulator()
    assert ident.margin == 0
    assert torch.equal(ident(hm), hm)
    emu = emu_mod.ElevationMapEmulator(5, 3, 1.0, 0.01)
    assert emu.margin == 2 + 1 + 3
    assert emu(_with_margin(hm, emu.margin)).shape == (3, 45, 21)


def test_closing_fills_a_narrow_pit_and_keeps_wide_ground():
    hm = torch.full((1, 45, 21), 0.2)
    hm[0, 20:22, :] = 0.0          # 10 cm wide trench across the window
    hm[0, 30:, :] = 0.0            # wide low ground behind
    emu = emu_mod.ElevationMapEmulator(k_dilate=3, k_erode=3)
    out = emu(_with_margin(hm, emu.margin))
    assert torch.allclose(out[0, 20:22], torch.tensor(0.2))      # the 12 cm-scale closing bridges it
    assert torch.allclose(out[0, 35:], torch.tensor(0.0))        # wide ground untouched


def test_dilation_grows_a_step_and_offset_adds():
    hm = torch.zeros(1, 45, 21)
    hm[0, :10, :] = 0.3            # step ahead, edge between rows 9 and 10
    emu = emu_mod.ElevationMapEmulator(k_dilate=5, k_erode=3, height_offset=0.01)
    out = emu(_with_margin(hm, emu.margin))
    assert out[0, 10, 10] == pytest.approx(0.31)                 # grown by (5 - 3) / 2 = 1 cell
    assert out[0, 12, 10] == pytest.approx(0.01)


def test_emulator_is_per_env():
    a, b = torch.zeros(45, 21), torch.zeros(45, 21)
    b[20, 10] = 0.5
    emu = emu_mod.ElevationMapEmulator(3, 1, 1.0)
    out = emu(_with_margin(torch.stack([a, b]), emu.margin))
    assert torch.allclose(out[0], torch.zeros(45, 21))
    assert out[1].max() > 0


def test_history_fills_on_reset_then_rolls():
    h = vc.FeatureHistory(2, 3, (45, 21), "cpu")
    h.push(torch.full((2, 45, 21), 1.0), torch.full((2, vc.PROPRIO_DIM), 1.0))
    assert torch.all(h.proprio == 1.0)                          # first frame replicated
    h.push(torch.full((2, 45, 21), 2.0), torch.full((2, vc.PROPRIO_DIM), 2.0))
    assert h.proprio[:, :, 0].tolist() == [[1, 1, 2], [1, 1, 2]]
    h.reset(torch.tensor([1]))
    h.push(torch.full((2, 45, 21), 3.0), torch.full((2, vc.PROPRIO_DIM), 3.0))
    assert h.proprio[0, :, 0].tolist() == [1, 2, 3]
    assert h.proprio[1, :, 0].tolist() == [3, 3, 3]


def test_random_velocity_range_and_reset():
    c = vc.make_velocity_controller({"type": "random", "low": 0.2, "high": 0.6, "history": 5}, 64, "cpu")
    v0 = c(None, None)
    assert v0.shape == (64,) and v0.min() >= 0.2 and v0.max() <= 0.6
    assert torch.equal(c(None, None), v0)                       # constant within an episode
    c.reset(torch.arange(32))
    v1 = c(None, None)
    assert torch.equal(v1[32:], v0[32:]) and not torch.equal(v1[:32], v0[:32])


def test_il_velocity_loads_torchscript(tmp_path):
    class Net(torch.nn.Module):
        def forward(self, hm, proprio):
            return proprio[:, -1, 0] * 10.0                     # absurd speed -> must be clamped

    path = str(tmp_path / "v.pt")
    torch.jit.script(Net()).save(path)
    c = vc.make_velocity_controller({"type": "il", "weights": path, "v_max": 0.7}, 2, "cpu")
    proprio = torch.zeros(2, 5, vc.PROPRIO_DIM)
    proprio[0, -1, 0] = 0.03
    proprio[1, -1, 0] = 1.0
    assert c(torch.zeros(2, 5, 45, 21), proprio).tolist() == pytest.approx([0.3, 0.7])


def test_bearing_heading_matches_auto_ride():
    pos = torch.zeros(3, 3)
    tgt = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [-10.0, 0.01, 0.0]])
    w = vc.bearing_heading(pos, torch.zeros(3), tgt)
    assert w[0] == pytest.approx(0.0)
    assert w[1] == pytest.approx(1.0)                           # 1.5 * pi/2 saturates at max_w
    assert w[2] == pytest.approx(1.0)                           # error wraps to ~+pi, not -pi
    w = vc.bearing_heading(pos[:1], torch.tensor([0.1]), tgt[:1], gain=2.0, max_w=5.0)
    assert w[0] == pytest.approx(-0.2)
    with pytest.raises(ValueError):
        vc.make_velocity_controller({"type": "nope"}, 1, "cpu")
    assert math.isclose(float(vc.bearing_heading(pos[:1], torch.tensor([2 * math.pi]), tgt[:1])), 0.0, abs_tol=1e-5)
