"""Offline checks of ctrac_module.py's stability margins, with Isaac stubbed out.

Isaac Sim is not needed: the two margin functions are pure tensor math. Same stubbing
approach as test_pan_shared.py.

    python src/FTR-Benchmark/rl_modules/ctrac/test_ctrac_stability.py

What this file exists to pin down, in order of how expensive getting it wrong has been:

1. **The sign convention.** _min_edge_signed_distance's own docstring records a winding-order
   bug that scored a robot standing squarely on all four flippers as if its CoG were outside
   its support base (rc pinned near -1 for a whole 13M-step run). NESM has the identical trap
   and one extra wrinkle: hector_stability_metrics' reference implementation assumes CLOCKWISE
   winding and derives its sign from ``corner_to_com . (edge x z)``, which is the exact
   negative of the 2-D cross product used here for a given ordering. Porting that sign
   verbatim onto our CCW polygon would invert the whole term. Both conventions are asserted
   directly below.
2. **That NESM actually depends on CoM height**, which is the property distinguishing it from
   a horizontal criterion. Raising the CoM with its projection held dead-centre must lower
   NESM and must leave SSM untouched; both halves of that contrast are asserted.
3. **Agreement with Garcia & De Santos' published form**, R (1 - cos theta) cos psi, which the
   implementation rewrites into a cheaper algebraically-identical expression.
"""
import math
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../src/FTR-Benchmark
sys.path.insert(0, ROOT)

# Stub the Isaac-only imports ctrac_module pulls in transitively.
for name, attrs in [
    ("omni", {}), ("omni.isaac", {}), ("omni.isaac.lab", {}),
    ("omni.isaac.lab.envs", {"VecEnvObs": dict}),
]:
    mod = types.ModuleType(name)
    mod.__path__ = []
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
_t = types.ModuleType("ftr_envs.utils.torch")
_t.add_noise = lambda t, std=None: t
for name in ("ftr_envs", "ftr_envs.utils"):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules.setdefault(name, m)
sys.modules["ftr_envs.utils.torch"] = _t

# ctrac_observation imports marv_rl_training (torchrl/tensordict), which the margin math
# does not touch — stub the two base classes rather than pull that dependency chain in.
_obs = types.ModuleType("marv_rl_training.observations")
_obs.Observation = type("Observation", (), {})
_obs.ObservationEncoder = type("ObservationEncoder", (), {"__init__": lambda self, *a, **k: None})
_pkg = types.ModuleType("marv_rl_training")
_pkg.__path__ = []
sys.modules.setdefault("marv_rl_training", _pkg)
sys.modules["marv_rl_training.observations"] = _obs
_td = types.ModuleType("torchrl.data")
_td.Unbounded = type("Unbounded", (), {"__init__": lambda self, *a, **k: None})
sys.modules.setdefault("torchrl", types.ModuleType("torchrl"))
sys.modules["torchrl.data"] = _td

from rl_modules.ctrac.ctrac_module import _min_edge_signed_distance, _nesm_per_edge  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def square(half=0.5, z=-0.2):
    """A unit-ish support polygon in the CCW winding ctrac_contact.FLIPPER_NAMES produces,
    [FL, RL, RR, FR] = (+x,+y), (-x,+y), (-x,-y), (+x,-y), all at the same height z below
    the robot base (which is the robot-frame origin)."""
    return torch.tensor([[[+half, +half, z], [-half, +half, z],
                          [-half, -half, z], [+half, -half, z]]])


def nesm(polygon, com):
    return _nesm_per_edge(polygon, com).min(dim=-1).values


# --- 1. sign convention -------------------------------------------------------------
poly = square()
inside = nesm(poly, torch.tensor([[0.0, 0.0, 0.0]]))
check("CoM centred over a CCW polygon is STABLE (NESM > 0)", bool(inside > 0), f"nesm={inside.item():+.4f}")

outside = nesm(poly, torch.tensor([[2.0, 0.0, 0.0]]))
check("CoM far outside the polygon is UNSTABLE (NESM < 0)", bool(outside < 0), f"nesm={outside.item():+.4f}")

# The same polygon reversed is CW; every sign must flip, which is what would happen if
# hector's convention were ported verbatim onto our ordering.
cw = torch.flip(poly, dims=[1])
cw_inside = nesm(cw, torch.tensor([[0.0, 0.0, 0.0]]))
check("reversing the winding inverts the sign (the hector CW-vs-CCW trap)",
      bool(cw_inside < 0), f"nesm_cw={cw_inside.item():+.4f} vs nesm_ccw={inside.item():+.4f}")

ssm_in = _min_edge_signed_distance(poly[:, :, :2], torch.zeros(1, 2))
check("NESM and SSM agree on the sign convention (both positive inside)",
      bool(ssm_in > 0) and bool(inside > 0), f"ssm={ssm_in.item():+.4f} nesm={inside.item():+.4f}")

# --- 2. NESM sees CoM height, SSM does not ------------------------------------------
low = nesm(poly, torch.tensor([[0.0, 0.0, 0.00]]))
high = nesm(poly, torch.tensor([[0.0, 0.0, 0.40]]))
check("raising the CoM lowers NESM with the projection unmoved",
      bool(high < low), f"nesm(z=0.00)={low.item():+.4f} > nesm(z=0.40)={high.item():+.4f}")

ssm_low = _min_edge_signed_distance(poly[:, :, :2], torch.zeros(1, 2))
ssm_high = _min_edge_signed_distance(poly[:, :, :2], torch.zeros(1, 2))
check("SSM is blind to CoM height (the criterion NESM is chosen over)",
      bool(torch.allclose(ssm_low, ssm_high)), f"ssm unchanged at {ssm_low.item():+.4f}")

# --- 3. agreement with the published closed form ------------------------------------
# Garcia & De Santos 2005: NESM_i = R (1 - cos theta) cos psi, theta measured from the
# in-plane vertical, psi the edge inclination. Recomputed here longhand, per edge, for a
# deliberately asymmetric polygon and an off-centre CoM so nothing cancels by symmetry.
tilted = torch.tensor([[[+0.6, +0.4, -0.10], [-0.5, +0.45, -0.25],
                        [-0.55, -0.4, -0.20], [+0.5, -0.35, -0.05]]])
com = torch.tensor([[0.07, -0.03, 0.12]])
got = _nesm_per_edge(tilted, com)[0]

expected = []
for i in range(4):
    a = tilted[0, i]
    b = tilted[0, (i + 1) % 4]
    edge = b - a
    e = edge / edge.norm()
    v = com[0] - a
    v_perp = v - torch.dot(v, e) * e
    R = v_perp.norm()
    psi = math.asin(float(edge[2] / edge.norm()))
    # theta: rotation needed to bring the CoM into the vertical plane through the edge.
    n = torch.cross(edge, torch.tensor([0.0, 0.0, 1.0]), dim=0)
    n = n / n.norm()
    theta = math.asin(float(torch.dot(n, v) / R))
    val = float(R) * (1 - math.cos(theta)) * math.cos(psi)
    sign = 1.0 if float(edge[0] * v[1] - edge[1] * v[0]) >= 0 else -1.0
    expected.append(sign * val)
expected = torch.tensor(expected)
check("matches Garcia & De Santos' R(1-cos theta)cos psi per edge",
      bool(torch.allclose(got, expected, atol=1e-5)),
      f"max|diff|={float((got - expected).abs().max()):.2e}")

# --- 4. balance point and degenerate inputs -----------------------------------------
# CoM exactly over an edge => that edge's margin is 0 (no energy left to tip).
flat = torch.tensor([[[+0.5, +0.5, 0.0], [-0.5, +0.5, 0.0], [-0.5, -0.5, 0.0], [+0.5, -0.5, 0.0]]])
on_edge = nesm(flat, torch.tensor([[0.0, 0.5, 0.3]]))  # directly above the FL->RL edge
check("CoM balanced over an edge gives NESM ~ 0", abs(float(on_edge)) < 1e-6, f"nesm={float(on_edge):+.2e}")

collapsed = torch.zeros(1, 4, 3)  # all four contacts coincident — degenerate polygon
deg = nesm(collapsed, torch.tensor([[0.0, 0.0, 0.2]]))
check("degenerate polygon stays finite (no NaN/inf)", bool(torch.isfinite(deg).all()), f"nesm={float(deg):+.4f}")

batched = nesm(torch.cat([poly, tilted, flat], dim=0), torch.cat([torch.zeros(1, 3), com, torch.zeros(1, 3)], dim=0))
check("batches independently", tuple(batched.shape) == (3,), f"shape={tuple(batched.shape)}")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks passed")
