"""Offline checks of rl_modules/pan_shared.py's geometry, with a stubbed FtrEnv.

Isaac Sim is not needed: PanRewardMixin only reads plain tensors/attributes off the env.
Run with any python that has torch + einops, e.g. the isaaclab conda env:
    python src/FTR-Benchmark/rl_modules/test_pan_shared.py
"""
import sys, types, math
import torch

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src/FTR-Benchmark
sys.path.insert(0, ROOT)

# Stub the two Isaac-only imports pan_shared pulls in transitively.
m = types.ModuleType("ftr_envs.utils.torch")
m.add_noise = lambda t, std=None: t
pkg = types.ModuleType("ftr_envs"); pkg.__path__ = []
utils = types.ModuleType("ftr_envs.utils"); utils.__path__ = []
sys.modules.setdefault("ftr_envs", pkg)
sys.modules.setdefault("ftr_envs.utils", utils)
sys.modules["ftr_envs.utils.torch"] = m

from rl_modules.pan_shared import PanRewardMixin, FLIPPER_UP_SIGN

N = 3
class Cfg:
    auxiliary_wheel_radius = 0.078
class Env:
    def __init__(self, hmap):
        self.num_envs = N
        self.device = "cpu"
        self.cfg = Cfg()
        self.height_map_size = (45, 21)
        self.height_map_length = (2.25, 1.05)
        self.track_wheel_radius = 0.1165
        self.hmap_noise_std = None
        self.flipper_pos_noise_std = None
        self.current_frame_height_maps = hmap
        self.positions = torch.zeros(N, 3); self.positions[:, 2] = 0.3
        self.orientations_3 = torch.zeros(N, 3)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long)
        self.flipper_positions = torch.zeros(N, 2)
        self.flipper_torques = torch.zeros(N, 4)
        self.clearance = torch.zeros(N)

class Mod(PanRewardMixin):
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.init_pan_common()

class C(dict):
    __getattr__ = dict.get
    def get(self, k, d=None): return dict.get(self, k, d)

BASE = C(orientation_history_k=4, lambda_flipper=0.1, lambda_pitch=0.33,
         robot_wheel_base_length=0.5, rear_flipper_shaping=True,
         lateral_band_m=0.4, paper_contact_semantics=False,
         belly_clearance_min=-0.02, contact_torque_threshold=0.6,
         contact_clearance_threshold=0.03)

def make_hmap(front_h=0.0, rear_h=0.0, edge_col_h=None):
    """45x21 world-Z map. Rows 0..21 are FRONT (+x), 23..44 REAR."""
    hm = torch.full((N, 45, 21), 0.3 - 0.1165)  # flat ground at the wheel-bottom plane
    hm[:, :22, :] += front_h
    hm[:, 23:, :] += rear_h
    if edge_col_h is not None:              # bump only in the outer columns
        hm[:, :, :4] += edge_col_h
        hm[:, :, -4:] += edge_col_h
    return hm

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))
    if not cond: fails.append(name)

# ------------------------------------------- Isaac pitch sign (pins _PITCH_NOSE_DOWN_POSITIVE)
# Replicates omni.isaac.core.utils.rotations.matrix_to_euler_angles(extrinsic=True)'s
# `pitch = -arcsin(mat[2, 0])` exactly, so the constant pan_shared derives its [L]->[R]
# conversion from is asserted rather than assumed.
import numpy as np
def _quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def _isaac_pitch(b):
    q = np.array([np.cos(b/2), 0.0, np.sin(b/2), 0.0])
    return -np.arcsin(_quat_to_rot(q)[2, 0]), _quat_to_rot(q) @ np.array([1.0, 0, 0])
p_up, x_up = _isaac_pitch(np.deg2rad(-20.0))
p_dn, x_dn = _isaac_pitch(np.deg2rad(+20.0))
check("_PITCH_NOSE_DOWN_POSITIVE matches Isaac's quat_to_euler_angles",
      x_up[2] > 0 and p_up < 0 and x_dn[2] < 0 and p_dn > 0,
      f"nose-up pitch={np.rad2deg(p_up):+.1f}deg, nose-down pitch={np.rad2deg(p_dn):+.1f}deg")

# ---------------------------------------------------------------- bin x
env = Env(make_hmap()); mod = Mod(env, BASE)
x = mod.height_map_bin_x(15, "cpu", torch.float32)
check("bin 0 is the front (+x)", x[0].item() > 0 and x[-1].item() < 0, f"x[0]={x[0]:.3f} x[14]={x[-1]:.3f}")
check("bin x spans the map length", abs(x[0] - 1.05) < 1e-5 and abs(x[-1] + 1.05) < 1e-5)

# ---------------------------------------------------- height map + band
hm_band = make_hmap(edge_col_h=1.0)
env = Env(hm_band); mod = Mod(env, BASE)
narrow = mod.scanned_height_map()
mod_full = Mod(Env(hm_band), C({**BASE, "lateral_band_m": None}))
full = mod_full.scanned_height_map()
check("lateral band excludes the outer columns",
      abs(narrow.mean().item()) < 1e-5 and full.mean().item() > 0.3,
      f"narrow={narrow.mean():.4f} full={full.mean():.4f}")
check("height map is (N,15) relative to the wheel-bottom plane", tuple(narrow.shape) == (N, 15))

# ------------------------------------------------- candidate angle: step up ahead
env = Env(make_hmap(front_h=0.4)); mod = Mod(env, BASE)
mod.update_orientation_history()
hmap = mod.scanned_height_map()
ts = mod.candidate_flipper_angles(hmap)
tf_paper = mod.pan_diagnostics["theta_star_front"][0].item()
check("step ahead -> front candidate angle points UP (paper convention, +)",
      tf_paper > 0.2, f"theta*_front={tf_paper:.3f} rad")
check("front target is negative in MARV convention (front up = negative)",
      ts[0, 0].item() < 0, f"marv theta*_front={ts[0,0]:.3f}")
check("flat behind -> rear candidate angle is not raised",
      mod.pan_diagnostics["theta_star_rear"][0].item() <= 0.05,
      f"theta*_rear={mod.pan_diagnostics['theta_star_rear'][0]:.3f}")

# ------------------------------------------------- candidate angle: step up behind
env = Env(make_hmap(rear_h=0.4)); mod = Mod(env, BASE)
mod.update_orientation_history()
ts = mod.candidate_flipper_angles(mod.scanned_height_map())
tr_paper = mod.pan_diagnostics["theta_star_rear"][0].item()
check("step behind -> rear candidate angle points UP", tr_paper > 0.2, f"theta*_rear={tr_paper:.3f}")
check("rear target is positive in MARV convention (rear up = positive)", ts[0, 1].item() > 0)

# ---------------------------- pitched chassis on a matching ramp -> flipper stays flat
# The heightmap is gravity-aligned but flipper angles are chassis-relative, so the [L]->[R]
# conversion (and the sign of orientations_3[:, 1]) is the one remaining assumption in
# candidate_flipper_angles. On a ramp the robot is already aligned with, the correct
# candidate angle is ~0: nothing for the flipper to do.
SLOPE = 0.35  # rad, nose-up
xs = torch.linspace(1.125, -1.125, 45).view(1, 45, 1)
ramp = (0.3 - 0.1165) + torch.tan(torch.tensor(SLOPE)) * xs.clamp(min=0.0)
env = Env(ramp.expand(N, 45, 21).contiguous()); mod = Mod(env, BASE)
env.orientations_3[:, 1] = -SLOPE  # nose-down-positive convention -> nose UP
mod.update_orientation_history()
mod.candidate_flipper_angles(mod.scanned_height_map())
tf = mod.pan_diagnostics["theta_star_front"][0].item()
check("chassis aligned with an upward ramp -> front candidate angle ~ 0 (chassis frame)",
      abs(tf) < 0.12, f"theta*_front={tf:.3f} rad (ramp slope {SLOPE})")
# Same ramp, chassis level: the flipper now has to reach up the slope.
env2 = Env(ramp.expand(N, 45, 21).contiguous()); mod2 = Mod(env2, BASE)
mod2.update_orientation_history()
mod2.candidate_flipper_angles(mod2.scanned_height_map())
tf2 = mod2.pan_diagnostics["theta_star_front"][0].item()
check("same ramp with a level chassis -> front candidate angle rises toward the slope",
      tf2 > tf + 0.2, f"level={tf2:.3f} vs aligned={tf:.3f}")

# ------------------------------- ride-height runaway (candidate_angle_ground_referenced)
# Chassis-referenced theta* is a function of the robot's OWN height, which on a robot with
# no main tracks closes a positive feedback loop: flippers down -> robot rises -> terrain
# looks lower -> theta* points down. Ground-referencing must make theta* depend on terrain
# geometry alone WITHOUT disturbing the case that matters (obstacle ahead -> flipper up).
GROUND_Z = 0.3 - 0.1165          # make_hmap's flat surface, in the same units as positions[2]
def _theta_star(step_h, ride, ground_ref):
    env = Env(make_hmap(front_h=step_h))
    env.positions[:, 2] = 0.3 + ride
    env.clearance = torch.full((N,), float(ride))
    mod = Mod(env, C({**BASE, "candidate_angle_ground_referenced": ground_ref}))
    mod.update_orientation_history()
    mod.candidate_flipper_angles(mod.scanned_height_map())
    return mod.pan_diagnostics["theta_star_front"][0].item()

flat_chassis = [_theta_star(0.0, r, False) for r in (0.0, 0.15, 0.30)]
flat_ground  = [_theta_star(0.0, r, True)  for r in (0.0, 0.15, 0.30)]
check("chassis-referenced theta* runs away with ride height (the bug)",
      flat_chassis[2] < flat_chassis[1] < flat_chassis[0] - 0.1,
      "theta* = %s at ride 0.00/0.15/0.30" % [round(v, 3) for v in flat_chassis])
check("ground-referenced theta* is invariant to ride height (the fix)",
      max(flat_ground) - min(flat_ground) < 1e-6,
      "theta* = %s" % [round(v, 3) for v in flat_ground])
check("the fix does not disturb the obstacle case",
      abs(_theta_star(0.40, 0.0, True) - _theta_star(0.40, 0.0, False)) < 1e-6
      and _theta_star(0.40, 0.0, True) > 0.2)
check("standing tall no longer under-commands the raise over an obstacle",
      _theta_star(0.40, 0.30, True) > _theta_star(0.40, 0.30, False) + 0.1,
      "ground=%.3f vs chassis=%.3f" % (_theta_star(0.40, 0.30, True), _theta_star(0.40, 0.30, False)))

# ------------------------------- near-hinge bin domination (min_hinge_reach_m)
# theta* is a max over bins and atan2(dz, reach) blows up as reach -> 0, so the bin nearest
# a hinge dominates. A modest bump close behind the robot must not produce an extreme rear
# target once the guard is on.
def _rear_target(bump_h, min_reach):
    hm = make_hmap()
    hm[:, 23:30, :] += bump_h            # raise the terrain just behind the robot
    env = Env(hm)
    mod = Mod(env, C({**BASE, "min_hinge_reach_m": min_reach}))
    mod.update_orientation_history()
    mod.candidate_flipper_angles(mod.scanned_height_map())
    return (mod.pan_diagnostics["theta_star_rear"][0].item(),
            int(mod.pan_diagnostics["argmax_bin_rear"][0].item()))

unguarded, arg_un = _rear_target(0.12, 0.0)
guarded, arg_g = _rear_target(0.12, None)   # None -> one bin width
check("unguarded theta*_rear is dominated by the bin next to the hinge",
      unguarded > 0.5, "theta*_rear=%.3f rad (%.1f deg) from bin %d" % (unguarded, math.degrees(unguarded), arg_un))
check("min_hinge_reach_m tames the near-hinge bin",
      guarded < unguarded - 0.2, "guarded=%.3f rad (%.1f deg) from bin %d vs unguarded %.3f"
      % (guarded, math.degrees(guarded), arg_g, unguarded))
check("the guard pushes the argmax to a bin further from the hinge", arg_g != arg_un,
      "argmax %d -> %d" % (arg_un, arg_g))

# a far obstacle must be unaffected by the guard
env = Env(make_hmap(front_h=0.4)); mod = Mod(env, C({**BASE, "min_hinge_reach_m": None}))
mod.update_orientation_history(); mod.candidate_flipper_angles(mod.scanned_height_map())
guard_front = mod.pan_diagnostics["theta_star_front"][0].item()
env = Env(make_hmap(front_h=0.4)); mod = Mod(env, C({**BASE, "min_hinge_reach_m": 0.0}))
mod.update_orientation_history(); mod.candidate_flipper_angles(mod.scanned_height_map())
check("the guard leaves a genuine obstacle ahead untouched",
      abs(guard_front - mod.pan_diagnostics["theta_star_front"][0].item()) < 1e-6)

# ------------------------------------------- R_flipper scaled to 75% of the terminal reward
STEPS, TERMINAL, KAPPA = 300, 2.0, 0.005
sat_cfg = C({**BASE, "lambda_flipper_saturation_rad": 1.5708})
env = Env(make_hmap()); mod = Mod(env, sat_cfg); mod.update_orientation_history()
ts = mod.candidate_flipper_angles(mod.scanned_height_map())
env.flipper_positions = ts + torch.tensor([[1.6689, 1.6689]] * N)   # 95.6 deg off, both ends
worst = float(mod.flipper_reward(ts)[0])
frac = KAPPA * STEPS * abs(worst) / TERMINAL
check("a maximally-wrong flipper costs ~75% of the terminal reward",
      0.70 < frac < 0.80, "R=%.3f -> %.3f over %d steps = %.0f%% of +/-%.0f"
      % (worst, KAPPA * STEPS * abs(worst), STEPS, 100 * frac, TERMINAL))

env.flipper_positions = ts + torch.tensor([[0.255, 0.255]] * N)     # 14.6 deg, ICM's tracked rear
mid = abs(float(mod.flipper_reward(ts)[0]))
check("a well-tracked flipper still costs little (gradient retained, not clipped)",
      0.05 < KAPPA * STEPS * mid / TERMINAL < 0.20,
      "%.0f%% of terminal" % (100 * KAPPA * STEPS * mid / TERMINAL))

env.flipper_positions = ts + torch.tensor([[1.21, 1.21]] * N)       # 69.3 deg, AT's deviation
at_new = abs(float(mod.flipper_reward(ts)[0])) * KAPPA * STEPS
at_old = abs(float(Mod(env, BASE).flipper_reward(ts)[0])) * KAPPA * STEPS
check("the scaling is a large increase over the paper-literal lambda",
      at_new > 5 * at_old, "69.3 deg costs %.3f now vs %.3f before" % (at_new, at_old))

# ------------------------------------------------- flat terrain -> no shaping pressure
env = Env(make_hmap()); mod = Mod(env, BASE)
mod.update_orientation_history()
ts = mod.candidate_flipper_angles(mod.scanned_height_map())
env.flipper_positions = ts.clone()
r = mod.flipper_reward(ts)
check("R_flipper == 0 when both flippers sit on their candidate angle",
      float(r.abs().max()) < 1e-6, f"max|R|={float(r.abs().max()):.2e}")

# ------------------------------------------------- R_flipper range and sign
env.flipper_positions = ts + torch.tensor([[2.0, 2.0]] * N)
r = mod.flipper_reward(ts)
# The papers' lambda_1 = 0.1 puts the saturation point at Delta = 10 rad, which a +/-pi/3
# flipper cannot reach — the term floors near -0.21, not -1. Documented in
# pan_shared._saturating_lambda; this asserts the paper-literal behaviour.
check("paper lambda_1 leaves the -1 branch unreachable (floor ~ -0.21)",
      -0.25 < float(r[0]) < -0.15, f"R={float(r[0]):.4f}")
mod_sat = Mod(env, C({**BASE, "lambda_flipper_saturation_rad": 0.52}))
mod_sat.update_orientation_history()
check("lambda_flipper_saturation_rad makes the -1 branch reachable",
      abs(float(mod_sat.flipper_reward(ts)[0]) + 1.0) < 1e-6,
      f"R={float(mod_sat.flipper_reward(ts)[0]):.4f}")
env.flipper_positions = ts + torch.tensor([[0.2, 0.0]] * N)
r = mod.flipper_reward(ts)
expected = 0.5 * (-0.1 * (0.2 - math.pi / 36) + 0.0)
check("R_flipper averages front and rear (keeps [-1,0] range)",
      abs(float(r[0]) - expected) < 1e-6, f"R={float(r[0]):.5f} expected={expected:.5f}")

# ------------------------------------------------- front-only mode is still reachable
mod_front = Mod(env, C({**BASE, "rear_flipper_shaping": False}))
mod_front.update_orientation_history()
r_front = mod_front.flipper_reward(ts)
check("rear_flipper_shaping: false gives the paper's front-only term",
      abs(float(r_front[0]) - (-0.1 * (0.2 - math.pi / 36))) < 1e-6, f"R={float(r_front[0]):.5f}")

# ------------------------------------------------- R_contact
env = Env(make_hmap()); mod = Mod(env, BASE)
env.flipper_torques = torch.full((N, 4), 900.0)
env.cfg.flipper_contact_effort_limit = 1000.0
env.clearance = torch.zeros(N)
check("R_contact == 0 with all four flippers loaded and belly clear",
      float(mod.contact_reward().abs().max()) < 1e-9)
env.flipper_torques = torch.tensor([[900.0, 900.0, 10.0, 10.0]] * N)
check("R_contact == -1 when the rear pair is off the ground",
      abs(float(mod.contact_reward()[0]) + 1.0) < 1e-9)
env.flipper_torques = torch.full((N, 4), 900.0)
env.clearance = torch.full((N,), -0.10)
check("R_contact == -1 when the belly grounds out",
      abs(float(mod.contact_reward()[0]) + 1.0) < 1e-9)

# ------------------------------------------------- paper semantics still reachable
mod_p = Mod(env, C({**BASE, "paper_contact_semantics": True}))
env.clearance = torch.full((N,), 0.5)
check("paper_contact_semantics: true restores the |clearance| proxy",
      abs(float(mod_p.contact_reward()[0]) + 1.0) < 1e-9)

# ------------------------------------------------- R_pitch unchanged / bounded
env = Env(make_hmap()); mod = Mod(env, BASE)
for v in (0.0, 0.05, 0.10, 0.15):
    env.orientations_3[:, 1] = v; mod.update_orientation_history()
rp = mod.pitch_reward()
check("R_pitch is in [-1, 0]", bool((rp <= 1e-9).all() and (rp >= -1 - 1e-9).all()), f"R={float(rp[0]):.4f}")

# --------------------------------------- module_cfg_overrides (RLModule.load_module_cfg)
# Exercises the real loader against the real shipped YAMLs, so the two *_paper.yaml
# ablation configs are checked to actually flip the switches they claim to.
omni = types.ModuleType("omni"); omni.__path__ = []
omni_isaac = types.ModuleType("omni.isaac"); omni_isaac.__path__ = []
omni_lab = types.ModuleType("omni.isaac.lab"); omni_lab.__path__ = []
omni_envs = types.ModuleType("omni.isaac.lab.envs"); omni_envs.VecEnvObs = dict
for name, mod in [("omni", omni), ("omni.isaac", omni_isaac),
                  ("omni.isaac.lab", omni_lab), ("omni.isaac.lab.envs", omni_envs)]:
    sys.modules.setdefault(name, mod)

from rl_modules.rl_module import RLModule
import yaml as _yaml

class _EnvCfg:
    def __init__(self, ov): self.module_cfg_overrides = ov
class _Env:
    def __init__(self, ov): self.cfg = _EnvCfg(ov)
class _Loader(RLModule):
    def __init__(self, ov): self.env = _Env(ov)

for mod_name in ("atd3qn", "icmd3qn"):
    ypath = os.path.join(ROOT, "rl_modules", mod_name, f"{mod_name}_module.yaml")
    cfgpath = os.path.join(ROOT, "..", "..", "configs", "baselines",
                           f"marv_config_{mod_name}_paper.yaml")
    base = _Loader({}).load_module_cfg(ypath)
    check(f"{mod_name}: R_contact is enabled by default (kappa_3 = 0.005, ICM-D3QN Table 2)",
          float(base.kappa_contact) == 0.005,
          f"kappa_contact={base.kappa_contact}")
    check(f"{mod_name}: defaults keep the MARV adaptations on",
          bool(base.rear_flipper_shaping) and base.lateral_band_m is not None
          and not bool(base.paper_contact_semantics)
          and bool(base.candidate_angle_ground_referenced)
          and base.lambda_flipper_saturation_rad is not None)

    if os.path.exists(cfgpath):
        ov = _yaml.safe_load(open(cfgpath))["env_cfg_overrides"]["module_cfg_overrides"]
        merged = _Loader(ov).load_module_cfg(ypath)
        check(f"{mod_name}_paper.yaml disables every MARV adaptation",
              not bool(merged.rear_flipper_shaping) and merged.lateral_band_m is None
              and bool(merged.paper_contact_semantics)
              and not bool(merged.candidate_angle_ground_referenced)
              and merged.lambda_flipper_saturation_rad is None
              and float(merged.min_hinge_reach_m) == 0.0
              and float(merged.kappa_contact) == (0.0 if mod_name == "atd3qn" else 0.005),
              f"rear={merged.rear_flipper_shaping} band={merged.lateral_band_m} "
              f"paper_contact={merged.paper_contact_semantics} "
              f"ground_ref={merged.candidate_angle_ground_referenced} "
              f"sat={merged.lambda_flipper_saturation_rad} min_reach={merged.min_hinge_reach_m}")
        check(f"{mod_name}_paper.yaml leaves the paper constants untouched",
              float(merged.lambda_flipper) == float(base.lambda_flipper)
              and float(merged.lambda_pitch) == float(base.lambda_pitch)
              and float(merged.kappa_pitch) == float(base.kappa_pitch))
    else:
        check(f"{mod_name}_paper.yaml exists", False, cfgpath)

# the flipper-only arm must remove R_contact and boost R_flipper, nothing else
fo = os.path.join(ROOT, "..", "..", "configs", "baselines", "marv_config_atd3qn_flipperonly.yaml")
if os.path.exists(fo):
    ov = _yaml.safe_load(open(fo))["env_cfg_overrides"]["module_cfg_overrides"]
    ypath = os.path.join(ROOT, "rl_modules", "atd3qn", "atd3qn_module.yaml")
    base = _Loader({}).load_module_cfg(ypath)
    m = _Loader(ov).load_module_cfg(ypath)
    check("flipperonly: R_contact off, R_flipper boosted 3x",
          float(m.kappa_contact) == 0.0 and abs(float(m.kappa_flipper) - 3 * float(base.kappa_flipper)) < 1e-9,
          f"kappa_contact={m.kappa_contact} kappa_flipper={m.kappa_flipper}")
    check("flipperonly: every MARV adaptation still active",
          bool(m.rear_flipper_shaping) and bool(m.candidate_angle_ground_referenced)
          and m.lateral_band_m is not None and not bool(m.paper_contact_semantics)
          and m.lambda_flipper_saturation_rad is not None)
else:
    check("marv_config_atd3qn_flipperonly.yaml exists", False, fo)

try:
    _Loader({"rear_flipper_shapingg": False}).load_module_cfg(
        os.path.join(ROOT, "rl_modules", "icmd3qn", "icmd3qn_module.yaml"))
    check("a typo'd override key is rejected", False, "no error raised")
except ValueError as e:
    check("a typo'd override key is rejected", "rear_flipper_shapingg" in str(e))

print()
print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
