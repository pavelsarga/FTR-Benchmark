"""Ground-truth per-flipper contact point/existence extraction for C-TRAC (Pan et al.
2025, "C-TRAC: Terrain-Adaptive Control for Articulated Tracked Robots via Contact-Aware
Reinforcement Learning"). Nothing else in this project computes real per-flipper contact
points or reads PhysX contact forces — the existing "flipper_contact" reward term
(marv_rl_module.py) is a torque proxy (env.flipper_torques == applied joint torque, not a
measured contact force/position), and icmd3qn_module.py's contact term is a single-point
chassis-clearance proxy. Neither gives what the paper's c_t (Eq. 2, 4x3 positions) /
c_prob_t (Eq. 2, 4-vector existence) need.

This combines two IsaacLab data sources, each used for the part it's actually good at:
  - omni.isaac.lab.sensors.contact_sensor.ContactSensor gives REAL per-wheel-body contact
    force (net_forces_w) straight from PhysX — used here for contact EXISTENCE. It does
    NOT give the exact point of application on the body, though.
  - Articulation.data.body_pos_w gives the REAL world-frame position of every rigid body
    in the articulation (including every flipper wheel link) straight from PhysX — no
    manual forward kinematics needed.
Combined with each wheel's known radius (MARV's wheel geometry is analytic, see
MarvWheelArticulation in ftr_envs/assets/articulation/marv.py), a wheel in contact touches
ground at approximately (wheel_center - radius along local "down"). This assumes a locally
near-vertical contact normal (a reasonable first-order approximation given wheel radii of
~0.08-0.12 m relative to typical terrain slope scale) rather than projecting along the true
terrain surface normal — documented simplification, kept deliberately simple since contact
EXISTENCE (the harder-to-fake part) already comes from real sensor data, not this
approximation; only the Z offset of an already-known-correct XY point is affected.

Per-flipper (matching the paper's 4-point c_t, i=1..4, not per-wheel): 5 wheels per flipper
are aggregated into one point (force-weighted average of the wheels currently in contact,
falling back to the lowest/closest-to-ground wheel if none register contact so the signal
degrades gracefully instead of going NaN) and one binary existence flag (any wheel's contact
force exceeds force_threshold).
"""
import torch

# Matches env.flipper_positions' column order [FL, FR, RL, RR] (see ftr.py / marv.py's
# flipper_dof_idx_list unpacking: front_left_dof, front_right_dof, rear_left_dof, rear_right_dof).
FLIPPER_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
NUM_WHEELS_PER_FLIPPER = 5

# MARV wheel geometry (matches MarvWheelArticulation._BIG_RADIUS/_SMALL_RADIUS in
# ftr_envs/assets/articulation/marv.py — "wheel 1 is always the big-radius end, wheel n is
# the small-radius end, for both front and rear flippers", so the same radii sequence
# applies to every one of the 4 flippers, not just a track-side-grouped subset).
_BIG_RADIUS = 0.1165
_SMALL_RADIUS = 0.0780

# Wheel *body* prim names, siblings of the robot's own container prim (not nested under
# their parent flipper link) — confirmed directly against MarvWheelArticulation.set_robot_env
# ("front_left_flipper_wheel1", ... "front_left_flipper_wheel5", one per flipper x 5 wheels).
_WHEEL_BODY_PATTERN = "{flipper}_flipper_wheel[1-5]"


def resolve_flipper_wheel_body_ids(robot) -> dict[str, torch.Tensor]:
    """Per-flipper LongTensor(5,) of body indices into robot.data.body_pos_w's body axis,
    ordered wheel1..wheel5 (regex match order from find_bodies is not guaranteed sorted,
    so this explicitly sorts by the trailing wheel-number digit in each matched body name).
    """
    result: dict[str, torch.Tensor] = {}
    for flipper in FLIPPER_NAMES:
        body_ids, body_names = robot.find_bodies(_WHEEL_BODY_PATTERN.format(flipper=flipper))
        if len(body_ids) != NUM_WHEELS_PER_FLIPPER:
            raise RuntimeError(
                f"resolve_flipper_wheel_body_ids: expected {NUM_WHEELS_PER_FLIPPER} wheel bodies "
                f"for {flipper!r}, found {len(body_ids)} ({body_names!r}). Wheel body prim naming "
                f"may have changed — see MarvWheelArticulation.set_robot_env."
            )
        order = sorted(range(len(body_names)), key=lambda i: int(body_names[i][-1]))
        result[flipper] = torch.as_tensor([body_ids[i] for i in order], dtype=torch.long)
    return result


class CTRACContactExtractor:
    """Owns the resolved wheel body/sensor indices for one env and computes ground-truth
    (contact_points, contact_prob) each call. Constructed once by CTRACModule.__init__
    (after the env's scene/robot/sensor already exist — CrossingEnv builds self.rl_module
    only after FtrEnv.__init__/_setup_scene has run).

    env._ctrac_contact_sensor is set by ftr_env.py's _setup_scene (gated on
    module_name == "ctrac" — see that file). If it's absent (e.g. sensor attachment failed,
    or this extractor is unit-tested outside Isaac Sim with a stub env), contact_prob is
    always 0 and points fall back to the lowest wheel per flipper — a graceful degradation,
    not a crash, since existence is the harder-to-fake half of this signal.
    """

    def __init__(self, env, force_threshold: float = 1.0):
        self.env = env
        self.force_threshold = force_threshold
        self._wheel_body_ids = resolve_flipper_wheel_body_ids(env._robot)
        self._wheel_radii = torch.linspace(
            _BIG_RADIUS, _SMALL_RADIUS, NUM_WHEELS_PER_FLIPPER, device=env.device
        )  # (5,), wheel1..wheel5 — same sequence for every flipper

        sensor = getattr(env, "_ctrac_contact_sensor", None)
        self._sensor_body_ids: dict[str, torch.Tensor] | None = None
        if sensor is not None:
            self._sensor_body_ids = {}
            for flipper in FLIPPER_NAMES:
                sensor_ids, sensor_names = sensor.find_bodies(_WHEEL_BODY_PATTERN.format(flipper=flipper))
                order = sorted(range(len(sensor_names)), key=lambda i: int(sensor_names[i][-1]))
                self._sensor_body_ids[flipper] = torch.as_tensor([sensor_ids[i] for i in order], dtype=torch.long)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (contact_points (N,4,3) ROBOT-frame, contact_prob (N,4) in {0.0, 1.0}),
        flipper order matching FLIPPER_NAMES / env.flipper_positions ([FL,FR,RL,RR]).

        Points are relative to the robot base and yaw-derotated (x forward, y left, z the
        height of the contact below the base), matching the paper's "4x2 relative coords of
        contacts" — c_t is a robot-relative quantity there, not a world position. We keep
        the z component too (a superset of the paper's 2-D c_t); it is the same small,
        robot-relative scale as x/y and the NESM reward reads the xy slice only.

        Returning raw world-frame body_pos_w here (as this did originally) is unusable
        downstream: the contact points are both a C-VAE *regression target* and part of the
        privileged critic observation, and no vecnorm runs on the ctrac obs. On a course
        with world coordinates in the ~10-20 m range that made L_est ~= 268 (RMS ~16 m, an
        absolute position that is simply not inferable from a robot-centric heightmap), so
        the C-VAE spent all its capacity on an unlearnable target while feeding the critic
        huge unnormalised inputs. The same yaw-only convention is used for goal_xy in
        CTRACModule.get_observations, so the two agree.
        """
        env = self.env
        sensor = getattr(env, "_ctrac_contact_sensor", None)
        body_pos_w = env._robot.data.body_pos_w  # (N, num_bodies, 3) — real PhysX positions

        points, probs = [], []
        for flipper in FLIPPER_NAMES:
            ids = self._wheel_body_ids[flipper].to(body_pos_w.device)
            wheel_pos = body_pos_w[:, ids, :]  # (N, 5, 3)
            contact_pts = wheel_pos.clone()
            contact_pts[..., 2] = contact_pts[..., 2] - self._wheel_radii.unsqueeze(0)  # project down by radius

            if sensor is not None and self._sensor_body_ids is not None:
                sensor_ids = self._sensor_body_ids[flipper].to(body_pos_w.device)
                force_mag = sensor.data.net_forces_w[:, sensor_ids, :].norm(dim=-1)  # (N, 5)
            else:
                force_mag = torch.zeros(wheel_pos.shape[0], NUM_WHEELS_PER_FLIPPER, device=wheel_pos.device)

            in_contact = force_mag > self.force_threshold  # (N, 5)
            any_contact = in_contact.any(dim=-1)  # (N,)
            weight = torch.where(in_contact, force_mag, torch.zeros_like(force_mag))
            weight_sum = weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            weighted_point = (contact_pts * weight.unsqueeze(-1)).sum(dim=1) / weight_sum  # (N, 3)

            lowest_idx = contact_pts[..., 2].argmin(dim=-1)  # (N,) — fallback: closest-to-ground wheel
            fallback_point = contact_pts.gather(1, lowest_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
            point = torch.where(any_contact.unsqueeze(-1), weighted_point, fallback_point)

            points.append(point)
            probs.append(any_contact.float())

        contact_points = torch.stack(points, dim=1)  # (N, 4, 3), still world frame here
        contact_prob = torch.stack(probs, dim=1)  # (N, 4)

        # World -> robot frame: translate to the base, then de-rotate by yaw only (same
        # convention as goal_xy in CTRACModule.get_observations).
        rel = contact_points - env.positions.unsqueeze(1)  # (N,4,3)
        yaw = env.orientations_3[:, 2]
        cos_y, sin_y = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)  # (N,1)
        x_b = cos_y * rel[..., 0] + sin_y * rel[..., 1]
        y_b = -sin_y * rel[..., 0] + cos_y * rel[..., 1]
        contact_points = torch.stack([x_b, y_b, rel[..., 2]], dim=-1)  # (N,4,3)
        return contact_points, contact_prob
