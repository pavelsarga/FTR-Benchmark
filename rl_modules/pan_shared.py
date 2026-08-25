"""Shared terrain geometry and reward primitives for the two Pan et al. 2023 flipper-control
reproductions — AT-D3QN (``rl_modules/atd3qn``) and ICM-D3QN (``rl_modules/icmd3qn``) — plus
the MARV-morphology adaptations both of them need.

Why this file exists
--------------------
Both papers were written for the authors' NuBot-Rescue platform: a **six-track** robot, i.e.
two large *main* body tracks that carry the driving force, plus four auxiliary flippers whose
job is posture adjustment. Every reward in both papers is stated in terms of that split:

  * ``R_flipper`` (Eq. 4) shapes the **front** flipper only, because the rear flipper is
    auxiliary and the main track behind the CoM already carries the load.
  * ``R_contact`` (ICM paper, Eq. 7-8) **penalises** the pose where the extreme contact
    points are both on the flippers, because that means the main tracks — the propulsion —
    are off the ground.

MARV has **no main tracks**. Its four flipper units are the entire drivetrain: ``MARV_CFG``
actuates only ``{front,rear}_{left,right}_flipper_wheel{1..5}_j`` (see
``ftr_envs/assets/marv.py``; FTR's ``baselink_wheel`` actuator block is commented out in
``ftr_envs/assets/ftr.py``). Consequently:

  * the rear flipper is *not* auxiliary — it is half the drivetrain and half the
    terrain-conforming surface, so shaping only the front leaves half the robot unsupervised;
  * "flippers holding the chassis off the ground" is the **correct** driving pose on MARV,
    not the failure mode. The failure mode is the opposite: the belly grounding out, or a
    flipper pair losing ground contact and with it its share of the traction.

The adaptations below keep each paper equation's *shape* (same functional form, same
``[-1, 0]`` range, same ``lambda``/``kappa`` meaning) and change only which body part the
equation refers to. Every one of them is gated by a key in the owning module's YAML so the
literal, paper-faithful behaviour is still reachable for a baseline run.

Frames and sign conventions (the part that is easy to get wrong)
---------------------------------------------------------------
``FtrEnv.calc_current_frame_height_maps`` yaw-derotates the terrain around the robot but does
**not** un-pitch it, and subtracts ``positions[:, 2] - track_wheel_radius``. The 45x21 map is
therefore already in the papers' gravity-aligned, robot-centred ``[L]`` frame, with heights
measured from the plane through the bottom of the drive wheels, and (after that function's
``local_map.flip(0)``) **row 0 = front (+x), row 44 = rear (-x)** — bin index *decreases* in x.

Flipper angles, by contrast, live in the chassis ``[R]`` frame, so a candidate angle computed
in ``[L]`` must be converted by the chassis pitch before it can be compared to one
(``_PITCH_NOSE_DOWN_POSITIVE`` below).

MARV's own flipper sign convention (``ftr_env.py``, ``marv_flipper_*_deg``) is *not* the
papers': MARV uses front-up = **negative**, rear-up = **positive**, while both papers define
theta_f as positive when the flipper is above the chassis for *both* ends. ``FLIPPER_UP_SIGN``
below carries that mapping; comparing a paper-convention candidate angle against a raw MARV
front-flipper angle drives the front flipper the wrong way.
"""
from __future__ import annotations

import logging

import einops
import torch

_log = logging.getLogger(__name__)


# VERIFIED against Isaac's own implementation, not assumed: omni.isaac.core.utils.rotations
# .quat_to_euler_angles -> matrix_to_euler_angles(extrinsic=True) computes
# ``pitch = -arcsin(mat[2, 0])``, and mat[2, 0] is the world-z component of the body +x
# axis. Nose up therefore gives mat[2, 0] > 0 and a NEGATIVE pitch:
#
#     nose UP   20 deg -> body +x -> [0.940, 0, +0.342] -> pitch = -20.00 deg
#     nose DOWN 20 deg -> body +x -> [0.940, 0, -0.342] -> pitch = +20.00 deg
#
# So orientations_3[:, 1] > 0 means nose-down. Consequence for the candidate angle: a
# terrain direction sitting at theta above the horizon in the gravity-aligned [L] frame
# sits at theta + pitch relative to a chassis pitched nose-down by `pitch`.
# test_pan_shared.py asserts this arithmetic directly, so it cannot regress silently.
_PITCH_NOSE_DOWN_POSITIVE = True

# Maps the papers' "positive = flipper above the chassis" convention onto MARV's
# per-end convention (ftr_env.py:83-92): front up = negative angle, rear up = positive.
# Indexing matches env.flipper_positions' column order under sync_flipper_control:
# [front, rear].
FLIPPER_UP_SIGN = (-1.0, +1.0)

# Papers' tolerance band on the candidate angle (Eq. 4's "theta*_f1 +/- pi/36").
ANGLE_TOLERANCE = torch.pi / 36

# N in Eq. 1 — the number of terrain sub-point-sets, and hence the width of H. Both papers
# use 15, and both observation classes hardcode it in their `dim`.
PAN_N_BINS = 15


class PanRewardMixin:
    """Terrain/reward primitives shared by ATD3QNModule and ICMD3QNModule.

    Expects the host class to set ``self.env`` (an ``FtrEnv``) and ``self.cfg`` (the module's
    own OmegaConf YAML) before calling ``init_pan_common()``.
    """

    # ---------------------------------------------------------------- setup

    def init_pan_common(self) -> None:
        env, cfg = self.env, self.cfg
        self.orientations_3_history = torch.zeros(
            env.num_envs, cfg.orientation_history_k, 3, device=env.device
        )
        # Populated lazily on first use — body_pos_w is not meaningful until the sim has
        # stepped, and RLModule is constructed before that (crossing_env.py:200).
        self._pivot_half_base: float | None = None
        self._contact_extractor = None
        self._contact_extractor_tried = False
        # Last-computed diagnostics, surfaced through get_reward_components() so a short
        # debug run makes every sign/frame assumption in this file directly checkable.
        self.pan_diagnostics: dict[str, torch.Tensor] = {}

    def update_orientation_history(self) -> None:
        """Roll in env.orientations_3 as the newest entry; re-seed envs that just reset."""
        env = self.env
        self.orientations_3_history = torch.cat(
            [self.orientations_3_history[:, 1:], env.orientations_3.unsqueeze(1)], dim=1
        )
        fresh = env.episode_length_buf == 0
        if fresh.any():
            k = self.cfg.orientation_history_k
            self.orientations_3_history[fresh] = env.orientations_3[fresh].unsqueeze(1).expand(-1, k, -1)

    # ------------------------------------------------------- robot geometry

    def pivot_half_base(self) -> float:
        """Half the front-to-rear flipper *pivot* distance, in metres — the papers' hinge
        offset ``p_bf`` (Fig. 4), measured from the actual articulation instead of the
        hardcoded ``robot_wheel_base_length: 0.5`` FTR literal the module YAMLs used to
        carry. Wheel 1 of each flipper sits exactly at that flipper's pivot (see
        ``MarvWheelArticulation``'s ``_AXLE_DISTANCE`` comment), so the front/rear wheel-1
        body positions give the pivot base directly.

        Falls back to the YAML's ``robot_wheel_base_length`` if the articulation cannot be
        queried (non-MARV robot, or a stubbed env in a unit test).
        """
        if self._pivot_half_base is not None:
            return self._pivot_half_base

        fallback = float(self.cfg.robot_wheel_base_length) / 2.0
        try:
            from rl_modules.ctrac.ctrac_contact import resolve_flipper_wheel_body_ids

            ids = resolve_flipper_wheel_body_ids(self.env._robot)
            body_pos = self.env._robot.data.body_pos_w  # (N, num_bodies, 3), world frame
            # Wheel 1 == the pivot. Average the left/right member of each pair, then take
            # the front-rear separation of env 0 (identical across envs by construction).
            front = 0.5 * (body_pos[0, ids["front_left"][0], :2] + body_pos[0, ids["front_right"][0], :2])
            rear = 0.5 * (body_pos[0, ids["rear_left"][0], :2] + body_pos[0, ids["rear_right"][0], :2])
            base = float((front - rear).norm().item())
            self._pivot_half_base = base / 2.0 if 0.05 < base < 3.0 else fallback
        except Exception:
            self._pivot_half_base = fallback
        return self._pivot_half_base

    def hinge_height(self) -> float:
        """Flipper pivot height above the heightmap's reference plane. The map is measured
        from ``positions[:, 2] - track_wheel_radius`` (the plane through the bottom of the
        drive wheels) and the pivot sits at the drive-wheel axle, one drive radius up."""
        return float(self.env.track_wheel_radius)

    def tip_radius(self) -> float:
        """Radius of the wheel at the far end of a flipper, used to "expand" the terrain
        outline the way the papers expand it by the flipper's arc radius (Fig. 4). Replaces
        the ``robot_wheel_diam: 0.24`` FTR literal, which the two modules were additionally
        using inconsistently (one added it whole, the other halved it)."""
        return float(getattr(self.env.cfg, "auxiliary_wheel_radius", 0.078))

    # -------------------------------------------------------- terrain (Eq. 1)

    def scanned_height_map(self, base_robot_frame: bool = True) -> torch.Tensor:
        """H (Eq. 1): N=15 terrain-height averages, (N, 15), row 0 = front.

        Adaptation (``lateral_band_m``): the papers average each sub-point-set over "the
        terrain in front, back and below the robot". The unmodified implementation averaged
        the full 1.05 m map width — nearly three times MARV's 0.36 m track footprint — so
        each bin mixed in terrain the robot never touches. Restricting the average to a band
        the width of the drivetrain keeps Eq. 1 exactly as written and just stops widening
        its point set beyond the robot. ``null`` restores the full-width behaviour.
        """
        from ftr_envs.utils.torch import add_noise

        env = self.env
        h, w = env.height_map_size
        shaped_map = torch.reshape(env.current_frame_height_maps, (-1, 1, h, w))
        if base_robot_frame:
            shaped_map = shaped_map - einops.repeat(
                env.positions[:, 2] - env.track_wheel_radius, "n -> n c rh rw", c=1, rh=h, rw=w
            )
        height_maps = shaped_map.squeeze(1)

        band_m = self.cfg.get("lateral_band_m", None)
        if band_m is not None:
            cell = env.height_map_length[1] / w
            half_cols = max(0, int(round((float(band_m) / 2.0) / cell)))
            centre = w // 2
            lo, hi = max(0, centre - half_cols), min(w, centre + half_cols + 1)
            height_maps = height_maps[:, :, lo:hi]

        if h % PAN_N_BINS != 0:
            raise ValueError(
                f"Eq. 1 needs the heightmap's {h} rows to divide into N={PAN_N_BINS} equal "
                "sub-point-sets; the atd3qn/icmd3qn observation width is fixed at 15 "
                "(ATD3QNObservation.dim / ICMD3QNObservation.dim). Check height_map_size — "
                "flipper_style's 65x65 map is not compatible with these modules."
            )
        height_maps = einops.reduce(height_maps, "n (h k) w -> n h", reduction="mean", k=h // PAN_N_BINS)
        return add_noise(height_maps, std=env.hmap_noise_std)

    def height_map_bin_x(self, n_bins: int, device, dtype) -> torch.Tensor:
        """Body-frame x of each H bin centre, in metres. Row 0 is the front (+x), so x
        *decreases* with bin index — the opposite of what the previous ``hmap_x_increasing:
        true`` assumed, which is why the ICM candidate angle was being computed from the
        rear half of the map."""
        span = self.env.height_map_length[0]
        bin_w = span / n_bins
        idx = torch.arange(n_bins, device=device, dtype=dtype)
        return span / 2.0 - (idx + 0.5) * bin_w

    # ------------------------------------------------- candidate angle (Eq. 4)

    def candidate_flipper_angles(self, height_map: torch.Tensor) -> torch.Tensor:
        """theta*_f (Fig. 4) for the front and rear flipper, returned as (N, 2) **already in
        MARV's own per-end angle convention**, so it can be differenced directly against
        ``env.flipper_positions``.

        Construction, per end, exactly as the papers draw it: take the flipper hinge as the
        reference point, form the vector from it to every "expanded" terrain point beyond it,
        and keep the largest angle. Three things differ from the previous implementations,
        all of them corrections rather than departures:

        1. **Direction.** Bins ahead of the front hinge / behind the rear hinge are selected
           by their actual body-frame x (``height_map_bin_x``), not by a fixed ``[:5]`` /
           ``[half:]`` slice under an inverted ``hmap_x_increasing`` assumption.
        2. **One frame throughout.** The heightmap is already relative to the plane under
           the drive wheels, so the hinge is placed in that same frame at
           ``hinge_height()``. AT-D3QN previously mixed a relative ``max_z`` with an absolute
           world ``z``, which offset the atan2 numerator by roughly the robot's ride height
           and pinned the result near -pi/2.
        3. **Rear end included.** The papers define theta* for the front only; on MARV the
           rear flipper is drivetrain, so the same construction is mirrored about the CoM.
           ``rear_flipper_shaping: false`` reverts to front-only.
        4. **Evaluated at the nominal ride height**, not the current one
           (``candidate_angle_ground_referenced``). Without this the term has a runaway
           fixed point on MARV -- see below.

        The ride-height runaway
        -----------------------
        Both papers place the reference frame at the chassis centre, which is well-defined
        for them because their robot's **main tracks pin the chassis height**: terrain
        measured relative to the chassis is then a function of terrain shape alone.

        MARV has no main tracks, so its chassis height is a free variable that the flippers
        themselves set. Measured chassis-relative, standing tall makes *all* terrain appear
        to sink, so theta* rotates downward, so the reward drives the flippers down, so the
        robot stands taller still. Measured on the shipped code (flat ground, front flipper):

            ride height  +0.00 m -> theta* = -0.048 rad   (target: slightly down)
            ride height  +0.15 m -> theta* = -0.231 rad   (target: down)
            ride height  +0.30 m -> theta* = -0.400 rad   (target: hard down)

        The loop is visible in every run: theta*_front is anti-correlated with
        ``state/clearance_mean`` (r = -0.27 to -0.60 across the four D3QN runs), strongest in
        ``logs/train_marv_icmd3qn_11366234``, which converged to all four flippers pinned at
        their down limit with the robot balanced on the flipper tips.

        Note this is NOT a sign error in Eq. 4's ``|theta_f1 - theta*|``: with an obstacle
        ahead at nominal ride height theta* is correctly +1.047 rad (flipper up). Negating
        theta* would fix the degenerate case only by inverting the correct one.

        Referencing the terrain to the ground under the robot and keeping the hinge at its
        nominal height makes theta* a function of terrain geometry alone, which is what the
        papers' construction means on a robot whose ride height does not move.
        ``candidate_angle_ground_referenced: false`` restores the paper-literal
        chassis-referenced behaviour.
        """
        env = self.env
        n = height_map.shape[-1]
        x = self.height_map_bin_x(n, height_map.device, height_map.dtype)  # (PAN_N_BINS,)
        a = self.pivot_half_base()
        pitch = self.orientations_3_history[:, -1, 1]  # (N,)

        if self.cfg.get("candidate_angle_ground_referenced", True):
            # Re-reference the terrain from the CHASSIS to the GROUND under the robot, i.e.
            # evaluate Fig. 4's construction for the nominal driving pose rather than the
            # robot's current one. See _ground_reference_note below for why this is required
            # on MARV. h_ground = h_chassis + clearance, since the map is measured from
            # (base_z - track_wheel_radius) and clearance is exactly that plane's height
            # above the terrain (crossing_env._get_rewards).
            height_map = height_map + env.clearance.unsqueeze(-1)

        # Hinge positions in the gravity-aligned [L] frame: the chassis is pitched, so the
        # pivots swing out of the body-x axis.
        pitch_l = pitch if _PITCH_NOSE_DOWN_POSITIVE else -pitch
        hx_front = a * torch.cos(pitch_l)
        hz_front = self.hinge_height() - a * torch.sin(pitch_l)
        hx_rear = -a * torch.cos(pitch_l)
        hz_rear = self.hinge_height() + a * torch.sin(pitch_l)

        heights = height_map + self.tip_radius()  # (N, 15)

        # Minimum hinge-to-bin lever arm admitted into the max. `null` = one H bin width.
        #
        # theta* is a max over bins, so whichever bin sits nearest a hinge dominates it: the
        # angle to a bin `reach` away goes as atan2(dz, reach), which blows up as reach -> 0.
        # With a pivot half-base near 0.25 m against 0.15 m bins, the nearest bin lands just
        # 0.05 m from each hinge, where terrain only 0.10 m high already yields a +50.9 deg
        # target (vs +7.0 deg from a bin at a 0.50 m lever arm). Measured on
        # logs/train_marv_icmd3qn_11369833, theta*_rear averaged +41.7 deg -- squarely in that
        # regime -- and the rear flipper dutifully tracked it to within 15 deg, i.e. the term
        # worked and the target was wrong.
        #
        # The papers take the max over a dense terrain point cloud, where a genuine point near
        # the hinge means the robot is right up against a step and a steep flipper really is
        # correct. Here each "point" is a 0.15 m bin average, so a bin centred 0.05 m from the
        # pivot is a discretisation artifact rather than a measurement -- and one whose lever
        # arm depends on the pivot half-base, so the target would change completely if that
        # were 0.30 m instead of 0.25 m. Excluding sub-bin-width reaches removes the artifact
        # and that fragility with it. 0.0 restores the unguarded behaviour.
        min_reach = self.cfg.get("min_hinge_reach_m", None)
        if min_reach is None:
            min_reach = self.env.height_map_length[0] / n
        min_reach = max(float(min_reach), 1e-3)

        def _end(hx: torch.Tensor, hz: torch.Tensor, forward: bool):
            dx = x.unsqueeze(0) - hx.unsqueeze(-1)  # (N, 15)
            reach = dx if forward else -dx
            dz = heights - hz.unsqueeze(-1)
            ang = torch.atan2(dz, reach.clamp_min(1e-3))
            # Only bins genuinely beyond the hinge, and far enough out to give a
            # well-conditioned angle, participate in the max.
            usable = reach > min_reach
            ang = torch.where(usable, ang, torch.full_like(ang, -torch.pi / 2))
            best = ang.max(dim=-1)
            return best.values, best.indices

        theta_l_front, arg_front = _end(hx_front, hz_front, forward=True)
        theta_l_rear, arg_rear = _end(hx_rear, hz_rear, forward=False)

        # [L] -> chassis [R]: subtract the chassis tilt. With nose-down-positive pitch, a
        # direction theta above the horizon in front of a nose-down chassis is theta + pitch
        # off the chassis x-axis; behind it, theta - pitch.
        theta_r_front = theta_l_front + pitch_l
        theta_r_rear = theta_l_rear - pitch_l

        limit = torch.pi / 3  # the papers' theta_f domain (Eq. 2)
        theta_r_front = theta_r_front.clamp(-limit, limit)
        theta_r_rear = theta_r_rear.clamp(-limit, limit)

        self.pan_diagnostics["theta_star_front"] = theta_r_front.detach()
        self.pan_diagnostics["theta_star_rear"] = theta_r_rear.detach()
        # Which bin won the max, and the hinge geometry it was measured against. Without
        # these the near-hinge domination above can only be inferred, not confirmed: a mean
        # argmax pinned at the bin adjacent to the hinge is the signature.
        self.pan_diagnostics["argmax_bin_front"] = arg_front.detach().to(theta_r_front.dtype)
        self.pan_diagnostics["argmax_bin_rear"] = arg_rear.detach().to(theta_r_rear.dtype)
        self.pan_diagnostics["pivot_half_base_m"] = torch.full_like(theta_r_front, a)
        self.pan_diagnostics["min_hinge_reach_m"] = torch.full_like(theta_r_front, min_reach)

        # Paper convention (up = positive) -> MARV convention (front up = negative).
        return torch.stack(
            [FLIPPER_UP_SIGN[0] * theta_r_front, FLIPPER_UP_SIGN[1] * theta_r_rear], dim=-1
        )

    # --------------------------------------------------- reward scaling (lambda)

    def _saturating_lambda(self, term: str) -> float:
        """Effective lambda for R_flipper / R_pitch.

        Both papers describe lambda as "the threshold coefficient of Delta": the term reads
        ``-lambda * Delta``, clamped to -1 once ``Delta > 1/lambda``, so ``1/lambda`` is the
        deviation the authors considered maximally bad. Table 2's values put those thresholds
        at **Delta_flipper = 10 rad** and **mean per-step pitch change = 3.03 rad**, neither of
        which this setup can reach: the flipper range is +/-pi/3, so Delta tops out near 2.1 rad
        (term floor -0.21), and a 0.1 s control step produces per-step pitch changes of
        1e-3..1e-2 rad (term floor about -3e-3).

        Measured on logs/train_marv_icmd3qn_11203271/attempt_2 over 73 M frames, that left
        ``rew/pitch_penalty`` averaging -2.1e-5 and ``rew/flipper_penalty`` -7.9e-4 against
        ``rew/contact_penalty``'s -3.6e-3 — i.e. R_pitch was numerically inert and the dense
        reward was essentially R_contact alone.

        ``lambda_*_saturation_rad: null`` (the default) keeps the papers' literal values.
        Setting it to a Delta that *is* physically meaningful here reinterprets lambda as
        exactly what the papers call it, without touching the equation or the kappa weights.
        """
        cfg = self.cfg
        lam = float(cfg[f"lambda_{term}"])
        sat = cfg.get(f"lambda_{term}_saturation_rad", None)
        return 1.0 / float(sat) if sat else lam

    # ------------------------------------------------------ R_flipper (Eq. 4)

    def flipper_reward(self, theta_star: torch.Tensor) -> torch.Tensor:
        """R_flipper (Eq. 4), in [-1, 0].

        ``delta = |theta_f - (theta*_f +/- pi/36)|``, i.e. the distance to the nearest edge of
        the tolerance band (zero inside it), then ``-lambda_1 * delta`` clamped at -1.

        With ``rear_flipper_shaping: true`` the front and rear terms are averaged, which
        keeps the term's [-1, 0] range and therefore keeps ``kappa_flipper``'s meaning
        unchanged — the rear end is brought under supervision without silently doubling the
        weight of flipper shaping relative to R_pitch and R_contact.
        """
        env = self.env
        lam = self._saturating_lambda("flipper")
        theta = env.flipper_positions  # (N, 2) = [front, rear] under sync_flipper_control

        def _term(col: int) -> torch.Tensor:
            d = theta[:, col] - theta_star[:, col]
            # Distance to the band [-tol, +tol] around theta*: 0 inside, |d| - tol outside.
            delta = (d.abs() - ANGLE_TOLERANCE).clamp_min(0.0)
            return torch.where(delta > 1.0 / lam, -torch.ones_like(delta), -lam * delta)

        front = _term(0)
        self.pan_diagnostics["flipper_delta_front"] = (theta[:, 0] - theta_star[:, 0]).detach()

        if not self.cfg.get("rear_flipper_shaping", False) or theta.shape[-1] < 2:
            return front

        rear = _term(1)
        self.pan_diagnostics["flipper_delta_rear"] = (theta[:, 1] - theta_star[:, 1]).detach()
        return 0.5 * (front + rear)

    # -------------------------------------------------------- R_pitch (Eq. 6)

    def pitch_reward(self) -> torch.Tensor:
        """R_pitch (Eq. 6): smoothness on the chassis pitch, in [-1, 0]. Unchanged from the
        papers — it is morphology-neutral, and both modules had it right already."""
        env = self.env
        hist = self.orientations_3_history[:, :, 1]
        if hist.shape[1] <= 1:
            return torch.zeros(env.num_envs, device=env.device)

        delta_abs = hist[:, -1].abs() - hist[:, -2].abs()          # Delta|theta_R(t)|
        mean_delta_k = (hist[:, 1:] - hist[:, :-1]).abs().mean(dim=1)  # Delta theta_R^k(t)
        lam = self._saturating_lambda("pitch")
        danger = ((hist[:, -1].abs() > torch.pi / 4) & (delta_abs > 0)) | (mean_delta_k > 1.0 / lam)
        return torch.where(danger, -torch.ones_like(mean_delta_k), -lam * mean_delta_k)

    # ----------------------------------------------------- R_contact (Eq. 7-8)

    def _flipper_contact(self) -> torch.Tensor:
        """(N, 4) bool per-flipper ground contact in ``[FL, FR, RL, RR]`` order.

        Prefers the ground-truth PhysX contact forces already implemented for C-TRAC
        (``rl_modules/ctrac/ctrac_contact.py``), which is why ``ftr_env._setup_scene``'s
        ContactSensor gate now also covers atd3qn/icmd3qn. Falls back to the joint-torque
        proxy ``env.flipper_torques`` (computed for every module in
        ``crossing_env.py``) when the sensor is unavailable.
        """
        env = self.env
        if self._contact_extractor is None and not self._contact_extractor_tried:
            self._contact_extractor_tried = True
            if getattr(env, "_ctrac_contact_sensor", None) is not None:
                try:
                    from rl_modules.ctrac.ctrac_contact import CTRACContactExtractor

                    self._contact_extractor = CTRACContactExtractor(env)
                except Exception as e:
                    _log.warning("R_contact: ContactSensor extractor failed (%s); "
                                 "falling back to the joint-torque proxy.", e)
                    self._contact_extractor = None
            # Log which source R_contact ended up on exactly once — the two differ in
            # quality (ground-truth PhysX forces vs a saturating torque proxy) and the
            # difference is otherwise invisible in the logs.
            _log.info(
                "R_contact source: %s",
                "ground-truth PhysX ContactSensor (ctrac_contact.CTRACContactExtractor)"
                if self._contact_extractor is not None
                else f"joint-torque proxy (threshold "
                     f"{float(self.cfg.get('contact_torque_threshold', 0.0)):.2f} x effort limit)",
            )

        if self._contact_extractor is not None:
            _, prob = self._contact_extractor.compute()  # (N, 4) in {0., 1.}
            return prob > 0.5

        threshold = float(self.cfg.get("contact_torque_threshold", 0.0)) * float(
            env.cfg.flipper_contact_effort_limit
        )
        return env.flipper_torques > threshold

    def contact_reward(self) -> torch.Tensor:
        """R_contact (Eq. 7-8), in {-1, 0} — same discrete shape and same kappa as the paper,
        with the *body parts* swapped for MARV's morphology.

        The paper's case analysis asks whether the extreme contact points straddle the CoM
        with at least one of them on a **main track**, and returns -1 for case4 (both on the
        flippers, chassis bridged off the ground). MARV has no main tracks, so:

        * the "at least one contact on each side of the CoM" requirement is kept verbatim,
          but evaluated over the front and rear **flipper pairs**, which are MARV's
          drivetrain — this is the term doing the work the paper intended, namely keeping
          the propulsion on the ground;
        * the case4 penalty is replaced by its MARV equivalent, the belly grounding out
          (``clearance < belly_clearance_min``), which is the pose that actually costs
          traction here.

        The previous proxy — ``|clearance| > 0.03 -> -1`` — is a *symmetric* band around a
        nominal ride height, so it penalises riding up on the flipper tracks (MARV's correct
        driving pose) just as hard as sinking into the terrain, and never rewards having the
        drivetrain on the ground at all. Measured over 73 M frames in
        ``logs/train_marv_icmd3qn_11203271/attempt_2`` it fired on 58.5-89.9% of steps
        (mean 72.6%): not literally constant, but a large, mostly-unavoidable offset pointed
        at the wrong quantity. The redefined term fired on 13-54% of steps in a short
        8-env debug run — the two are not directly comparable (different policy maturity and
        env count), so treat that only as evidence the term is not saturated.
        Setting ``paper_contact_semantics: true`` restores the old proxy.
        """
        env = self.env
        if self.cfg.get("paper_contact_semantics", False):
            ungrounded = env.clearance.abs() > float(self.cfg.contact_clearance_threshold)
            return torch.where(ungrounded, -torch.ones_like(env.clearance), torch.zeros_like(env.clearance))

        contact = self._flipper_contact()  # (N, 4) [FL, FR, RL, RR]
        front_down = contact[:, 0:2].any(dim=-1)
        rear_down = contact[:, 2:4].any(dim=-1)
        straddles_com = front_down & rear_down

        bellied = env.clearance < float(self.cfg.get("belly_clearance_min", -0.02))

        ok = straddles_com & ~bellied
        self.pan_diagnostics["contact_front"] = front_down.float().detach()
        self.pan_diagnostics["contact_rear"] = rear_down.float().detach()
        self.pan_diagnostics["contact_bellied"] = bellied.float().detach()
        return torch.where(ok, torch.zeros_like(env.clearance), -torch.ones_like(env.clearance))

    # --------------------------------------------------------- R_end (Eq. 9)

    def apply_terminal(self, components: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Zero every shaping term on failure/explosion, then add R_end (Eq. 9)."""
        env = self.env
        cfg = env.cfg
        terminal = env._explosion_mask | env._fail_mask
        for name, comp in components.items():
            components[name] = torch.where(terminal, torch.zeros_like(comp), comp)

        components["terminal_bonus"] = (
            env._success_mask.float() * cfg.goal_reached_reward
            + env._fail_mask.float() * cfg.failed_reward
            + env._timeout_mask.float() * (cfg.timeout_penalty if cfg.timeout_penalty is not None else 0.0)
        )
        return components
