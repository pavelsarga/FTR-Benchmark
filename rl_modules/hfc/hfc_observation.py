from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from marv_rl_training.observations import Observation, ObservationEncoder


@dataclass
class HFCObservation(Observation):
    """Terrain heightmap + robot state for the Hybrid Flipper Controller (HFC), adapted
    from Azayev & Zimmermann 2022, "Autonomous State-Based Flipper Control for Articulated
    Tracked Robots in Urban Environments" (IEEE RA-L).

    dim = 24 = 15 (downsampled terrain, matches atd3qn/icmd3qn) + 4 (front_left, front_right,
    rear_left, rear_right flipper angles, RAW radians — this module now requires
    sync_flipper_control: false, unlike the old 2-flipper-synced version, since Eq. 7's roll
    stabilization and Eq. 8's escape maneuvers both act asymmetrically on left vs right /
    front vs rear flippers) + 3 (roll, roll rate, pitch, all RAW radians/rad-per-s) + 1
    (actual forward velocity, RAW m/s — needed for HFCActionDecoder's stagnation feature,
    see hfc_policy.py) + 1 (episode-just-reset flag, 0.0/1.0 — lets HFCSDSMBelief's persistent
    per-env belief buffer reset itself on a fresh episode without the policy needing direct
    env access; see hfc_module.py's get_observations()).

    supports_vecnorm = False (matching atd3qn/icmd3qn/mitriakov): roll/roll-rate/velocity must
    stay raw and physically meaningful for HFCActionDecoder's PD roll-stabilization (Eq. 7)
    and stagnation-driven escape-maneuver (Eq. 8) formulas — VecNorm's drifting running
    mean/std would silently detune kp/kd/escape_mods over the course of training.
    """

    supports_vecnorm = False
    dim = 24  # 15 (terrain) + 4 (flipper angles, raw) + 3 (roll, roll rate, pitch, raw) + 1 (fwd vel, raw) + 1 (reset flag)

    def __call__(self, prev_state, action, prev_state_der, curr_state):
        raise NotImplementedError("HFCObservation is populated directly by FtrTorchRLEnv._step / _reset.")

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.batch_size[0], self.dim),
            device=self.env.device,
            dtype=torch.float32,
        )

    def get_encoder(self) -> ObservationEncoder:
        return HFCIdentityEncoder(**(self.encoder_opts or {}))


class HFCIdentityEncoder(ObservationEncoder):
    """Identity passthrough — provided only so HFCObservation conforms to the standard
    Observation/ObservationEncoder interface (used e.g. by FtrTorchRLEnv to size its
    observation spec). Not used by HFCPolicyConfig (hfc_policy.py), which is a fully
    self-contained PolicyConfig with its own terrain/state encoder reading OBS_KEY directly
    — this encoder plays no role in that path (same convention as MitriakovEncoder).
    """

    def __init__(self, **kwargs):
        super().__init__(output_dim=HFCObservation.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
