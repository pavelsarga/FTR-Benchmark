from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from marv_rl_training.observations import Observation, ObservationEncoder


@dataclass
class MitriakovObservation(Observation):
    """State vector from Mitriakov et al. 2021, "Reinforcement Learning Based, Staircase
    Negotiation Learning": s = (p_x^front, p_y^front, p_x^rear, p_y^rear, v_s, psi^front,
    psi^rear) (Eq. 2, 3-DoF/no-arm variant — this project doesn't model the paper's
    optional 2-DoF robotic arm, per the user's notes: "uses 2DOF robotic arm ... we
    won't"). p^front/p^rear are the (along-travel, vertical) offsets from the front/rear
    flipper reference point to the next/previous staircase step edge respectively; v_s is
    forward linear velocity; psi^front/psi^rear are the current flipper angles. See
    mitriakov_module.py's get_observations() for exactly how these are computed.

    An 8th feature, is_ascending, is appended beyond the paper's own 7-D Eq. 2 — this
    project's own addition (not in the paper), needed because "mitriakov_stairs" mixes
    ascending and descending tiles across envs in one training run (see
    mitriakov_module.py's _ascending_mask()) and MitriakovPolicyConfig's per-env flipper
    angle limits (mitriakov_policy.py) need to know which one a given env is on.

    supports_vecnorm = False: is_ascending must stay an exact 0.0/1.0 for
    MitriakovPolicyConfig's low/high selection (mitriakov_policy.py) to threshold it
    reliably — VecNorm's running normalisation would still preserve the sign distinction,
    but there's no need for that indirection when the flag can just stay exact.
    """

    supports_vecnorm = False
    dim = 8

    def __call__(self, prev_state, action, prev_state_der, curr_state):
        raise NotImplementedError("MitriakovObservation is populated directly by FtrTorchRLEnv._step / _reset.")

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.batch_size[0], self.dim),
            device=self.env.device,
            dtype=torch.float32,
        )

    def get_encoder(self) -> ObservationEncoder:
        return MitriakovEncoder(**(self.encoder_opts or {}))


class MitriakovEncoder(ObservationEncoder):
    """Identity passthrough — provided only so MitriakovObservation conforms to the
    standard Observation/ObservationEncoder interface (used e.g. by FtrTorchRLEnv to size
    its observation spec). Not used by MitriakovPolicyConfig (mitriakov_policy.py), which
    is a fully self-contained PolicyConfig with its own MitriakovMLP actor/critic
    networks reading OBS_KEY directly — this encoder plays no role in that path.
    """

    def __init__(self, **kwargs):
        super().__init__(output_dim=MitriakovObservation.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
