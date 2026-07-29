from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from marv_rl_training.observations import Observation, ObservationEncoder


@dataclass
class CREPSObservation(Observation):
    """State vector s = (pitch, height_ahead) for Pecka et al. 2016, "Autonomous Flipper
    Control with Safety Constraints" (Constrained REPS / CREPS). This is the 2-D per-step
    input to the lower-level linear policy (CREPSLowerLevelPolicy, creps_policy.py):
    front = omega1 + omega2*pitch + omega3*height_ahead, rear = omega4 + omega5*pitch +
    omega6*height_ahead. Both features are RAW (radians / meters), unnormalized — see
    CREPSModule.get_observations() (creps_module.py) for exactly how they're computed.

    supports_vecnorm = False: the omega coefficients optimized by CREPS (train_creps.py)
    are literal physical-unit linear-regression sensitivities (rad^-1, m^-1). If VecNorm
    kept rescaling pitch/height_ahead under a running mean/std that drifts every outer
    iteration, a fixed omega sampled at iteration k would mean something different by the
    time its (reward, safety, mechanical) triple is used to refit q(omega) at the end of
    that same iteration — corrupting the weighted maximum-likelihood update. Same
    reasoning as mitriakov's is_ascending / hfc's flipper angles.
    """

    supports_vecnorm = False
    dim = 2

    def __call__(self, prev_state, action, prev_state_der, curr_state):
        raise NotImplementedError("CREPSObservation is populated directly by FtrTorchRLEnv._step / _reset.")

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.batch_size[0], self.dim),
            device=self.env.device,
            dtype=torch.float32,
        )

    def get_encoder(self) -> ObservationEncoder:
        return CREPSEncoder(**(self.encoder_opts or {}))


class CREPSEncoder(ObservationEncoder):
    """Identity passthrough, exists only so CREPSObservation conforms to the standard
    Observation/ObservationEncoder interface (used e.g. by FtrTorchRLEnv to size its
    observation spec). CREPSLowerLevelPolicy (creps_policy.py) is a fully self-contained
    deterministic linear policy reading the raw 2-D obs directly — this encoder plays no
    role in that path.
    """

    def __init__(self, **kwargs):
        super().__init__(output_dim=CREPSObservation.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
