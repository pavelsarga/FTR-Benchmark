from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from marv_rl_training.observations import Observation, ObservationEncoder
from marv_rl_training.observations.ftr_flat_obs import FtrFlipperStyleEncoder
from rl_modules.marv_rl.marv_rl_cnn_flat_encoder import MarvRLCNNFlatEncoder


@dataclass(kw_only=True)
class MarvRLFlatObservation(Observation):
    """Flat observation vector produced by FTR-Benchmark CrossingEnv.

    Default (flipper_style=False): 966-D  (945 hmap + 21 state, 6 prev_action)
    flipper_style=True:           4119-D  (4096 hmap + 15 state + 8 prev_action)

    This observation is filled directly by FtrTorchRLEnv; the __call__ method is never
    invoked during FTR training.
    """

    supports_vecnorm = True
    dim: int = 966  # default; overridden in __post_init__ for flipper_style

    def __post_init__(self):
        opts = self.encoder_opts or {}
        if opts.get("flipper_style", False):
            self.dim = 4119  # 4096 hmap + 15 state + 8 prev_action (4 track vels + 4 flipper angles)

    def __call__(self, prev_state, action, prev_state_der, curr_state):
        raise NotImplementedError("MarvRLFlatObservation is populated directly by FtrTorchRLEnv._step / _reset.")

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.batch_size[0], self.dim),
            device=self.env.device,
            dtype=torch.float32,
        )

    def get_encoder(self) -> ObservationEncoder:
        opts: dict = {**(self.encoder_opts or {})}
        if opts.pop("flipper_style", False):
            return FtrFlipperStyleEncoder(input_dim=4117, **opts)
        return MarvRLCNNFlatEncoder(input_dim=self.dim, **opts)
