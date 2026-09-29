from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from marv_rl_training.observations import Observation, ObservationEncoder
from marv_rl_training.observations.ftr_flat_obs import FtrFlipperStyleEncoder
from rl_modules.marv_rl.marv_rl_cnn_flat_encoder import MarvRLCNNFlatEncoder


@dataclass(kw_only=True)
class MarvRLFlatObservation(Observation):
    """Flat observation vector produced by FTR-Benchmark CrossingEnv.

    Default (flipper_style=False): 945 hmap + 15 state + num_actions prev_action
                                   (966-D with the usual 6-D action, 969-D in flipper state mode)
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
        else:
            # FtrTorchRLEnv has no action_spec yet at this point; the Isaac-free inference stubs
            # have no ftr_env.
            ftr_env = getattr(self.env, "ftr_env", None)
            num_actions = ftr_env.unwrapped.cfg.num_actions if ftr_env is not None else self.env.action_spec.shape[-1]
            self.dim = 945 + 15 + num_actions

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
