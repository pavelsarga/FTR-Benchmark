from dataclasses import dataclass

import torch
import torch.nn as nn
from torchrl.data import Unbounded

from marv_rl_training.observations import Observation, ObservationEncoder


@dataclass
class ICMD3QNObservation(Observation):
    """Local terrain heightmap H (Eq. 1) + robot state E = {theta_f1, theta_f2, theta_R}
    (Eq. 2) from Pan et al. 2023, "Deep Reinforcement Learning for Flipper Control of
    Tracked Robots in Urban Rescuing Environments" (ICM-D3QN).

    dim = 15 (H) + 3 (E) = 18, matching the S_t input width drawn for the paper's ICM
    encoder psi (Fig. 7, block (3)) — see icmd3qn_module.get_observations() for how the
    18 values are populated, and icmd3qn_icm.py for psi/F/I.
    """

    supports_vecnorm = False
    dim = 18

    def __call__(self, prev_state, action, prev_state_der, curr_state):
        raise NotImplementedError("ICMD3QNObservation is populated directly by FtrTorchRLEnv._step / _reset.")

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.batch_size[0], self.dim),
            device=self.env.device,
            dtype=torch.float32,
        )

    def get_encoder(self) -> ObservationEncoder:
        return ICMD3QNEncoder(**(self.encoder_opts or {}))


class ICMD3QNEncoder(ObservationEncoder):
    """
    Feature-extraction/fusion + Dueling D3QN head from Pan et al. 2023's ICM-D3QN
    (Fig. 7, blocks (1) and (2) — the Q-network). The curiosity module (block (3)) is a
    separate network, ICMD3QNCuriosityModule in icmd3qn_icm.py, owned by the trainer.

    Input:  x of shape (..., HM_DIM + STATE_DIM) = (..., 15 + 3)
      - H = x[..., :15]  local terrain heights, N=15 downsampled point-set averages (Eq. 1)
      - E = x[..., 15:]  robot state {theta_f1, theta_f2, theta_R} (Eq. 2), normalised to
        roughly [-1, 1] by ICMD3QNModule.get_observations()
    Output: Q(S'_t, a_t) of shape (..., NUM_ACTIONS) = (..., 9), the combination of
      {front flipper CW/hold/CCW} x {rear flipper CW/hold/CCW} (Eq. 3).

    Deviation from Fig. 7: the figure draws the fusion Dense layer as Dense(8, 16), i.e.
    H'(4) spliced with a 4-dim E, but Eq. 2 formally defines E as the 3-dim
    {theta_f1, theta_f2, theta_R} — the same E used for the block-(3) encoder input
    (S_t = 15 + 3 = 18, matching that figure's own "18" label). We follow the equation
    over the figure's unexplained "4", using Dense(hm_out + 3, fusion_hidden) here.
    """

    HM_DIM = 15
    STATE_DIM = 3
    NUM_ACTIONS = 9

    def __init__(self, hm_hidden: int = 32, hm_out: int = 4, fusion_hidden: int = 16, **kwargs):
        super().__init__(self.NUM_ACTIONS)
        self.terrain_mlp = nn.Sequential(
            nn.Linear(self.HM_DIM, hm_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hm_hidden, hm_out),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hm_out + self.STATE_DIM, fusion_hidden),
            nn.LeakyReLU(inplace=True),
        )
        self.value_head = nn.Linear(fusion_hidden, 1)
        self.advantage_head = nn.Linear(fusion_hidden, self.NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x[..., : self.HM_DIM]
        e = x[..., self.HM_DIM :]
        h_prime = self.terrain_mlp(h)
        s_prime = self.fusion_mlp(torch.cat([h_prime, e], dim=-1))
        value = self.value_head(s_prime)
        advantage = self.advantage_head(s_prime)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))
