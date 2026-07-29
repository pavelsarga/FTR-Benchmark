from dataclasses import dataclass

import torch
import torch.nn as nn
from torchrl.data import Unbounded

from marv_rl_training.observations import Observation, ObservationEncoder


@dataclass
class ATD3QNObservation(Observation):
    supports_vecnorm = False
    dim = 17   # 15 (terrain) + 2 (front/rear flipper angle) — requires sync_flipper_control: true

    def __call__(self, prev_state, action, prev_state_der, curr_state):
        raise NotImplementedError("FtrFlatObservation is populated directly by FtrTorchRLEnv._step / _reset.")

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.batch_size[0], self.dim),
            device=self.env.device,
            dtype=torch.float32,
        )

    def get_encoder(self) -> ObservationEncoder:
        return ATD3QNEncoder(**(self.encoder_opts or {}))


class ATD3QNEncoder(ObservationEncoder):
    """
    Full AT-D3QN network from Pan et al. 2023, "Deep Reinforcement Learning for Flipper
    Control of Tracked Robots": a robot-terrain interaction feature extraction module
    (block (1), green) feeding a Double-Dueling DQN head (block (2), red).

    Input:  x of shape (..., HM_DIM + STATE_DIM) = (..., 15 + 2)
      - H = x[..., :15]  local terrain heights (N downsampled point-set averages)
      - E = x[..., 15:]  robot state (front/rear flipper angle) — requires
        sync_flipper_control: true, which halves FtrEnv.flipper_num from 4 to 2
    Output: Q(S'_t, a_t) of shape (..., NUM_ACTIONS) = (..., 9), the combination of
      {front flipper CW/hold/CCW} x {rear flipper CW/hold/CCW}.

    Batch dims are arbitrary leading dims; the paper's 10x256 in the figure is a
    visualization of a different batching scheme and is not modeled here (input is
    plain (N, 15) / (N, 2)).
    """

    HM_DIM = 15
    STATE_DIM = 2
    NUM_ACTIONS = 9

    def __init__(
        self,
        hm_hidden: int = 32,
        hm_out: int = 4,
        fusion_hidden: int = 16,
        **kwargs,
    ):
        super().__init__(self.NUM_ACTIONS)

        # Block (1): H -> Dense -> LeakyReLU -> Dense -> H', then concat with E and fuse
        self.terrain_mlp = nn.Sequential(
            nn.Linear(self.HM_DIM, hm_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hm_hidden, hm_out),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hm_out + self.STATE_DIM, fusion_hidden),
            nn.LeakyReLU(inplace=True),
        )

        # Block (2): dueling D3QN head, Q = V + (A - mean(A))
        self.value_head = nn.Linear(fusion_hidden, 1)
        self.advantage_head = nn.Linear(fusion_hidden, self.NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x[..., : self.HM_DIM]  # (..., 15)
        e = x[..., self.HM_DIM :]  # (..., 2)

        h_prime = self.terrain_mlp(h)  # (..., 4)
        s_prime = self.fusion_mlp(torch.cat([h_prime, e], dim=-1))  # (..., 16)

        value = self.value_head(s_prime)  # (..., 1)
        advantage = self.advantage_head(s_prime)  # (..., 9)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))
