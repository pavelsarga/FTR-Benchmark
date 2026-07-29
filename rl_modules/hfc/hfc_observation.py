from dataclasses import dataclass

import torch
import torch.nn as nn
from torchrl.data import Unbounded

from marv_rl_training.observations import Observation, ObservationEncoder


@dataclass
class HFCObservation(Observation):
    """Terrain heightmap + robot state for the Hybrid Flipper Controller (HFC), adapted
    from Azayev & Zimmermann 2022, "Autonomous State-Based Flipper Control for Articulated
    Tracked Robots in Urban Environments" (IEEE RA-L). dim = 15 (terrain) + 3 (front/rear
    flipper angle, chassis pitch — all RAW radians, unnormalized).

    The paper trains its state-transition classifier via imitation learning from human
    demonstrations (cross-entropy loss, Eq. 2); we have no such demonstrations in this
    automated Isaac Sim pipeline, so HFCModule/train_hfc.py instead train the discrete
    pose-selection Q-network via reward-driven D3QN (see train_hfc.py's docstring).

    supports_vecnorm = False (matching atd3qn/icmd3qn): the front/rear flipper angles must
    stay in raw radians, unrescaled by VecNorm, because HFCPolicy's P-controller decode
    (target pose angle -> per-step delta command) needs the true physical angle.
    """

    supports_vecnorm = False
    dim = 18   # 15 (terrain) + 2 (front/rear flipper angle, raw radians) + 1 (chassis pitch, raw radians)

    def __call__(self, prev_state, action, prev_state_der, curr_state):
        raise NotImplementedError("HFCObservation is populated directly by FtrTorchRLEnv._step / _reset.")

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.batch_size[0], self.dim),
            device=self.env.device,
            dtype=torch.float32,
        )

    def get_encoder(self) -> ObservationEncoder:
        return HFCEncoder(**(self.encoder_opts or {}))


class HFCEncoder(ObservationEncoder):
    """Dueling D3QN head over the paper's 7 named poses (NEUTRAL, ASCENDING_FRONT,
    UP_STAIRS, ASCENDING_REAR, DESCENDING_FRONT, DOWN_STAIRS, DESCENDING_REAR — Fig. 1),
    using the same terrain-fusion architecture as ATD3QNEncoder/ICMD3QNEncoder (this
    project's D3QN convention) rather than the paper's own small per-state MLP classifiers
    — since training already departs from the paper's imitation-learning method (see
    HFCObservation's docstring), matching our existing D3QN network family is more
    consistent than reproducing the paper's tiny linear/MLP transition heads.

    Input:  x of shape (..., HM_DIM + STATE_DIM) = (..., 15 + 3)
      - H = x[..., :15]  local terrain heights (N downsampled point-set averages)
      - E = x[..., 15:]  robot state: front angle, rear angle, chassis pitch (raw radians)
    Output: Q(S'_t, a_t) of shape (..., NUM_ACTIONS) = (..., 7), one Q-value per pose.
    HFCPolicy decodes the selected pose into a per-step delta command toward that pose's
    fixed target angle (see hfc_policy.py) — it does not select flipper deltas directly.
    """

    HM_DIM = 15
    STATE_DIM = 3
    NUM_ACTIONS = 7

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
