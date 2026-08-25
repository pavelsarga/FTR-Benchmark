from dataclasses import dataclass

import torch
import torch.nn as nn
from torchrl.data import Unbounded

from marv_rl_training.observations import Observation, ObservationEncoder


@dataclass
class ATD3QNObservation(Observation):
    """Local terrain heightmap H (Eq. 1) + robot state E = {theta_f1, theta_f2, theta_R}
    (Eq. 2) from Pan et al. 2023, "Deep Reinforcement Learning for Flipper Control of
    Tracked Robots" (AT-D3QN, arXiv:2306.10352).

    dim = 15 (H) + 3 (E) = 18. The chassis pitch theta_R was previously missing: this class
    was 17-D with E = {theta_f1, theta_f2} only. That was an undocumented omission, not a
    deviation from AT-D3QN in favour of something else -- the paper's Eq. 2 reads
    "E = {theta_f1, theta_f2, theta_R}", and its text is explicit: "The robot state E
    consists of the angle of the robot's front flipper theta_f1, the angle of the robot's
    rear flipper theta_f2, and the chassis pitch angle theta_R." ICM-D3QN's Eq. 2 is
    word-for-word identical, and icmd3qn_observation.py already followed it.

    The omission mattered more than one feature usually would: pitch was the ONLY pose
    quantity in this observation, so AT-D3QN was blind to its own attitude entirely (no
    pitch, no roll, and -- because the heightmap is chassis-referenced -- no ride height
    either). R_pitch penalises pitch dynamics the network had no way to perceive.
    """

    supports_vecnorm = False
    dim = 18   # 15 (terrain) + 2 (front/rear flipper angle) + 1 (chassis pitch)
               # requires sync_flipper_control: true

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

    Input:  x of shape (..., HM_DIM + STATE_DIM) = (..., 15 + 3)
      - H = x[..., :15]  local terrain heights (N downsampled point-set averages)
      - E = x[..., 15:]  robot state {theta_f1, theta_f2, theta_R} (Eq. 2) — the flipper
        pair angles require sync_flipper_control: true, which halves FtrEnv.flipper_num
        from 4 to 2, plus the chassis pitch
    Output: Q(S'_t, a_t) of shape (..., NUM_ACTIONS) = (..., 9), the combination of
      {front flipper CW/hold/CCW} x {rear flipper CW/hold/CCW}.

    Batch dims are arbitrary leading dims; the paper's 10x256 in the figure is a
    visualization of a different batching scheme and is not modeled here (input is
    plain (N, 15) / (N, 2)).
    """

    HM_DIM = 15
    STATE_DIM = 3
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
