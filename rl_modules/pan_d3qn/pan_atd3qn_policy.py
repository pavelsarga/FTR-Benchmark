import torch
import torch.nn as nn

from rl_modules.pan_d3qn.pan_atd3qn_observation import PanATD3QNEncoder


class PanATD3QNPolicy(nn.Module):
    """
    Wraps PanATD3QNEncoder's Q(S'_t, a_t) with (epsilon-greedy) discrete action
    selection and decodes the chosen action into the continuous action vector
    FtrEnv._pre_physics_step expects: [track_v, track_w, front_frac, rear_frac].

    FtrEnv treats actions as continuous fractions in [-1, 1] and scales them by
    `flipper_dt` (degrees/step) before integrating into the flipper position target
    (see ftr_env.py:_pre_physics_step). The paper's discrete {-1, 0, 1} = {CW, hold,
    CCW} action already lives in exactly that range, so no further rescaling is
    needed here — only the index -> (i, j) decode.

    Track velocity/turn rate are not part of the paper's action space (a human
    operator/planner is assumed to drive the tracks); `track_vel` fixes a constant
    forward speed and `track_w` is left at 0.
    """

    NUM_ACTIONS = 9

    def __init__(self, track_vel: float = 0.5, epsilon: float = 0.0, **encoder_kwargs):
        super().__init__()
        self.q_network = PanATD3QNEncoder(**encoder_kwargs)
        self.track_vel = track_vel
        self.epsilon = epsilon

    @staticmethod
    def index_to_ij(idx: torch.Tensor) -> torch.Tensor:
        """idx: (...,) int64 in [0, 8] -> (..., 2) float in {-1, 0, 1}, [front, rear].

        index = (i + 1) * 3 + (j + 1), i.e. idx // 3 - 1 = i (front), idx % 3 - 1 = j (rear).
        """
        i = torch.div(idx, 3, rounding_mode="floor") - 1
        j = idx.remainder(3) - 1
        return torch.stack([i, j], dim=-1).float()

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, action_idx) so a TensorDictModule can expose both:
        `action` (N, 4) drives the env, `action_idx` (N,) int64 is what the DQN
        loss trains on — decoding action back to an index would be lossy/ambiguous.
        """
        q = self.q_network(obs)  # (..., 9)

        greedy_idx = q.argmax(dim=-1)
        if self.training and self.epsilon > 0:
            random_idx = torch.randint(0, self.NUM_ACTIONS, greedy_idx.shape, device=obs.device)
            explore = torch.rand(greedy_idx.shape, device=obs.device) < self.epsilon
            idx = torch.where(explore, random_idx, greedy_idx)
        else:
            idx = greedy_idx

        flipper_frac = self.index_to_ij(idx)  # (..., 2): [front, rear] in {-1, 0, 1}

        lead_shape = flipper_frac.shape[:-1]
        track_v = torch.full((*lead_shape, 1), self.track_vel, device=obs.device, dtype=flipper_frac.dtype)
        track_w = torch.zeros((*lead_shape, 1), device=obs.device, dtype=flipper_frac.dtype)

        action = torch.cat([track_v, track_w, flipper_frac], dim=-1)  # (..., 4)
        return action, idx
