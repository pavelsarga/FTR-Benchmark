import torch
import torch.nn as nn
import torch.nn.functional as F


class ICMD3QNCuriosityModule(nn.Module):
    """Intrinsic Curiosity Module (ICM) from Pan et al. 2023, Fig. 6 / Fig. 7 block (3):
    an encoder psi, a forward-prediction model F, and an inverse-prediction model I,
    used to compute the intrinsic exploration reward R^i_t (Eq. 11-13).

    Not wired into ICMD3QNModule's reward computation: R^i_t needs psi(s_t) and
    psi(s_t+1), i.e. this network's forward pass over full transitions (s_t, a_t, s_t+1),
    not just the single post-step env state ICMD3QNModule.get_reward_components() sees.
    The D3QN trainer (owning both this module and the replay buffer) is responsible for
    running it and adding R^i_t to the extrinsic reward from get_reward_components()
    before building the Double-DQN target (Eq. 13: R_t = R^e_t + R^i_t).

    STATE_DIM (18) matches ICMD3QNObservation.dim, i.e. S_t = [H (15), E (3)] with
    E = {theta_f1, theta_f2, theta_R} (Eq. 2) — consistent with the paper's Fig. 7
    S_t = 18 label for this same encoder.
    """

    STATE_DIM = 18
    NUM_ACTIONS = 9
    FEATURE_DIM = 10

    def __init__(
        self,
        encoder_hidden: tuple[int, int] = (32, 64),
        feature_dim: int = FEATURE_DIM,
        forward_hidden: int = 32,
        inverse_hidden: int = 32,
        eta: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.eta = eta

        h1, h2 = encoder_hidden
        # psi: encoder model (Fig. 7 block (3), 3 Dense/LeakyReLU layers: 18 -> 32 -> 64 -> 10)
        self.encoder = nn.Sequential(
            nn.Linear(self.STATE_DIM, h1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h2, feature_dim),
        )
        # F: forward-prediction model, psi(s_t) + one-hot(a_t) -> psi_hat(s_t+1) (Eq. 11-12)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + self.NUM_ACTIONS, forward_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(forward_hidden, feature_dim),
        )
        # I: inverse-prediction model, psi(s_t) + psi(s_t+1) -> action logits a_hat_t (Eq. 14)
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * feature_dim, inverse_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(inverse_hidden, self.NUM_ACTIONS),
        )

    def forward(
        self, s_t: torch.Tensor, action_idx: torch.Tensor, s_t1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (psi_t1, psi_t1_hat, action_logits) so the trainer can compute both
        losses (Eq. 11, 14) and the intrinsic reward (Eq. 12) from one forward pass.
        """
        psi_t = self.encoder(s_t)
        psi_t1 = self.encoder(s_t1)

        action_onehot = F.one_hot(action_idx, self.NUM_ACTIONS).to(psi_t.dtype)
        psi_t1_hat = self.forward_model(torch.cat([psi_t, action_onehot], dim=-1))

        action_logits = self.inverse_model(torch.cat([psi_t, psi_t1], dim=-1))

        return psi_t1, psi_t1_hat, action_logits

    def intrinsic_reward(self, psi_t1: torch.Tensor, psi_t1_hat: torch.Tensor) -> torch.Tensor:
        """R^i_t = eta / M * ||psi_hat(s_t+1) - psi(s_t+1)||_2^2 (Eq. 12), M = feature_dim."""
        return self.eta * (psi_t1_hat - psi_t1).pow(2).mean(dim=-1)

    def losses(
        self,
        psi_t1: torch.Tensor,
        psi_t1_hat: torch.Tensor,
        action_logits: torch.Tensor,
        action_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(L_F, L_I): forward MSE loss (Eq. 11) and inverse cross-entropy loss (Eq. 14)."""
        l_forward = F.mse_loss(psi_t1_hat, psi_t1)
        l_inverse = F.cross_entropy(action_logits, action_idx)
        return l_forward, l_inverse
