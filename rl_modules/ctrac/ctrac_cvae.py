"""C-VAE (Eq. 9-14) for C-TRAC (Pan et al. 2025): a beta-VAE with a single encoder and a
two-head decoder — head 1 estimates contact points/existence (c-tilde_t, c-tilde-prob_t),
head 2 reconstructs/denoises the next partial observation (o-hat_{t+1}). Encoder/decoder
sizes (512x256x128 / 128x256x128) match the paper's Fig. 2.

Ground-truth contact targets (ctrac_contact.py's CTRACContactExtractor output) are used
directly as this network's supervised training targets — the C-VAE's job is to let the
*deployed* actor estimate them from noisy partial observations alone (sim-to-real bridge);
the reward's own stabilization term (ctrac_module.py) always uses the real ground truth,
never this network's estimate, matching the paper (c_t is privileged state).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_modules.ctrac.ctrac_observation import PARTIAL_DIM

NUM_FLIPPERS = 4


class _MLPTrunk(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU()]
            prev = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CVAEEncoder(nn.Module):
    """P_en(z_t | o^H_t) (Eq. 9): flattens the last history_len partial-obs frames and maps
    to a diagonal-Gaussian latent z_t."""

    def __init__(self, history_len: int, partial_dim: int = PARTIAL_DIM,
                 hidden_dims: tuple[int, ...] = (512, 256, 128), latent_dim: int = 32):
        super().__init__()
        self.history_len = history_len
        self.trunk = _MLPTrunk(history_len * partial_dim, hidden_dims)
        self.mu = nn.Linear(self.trunk.output_dim, latent_dim)
        self.logvar = nn.Linear(self.trunk.output_dim, latent_dim)

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs_history: (N, history_len, partial_dim) -> (mu, logvar), each (N, latent_dim)."""
        h = self.trunk(obs_history.reshape(obs_history.shape[0], -1))
        return self.mu(h), self.logvar(h)


class CVAEDecoder(nn.Module):
    """P_de(s-tilde_t | z_t) (Eq. 9), two heads off a shared trunk: contact estimation
    (c-tilde_t, c-tilde-prob_t) and denoised next-partial-obs reconstruction."""

    def __init__(self, latent_dim: int, partial_dim: int = PARTIAL_DIM,
                 hidden_dims: tuple[int, ...] = (128, 256, 128)):
        super().__init__()
        self.trunk = _MLPTrunk(latent_dim, hidden_dims)
        self.contact_head = nn.Linear(self.trunk.output_dim, NUM_FLIPPERS * 3)
        self.prob_head = nn.Linear(self.trunk.output_dim, NUM_FLIPPERS)
        self.recon_head = nn.Linear(self.trunk.output_dim, partial_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        contact = self.contact_head(h).view(z.shape[0], NUM_FLIPPERS, 3)
        prob = torch.sigmoid(self.prob_head(h))
        recon = self.recon_head(h)
        return contact, prob, recon


class CTRACCVAE(nn.Module):
    def __init__(self, history_len: int, partial_dim: int = PARTIAL_DIM,
                 encoder_hidden: tuple[int, ...] = (512, 256, 128),
                 decoder_hidden: tuple[int, ...] = (128, 256, 128), latent_dim: int = 32):
        super().__init__()
        self.history_len = history_len
        self.latent_dim = latent_dim
        self.encoder = CVAEEncoder(history_len, partial_dim, encoder_hidden, latent_dim)
        self.decoder = CVAEDecoder(latent_dim, partial_dim, decoder_hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def forward(self, obs_history: torch.Tensor, sample: bool = True):
        """Returns (z, mu, logvar, contact, prob, recon)."""
        mu, logvar = self.encoder(obs_history)
        z = self.reparameterize(mu, logvar) if sample else mu
        contact, prob, recon = self.decoder(z)
        return z, mu, logvar, contact, prob, recon


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_VAE — reconstruction MSE + beta * KL(q(z|o^H) || N(0,I)). Returns (total, recon, kl)."""
    recon_loss = F.mse_loss(recon, target, reduction="none").mean(dim=-1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()
    return recon_loss + beta * kl, recon_loss, kl


def contact_prob_loss(pred_prob: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    """L_prob (Eq. 12) — BCE between estimated and ground-truth contact existence."""
    return F.binary_cross_entropy(pred_prob.clamp(1e-6, 1 - 1e-6), target_prob, reduction="none").mean(dim=-1).mean()


def _dynamic_mask(target_prob: torch.Tensor) -> torch.Tensor:
    """m_i (Eq. 13): normalized ground-truth contact-existence weighting, so a flipper that
    isn't actually touching ground doesn't supervise its (meaningless) point estimate."""
    return target_prob / target_prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def contact_est_loss(pred_points: torch.Tensor, target_points: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    """L_est (Eq. 13) — dynamic mask-weighted MSE on contact points."""
    mask = _dynamic_mask(target_prob)
    mse = ((pred_points - target_points) ** 2).mean(dim=-1)  # (N, 4)
    return (mask * mse).sum(dim=-1).mean()


def contact_geo_loss(pred_points: torch.Tensor, target_prob: torch.Tensor, robot_pos: torch.Tensor,
                      max_reach: float) -> torch.Tensor:
    """L_geo (Eq. 14) — spatial-feasibility penalty. The paper's region Omega ("contact
    points are located on the robot") is approximated here as a sphere of radius max_reach
    around the robot base (world-frame contact points vs. a fixed per-flipper box would need
    per-flipper body-frame geometry this project doesn't expose at this level) — d(c,
    boundary of Omega) becomes the excess distance beyond max_reach."""
    mask = _dynamic_mask(target_prob)
    dist_from_robot = (pred_points - robot_pos.unsqueeze(1)).norm(dim=-1)  # (N, 4)
    excess = (dist_from_robot - max_reach).clamp_min(0.0)
    return (mask * excess).sum(dim=-1).mean()
