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
             beta: float = 1.0, free_bits: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_VAE — reconstruction MSE + beta * KL(q(z|o^H) || N(0,I)). Returns (total, recon, kl).

    The returned `kl` is always the TRUE, unpenalised KL, so the logged value stays a
    faithful collapse indicator regardless of `free_bits`.

    free_bits (Kingma et al. 2016, "Improving Variational Inference with Inverse
    Autoregressive Flow", Sec. 2.3) floors the *per-latent-dimension* KL before summing, so
    the optimiser gets no gradient from a dimension that is already below the floor and
    therefore cannot profit from switching it off entirely. Without it this objective
    posterior-collapses hard here: the reconstruction term's natural scale on this data is
    ~0.08 while beta was 1.0, so zeroing the latent buys far more than it costs. A real
    22M-frame Stage II run drove KL from 3.06 to ~1e-7 within the first ~300k frames and
    never recovered — mu ~ 0 and logvar ~ 0 on all 32 dims, i.e. the encoder emitted pure
    N(0, I) noise. Since CVAEDecoder consumes z ALONE, that made the contact estimate
    (c-tilde_t, c-tilde-prob_t) a constant and left the actor running on its partial
    observation plus 44 dead input dimensions — the whole contact-estimation architecture
    silently contributed nothing. Set free_bits > 0 (~0.5 nats/dim is the usual starting
    point) and keep beta small relative to the recon scale.
    """
    recon_loss = F.mse_loss(recon, target, reduction="none").mean(dim=-1).mean()
    # Per-dimension KL, averaged over the batch: (latent_dim,)
    kl_per_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)
    kl = kl_per_dim.sum()
    kl_penalised = kl_per_dim.clamp_min(free_bits).sum() if free_bits > 0.0 else kl
    return recon_loss + beta * kl_penalised, recon_loss, kl


@torch.no_grad()
def latent_diagnostics(mu: torch.Tensor, logvar: torch.Tensor, active_threshold: float = 0.01) -> dict[str, float]:
    """Cheap per-batch collapse indicators for the C-VAE latent.

    `active_dims` is the count of latent dimensions whose batch-mean KL exceeds
    `active_threshold` nats — the number that still carry information. It going to 0 (or
    `posterior_std` going to 1.0 while `mu_abs` goes to 0) IS posterior collapse, and is
    worth watching from the first logged step rather than discovering hours in from a
    flat success curve.
    """
    kl_per_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)
    return {
        "cvae_latent_active_dims": float((kl_per_dim > active_threshold).sum().item()),
        "cvae_latent_kl_max_dim": float(kl_per_dim.max().item()),
        "cvae_latent_mu_abs": float(mu.abs().mean().item()),
        "cvae_latent_posterior_std": float((0.5 * logvar).exp().mean().item()),
    }


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


def contact_geo_loss(pred_points: torch.Tensor, target_prob: torch.Tensor,
                     max_reach: float) -> torch.Tensor:
    """L_geo (Eq. 14) — spatial-feasibility penalty. The paper's region Omega ("contact
    points are located on the robot") is approximated here as a sphere of radius max_reach
    centred on the robot base (a fixed per-flipper box would need per-flipper body-frame
    geometry this project doesn't expose at this level) — d(c, boundary of Omega) becomes
    the excess distance beyond max_reach.

    ctrac_contact.py returns ROBOT-frame points, so the base is the origin by construction
    and Omega needs no centre argument. This used to take a `robot_pos`, left over from when
    the points were world-frame; callers were passing target_points.mean(dim=1) — the
    centroid of the ground-truth contacts — which centres Omega on a quantity that itself
    moves with the prediction target instead of on the robot. Numerically it was near-inert
    on clean data (measured max ||c|| = 0.674 m against max_reach 0.8, so both spellings
    give 0), but it fired on corrupted rows and was simply the wrong region."""
    mask = _dynamic_mask(target_prob)
    dist_from_robot = pred_points.norm(dim=-1)  # (N, 4) — base is the origin in robot frame
    excess = (dist_from_robot - max_reach).clamp_min(0.0)
    return (mask * excess).sum(dim=-1).mean()
