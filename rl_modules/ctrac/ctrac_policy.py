"""Asymmetric SAC actor + Q-critic for C-TRAC (Pan et al. 2025), built around the C-VAE in
ctrac_cvae.py. Construction contract deliberately deviates from the project's usual
PolicyConfig (which returns a single PPO-shaped ActorCriticWrapper) the same way
train_d3qn.py already bypasses it for D3QN: SAC needs an actor + a Q(s,a) critic, not an
actor + V(s) critic, and torchrl's SACLoss takes those as separate modules (see
train_sac.py). CTRACPolicyConfig.create() is therefore consumed only by train_sac.py /
collect_ctrac_dataset.py / eval_sac.py, never by train_ftr.py's generic PPO path.

Actor input matches the paper's exact signature pi_psi(a_t | o_t, z_t, c-tilde_t,
c-tilde-prob_t) (Sec. IV-A.3): the partial-obs slice plus the C-VAE's latent + estimated
contact. The critic is asymmetric per the paper (Sec. IV-A.3, "the critic ... utilizing the
complete state-action pair (s_t, a_t) that includes privileged observations") — it reads
the FULL packed observation (partial + privileged, see ctrac_observation.py), not just the
actor's slice.

Action space: env's native 6-D [v, w, FL, FR, RL, RR] (see marv_config_ctrac.yaml — no
fixed_forward_vel override, unlike hfc/creps/atd3qn/icmd3qn: the paper's action Eq. 3
genuinely includes desired velocity v_t as a learned action, not a fixed constant). w has
no equivalent in the paper's 5-D action [v, dtheta_fl, dtheta_rl, dtheta_rr, dtheta_fr]
(straight-path traversal only) and the env has no fixed-w override analogous to
fixed_forward_vel, so it's pinned to a near-zero mean with a small fixed std directly in
the actor's output distribution instead — same "paper has no action for this env slot"
situation hfc_policy.py already solved, adapted here since (unlike HFC) v itself is genuinely
learned rather than also fixed.
"""
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal, ValueOperator

from marv_rl_training.environment.ftr_env_adapter import OBS_KEY
from marv_rl_training.policies import MLP, PolicyConfig

from rl_modules.ctrac.ctrac_cvae import NUM_FLIPPERS, CTRACCVAE
from rl_modules.ctrac.ctrac_observation import PARTIAL_DIM

_log = logging.getLogger(__name__)

# SB3 SAC's LOG_STD_MIN / LOG_STD_MAX, as std bounds (see CTRACActor.forward). Only the
# upper bound actually binds: NormalParamExtractor's biased-softplus floor (0.01) already
# sits well above exp(-20), so the lower bound is inert — kept only to state the full SB3
# interval explicitly.
_MIN_SCALE = 2.061153622438558e-09  # exp(-20)
_MAX_SCALE = 7.38905609893065       # exp(2)


class CTRACObsHistory(nn.Module):
    """Per-env ring buffer of the last history_len partial-obs frames (o^H_t, paper Eq. 9),
    feeding the C-VAE's encoder. Non-persistent (rollout state, not a trained parameter) —
    same detach-then-clone / shape-mismatch-fallback pattern as HFCSDSMBelief
    (hfc_policy.py), including the same in-place-mutation pitfall it documents: .detach()
    alone shares storage with the buffer, so a later .copy_() would corrupt whatever this
    step's forward pass saved for backward — must .clone() too.
    """

    def __init__(self, num_envs: int, history_len: int, partial_dim: int = PARTIAL_DIM):
        super().__init__()
        self.history_len = history_len
        self.register_buffer("hist", torch.zeros(num_envs, history_len, partial_dim), persistent=False)

    def forward(self, partial_obs: torch.Tensor, fresh_mask: torch.Tensor) -> torch.Tensor:
        prev = self.hist.detach().clone()
        if prev.shape[0] != partial_obs.shape[0]:
            # Batch came from a differently-sized source (e.g. an off-policy minibatch
            # reconstructed outside the live rollout) — fall back to a flat history of the
            # current frame repeated, rather than crashing on a shape mismatch.
            prev = partial_obs.unsqueeze(1).expand(-1, self.history_len, -1).clone()
        else:
            fresh = fresh_mask.bool().reshape(-1)
            if fresh.any():
                reset_hist = partial_obs.unsqueeze(1).expand(-1, self.history_len, -1)
                prev = torch.where(fresh.view(-1, 1, 1), reset_hist, prev)
        new_hist = torch.cat([prev[:, 1:], partial_obs.unsqueeze(1)], dim=1)
        if self.hist.shape[0] == partial_obs.shape[0]:
            self.hist.copy_(new_hist.detach())
        return new_hist


class CTRACActorNet(nn.Module):
    """Encoder(history) -> C-VAE -> [o_t, z_t, c-tilde_t, c-tilde-prob_t] -> MLP -> TanhNormal
    (loc, scale) over the env's 6-D action. Also emits "obs_history" — the ring buffer's
    current window — as an extra TensorDictModule output key so it lands in the
    SyncDataCollector's rollout tensordict and, from there, the replay buffer: SAC samples
    random (non-sequential) transitions, which would otherwise make it impossible to
    reconstruct each transition's correct temporal obs-history window for the C-VAE's
    training step (see train_sac.py's _cvae_update) — precomputing it once, in the correct
    temporal order, at collection time and carrying it along with the transition sidesteps
    that entirely.
    """

    def __init__(self, num_envs: int, cvae: CTRACCVAE, hidden_dims: tuple[int, ...] = (512, 128),
                 w_log_std: float = -4.0):
        super().__init__()
        self.cvae = cvae
        self.obs_history = CTRACObsHistory(num_envs, cvae.history_len)
        in_dim = (PARTIAL_DIM - 1) + cvae.latent_dim + NUM_FLIPPERS * 3 + NUM_FLIPPERS  # o_t (no reset flag) + z_t + c~_t + c~prob_t
        self.trunk = nn.Sequential(
            MLP(in_dim=in_dim, hidden_dim=list(hidden_dims), num_hidden=len(hidden_dims), out_dim=2 * 5, layernorm=False),
            NormalParamExtractor(),
        )
        self.w_log_std = float(w_log_std)

    def forward(self, obs: torch.Tensor):
        partial = obs[..., :PARTIAL_DIM]
        fresh_mask = partial[..., -1:]  # reset flag is the last partial column
        obs_hist = self.obs_history(partial, fresh_mask)  # (N, H, PARTIAL_DIM)
        z, _mu, _logvar, contact, prob, _recon = self.cvae(obs_hist, sample=self.training)

        o_t = partial[..., :-1]  # drop reset flag before feeding the actor trunk
        actor_in = torch.cat([o_t, z, contact.reshape(z.shape[0], -1), prob], dim=-1)
        loc5, scale5 = self.trunk(actor_in)  # each (N,5): [v, d_FL, d_FR, d_RL, d_RR]

        # SB3's SAC (the implementation the paper states it builds on) clamps log_std to
        # [-20, 2]; NormalParamExtractor's softplus output is unbounded above, so without
        # this the entropy term can grow the std without limit — observed in run
        # train_ctrac_11305386 as a monotonically diverging actor loss.
        scale5 = scale5.clamp(_MIN_SCALE, _MAX_SCALE)

        n = loc5.shape[0]
        w_loc = torch.zeros(n, 1, device=loc5.device, dtype=loc5.dtype)
        w_scale = torch.full((n, 1), float(torch.tensor(self.w_log_std).exp()), device=loc5.device, dtype=loc5.dtype)
        loc = torch.cat([loc5[..., 0:1], w_loc, loc5[..., 1:5]], dim=-1)      # [v, w, FL, FR, RL, RR]
        scale = torch.cat([scale5[..., 0:1], w_scale, scale5[..., 1:5]], dim=-1)
        return loc, scale, obs_hist


class CTRACQNet(nn.Module):
    """Asymmetric Q(s_t, a_t): full packed observation (partial + privileged) + action."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        self.mlp = MLP(in_dim=obs_dim + action_dim, hidden_dim=list(hidden_dims), num_hidden=len(hidden_dims), out_dim=1, layernorm=False)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([obs, action], dim=-1))


@dataclass
class CTRACPolicyConfig(PolicyConfig):
    """Builds the actor (with its embedded, jointly-trained C-VAE) and a single Q-network
    template — torchrl's SACLoss internally ensembles it into num_qvalue_nets=2 twin
    critics (Sec V-A: "asymmetric Soft Actor-Critic"), so this config only constructs one
    Q-network architecture, not two by hand.

    cvae_weights_path optionally warm-starts the C-VAE from pretrain_ctrac_cvae.py's
    checkpoint (Stage I) — it keeps training jointly afterwards (not frozen), per the
    paper's own "jointly optimized" design and the user's explicit decision.
    """

    actor_optimizer_opts: dict[str, Any]
    qvalue_optimizer_opts: dict[str, Any]
    cvae_optimizer_opts: dict[str, Any]
    history_len: int = 8
    latent_dim: int = 32
    actor_hidden: tuple = (512, 128)
    qvalue_hidden: tuple = (512, 256, 128)
    cvae_encoder_hidden: tuple = (512, 256, 128)
    cvae_decoder_hidden: tuple = (128, 256, 128)
    w_log_std: float = -4.0
    cvae_weights_path: str | None = None
    extra_distribution_kwargs: dict = field(default_factory=dict)

    def create(self, env, **kwargs):
        num_envs = env.batch_size[0]
        action_spec = env.action_spec
        obs_dim = env.observations[0].dim
        action_dim = action_spec.shape[-1]
        device = kwargs.get("device", None)

        cvae = CTRACCVAE(
            history_len=self.history_len, partial_dim=PARTIAL_DIM,
            encoder_hidden=self.cvae_encoder_hidden, decoder_hidden=self.cvae_decoder_hidden,
            latent_dim=self.latent_dim,
        )
        if self.cvae_weights_path:
            sd = torch.load(self.cvae_weights_path, map_location=device or "cpu")
            missing_unexpected = cvae.load_state_dict(sd, strict=False)
            _log.info(f"Loaded C-VAE weights from {self.cvae_weights_path}")
            if missing_unexpected.missing_keys:
                _log.warning(f"C-VAE missing keys: {missing_unexpected.missing_keys}")
            if missing_unexpected.unexpected_keys:
                _log.warning(f"C-VAE unexpected keys: {missing_unexpected.unexpected_keys}")

        actor_net = CTRACActorNet(num_envs=num_envs, cvae=cvae, hidden_dims=self.actor_hidden, w_log_std=self.w_log_std)
        actor_module = TensorDictModule(actor_net, in_keys=[OBS_KEY], out_keys=["loc", "scale", "obs_history"])
        policy_operator = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low[0],
                "high": action_spec.space.high[0],
                **self.extra_distribution_kwargs,
            },
            return_log_prob=True,
        )

        qnet = CTRACQNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=self.qvalue_hidden)
        qvalue_operator = ValueOperator(module=qnet, in_keys=[OBS_KEY, "action"], out_keys=["state_action_value"])

        if device is not None:
            policy_operator.to(device)
            qvalue_operator.to(device)
            cvae.to(device)

        # Only the actor (which embeds the C-VAE) is meaningful at inference/eval time — the
        # Q-network exists purely to train the actor and eval_sac.py never loads it, so
        # weights_path (matching eval_ftr.py's contract for every other module) loads a
        # flat policy_operator.state_dict() here, not a Q-network-inclusive bundle.
        if weights_path := kwargs.get("weights_path", None):
            sd = torch.load(weights_path, map_location=device or "cpu")
            missing_unexpected = policy_operator.load_state_dict(sd, strict=False)
            _log.info(f"Loaded policy weights from {weights_path}")
            if missing_unexpected.missing_keys:
                _log.warning(f"Missing keys: {missing_unexpected.missing_keys}")
            if missing_unexpected.unexpected_keys:
                _log.warning(f"Unexpected keys: {missing_unexpected.unexpected_keys}")

        # actor_net.cvae is a submodule of policy_operator (it's inside CTRACActorNet), so
        # policy_operator.parameters() would otherwise double-count it alongside the
        # separate cvae optimizer train_sac.py builds from the returned `cvae` module
        # directly (with its own, deliberately different rising-LR schedule) — exclude it
        # here so each parameter is owned by exactly one optimizer.
        actor_only_params = [p for n, p in policy_operator.named_parameters() if "cvae" not in n]
        optim_groups = [
            {"params": actor_only_params, "name": "policy_operator", **self.actor_optimizer_opts},
            {"params": qvalue_operator.parameters(), "name": "qvalue_operator", **self.qvalue_optimizer_opts},
        ]
        return policy_operator, qvalue_operator, cvae, optim_groups
