import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import ActorCriticWrapper, NormalParamExtractor, ProbabilisticActor, TanhNormal, ValueOperator

from marv_rl_training.environment.ftr_env_adapter import OBS_KEY
from marv_rl_training.policies import PolicyConfig


class MitriakovMLP(nn.Module):
    """Dense(in)->64->Tanh->64->Tanh->Dense(out), exactly Fig. 1 of Mitriakov et al. 2021
    (reduced from the paper's 9-in/5-out Jaguar-with-arm variant to this project's
    3-DoF/no-arm state — see mitriakov_observation.py). Standalone by design (not
    marv_rl_training.policies.MLP, not MLPPolicyConfig's EncoderCombiner path) — this
    module's observation has no heightmap/encoder stage at all, and reusing the generic
    encoder-keyed-by-observation-name machinery would silently break: FtrTorchRLEnv
    always writes the observation under the fixed OBS_KEY ("MarvRLFlatObservation"), not
    under each Observation subclass's own class name, so EncoderCombiner's name-based
    lookup only happens to work for the marv_rl module (whose Observation class is
    literally named MarvRLFlatObservation) and would raise a KeyError for
    MitriakovObservation.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class MitriakovFlipperBounds(nn.Module):
    """Computes per-env TanhNormal (low, high) for the 4 action dims (track_v, track_w,
    front, rear) from each env's is_ascending observation feature (obs[..., -1], the
    project-specific 8th feature — see mitriakov_observation.py). Deterministic, no
    learnable parameters.

    Implements the paper's asymmetric per-direction flipper limits (rear constrained to
    psi_s^rear in [-pi/4, 0] during ascent, front the same way during descent) as a
    TIGHTER sub-range within the env's own (looser) marv_flipper_*_deg bounds — i.e.
    "mitriakov" is the inner constraint, "marv" is the outer one it can never exceed.
    Computed once at construction via the inverse of FtrEnv's "position" control mapping
    (target = low + unit * (high - low), ftr_env.py:_pre_physics_step) — the same
    approach hfc_policy.py uses for its templates — then selected per env at each forward
    pass by whichever direction that env is currently on, unlike hfc_policy.py's fixed
    templates (there's no per-env direction to select there).

    track_v/track_w are always left at the env-wide [-1, 1] full range (untouched by
    either constraint).
    """

    def __init__(
        self,
        front_up_deg: float,
        front_down_deg: float,
        back_up_deg: float,
        back_down_deg: float,
        mitriakov_limit_front_deg: float,
        mitriakov_limit_rear_deg: float,
    ):
        super().__init__()
        front_up_rad, front_down_rad = math.radians(front_up_deg), math.radians(front_down_deg)
        back_up_rad, back_down_rad = math.radians(back_up_deg), math.radians(back_down_deg)
        front_low_rad, front_high_rad = -front_up_rad, front_down_rad
        rear_low_rad, rear_high_rad = -back_down_rad, back_up_rad

        # Tighter target ranges, mirrored between front/rear's opposite sign conventions
        # (rear up = positive, front up = negative): rear in [0, +limit] (up to limit,
        # never down), front in [-limit, 0] (up to limit UP, never down).
        rear_limit_rad = math.radians(mitriakov_limit_rear_deg)
        front_limit_rad = math.radians(mitriakov_limit_front_deg)

        def inv_map(target: float, low: float, high: float) -> float:
            """Inverse of target = low + unit * (high - low), unit = (cmd + 1) / 2."""
            return 2.0 * (target - low) / (high - low) - 1.0

        self.rear_tight_low = inv_map(0.0, rear_low_rad, rear_high_rad)
        self.rear_tight_high = inv_map(rear_limit_rad, rear_low_rad, rear_high_rad)
        self.front_tight_low = inv_map(-front_limit_rad, front_low_rad, front_high_rad)
        self.front_tight_high = inv_map(0.0, front_low_rad, front_high_rad)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_ascending = obs[..., -1] > 0.5

        full_low = torch.full_like(is_ascending, -1.0, dtype=obs.dtype)
        full_high = torch.full_like(is_ascending, 1.0, dtype=obs.dtype)

        rear_low = torch.where(is_ascending, torch.full_like(full_low, self.rear_tight_low), full_low)
        rear_high = torch.where(is_ascending, torch.full_like(full_high, self.rear_tight_high), full_high)
        front_low = torch.where(is_ascending, full_low, torch.full_like(full_low, self.front_tight_low))
        front_high = torch.where(is_ascending, full_high, torch.full_like(full_high, self.front_tight_high))

        low = torch.stack([full_low, full_low, front_low, rear_low], dim=-1)      # (..., 4)
        high = torch.stack([full_high, full_high, front_high, rear_high], dim=-1)  # (..., 4)
        return low, high


@dataclass
class MitriakovPolicyConfig(PolicyConfig):
    """PPO actor + critic for Mitriakov et al. 2021, built from two independent
    MitriakovMLPs (the paper diagrams only the deployed actor, Fig. 1; PPO itself always
    needs a separate critic for GAE, which the paper doesn't specify an architecture for
    — mirroring the actor's shape is the standard default, not something from the paper).
    No shared trunk/encoder between them (the paper gives no indication of parameter
    sharing).

    front_up_deg/front_down_deg/back_up_deg/back_down_deg must match
    env_cfg_overrides.marv_flipper_*_deg in the training config (the outer, looser
    bound); mitriakov_limit_front_deg/mitriakov_limit_rear_deg is the tighter inner
    constraint applied only for the direction each one matters for (see
    MitriakovFlipperBounds).
    """

    actor_optimizer_opts: dict[str, Any]
    value_optimizer_opts: dict[str, Any]
    hidden_dim: int = 64
    extra_distribution_kwargs: dict = field(default_factory=dict)
    front_up_deg: float = 90.0
    front_down_deg: float = 90.0
    back_up_deg: float = 90.0
    back_down_deg: float = 90.0
    mitriakov_limit_front_deg: float = 45.0
    mitriakov_limit_rear_deg: float = 45.0

    def create(self, env, **kwargs):
        obs_dim = env.observations[0].dim
        action_spec = env.action_spec
        action_dim = action_spec.shape[1]

        actor_net = nn.Sequential(
            MitriakovMLP(in_dim=obs_dim, out_dim=2 * action_dim, hidden_dim=self.hidden_dim),
            NormalParamExtractor(),
        )
        loc_scale_module = TensorDictModule(actor_net, in_keys=[OBS_KEY], out_keys=["loc", "scale"])
        bounds_module = TensorDictModule(
            MitriakovFlipperBounds(
                self.front_up_deg, self.front_down_deg, self.back_up_deg, self.back_down_deg,
                self.mitriakov_limit_front_deg, self.mitriakov_limit_rear_deg,
            ),
            in_keys=[OBS_KEY], out_keys=["low", "high"],
        )
        actor_module = TensorDictSequential(loc_scale_module, bounds_module)
        policy_operator = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=["loc", "scale", "low", "high"],
            distribution_class=TanhNormal,
            distribution_kwargs=self.extra_distribution_kwargs,
            return_log_prob=True,
        )

        value_net = MitriakovMLP(in_dim=obs_dim, out_dim=1, hidden_dim=self.hidden_dim)
        value_operator = ValueOperator(module=value_net, in_keys=[OBS_KEY])

        actor_value_wrapper = ActorCriticWrapper(policy_operator=policy_operator, value_operator=value_operator)
        if kwargs.get("device", None) is not None:
            actor_value_wrapper.to(kwargs["device"])

        optim_groups = [
            {"params": policy_operator.parameters(), "name": "policy_operator", **self.actor_optimizer_opts},
            {"params": value_operator.parameters(), "name": "value_operator", **self.value_optimizer_opts},
        ]
        return actor_value_wrapper, optim_groups, []
