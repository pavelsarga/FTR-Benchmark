import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ActorCriticWrapper, ProbabilisticActor, TanhNormal, ValueOperator

from marv_rl_training.environment.ftr_env_adapter import OBS_KEY
from marv_rl_training.policies import MLP, PolicyConfig

_log = logging.getLogger(__name__)

# Pose order matches Azayev & Zimmermann 2022 Fig. 1 / the reference repo's state_list
# (paper_repos/augmented_robot_trackers/src/control/marv_flipper_controller.py).
POSE_NAMES = [
    "NEUTRAL",
    "ASCENDING_FRONT",
    "UP_STAIRS",
    "ASCENDING_REAR",
    "DESCENDING_FRONT",
    "DOWN_STAIRS",
    "DESCENDING_REAR",
]
NUM_POSES = len(POSE_NAMES)
ASCENDING_REAR_IDX = POSE_NAMES.index("ASCENDING_REAR")

# [front_frac, rear_frac] per pose (see hfc_policy.py's module docstring for the
# normalisation method against the reference repo's raw radian templates).
# front_frac: negative = up (scaled by front_up_deg), positive = down (front_down_deg).
# rear_frac: positive = up (back_up_deg), negative = down (back_down_deg).
POSE_FRACTIONS = [
    (-1.0, 1.0),        # NEUTRAL
    (-0.2, 0.0),        # ASCENDING_FRONT
    (0.2857, -0.1429),  # UP_STAIRS
    (0.2857, -0.8571),  # ASCENDING_REAR
    (1.0, -1.0),        # DESCENDING_FRONT
    (0.0, 0.0333),      # DOWN_STAIRS
    (-0.15, 0.2667),    # DESCENDING_REAR
]

# HFCObservation's fixed field layout (hfc_module.py:get_observations) — 24-D total.
HM_DIM = 15
_FLIP_LO, _FLIP_HI = HM_DIM, HM_DIM + 4          # [FL, FR, RL, RR], raw rad
_ROLL_IDX = _FLIP_HI                              # raw rad
_ROLL_RATE_IDX = _ROLL_IDX + 1                    # raw rad/s
_PITCH_IDX = _ROLL_RATE_IDX + 1                   # raw rad
_FWD_VEL_IDX = _PITCH_IDX + 1                     # raw m/s
_RESET_FLAG_IDX = _FWD_VEL_IDX + 1                # 0.0 / 1.0


class HFCTerrainStateEncoder(nn.Module):
    """Terrain + robot-state fusion trunk feeding both the SDSM transition heads
    (HFCTransitionHeads) and, in a separate instance, the value critic. Same
    terrain-then-fuse shape as the old D3QN HFCEncoder/ATD3QNEncoder, resized: input is
    terrain (15) + flippers (4, raw) + roll + pitch (not roll-rate/fwd-vel/reset-flag —
    those feed HFCActionDecoder's deterministic formulas directly, not the classifier).
    """

    STATE_DIM = 6  # 4 flippers + roll + pitch

    def __init__(self, hm_hidden: int = 32, hm_out: int = 4, fusion_hidden: int = 32):
        super().__init__()
        self.terrain_mlp = MLP(in_dim=HM_DIM, hidden_dim=hm_hidden, num_hidden=1, out_dim=hm_out, layernorm=False)
        self.fusion_mlp = MLP(
            in_dim=hm_out + self.STATE_DIM, hidden_dim=fusion_hidden, num_hidden=1, out_dim=fusion_hidden,
            layernorm=False, activate_last_layer=True,
        )
        self.output_dim = fusion_hidden

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        terrain = obs[..., :HM_DIM]
        flippers = obs[..., _FLIP_LO:_FLIP_HI]
        roll = obs[..., _ROLL_IDX : _ROLL_IDX + 1]
        pitch = obs[..., _PITCH_IDX : _PITCH_IDX + 1]
        h_terrain = self.terrain_mlp(terrain)
        return self.fusion_mlp(torch.cat([h_terrain, flippers, roll, pitch], dim=-1))


class HFCTransitionHeads(nn.Module):
    """The 7 per-state transition networks mu_phi_1 .. mu_phi_7 (Eq. 6): each maps the
    fused terrain/state features to a distribution over the 7 next-states. Kept as
    separate small heads (not one 7x7 matrix) to match the paper's own per-state mu_phi_i
    parameterisation.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, NUM_POSES))
            for _ in range(NUM_POSES)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns (..., NUM_POSES, NUM_POSES): trans[..., i, :] = softmax(mu_phi_i(h))."""
        logits = torch.stack([head(h) for head in self.heads], dim=-2)  # (..., 7, 7)
        return torch.softmax(logits, dim=-1)


class HFCSDSMBelief(nn.Module):
    """Soft-differentiable state machine belief propagation (Eq. 6):
    p_{t+1} = sum_i p_t^i * mu_phi_i(o_t). The previous belief is carried in a
    non-persistent buffer (not saved/loaded with the model — it's rollout state, not a
    trained parameter, and would size-mismatch a checkpoint reloaded at a different
    num_envs) and is DETACHED before combining, so only the current step's transition
    heads receive gradient — a 1-step truncation of the recursion, matching the paper's
    own Fig. 2 caption ("computed through 2 time steps for simplicity") rather than
    backpropagating through an entire episode.

    Reset per-env to one-hot NEUTRAL (Algorithm 1's p_1 = [1, 0, ..., 0]) whenever
    HFCObservation's reset flag is set for that env.
    """

    def __init__(self, num_envs: int):
        super().__init__()
        init = torch.zeros(num_envs, NUM_POSES)
        init[:, 0] = 1.0
        self.register_buffer("belief", init, persistent=False)

    def forward(self, trans: torch.Tensor, fresh_mask: torch.Tensor) -> torch.Tensor:
        # .clone() matters, not just .detach(): detach() shares storage with self.belief,
        # so the later in-place self.belief.copy_() below would otherwise silently
        # overwrite the very tensor this step's einsum saved for its backward pass.
        prev = self.belief.detach().clone()
        if prev.shape[0] != trans.shape[0]:
            # Batch came from a differently-sized env (e.g. a vmap'd sub-batch during PPO
            # updates) — fall back to a fresh uniform-NEUTRAL prior for this call only.
            prev = torch.zeros(trans.shape[0], NUM_POSES, device=trans.device, dtype=trans.dtype)
            prev[:, 0] = 1.0
        else:
            fresh = fresh_mask.bool().squeeze(-1)
            if fresh.any():
                reset_row = torch.zeros(NUM_POSES, device=prev.device, dtype=prev.dtype)
                reset_row[0] = 1.0
                prev = torch.where(fresh.unsqueeze(-1), reset_row, prev)
        p_next = torch.einsum("ni,nij->nj", prev, trans)
        if self.belief.shape[0] == trans.shape[0]:
            self.belief.copy_(p_next.detach())
        return p_next


class HFCActionDecoder(nn.Module):
    """Deterministic decode of belief p_t + raw robot state into the 4 flipper commands
    via Eq. 9's a_i = a_temp^i + a_stab^i + a_em^i (see this file's module docstring for
    the exact formulas and index convention: [FL, FR, RL, RR] == paper's [a1, a2, a3, a4]).

    kp/kd (Eq. 7, no paper-given numeric value — roll stabilization is unconditional,
    independent of pose) and escape_mods (Eq. 8, paper gives only one numeric example —
    the ascending_rear row is warm-started at it) are the parameters PPO trains using
    marv_rl's reward, per the user's explicit "train via RL instead of IL" decision.
    """

    def __init__(self, front_up_deg: float, front_down_deg: float, back_up_deg: float, back_down_deg: float):
        super().__init__()
        front_up_rad, front_down_rad = math.radians(front_up_deg), math.radians(front_down_deg)
        back_up_rad, back_down_rad = math.radians(back_up_deg), math.radians(back_down_deg)
        front_low, front_high = -front_up_rad, front_down_rad
        rear_low, rear_high = -back_down_rad, back_up_rad

        front_cmds, rear_cmds = [], []
        for front_frac, rear_frac in POSE_FRACTIONS:
            front_target = front_frac * (front_up_rad if front_frac < 0 else front_down_rad)
            rear_target = rear_frac * (back_up_rad if rear_frac > 0 else back_down_rad)
            # Inverse of FtrEnv's `target = low + unit * (high - low)`, unit = (cmd + 1) / 2.
            front_cmds.append(2.0 * (front_target - front_low) / (front_high - front_low) - 1.0)
            rear_cmds.append(2.0 * (rear_target - rear_low) / (rear_high - rear_low) - 1.0)
        self.register_buffer("pose_commands", torch.tensor([front_cmds, rear_cmds]).T.clamp(-1.0, 1.0))  # (7, 2)

        self.kp = nn.Parameter(torch.tensor(0.0))
        self.kd = nn.Parameter(torch.tensor(0.0))
        escape_mods = torch.zeros(NUM_POSES, 2)
        escape_mods[ASCENDING_REAR_IDX] = torch.tensor([-0.3, 0.5])  # paper's one given example
        self.escape_mods = nn.Parameter(escape_mods)

    def forward(self, p_t: torch.Tensor, roll: torch.Tensor, roll_rate: torch.Tensor,
                fwd_vel_actual: torch.Tensor, v_cmd: float) -> torch.Tensor:
        temp_2 = p_t @ self.pose_commands  # (N, 2): [front, rear]
        a_temp = torch.stack([temp_2[..., 0], temp_2[..., 0], temp_2[..., 1], temp_2[..., 1]], dim=-1)  # (N,4)

        x = roll.squeeze(-1) * self.kp - roll_rate.squeeze(-1) * self.kd  # (N,)
        a_stab = torch.stack([-x, x, -x, x], dim=-1)  # (N,4): [FL,FR,RL,RR]

        st = torch.clamp((v_cmd - fwd_vel_actual.squeeze(-1)) / max(v_cmd, 1e-6), 0.0, 1.0)  # (N,)
        em_2 = p_t @ self.escape_mods  # (N, 2): [front_mod, rear_mod]
        a_em = torch.stack([em_2[..., 0], em_2[..., 0], em_2[..., 1], em_2[..., 1]], dim=-1) * st.unsqueeze(-1)

        return a_temp + a_stab + a_em  # (N, 4), Eq. 9


class HFCActorNet(nn.Module):
    """Full HFC actor: encoder -> SDSM belief -> action decoder -> pre-squash TanhNormal
    (loc, scale). track_vel is a fixed policy hyperparameter (not learned) for action slot
    0 (v) — same role fixed_forward_vel plays for CREPS: the env_cfg_overrides value
    ultimately forces the real track velocity regardless, this just keeps the emitted
    action tensor consistent with it. w (slot 1) is always 0.

    The decoder's a_temp/a_stab/a_em sum is already meant to be the literal desired action
    in the env's raw [-1, 1] command space (not an arbitrary unconstrained network output),
    so it is atanh'd before use as TanhNormal's pre-squash loc — otherwise TanhNormal's own
    tanh squashing would silently distort the physically-meaningful decoder output.
    """

    def __init__(self, num_envs: int, track_vel: float, init_log_std: float, front_up_deg: float,
                 front_down_deg: float, back_up_deg: float, back_down_deg: float,
                 hm_hidden: int, hm_out: int, fusion_hidden: int, head_hidden: int):
        super().__init__()
        self.track_vel = track_vel
        self.encoder = HFCTerrainStateEncoder(hm_hidden=hm_hidden, hm_out=hm_out, fusion_hidden=fusion_hidden)
        self.transitions = HFCTransitionHeads(in_dim=self.encoder.output_dim, hidden_dim=head_hidden)
        self.belief = HFCSDSMBelief(num_envs=num_envs)
        self.decoder = HFCActionDecoder(front_up_deg, front_down_deg, back_up_deg, back_down_deg)
        self.log_std = nn.Parameter(torch.full((6,), float(init_log_std)))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        roll = obs[..., _ROLL_IDX : _ROLL_IDX + 1]
        roll_rate = obs[..., _ROLL_RATE_IDX : _ROLL_RATE_IDX + 1]
        fwd_vel = obs[..., _FWD_VEL_IDX : _FWD_VEL_IDX + 1]
        fresh_mask = obs[..., _RESET_FLAG_IDX : _RESET_FLAG_IDX + 1]

        h = self.encoder(obs)
        trans = self.transitions(h)
        p_t = self.belief(trans, fresh_mask)
        flippers_4 = self.decoder(p_t, roll, roll_rate, fwd_vel, self.track_vel)

        lead_shape = flippers_4.shape[:-1]
        v = torch.full((*lead_shape, 1), self.track_vel, device=obs.device, dtype=flippers_4.dtype)
        w = torch.zeros((*lead_shape, 1), device=obs.device, dtype=flippers_4.dtype)
        mean_action = torch.cat([v, w, flippers_4], dim=-1)  # (N, 6)

        loc = torch.atanh(mean_action.clamp(-0.999, 0.999))
        scale = self.log_std.exp().expand_as(loc)
        return loc, scale


class HFCValueNet(nn.Module):
    """Independent (non-shared) critic — own HFCTerrainStateEncoder instance/params, plus
    the raw roll/roll-rate/fwd-vel features the actor's decoder also sees, through a small
    MLP head. Mirrors MitriakovPolicyConfig's non-shared actor/critic design (the paper
    gives no critic architecture at all — PPO always needs one for GAE regardless).
    """

    def __init__(self, hm_hidden: int, hm_out: int, fusion_hidden: int, value_hidden: int):
        super().__init__()
        self.encoder = HFCTerrainStateEncoder(hm_hidden=hm_hidden, hm_out=hm_out, fusion_hidden=fusion_hidden)
        self.head = MLP(in_dim=self.encoder.output_dim + 3, hidden_dim=value_hidden, num_hidden=1, out_dim=1, layernorm=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        roll_rate = obs[..., _ROLL_RATE_IDX : _ROLL_RATE_IDX + 1]
        fwd_vel = obs[..., _FWD_VEL_IDX : _FWD_VEL_IDX + 1]
        fresh = obs[..., _RESET_FLAG_IDX : _RESET_FLAG_IDX + 1]
        h = self.encoder(obs)
        return self.head(torch.cat([h, roll_rate, fwd_vel, fresh], dim=-1))


@dataclass
class HFCPolicyConfig(PolicyConfig):
    """PPO actor + critic for the Hybrid Flipper Controller (HFC), Azayev & Zimmermann
    2022 — see hfc_policy.py's module-level docstrings for the full design rationale
    (SDSM belief propagation, Eq. 7/8/9 decode, why loc is atanh'd, why the belief buffer
    is non-persistent). Trained via train_ftr.py (PPO) using HFCModule's reward, which is
    an intentional copy of marv_rl's reward formula (see hfc_module.py) — per the user's
    explicit decision to train the whole controller (classifier + stabilization gains +
    escape-maneuver magnitudes) via RL rather than the paper's imitation learning.

    front_up_deg/front_down_deg/back_up_deg/back_down_deg must match
    env_cfg_overrides.marv_flipper_*_deg. track_vel must match
    env_cfg_overrides.fixed_forward_vel (the env forces this regardless of the policy's
    own v output — see HFCActorNet's docstring).
    """

    actor_optimizer_opts: dict[str, Any]
    value_optimizer_opts: dict[str, Any]
    track_vel: float = 0.4
    init_log_std: float = -1.0
    front_up_deg: float = 90.0
    front_down_deg: float = 90.0
    back_up_deg: float = 90.0
    back_down_deg: float = 90.0
    hm_hidden: int = 32
    hm_out: int = 4
    fusion_hidden: int = 32
    head_hidden: int = 16
    value_hidden: int = 32
    extra_distribution_kwargs: dict = field(default_factory=dict)

    def create(self, env, **kwargs):
        num_envs = env.batch_size[0]
        action_spec = env.action_spec

        actor_net = HFCActorNet(
            num_envs=num_envs, track_vel=self.track_vel, init_log_std=self.init_log_std,
            front_up_deg=self.front_up_deg, front_down_deg=self.front_down_deg,
            back_up_deg=self.back_up_deg, back_down_deg=self.back_down_deg,
            hm_hidden=self.hm_hidden, hm_out=self.hm_out, fusion_hidden=self.fusion_hidden,
            head_hidden=self.head_hidden,
        )
        actor_module = TensorDictModule(actor_net, in_keys=[OBS_KEY], out_keys=["loc", "scale"])
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

        value_net = HFCValueNet(
            hm_hidden=self.hm_hidden, hm_out=self.hm_out, fusion_hidden=self.fusion_hidden, value_hidden=self.value_hidden,
        )
        value_operator = ValueOperator(module=value_net, in_keys=[OBS_KEY])

        actor_value_wrapper = ActorCriticWrapper(policy_operator=policy_operator, value_operator=value_operator)
        if kwargs.get("device", None) is not None:
            actor_value_wrapper.to(kwargs["device"])

        # HFCSDSMBelief's buffer is registered persistent=False (see hfc_policy.py's
        # docstring — it's ephemeral rollout state, not a trained parameter), so it never
        # appears as a missing/unexpected key here.
        if weights_path := kwargs.get("weights_path", None):
            sd = torch.load(weights_path, map_location=kwargs.get("device") or "cpu")
            missing_unexpected = actor_value_wrapper.load_state_dict(sd, strict=False)
            _log.info(f"Loaded weights from {weights_path}")
            if missing_unexpected.missing_keys:
                _log.warning(f"Missing keys: {missing_unexpected.missing_keys}")
            if missing_unexpected.unexpected_keys:
                _log.warning(f"Unexpected keys: {missing_unexpected.unexpected_keys}")

        optim_groups = [
            {"params": policy_operator.parameters(), "name": "policy_operator", **self.actor_optimizer_opts},
            {"params": value_operator.parameters(), "name": "value_operator", **self.value_optimizer_opts},
        ]
        return actor_value_wrapper, optim_groups, []
