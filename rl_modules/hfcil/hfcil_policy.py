"""HFC-IL: the HFC controller with its state-transition classifier trained by imitation
learning from the MARV reactive controller's mode-8 state machine, instead of by PPO.

Relationship to rl_modules/hfc
------------------------------
hfc_module.py's own docstring records why that module trains by PPO: "The paper trains its
state-transition classifier via imitation learning from human demonstrations (Eq. 2), which
this project has no data for." This module exists because that data now does exist — human
demonstrations recorded from `marv_flipper_controller` mode 8, whose 7 button-driven states
are exactly HFC's 7 poses (Control.cpp's ASCENDING_STAIRS/DESCENDING_STAIRS are the paper's
UP_STAIRS/DOWN_STAIRS), plus a BALLERINA state HFC has no pose for.

So this is closer to the paper than hfc is: the classifier is supervised, and only it is
learned. The deterministic parts of Eq. 9 (pose templates, roll stabilisation, escape
manoeuvres) are reused unchanged from hfc_policy.py by direct import — the one place this
project's "no cross-module imports" convention is deliberately broken, because hfcil is not
an independent method but the same controller with a different training signal for one
component. Sharing the decoder is the whole point; a copy would let the two drift.

Hard vs soft state
------------------
hfc runs a *soft* belief (HFCSDSMBelief, Eq. 6: p_{t+1} = sum_i p_t^i mu_phi_i) because PPO
needs a differentiable path to the transition heads. The demonstrator has no such belief:
it is in exactly one state, and the label in a recording is that integer. This module
therefore carries a HARD state index and trains the classifier by cross-entropy on the
demonstrator's next state. Rollout keeps the hard state and feeds a one-hot p_t into the
shared decoder, so the pose commands are exactly the demonstrator's pose templates rather
than a blend of them.

Transition legality
-------------------
Illegal transitions are masked to -inf from the current state's row of
hfcil_transitions.legal_mask(), in BOTH training and rollout. See that file for why this is
structural rather than learned. Consequences worth knowing:
  * the classifier never spends capacity on impossible outputs, and can never emit one;
  * cross-entropy is computed over the legal set only, so reported losses/accuracies are
    against the choice the operator actually faced, not against 8 free classes;
  * a label that is illegal given its previous state is a corrupt recording, not something
    to train through — the dataset loader rejects it rather than masking it away silently.
"""
import logging

import torch
import torch.nn as nn

from rl_modules.hfc.hfc_policy import HFCActionDecoder, HFCTerrainStateEncoder
from rl_modules.hfc import hfc_policy as _hfc
from rl_modules.hfcil.hfcil_transitions import NUM_STATES, STATE_SHORT_NAMES, legal_mask

_log = logging.getLogger(__name__)

# HFC's decoder only has pose templates for its 7 poses; the demonstrator's 8th state
# (BALLERINA) has no HFC counterpart. See hfcil_module.yaml's drop_ballerina.
NUM_HFC_POSES = _hfc.NUM_POSES


class HFCILClassifier(nn.Module):
    """o_t, s_t -> distribution over legal s_{t+1}.

    Conditioning on the current state matters and is not optional: the demonstrator's
    correct action is genuinely a function of it. The same terrain that means "start
    climbing" (N -> AF) also means "keep climbing" (AF -> AS) a second later, and the only
    thing separating those two is which state the machine is already in. A classifier fed
    the observation alone would see identical inputs with different labels.

    The current state enters twice, deliberately:
      * as a one-hot input feature, so the network can condition its features on it;
      * as the mask row selecting which logits survive.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_states: int = NUM_STATES):
        super().__init__()
        self.num_states = num_states
        self.net = nn.Sequential(
            nn.Linear(in_dim + num_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_states),
        )
        self.register_buffer("mask", legal_mask(num_states), persistent=False)

    def masked_logits(self, h: torch.Tensor, state_idx: torch.Tensor) -> torch.Tensor:
        """h: (N, in_dim) fused features. state_idx: (N,) long. -> (N, num_states) with
        illegal successors at -inf."""
        one_hot = torch.nn.functional.one_hot(state_idx, self.num_states).to(h.dtype)
        logits = self.net(torch.cat([h, one_hot], dim=-1))
        allowed = self.mask[state_idx]  # (N, num_states) bool
        return logits.masked_fill(~allowed, float("-inf"))

    def forward(self, h: torch.Tensor, state_idx: torch.Tensor) -> torch.Tensor:
        return self.masked_logits(h, state_idx)


class HFCILStateTracker(nn.Module):
    """Carries the hard current state per env across rollout steps.

    Non-persistent buffer for the same reason HFCSDSMBelief's is: it is rollout state, not
    a trained parameter, and would size-mismatch a checkpoint reloaded at a different
    num_envs. Reset to NEUTRAL (0) on the observation's reset flag, matching the
    demonstrator, which is powered up in NEUTRAL.
    """

    def __init__(self, num_envs: int):
        super().__init__()
        self.register_buffer("state_idx", torch.zeros(num_envs, dtype=torch.long), persistent=False)

    def get(self, batch: int, device) -> torch.Tensor:
        if self.state_idx.shape[0] != batch:
            # Differently-sized batch (e.g. an update sub-batch, or eval at another
            # num_envs) — fall back to NEUTRAL for this call rather than indexing garbage.
            return torch.zeros(batch, dtype=torch.long, device=device)
        return self.state_idx.clone()

    def set(self, new_state: torch.Tensor) -> None:
        if self.state_idx.shape[0] == new_state.shape[0]:
            self.state_idx.copy_(new_state.detach())

    def reset(self, fresh_mask: torch.Tensor) -> None:
        if self.state_idx.shape[0] != fresh_mask.shape[0]:
            return
        fresh = fresh_mask.bool().reshape(-1)
        if fresh.any():
            self.state_idx[fresh] = 0


class HFCILActorNet(nn.Module):
    """Full inference path: obs -> encoder -> masked classifier -> hard state -> one-hot
    -> HFC's own deterministic decoder -> 4 flipper commands.

    Only the encoder + classifier are trained (by imitation). The decoder's kp/kd and
    escape_mods are FIXED constants now (rl_modules.hfc.hfc_policy.DEFAULT_KP_RAD/
    DEFAULT_KD_RAD/DEFAULT_ESCAPE_RAD -- sourced from the real deployed reference controller
    in paper_repos/augmented_robot_trackers, the paper's own Eq. 8 value where it conflicts),
    not trained by anything, here or in hfc. Pass kp/kd to override them explicitly; there is
    no warm-start-from-a-PPO-run path any more -- that RL stage was dropped.
    """

    def __init__(self, num_envs: int, front_up_deg: float, front_down_deg: float,
                 back_up_deg: float, back_down_deg: float, hm_hidden: int = 32, hm_out: int = 4,
                 fusion_hidden: int = 32, hidden_dim: int = 64,
                 num_states: int = NUM_STATES, v_cmd: float = 0.0,
                 below_body_extra_deg: float = 10.0, kp: float = _hfc.DEFAULT_KP_RAD,
                 kd: float = _hfc.DEFAULT_KD_RAD):
        super().__init__()
        self.encoder = HFCTerrainStateEncoder(hm_hidden=hm_hidden, hm_out=hm_out, fusion_hidden=fusion_hidden)
        self.classifier = HFCILClassifier(in_dim=fusion_hidden, hidden_dim=hidden_dim, num_states=num_states)
        self.tracker = HFCILStateTracker(num_envs)
        self.decoder = HFCActionDecoder(front_up_deg, front_down_deg, back_up_deg, back_down_deg,
                                        below_body_extra_deg=below_body_extra_deg, kp=kp, kd=kd)
        self.num_states = num_states
        self.v_cmd = v_cmd

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """HFCTerrainStateEncoder slices the full 24-D observation itself (terrain /
        flippers / roll / pitch) using hfc_policy's own index constants, so pass the whole
        vector — re-slicing here would risk the two disagreeing about the layout."""
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        fresh = obs[..., _hfc._RESET_FLAG_IDX : _hfc._RESET_FLAG_IDX + 1]
        self.tracker.reset(fresh)

        h = self.encode(obs)
        state_idx = self.tracker.get(obs.shape[0], obs.device)
        logits = self.classifier(h, state_idx)
        next_state = logits.argmax(dim=-1)
        self.tracker.set(next_state)

        p_t = torch.nn.functional.one_hot(next_state, self.num_states).to(obs.dtype)
        # HFC's decoder is defined over its 7 poses; drop the 8th (BALLERINA) column, which
        # has no pose template there. A BALLERINA prediction therefore decodes as all-zero
        # weights -> only the stabilisation/escape terms act. hfcil_module.yaml's
        # drop_ballerina keeps it out of the label space entirely by default.
        p_hfc = p_t[..., :NUM_HFC_POSES]

        roll = obs[..., _hfc._ROLL_IDX : _hfc._ROLL_IDX + 1]
        roll_rate = obs[..., _hfc._ROLL_RATE_IDX : _hfc._ROLL_RATE_IDX + 1]
        fwd_vel = obs[..., _hfc._FWD_VEL_IDX : _hfc._FWD_VEL_IDX + 1]
        return self.decoder(p_hfc, roll, roll_rate, fwd_vel, self.v_cmd)
