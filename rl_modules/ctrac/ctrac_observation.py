from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from marv_rl_training.observations import Observation, ObservationEncoder

# Partial obs o_t (paper Eq. 1): fwd vel (1) + 4 flipper angles, raw rad, paper order
# [FL,RL,RR,FR] (2)  <- the privileged contact points/probs below use this SAME order + roll/pitch/yaw (3) + goal-relative XY, body frame (2) + local
# heightmap h^l_t, robot-centric x in [0.4,1.0] m ahead / y in [-0.5,0.5] m, 12x20 (240) +
# 1 episode-just-reset flag (0.0/1.0 — lets ctrac_cvae.py's per-env observation-history
# ring buffer and ctrac_module.py's posture-swing window reset themselves on a fresh
# episode without needing direct env access, same pattern hfc_module.py/hfc_policy.py use).
PARTIAL_DIM = 1 + 4 + 3 + 2 + 12 * 20 + 1  # = 251

# Privileged (critic-only, appended after the partial slice in the same flat vector — this
# project's obs pipeline only carries one flat tensor per env under OBS_KEY, see
# ftr_env_adapter.py, so "asymmetric" here means CTRACActor reads only the leading
# PARTIAL_DIM columns and CTRACCritic/CVAE-target-builders read the full vector, not two
# separate tensordict keys): larger heightmap h^f_t, x in [-1.0,1.4] m / y in [-0.5,0.5] m,
# same ~0.05 m/cell density as h^l_t -> 48x20 (960) + ground-truth contact points (4x3=12)
# + contact existence probability (4).
#
# The contact flipper axis is the paper's [FL,RL,RR,FR] (ctrac_contact.py's FLIPPER_NAMES),
# matching the flipper-ANGLE order in the partial slice above so index i means the same
# physical flipper in both. It used to be env.flipper_positions' native [FL,FR,RL,RR], i.e.
# obs index 1 was the rear-left angle while contact index 1 was the front-right point.
# ⚠ Any dataset shard or Stage-I C-VAE checkpoint built before that change is stale.
PRIVILEGED_HMAP_ROWS = 48
PRIVILEGED_HMAP_COLS = 20
PRIVILEGED_DIM = PRIVILEGED_HMAP_ROWS * PRIVILEGED_HMAP_COLS + 4 * 3 + 4  # = 976

TOTAL_DIM = PARTIAL_DIM + PRIVILEGED_DIM  # = 1227

# Offsets into the privileged slice — shared by ctrac_policy.py's C-VAE training-target
# slicing (train_sac.py/pretrain_ctrac_cvae.py) so the layout is defined in exactly one
# place. Privileged layout: [priv_hmap (PRIVILEGED_HMAP_ROWS*COLS), contact_points (4x3),
# contact_prob (4)].
PRIV_HMAP_SIZE = PRIVILEGED_HMAP_ROWS * PRIVILEGED_HMAP_COLS
CONTACT_POINTS_OFFSET = PARTIAL_DIM + PRIV_HMAP_SIZE
CONTACT_PROB_OFFSET = CONTACT_POINTS_OFFSET + 4 * 3


@dataclass
class CTRACObservation(Observation):
    """Partial + privileged observation for C-TRAC (Pan et al. 2025), packed into one flat
    OBS_KEY vector (see PARTIAL_DIM/PRIVILEGED_DIM above for the exact layout — mirrored in
    ctrac_policy.py's slicing constants, keep both in sync).

    supports_vecnorm = False, same reasoning as hfc/mitriakov: the privileged slice's
    ground-truth contact points/heightmap and the partial slice's roll/pitch/velocity must
    stay raw and physically meaningful for the NESM stabilization reward (ctrac_module.py)
    and the C-VAE's geometric-feasibility loss (ctrac_cvae.py) — VecNorm's drifting running
    mean/std would silently distort both.
    """

    supports_vecnorm = False
    dim = TOTAL_DIM

    def __call__(self, prev_state, action, prev_state_der, curr_state):
        raise NotImplementedError("CTRACObservation is populated directly by FtrTorchRLEnv._step / _reset.")

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.batch_size[0], self.dim),
            device=self.env.device,
            dtype=torch.float32,
        )

    def get_encoder(self) -> ObservationEncoder:
        return CTRACIdentityEncoder(**(self.encoder_opts or {}))


class CTRACIdentityEncoder(ObservationEncoder):
    """Identity passthrough — provided only so CTRACObservation conforms to the standard
    Observation/ObservationEncoder interface (used by FtrTorchRLEnv to size its observation
    spec). Not used by CTRACPolicyConfig (ctrac_policy.py), which is a fully self-contained
    policy construction reading OBS_KEY directly and slicing partial/privileged itself —
    same convention as HFCIdentityEncoder/MitriakovEncoder.
    """

    def __init__(self, **kwargs):
        super().__init__(output_dim=CTRACObservation.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
