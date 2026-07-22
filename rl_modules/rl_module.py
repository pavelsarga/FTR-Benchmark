from __future__ import annotations

from typing import TYPE_CHECKING

import einops
import torch
from omni.isaac.lab.envs import VecEnvObs


if TYPE_CHECKING:
    from ftr_envs.tasks.crossing.ftr_env import FtrEnv


class RLModule:
    """Base class for reward/observation computation, decoupled from env state management.

    Subclasses hold a reference to the owning env and read/write its state (positions,
    cfg, buffers, ...) rather than owning it themselves.
    """

    def __init__(self, env: "FtrEnv") -> None:
        self.env = env

    def get_observations(self) -> VecEnvObs:
        raise NotImplementedError

    def get_reward_components(self) -> dict[str, torch.Tensor]:
        """Return this module's individual reward terms, keyed by name.

        Each tensor is the term's actual signed contribution to the total reward
        (already negated for penalties), including step penalty and terminal
        masking/bonus — this module owns its full reward, terminal handling included.
        The caller (CrossingEnv) just sums the returned dict and does all logging.
        """
        raise NotImplementedError

    def calc_scanned_height_maps(self, base_robot_frame=True):
        raise NotImplementedError