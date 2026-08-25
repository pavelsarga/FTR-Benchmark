from __future__ import annotations

from pathlib import Path
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

    def load_module_cfg(self, path):
        """Load this module's own YAML and merge ``env.cfg.module_cfg_overrides`` over it.

        Each reproduction keeps its paper constants (and, where the target robot differs
        from the paper's, its adaptation switches) in ``rl_modules/<name>/<name>_module.yaml``
        rather than in ``configs/``, because they belong to the reproduction and not to a
        training run. This hook lets a training config vary them per experiment — e.g. run
        the paper-faithful reward semantics and the MARV-adapted ones as two configs —
        without editing a file inside the module.

        Unknown keys raise instead of being silently ignored: a typo'd switch would
        otherwise leave the run quietly on the default and produce a result attributed to
        the wrong condition.
        """
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        overrides = dict(getattr(self.env.cfg, "module_cfg_overrides", None) or {})
        if not overrides:
            return cfg

        unknown = sorted(set(overrides) - set(cfg.keys()))
        if unknown:
            raise ValueError(
                f"module_cfg_overrides for {type(self).__name__} names unknown key(s) "
                f"{unknown}; {Path(path).name} defines {sorted(cfg.keys())}."
            )
        return OmegaConf.merge(cfg, OmegaConf.create(overrides))

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