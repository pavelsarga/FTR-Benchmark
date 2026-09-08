"""Registers Ftr-Crossing-Direct-v0, the only task this fork uses.

Which observations and rewards the env computes is not chosen here — it comes from
`env_cfg_overrides.module_name` at env-creation time, via rl_modules/registry.py. The
upstream rl_games / skrl / ftr_algo entry points are gone along with those launchers.
"""

import gymnasium as gym

from .crossing_env import CrossingEnv, CrossingEnvCfg  # noqa: F401 — re-exported

gym.register(
    id="Ftr-Crossing-Direct-v0",
    entry_point="ftr_envs.tasks.crossing.crossing_env:CrossingEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": CrossingEnvCfg},
)
