# FTR-Benchmark (MARV fork)

Isaac Sim / IsaacLab simulation environments for the MARV tracked rover, forked from
[FTR-Bench](https://github.com/tianyudwang/FTR-Bench) (Zhang et al.) and reduced to the
parts the MARV_RL project actually uses. Training itself lives in the sibling
`flipper_training` submodule; this repository provides the environment it steps.

```
ftr_envs/
  tasks/crossing/       the one registered task, Ftr-Crossing-Direct-v0
                        ftr_env.py    physics, flipper control, heightmap sampling
                        crossing_env.py  episode logic, termination, the two stats dicts
  assets/articulation/  robot definitions (ftr.py, marv.py) and the USD conversion scripts
  assets/terrain/       generated terrains: usd/, map/, config/, birth/, gen_config/, plot/
  utils/                OmegaConf helpers shared by the env configs
rl_modules/             pluggable observation + reward implementations, selected at
                        env-creation time by env_cfg_overrides.module_name
scripts/debug.py        stand-alone env debug/plot runner
```

## rl_modules

`rl_modules/registry.py` maps `module_name` to an `RLModule` subclass. Each module owns its
full `get_observations()` / `get_reward_components()`; `CrossingEnv`/`FtrEnv` sum whatever
components the active module returns and log them, and know nothing about the reward formula.

| module | what it is |
|---|---|
| `marv_rl` | the project's own reward/obs: 4 independent flippers + track velocity |
| `atd3qn` | Pan et al. 2023 AT-D3QN — terrain+state fusion encoder, Dueling D3QN head, 9 discrete actions |
| `icmd3qn` | Pan et al. 2023 ICM-D3QN — same Q-network plus an Intrinsic Curiosity Module |
| `ctrac` | Pan et al. 2025 C-TRAC — asymmetric SAC with a C-VAE contact estimator |
| `creps` | Pecka et al. 2016 CREPS — contextual relative-entropy policy search |
| `hfc` / `hfcil` | heuristic flipper controller, and its imitation-learning variant |
| `mitriakov` | Mitriakov et al. step-edge-table baseline |

`atd3qn` and `icmd3qn` require `env_cfg_overrides: {sync_flipper_control: true}`, which
collapses the four independent flippers into a synced front/rear pair to match their 3x3
discrete action space.

## Notes carried over from upstream

Removed from this fork because nothing here used them: the `ftr_algo/` MARL algorithm suite
(HAPPO/MAPPO/IPPO/MADDPG/TRPO/DDPG/TD3/SAC/PPO), the `prey`, `push_cube`, `trans_cargo` and
`anymal_d` tasks, the rl_games / skrl / ftr_algo launcher scripts and their agent configs,
and the upstream README media. `ftr_envs/tasks/__init__.py` imports every task package it
finds, so an unused task was not merely dead — it was imported on every training job.

The upstream project's paper, videos and benchmark results are at the link above.
