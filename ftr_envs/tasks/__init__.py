"""Task registrations.

Every task package under this directory is imported so its ``gym.register`` call runs.
Only ``crossing`` remains — the upstream ``prey`` / ``push_cube`` / ``trans_cargo`` /
``anymal_d`` tasks were removed, and the auto-import is why an unused task was not free:
it was imported on every training and eval job.
"""

import importlib
import os
from pathlib import Path

_TASKS_DIR = Path(__file__).parent
for _name in sorted(os.listdir(_TASKS_DIR)):
    if (_TASKS_DIR / _name).is_dir() and not _name.startswith("_"):
        importlib.import_module(f"ftr_envs.tasks.{_name}")
