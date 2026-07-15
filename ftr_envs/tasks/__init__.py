# -*- coding: utf-8 -*-
"""
====================================
@File Name ：__init__.py
@Time ： 2024/9/29 PM12:15
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""

import importlib
import inspect
import os
from pathlib import Path

current_file_dir = Path(__file__).parent
for directory in os.listdir(current_file_dir):
    if os.path.isdir(current_file_dir / directory) and not directory.startswith('_'):
        importlib.import_module(f'ftr_envs.tasks.{directory}')