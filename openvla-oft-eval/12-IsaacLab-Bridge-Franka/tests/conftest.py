"""Pytest conftest: add parent directory to sys.path so tests can import
without triggering the package __init__.py relative imports."""

import sys
from pathlib import Path

# Ensure the tests can import modules directly without package-level __init__
parent = str(Path(__file__).resolve().parents[1])
if parent not in sys.path:
    sys.path.insert(0, parent)

# 需要 Isaac Lab 才能 import 的模块, 不能进收集器。
# collect_ignore 是 conftest 变量而不是 ini 选项 —— 写在 pytest.ini 里会被静默忽略。
collect_ignore = ["__init__.py", "register_env.py", "bridge_pick_place_env_cfg.py"]
