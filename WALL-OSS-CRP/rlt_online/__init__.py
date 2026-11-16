"""WALL-RLT Stage 2: online actor-critic on top of the frozen WALL + RL token.

Runs on the 5070 Ti host in the robot's Python environment (``dexgraspvla``),
next to ``yuhang_workspace/wall_oss_rlt/tools/wall_execute.py``. The H100
training bundle does not include this package. Plan and decisions:
``WALL_RLT_Stage2_Research_Plan.md``.

Modules:
  evo           vendored Evo-RLT core, collector and online trainer
  action_space  per-step action bounds, [-1, 1] codec, proprio normalizer
  policy        actor/critic stand-in for Evo-RLT's OnlineRLTrainer, configs
"""

__version__ = '0.1.0'
