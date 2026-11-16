"""Access to the vendored Evo-RLT sources (see ``_vendor/NOTICE.md``).

Importing this module puts ``rlt_online/_vendor`` at the front of ``sys.path``
so the unmodified upstream modules can keep their absolute ``evo_rlt.*``
imports. It refuses to continue if another ``evo_rlt`` installation would be
used instead, because the manifest only vouches for the vendored copy.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / '_vendor'


def _ensure_on_path() -> None:
    entry = str(VENDOR)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def verify_vendor() -> list[str]:
    """Return a list of problems; empty means every vendored file matches MANIFEST.json."""
    manifest = json.loads((VENDOR / 'MANIFEST.json').read_text(encoding='utf-8'))
    problems = []
    for entry in manifest['files']:
        path = VENDOR / entry['path']
        if not path.is_file():
            problems.append(f"missing {entry['path']}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            problems.append(f"modified {entry['path']}")
    listed = {entry['path'] for entry in manifest['files']}
    for path in VENDOR.rglob('*'):
        relative = path.relative_to(VENDOR).as_posix()
        if (path.is_file() and '__pycache__' not in path.parts
                and relative not in listed | {'MANIFEST.json', 'NOTICE.md'}):
            problems.append(f'unlisted {relative}')
    return problems


_ensure_on_path()

import evo_rlt  # noqa: E402

if Path(evo_rlt.__file__).resolve().parent != VENDOR / 'evo_rlt':
    raise ImportError(f'evo_rlt resolved to {evo_rlt.__file__}, not the vendored copy in {VENDOR}')

from evo_rlt.adapters.lerobot.online_collector import RLTOnlineCollector  # noqa: E402
from evo_rlt.adapters.lerobot.record.annotations import SOURCE_HUMAN, SOURCE_RL, SOURCE_VLA  # noqa: E402
from evo_rlt.adapters.lerobot.record.online_trainer import OnlineRLTrainer  # noqa: E402
from evo_rlt.core.actor import ChunkActor  # noqa: E402
from evo_rlt.core.critic import TwinCritic  # noqa: E402
from evo_rlt.core.interfaces import ChunkTransition  # noqa: E402
from evo_rlt.core.losses import actor_loss, critic_loss, q_action_sensitivity  # noqa: E402
from evo_rlt.core.replay_buffer import ReplayBuffer  # noqa: E402
from evo_rlt.core.utils import flatten_chunk, soft_update, unflatten_chunk  # noqa: E402

__all__ = ['VENDOR', 'verify_vendor', 'RLTOnlineCollector', 'OnlineRLTrainer', 'ChunkActor',
           'TwinCritic', 'ChunkTransition', 'ReplayBuffer', 'actor_loss', 'critic_loss',
           'q_action_sensitivity', 'flatten_chunk', 'unflatten_chunk', 'soft_update',
           'SOURCE_HUMAN', 'SOURCE_RL', 'SOURCE_VLA']
