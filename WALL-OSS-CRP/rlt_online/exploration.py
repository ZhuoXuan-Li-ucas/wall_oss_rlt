"""Exploration for WALL-RLT Stage 2 rollouts (decisions D6 / D9 / D14, 2026-11-10).

Applied to the actor's chunk *after* it has been decoded to physical relative
rows (``ChunkActionCodec.decode``) and before it is handed to the executor;
the executed rows are then re-encoded for the replay buffer. The actor's own
Gaussian noise is off (``actor_fixed_std = 0``).

- Position and orientation: one random offset per chunk, ramped linearly
  from 1/C at the first step to the full offset at step C, so the chunk
  stays continuous with the previous one and smooth inside. Position offset
  is per-axis N(0, position_std_m^2) in the AR5 base frame. Orientation is a
  rotation vector with per-axis N(0, rotation_std_deg^2), left-multiplied
  onto the relative rotation (``R_rel = R_abs R_obs^T``), i.e. a base-frame
  rotation of the commanded flange orientation.
- Gripper: with probability ``random_release_prob`` per chunk the whole
  chunk commands "open" (a release attempt). The operator's release key is
  handled by the caller as a human intervention, not here.
- Warmup (D9): release exploration on, pose noise off.
- Evaluation: nothing is added.

The noise distribution never enters a loss (off-policy: the critic uses the
executed chunk, the actor loss uses the actor mean), so this shape replaces
the paper's i.i.d. sigma^2 I without changing the algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from rlt_online.action_space import GRIPPER, ROT6D, STEP_DIM
from wall_rlt.data.ar5 import rot6d_to_matrix

OPEN = 1.0


@dataclass(kw_only=True)
class ExplorationConfig:
    position_std_m: float
    rotation_std_deg: float
    random_release_prob: float
    warmup_pose_noise: bool
    warmup_release_exploration: bool

    def __post_init__(self):
        if self.position_std_m < 0 or self.rotation_std_deg < 0:
            raise ValueError('noise scales must be >= 0')
        if not 0 <= self.random_release_prob <= 1:
            raise ValueError('random_release_prob must be in [0, 1]')

    @classmethod
    def from_meta(cls, meta: dict) -> 'ExplorationConfig':
        """Build from the ``exploration`` section returned by ``load_configs``."""
        section = meta['exploration']
        if section.get('form') != 'per_chunk_linear_ramp':
            raise ValueError(f"unsupported exploration form {section.get('form')!r}")
        gripper, warmup = section['gripper'], section['warmup']
        if gripper.get('release_source') != 'human_key_plus_random':
            raise ValueError(f"unsupported release_source {gripper.get('release_source')!r}")
        values = dict(position_std_m=section['position_std_m'],
                      rotation_std_deg=section['rotation_std_deg'],
                      random_release_prob=gripper['random_release_prob_per_chunk'],
                      warmup_pose_noise=warmup['pose_noise'],
                      warmup_release_exploration=warmup['release_exploration'])
        undecided = [k for k, v in values.items() if v is None]
        if undecided:
            raise ValueError(f'exploration has undecided fields {undecided}')
        return cls(**values)


class ChunkExplorer:
    """Adds the per-chunk exploration to physical relative rows ``[C, 10]``."""

    def __init__(self, config: ExplorationConfig, chunk_length: int, rng: np.random.Generator):
        self.config, self.chunk_length, self.rng = config, chunk_length, rng
        self.ramp = np.arange(1, chunk_length + 1, dtype=np.float64) / chunk_length

    def __call__(self, relative, *, mode: str) -> tuple[np.ndarray, dict]:
        """``mode`` is 'train', 'warmup' or 'eval'. Returns (executed rows, info)."""
        if mode not in ('train', 'warmup', 'eval'):
            raise ValueError(f'unknown mode {mode!r}')
        rows = np.array(relative, dtype=np.float64, copy=True)
        if rows.shape != (self.chunk_length, STEP_DIM) or not np.isfinite(rows).all():
            raise ValueError(f'expected finite [{self.chunk_length},{STEP_DIM}] rows, got {rows.shape}')
        cfg = self.config
        info = {'mode': mode, 'position_offset_m': [0.0, 0.0, 0.0],
                'rotation_offset_rad': [0.0, 0.0, 0.0], 'random_release': False}
        if mode == 'eval':
            return rows, info
        pose_noise = mode == 'train' or cfg.warmup_pose_noise
        release = mode == 'train' or cfg.warmup_release_exploration
        if pose_noise and (cfg.position_std_m > 0 or cfg.rotation_std_deg > 0):
            offset = self.rng.normal(0.0, cfg.position_std_m, 3)
            rotvec = self.rng.normal(0.0, np.radians(cfg.rotation_std_deg), 3)
            rows[:, :3] += self.ramp[:, None] * offset
            matrices = rot6d_to_matrix(rows[:, ROT6D])                 # orthonormalized
            perturbed = Rotation.from_rotvec(self.ramp[:, None] * rotvec).as_matrix() @ matrices
            rows[:, ROT6D] = perturbed[:, :2, :].reshape(-1, 6)
            info['position_offset_m'] = offset.tolist()
            info['rotation_offset_rad'] = rotvec.tolist()
        if release and self.rng.random() < cfg.random_release_prob:
            rows[:, GRIPPER] = OPEN
            info['random_release'] = True
        return rows, info
