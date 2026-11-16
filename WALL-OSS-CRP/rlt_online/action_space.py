"""RL action space for WALL-RLT Stage 2 on the AR5 EE contract.

The actor works in [-1, 1]. Physically, one step of a chunk is the WALL
contract's 10-dim row *relative to the observation pose at which the chunk
was requested*: flange position offset (m, base frame), relative rotation as
6D (first two rows of ``R_abs @ R_obs^T``) and the gripper command (0 closed,
1 open). This is exactly what the inference service returns in ``actions``
(32 x 10), so the VLA reference chunk needs no conversion beyond scaling.

Scaling uses per-step, per-dimension bounds ``lo[k, d]`` / ``hi[k, d]`` for
window step ``k`` (0..31), measured over the train split with a margin. Step
``k`` matters because the reachable offset grows with ``k``. A chunk that
starts at window step ``offset`` (for example after the steps that elapsed
during inference) uses rows ``offset .. offset + C - 1`` of the table.

The bounds are a *proposal* until the user signs off decision D2 of
``WALL_RLT_Stage2_Research_Plan.md``; ``ActionBounds.status`` records that.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wall_rlt.data.ar5 import (EE_REL6D_NEXT, EE_ROW_DIM, aligned_targets, ee_absolute_window,
                               ee_relative_window)

STEP_DIM = EE_ROW_DIM            # 10
GRIPPER = 9
ROT6D = slice(3, 9)
MIN_RANGE = 1e-6                 # guards dims that never move at a given k
ROT6D_LIMIT = 1.2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def episode_windows(npz_path, horizon: int = 32):
    """Yield ``(t, rel)`` for every observation frame: rel is ``[K, 10]``, K <= horizon.

    Identical construction to training: ``aligned_targets`` (observation t,
    targets t+1..) and ``ee_relative_window``; windows near the end are
    shorter instead of padded.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        targets, observations = aligned_targets(z['state'], z['action'], EE_REL6D_NEXT,
                                                ee_pose=z['ee_pose'])
    for t in range(len(observations)):
        k = min(horizon, len(targets) - t)
        yield t, ee_relative_window(targets[t:t + k], observations[t])


@dataclass
class ActionBounds:
    lo: np.ndarray                       # [H, 10]
    hi: np.ndarray                       # [H, 10]
    status: str = 'proposal'
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=np.float64)
        self.hi = np.asarray(self.hi, dtype=np.float64)
        if self.lo.shape != self.hi.shape or self.lo.ndim != 2 or self.lo.shape[1] != STEP_DIM:
            raise ValueError(f'bounds must be [H,{STEP_DIM}], got {self.lo.shape} / {self.hi.shape}')
        if not (np.isfinite(self.lo).all() and np.isfinite(self.hi).all()):
            raise ValueError('bounds must be finite')
        if not (self.hi - self.lo >= MIN_RANGE).all():
            raise ValueError('every bound needs hi - lo >= MIN_RANGE')

    @property
    def horizon(self) -> int:
        return self.lo.shape[0]

    def to_json(self, path) -> None:
        payload = {'schema_version': 1, 'status': self.status, 'step_dim': STEP_DIM,
                   'horizon': self.horizon, 'provenance': self.provenance,
                   'lo': self.lo.tolist(), 'hi': self.hi.tolist()}
        Path(path).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')

    @classmethod
    def from_json(cls, path) -> 'ActionBounds':
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('schema_version') != 1 or payload.get('step_dim') != STEP_DIM:
            raise ValueError(f'{path}: unsupported action bounds file')
        return cls(payload['lo'], payload['hi'], payload['status'], payload.get('provenance', {}))


def compute_bounds(contract_path, *, split: str = 'train', margin: float = 0.2,
                   horizon: int = 32) -> ActionBounds:
    """Per-(k, d) min/max over ``split`` with ``margin`` x range added on both sides.

    The gripper dimension is fixed to [0, 1] (the command is binary; WALL's
    continuous prediction is clipped there). Episode files are resolved and
    hash-checked through the contract's episode manifest.
    """
    if not 0 <= margin <= 1:
        raise ValueError('margin must be within [0, 1]')
    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    manifest_path = (contract_path.parent / contract['manifest_path']).resolve()
    if _sha256(manifest_path) != contract['manifest_sha256']:
        raise ValueError(f'{manifest_path} does not match the contract manifest_sha256')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    by_id = {episode['episode_id']: episode for episode in manifest['episodes']}
    root = Path(contract['root'])
    ids = list(contract['splits'][split])
    lo = np.full((horizon, STEP_DIM), np.inf)
    hi = np.full((horizon, STEP_DIM), -np.inf)
    rows = 0
    for episode_id in ids:
        entry = by_id[episode_id]
        path = root / entry['npz']
        if _sha256(path) != entry['npz_sha256']:
            raise ValueError(f'{path} does not match the manifest sha256')
        for _, rel in episode_windows(path, horizon):
            k = len(rel)
            lo[:k] = np.minimum(lo[:k], rel)
            hi[:k] = np.maximum(hi[:k], rel)
            rows += k
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError('some window steps never occur in the split')
    measured_lo, measured_hi = lo.copy(), hi.copy()
    width = np.maximum(hi - lo, MIN_RANGE)
    lo, hi = lo - margin * width, hi + margin * width
    # Plan D2 proposal: rot6D within [-1.2, 1.2]. Not [-1, 1]: WALL's rot6D
    # output is not exactly orthonormal (diagonal entries reach 1.0001), and the
    # zero-init actor must reproduce it without clipping; decoding orthonormalizes.
    lo[:, ROT6D] = np.maximum(lo[:, ROT6D], -ROT6D_LIMIT)
    hi[:, ROT6D] = np.minimum(hi[:, ROT6D], ROT6D_LIMIT)
    widen = hi - lo < MIN_RANGE
    hi[widen] = lo[widen] + MIN_RANGE
    lo[:, GRIPPER], hi[:, GRIPPER] = 0.0, 1.0
    provenance = {
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'contract': str(contract_path), 'contract_sha256': _sha256(contract_path),
        'target_mode': contract['target_mode'], 'fps': contract['fps'],
        'split': split, 'episode_ids': ids, 'window_rows': rows, 'horizon': horizon,
        'margin_fraction_of_range': margin,
        'rule': 'per window step k and dim d: train min/max widened by margin*range; '
                f'rot6D limited to [-{ROT6D_LIMIT},{ROT6D_LIMIT}]; gripper fixed to [0,1]',
        'measured_lo': measured_lo.tolist(), 'measured_hi': measured_hi.tolist(),
        'names': contract['action_names'],
    }
    return ActionBounds(lo, hi, 'proposal (decision D2 not confirmed)', provenance)


class ChunkActionCodec:
    """Maps C-step relative chunks to the actor's [-1, 1] space and back.

    ``offset`` selects the window steps the chunk occupies (see module doc).
    All arrays are ``[C, 10]``; flattening for the networks is time-major,
    matching Evo-RLT ``flatten_chunk``.
    """

    def __init__(self, bounds: ActionBounds, chunk_length: int, offset: int = 0):
        if chunk_length < 1 or offset < 0 or offset + chunk_length > bounds.horizon:
            raise ValueError(f'chunk [{offset}, {offset + chunk_length}) is outside the '
                             f'{bounds.horizon}-step bounds table')
        self.chunk_length, self.offset = chunk_length, offset
        self.lo = bounds.lo[offset:offset + chunk_length]
        self.hi = bounds.hi[offset:offset + chunk_length]

    def _check(self, chunk) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.shape != (self.chunk_length, STEP_DIM) or not np.isfinite(chunk).all():
            raise ValueError(f'chunk must be finite [{self.chunk_length},{STEP_DIM}], got {chunk.shape}')
        return chunk

    def encode(self, relative) -> tuple[np.ndarray, int]:
        """Physical relative rows -> [-1, 1]; returns (normalized, number of clipped entries)."""
        relative = self._check(relative)
        scaled = 2.0 * (relative - self.lo) / (self.hi - self.lo) - 1.0
        clipped = np.clip(scaled, -1.0, 1.0)
        return clipped, int(np.count_nonzero(clipped != scaled))

    def decode(self, normalized) -> np.ndarray:
        """[-1, 1] -> physical relative rows (input is clipped to [-1, 1] first)."""
        normalized = np.clip(self._check(normalized), -1.0, 1.0)
        return self.lo + (normalized + 1.0) * 0.5 * (self.hi - self.lo)

    @staticmethod
    def to_absolute(relative, observation_state) -> np.ndarray:
        """Relative rows + the 10-dim observation state -> ``[C, 7]`` flange targets.

        Same function the inference service uses for ``absolute_targets``:
        position (m), rotation vector (rad, rot6D Gram-Schmidt orthonormalized),
        continuous gripper value (the executor applies its own hysteresis).
        """
        return ee_absolute_window(np.asarray(relative, dtype=np.float64), observation_state)


class ProprioNormalizer:
    """10-dim observation state -> roughly [-1, 1] with the contract's train min/max.

    No clipping: values outside the train range stay visible to the networks.
    """

    def __init__(self, normalization_path):
        payload = json.loads(Path(normalization_path).read_text(encoding='utf-8'))
        stats = payload['norm_stats']['state']
        self.lo = np.asarray(stats['min'], dtype=np.float64)
        self.hi = np.asarray(stats['max'], dtype=np.float64)
        if self.lo.shape != (STEP_DIM,) or not (self.hi - self.lo > 0).all():
            raise ValueError('state min/max must be 10 non-degenerate values')
        self.source = str(normalization_path)

    def __call__(self, state) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape[-1] != STEP_DIM:
            raise ValueError(f'state must end in {STEP_DIM} values')
        return 2.0 * (state - self.lo) / (self.hi - self.lo) - 1.0


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Compute the proposed per-step action bounds (D2).')
    parser.add_argument('--contract', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--margin', type=float, default=0.2)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f'{output} exists; bounds files are versioned, write a new name')
    bounds = compute_bounds(args.contract, split=args.split, margin=args.margin)
    bounds.to_json(output)
    print(f"wrote {output}: {bounds.provenance['window_rows']} window rows, "
          f"{len(bounds.provenance['episode_ids'])} episodes, status={bounds.status!r}")


if __name__ == '__main__':
    main()
