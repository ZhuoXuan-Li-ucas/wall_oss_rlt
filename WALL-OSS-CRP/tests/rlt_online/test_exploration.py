"""Per-chunk ramped pose exploration and random release tries (D6 / D9 / D14)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlt_online.action_space import ActionBounds, ChunkActionCodec, episode_windows
from rlt_online.exploration import ChunkExplorer, ExplorationConfig
from wall_rlt.data.ar5 import rot6d_to_matrix

ROOT = Path(__file__).resolve().parents[2]
C = 12


def _config(**overrides):
    values = dict(position_std_m=0.002, rotation_std_deg=0.5, random_release_prob=0.08,
                  warmup_pose_noise=False, warmup_release_exploration=True)
    values.update(overrides)
    return ExplorationConfig(**values)


def _chunk(seed=0):
    """A realistic WALL-like chunk: dev episode rows 11..22 of one window, gripper closed."""
    path = Path('/home/crp-5070ti-01/atom_episodes/wall_unlocked_v2/bolt/episode_0014.npz')
    if not path.is_file():
        pytest.skip('bolt episodes are not available on this host')
    for t, rel in episode_windows(path):
        if t == 200 + seed:
            return rel[11:23].copy()


def test_eval_adds_nothing():
    rows = _chunk()
    out, info = ChunkExplorer(_config(), C, np.random.default_rng(0))(rows, mode='eval')
    np.testing.assert_array_equal(out, rows)
    assert not info['random_release']


def test_train_offset_is_a_linear_ramp_from_one_over_c():
    rows = _chunk()
    out, info = ChunkExplorer(_config(random_release_prob=0.0), C, np.random.default_rng(1))(rows, mode='train')
    ramp = np.arange(1, C + 1)[:, None] / C
    np.testing.assert_allclose(out[:, :3] - rows[:, :3], ramp * np.array(info['position_offset_m']), atol=1e-12)
    rel = Rotation.from_matrix(rot6d_to_matrix(out[:, 3:9])) * Rotation.from_matrix(rot6d_to_matrix(rows[:, 3:9])).inv()
    np.testing.assert_allclose(rel.as_rotvec(), ramp * np.array(info['rotation_offset_rad']), atol=1e-9)
    np.testing.assert_array_equal(out[:, 9], rows[:, 9])


def test_noise_and_release_statistics():
    rows = _chunk()
    explorer = ChunkExplorer(_config(), C, np.random.default_rng(2))
    offsets, angles, releases = [], [], 0
    for _ in range(20000):
        _, info = explorer(rows, mode='train')
        offsets.append(info['position_offset_m'])
        angles.append(info['rotation_offset_rad'])
        releases += info['random_release']
    assert np.std(offsets, axis=0) == pytest.approx([0.002] * 3, rel=0.03)
    assert np.degrees(np.std(angles, axis=0)) == pytest.approx([0.5] * 3, rel=0.03)
    assert releases / 20000 == pytest.approx(0.08, abs=0.006)


def test_release_try_opens_the_whole_chunk():
    rows = _chunk()
    out, info = ChunkExplorer(_config(random_release_prob=1.0), C, np.random.default_rng(3))(rows, mode='train')
    assert info['random_release'] and (out[:, 9] == 1.0).all()


def test_warmup_releases_but_keeps_the_pose():
    rows = _chunk()
    explorer = ChunkExplorer(_config(random_release_prob=1.0), C, np.random.default_rng(4))
    out, info = explorer(rows, mode='warmup')
    np.testing.assert_array_equal(out[:, :9], rows[:, :9])
    assert info['random_release'] and info['position_offset_m'] == [0.0, 0.0, 0.0]


def test_explored_chunks_stay_inside_the_d1_bounds():
    codec = ChunkActionCodec(ActionBounds.from_json(ROOT / 'configs/rlt_online/action_bounds.bolt.20261110.proposal.json'),
                             C, 11)
    explorer = ChunkExplorer(_config(), C, np.random.default_rng(5))
    clipped = 0
    for seed in range(50):
        rows = _chunk(seed)
        out, _ = explorer(codec.decode(codec.encode(rows)[0]), mode='train')
        normalized, n = codec.encode(out)
        clipped += n
        np.testing.assert_allclose(codec.decode(normalized), out, atol=1e-12)
    assert clipped == 0


def test_config_section_from_the_proposal_file():
    import yaml
    meta = yaml.safe_load((ROOT / 'configs/rlt_online/arm1.bolt.yaml').read_text())
    cfg = ExplorationConfig.from_meta(meta)
    assert (cfg.position_std_m, cfg.rotation_std_deg, cfg.random_release_prob) == (0.002, 0.5, 0.08)
    assert cfg.warmup_release_exploration and not cfg.warmup_pose_noise
