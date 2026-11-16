"""Action-space codec against the real bolt episodes and the frozen contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlt_online.action_space import (GRIPPER, ActionBounds, ChunkActionCodec, ProprioNormalizer,
                                     episode_windows)
from wall_rlt.data.ar5 import EE_REL6D_NEXT, aligned_targets, rot6d_to_matrix

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'configs/rlt/wall_v2_bolt_ar5_ee_rel6d_next_v1.json'
BOUNDS = ROOT / 'configs/rlt_online/action_bounds.bolt.20261110.proposal.json'
NORMALIZATION = ROOT / 'metadata/rlt-wall-v2-bolt50-20261103/normalization.ar5_ee_rel6d_next_v1.json'


def _episodes(split):
    contract = json.loads(CONTRACT.read_text())
    manifest = json.loads((CONTRACT.parent / contract['manifest_path']).resolve().read_text())
    by_id = {e['episode_id']: e for e in manifest['episodes']}
    root = Path(contract['root'])
    paths = [root / by_id[i]['npz'] for i in contract['splits'][split]]
    if not all(p.is_file() for p in paths):
        pytest.skip('bolt episodes are not available on this host')
    return paths


@pytest.fixture(scope='module')
def bounds():
    return ActionBounds.from_json(BOUNDS)


def test_bounds_file_matches_contract_statistics(bounds):
    norm = json.loads(NORMALIZATION.read_text())
    assert bounds.horizon == 32
    assert 'proposal' in bounds.status
    assert bounds.provenance['window_rows'] == norm['norm_stats']['action']['count']
    measured_lo = np.array(bounds.provenance['measured_lo'])
    measured_hi = np.array(bounds.provenance['measured_hi'])
    # The per-step table must contain the contract's all-step min/max.
    np.testing.assert_allclose(measured_lo[:, :GRIPPER].min(0), norm['norm_stats']['action']['min'][:GRIPPER],
                               atol=1e-6)
    np.testing.assert_allclose(measured_hi[:, :GRIPPER].max(0), norm['norm_stats']['action']['max'][:GRIPPER],
                               atol=1e-6)
    assert (bounds.lo <= measured_lo).all() and (bounds.hi >= measured_hi).all()


@pytest.mark.parametrize('chunk_length,offset', [(12, 11), (10, 0), (10, 8), (32, 0)])   # (12, 11) = D1
def test_train_windows_round_trip_without_clipping(bounds, chunk_length, offset):
    codec = ChunkActionCodec(bounds, chunk_length, offset)
    checked = 0
    for path in _episodes('train')[::5]:
        for _, rel in episode_windows(path):
            if len(rel) < offset + chunk_length:
                continue
            chunk = rel[offset:offset + chunk_length]
            normalized, clipped = codec.encode(chunk)
            assert clipped == 0
            assert np.abs(normalized).max() <= 1.0
            np.testing.assert_allclose(codec.decode(normalized), chunk, atol=1e-12)
            np.testing.assert_allclose(codec.decode(normalized.astype(np.float32)), chunk, atol=2e-6)
            checked += 1
    assert checked > 1000


def test_decoded_chunk_reproduces_recorded_absolute_poses(bounds):
    codec = ChunkActionCodec(bounds, 10, 0)
    path = _episodes('dev')[0]
    with np.load(path) as z:
        targets, observations = aligned_targets(z['state'], z['action'], EE_REL6D_NEXT, ee_pose=z['ee_pose'])
        ee_pose = z['ee_pose'].astype(np.float64)
    from scipy.spatial.transform import Rotation
    for t, rel in episode_windows(path):
        if len(rel) < 10 or t % 7:
            continue
        normalized, _ = codec.encode(rel[:10])
        absolute = codec.to_absolute(codec.decode(normalized), observations[t])
        np.testing.assert_allclose(absolute[:, :3], ee_pose[t + 1:t + 11, :3], atol=1e-6)
        got = Rotation.from_rotvec(absolute[:, 3:6]).as_matrix()
        want = Rotation.from_rotvec(ee_pose[t + 1:t + 11, 3:6]).as_matrix()
        np.testing.assert_allclose(got, want, atol=1e-5)
        np.testing.assert_array_equal(absolute[:, 6], targets[t:t + 10, 9])


def test_decoded_rot6d_is_orthonormalized_for_any_actor_output(bounds):
    codec = ChunkActionCodec(bounds, 10, 0)
    rng = np.random.default_rng(0)
    state = np.array([0.25, -0.25, 0.35, *rot6d_to_matrix(np.array([[1, 0, 0, 0, 1, 0.]]))[0][:2].ravel(), 0.5])
    absolute = codec.to_absolute(codec.decode(rng.uniform(-1, 1, (10, 10))), state)
    assert np.isfinite(absolute).all()


def test_held_out_episodes_rarely_leave_the_bounds(bounds):
    codec = ChunkActionCodec(bounds, 10, 0)
    total = clipped = 0
    for path in _episodes('dev') + _episodes('test'):
        for _, rel in episode_windows(path):
            if len(rel) >= 10:
                clipped += codec.encode(rel[:10])[1]
                total += rel[:10].size
    assert clipped / total < 1e-3


def test_codec_rejects_chunks_outside_the_table(bounds):
    with pytest.raises(ValueError):
        ChunkActionCodec(bounds, 10, 23)
    codec = ChunkActionCodec(bounds, 10, 22)
    with pytest.raises(ValueError):
        codec.encode(np.zeros((9, 10)))


def test_proprio_normalizer_maps_train_extremes():
    norm = json.loads(NORMALIZATION.read_text())
    proprio = ProprioNormalizer(NORMALIZATION)
    np.testing.assert_allclose(proprio(norm['norm_stats']['state']['min']), -1.0)
    np.testing.assert_allclose(proprio(norm['norm_stats']['state']['max']), 1.0)
