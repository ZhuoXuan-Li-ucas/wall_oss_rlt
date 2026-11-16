"""A memory lower bound must retain excluded costs and never read weight bodies."""
import json
import struct

import pytest

from wall_rlt.config import WallRLTConfig
from wall_rlt.resources import estimate_training_memory


def tiny_config():
    return WallRLTConfig(input_dim=4, embed_dim=4, num_heads=1, num_layers=1,
                         prefix_seq_len=8, rlt_alpha=2, action_horizon=2,
                         model_action_dim=2, action_token_id=9)


def test_memory_uses_only_header_and_preserves_exclusions(tmp_path):
    header = json.dumps({'weights': {'shape': [2, 3]},
                         'action_preprocessor.normalizer_action.mean': {'shape': [100]},
                         '__metadata__': {'note': 'weight body intentionally absent'}}).encode()
    path = tmp_path / 'model.safetensors'
    path.write_bytes(struct.pack('<Q', len(header)) + header)
    one = estimate_training_memory(tmp_path, tiny_config())
    two = estimate_training_memory(path, tiny_config(), world_size=2, sharding_strategy='full_shard')
    assert one['wall_serialized_scalars_excluding_normalizers'] == 6
    assert one['persistent_per_rank_bytes'] == one['total_scalar_count'] * 16
    assert two['persistent_per_rank_bytes'] * 2 == one['persistent_per_rank_bytes']
    assert one['bf16_whole_model_working_copy_bytes_excluded'] == one['total_scalar_count'] * 2
    assert one['measured_gpu_peak_bytes'] is None and one['capacity_fit_verified'] is False
    assert 'Adam update temporaries' in one['not_included']


@pytest.mark.parametrize('payload', [b'', struct.pack('<Q', 2**32), struct.pack('<Q', 32) + b'{}'])
def test_memory_rejects_truncated_or_unbounded_header(tmp_path, payload):
    path = tmp_path / 'broken.safetensors'
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        estimate_training_memory(path, tiny_config())
