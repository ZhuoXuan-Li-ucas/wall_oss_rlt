"""Replay evidence checks with local sockets and explicit test-only model/data fixtures."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest
import torch

from wall_rlt.data.contract import DataContract
from wall_rlt.inference.server import create_server

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('wall_rlt_inference_replay_check',
                                             ROOT / 'scripts/check_wall_rlt_inference.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.fixture
def replay_data(tmp_path, monkeypatch):
    for key in ('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TOKENIZERS_PARALLELISM'):
        monkeypatch.setenv(key, replay.os.environ.get(key, ''))
    stats = {'train_episode_ids': [7], 'norm_stats': {
        'action': {'q01': [-1.] * 8, 'q99': [1.] * 8},
        'observation.state': {'q01': [-1.] * 8, 'q99': [1.] * 8}}}
    normalization = tmp_path / 'normalization.json'
    normalization.write_text(json.dumps(stats))
    contract = DataContract(root=str(tmp_path), repo_id='test-only/replay-fixture', fps=30,
        action_names=[f'action_{i}' for i in range(8)],
        state_names=[f'state_{i}' for i in range(8)],
        camera_mapping={'cam_high': 'top_view', 'cam_front': 'wrist_view'}, splits={'train': [7]},
        action_slots=list(range(8)), state_slots=list(range(8)),
        action_units=['rad'] * 7 + ['binary_0_close_1_open'], state_units=['rad'] * 7 + ['m'],
        action_convention='absolute', coordinate_frame='fixture_joint',
        alignment='observation_t_target_t_plus_1_drop_last', semantics_confirmed=True,
        normalization_path=str(normalization),
        normalization_sha256=hashlib.sha256(normalization.read_bytes()).hexdigest())
    contract_path = tmp_path / 'contract.json'
    contract_path.write_text(json.dumps(asdict(contract)))

    class TestOnlySource:
        # Deliberately contains another episode before the requested episode.
        # LeRobotFrameSource may contain all split episodes even when episode_ids is passed.
        episode_indices = np.asarray([3, 7, 7])
        frame_indices = np.asarray([0, 0, 1])
        seen = []
        def observation(self, index):
            self.seen.append(index)
            return {'instruction': 'synthetic replay transport test',
                    'state': torch.ones(8) * index,
                    'images': {name: torch.full((3, 28, 28), index, dtype=torch.uint8)
                               for name in contract.camera_mapping}}
    source = TestOnlySource()
    def build_source(supplied_contract, *, episode_ids=None):
        assert supplied_contract.fingerprint() == contract.fingerprint() and episode_ids == [7]
        return source
    monkeypatch.setattr('wall_rlt.data.npz_source.build_source', build_source)
    return contract, contract_path, source


class ReplayMetadataFixtureBackend:
    """Test-only acceptance-schema metadata fixture, not a trained checkpoint.

    Its metadata deliberately exercises the production replay parser. It is
    created only here and never exposed as a production backend or fallback.
    """
    def __init__(self, contract):
        self.contract = contract
        self.calls = 0
        self.settings = {}

    def metadata(self):
        contract = self.contract
        fields = {key: getattr(contract, key) for key in (
            'action_horizon', 'action_names', 'action_units', 'action_convention',
            'coordinate_frame', 'camera_mapping', 'state_names', 'state_units')}
        return {**fields, 'backend': 'wall_rlt_stage1', 'model_origin': 'stage1_checkpoint',
            'artifact_id': 'a' * 64, 'runtime_model_sources_sha256': 'b' * 64,
            'data_contract_hash': contract.fingerprint(), 'training_step': 1,
            'action_dim': 8, 'feature_dim': 2, 'device': 'cpu', 'inference_steps': 2,
            'image_size': 56, 'max_length': 2048, 'vision_attention': 'eager',
            'wall_attention': 'sdpa', 'includes_actor': False, 'includes_decoder': False,
            'robot_execution_supported': False, 'action_source': 'WALL_reference_prediction',
            'gripper_output': 'continuous_prediction_of_binary_target',
            'completed_requests': self.calls, 'last_inference_seconds': .01 if self.calls else None,
            'peak_allocated_bytes': self.calls * 1024, 'peak_reserved_bytes': self.calls * 2048,
            **self.settings}

    def infer(self, observation, *, return_features=False, seed=None):
        assert return_features and seed == 7
        self.calls += 1
        return {'actions': [[float(seed)] * 8 for _ in range(32)],
                'z_rl': [float(observation['state'][0]), float(seed)]}


@contextmanager
def fixture_server(contract):
    backend = ReplayMetadataFixtureBackend(contract)
    server = create_server(backend, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    thread.start()
    try:
        yield backend, f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)
        assert not thread.is_alive()


def run_replay(contract_path, endpoint, output, *extra):
    status = replay.main(['--contract', str(contract_path), '--endpoint', endpoint,
                          '--episode-id', '7', '--frame-index', '0', '--seed', '7',
                          '--output', str(output), *extra])
    return status, json.loads(output.read_text())


def test_real_socket_cli_chooses_episode_frame_and_records_after_metadata(replay_data, tmp_path):
    contract, path, source = replay_data
    with fixture_server(contract) as (_, endpoint):
        code, report = run_replay(path, endpoint, tmp_path / 'report.json')
    assert code == 0 and report['status'] == 'PASS' and report['schema_version'] == 2
    assert source.seen == [1]
    assert report['repeat_checked'] and report['restart_checked'] is False
    assert report['requests'][0]['response']['z_rl'] == [1., 7.]
    assert report['metadata']['completed_requests'] == 0
    assert report['metadata_after']['completed_requests'] == 2
    assert report['metadata_after']['peak_allocated_bytes'] == 2048
    assert report['metadata_after']['peak_reserved_bytes'] == 4096
    assert report['metadata_after']['last_inference_seconds'] == .01
    assert len(report['observation_sha256']) == 64
    assert report['runtime_settings']['runtime_model_sources_sha256'] == 'b' * 64


def test_cli_requires_distinct_service_for_restart_comparison(replay_data, tmp_path):
    contract, path, _ = replay_data
    first_path = tmp_path / 'first.json'
    with fixture_server(contract) as (_, endpoint):
        first_code, first = run_replay(path, endpoint, first_path)
        same_code, same = run_replay(path, endpoint, tmp_path / 'same.json', '--compare-report', str(first_path))
    with fixture_server(contract) as (_, endpoint):
        second_code, second = run_replay(path, endpoint, tmp_path / 'second.json', '--compare-report', str(first_path))
    assert first_code == second_code == 0
    assert same_code == 1 and 'different server_instance_id' in same['error']
    assert same['metadata_after']['completed_requests'] == 4
    assert first['server_instance_id'] != second['server_instance_id']
    assert second['restart_checked'] is True
    assert second['restart_difference'] == {'action_max_abs': 0., 'feature_max_abs': 0.}


@pytest.mark.parametrize('field', ['inference_steps', 'image_size', 'max_length',
                                  'vision_attention', 'runtime_model_sources_sha256'])
def test_restart_rejects_changed_runtime_settings(replay_data, tmp_path, field):
    contract, path, _ = replay_data
    with fixture_server(contract) as (_, endpoint):
        _, current = run_replay(path, endpoint, tmp_path / 'current.json')
    previous = deepcopy(current)
    previous['server_instance_id'] = 'f' * 32
    previous['runtime_settings'][field] = 'changed'
    with pytest.raises(ValueError, match='matching runtime_settings'):
        replay.compare_restart(previous, current)


@pytest.mark.parametrize('field', ['artifact_id', 'contract_hash', 'episode_id', 'frame_index',
                                  'seed', 'observation_sha256', 'atol'])
def test_restart_rejects_changed_evidence_binding(replay_data, tmp_path, field):
    contract, path, _ = replay_data
    with fixture_server(contract) as (_, endpoint):
        _, current = run_replay(path, endpoint, tmp_path / 'current.json')
    previous = deepcopy(current)
    previous['server_instance_id'] = 'f' * 32
    previous[field] = 'changed'
    with pytest.raises(ValueError, match=f'matching {field}'):
        replay.compare_restart(previous, current)


def test_restart_rejects_invalid_old_prediction(replay_data, tmp_path):
    contract, path, _ = replay_data
    with fixture_server(contract) as (_, endpoint):
        _, current = run_replay(path, endpoint, tmp_path / 'current.json')
    for bad_value in (float('nan'), float('inf')):
        previous = deepcopy(current)
        previous['server_instance_id'] = 'f' * 32
        previous['requests'][0]['response']['actions'][0][0] = bad_value
        with pytest.raises(ValueError, match='finite'):
            replay.compare_restart(previous, current)
    previous = deepcopy(current)
    previous['server_instance_id'] = 'f' * 32
    previous['requests'][0]['response']['data_contract_hash'] = 'c' * 64
    with pytest.raises(ValueError, match='provenance'):
        replay.compare_restart(previous, current)


def test_observation_fingerprint_binds_pixels_geometry_state_and_instruction():
    observation = {'instruction': 'fixture', 'state': [0.] * 8,
        'images': {'cam_high': Image.new('RGB', (28, 28)), 'cam_front': Image.new('RGB', (28, 28))}}
    first, _ = replay.bind_observation(observation)
    changed = deepcopy(observation); changed['state'][0] = 1.
    assert replay.bind_observation(changed)[0] != first
    changed = deepcopy(observation); changed['instruction'] = 'changed'
    assert replay.bind_observation(changed)[0] != first
    changed = deepcopy(observation); changed['images']['cam_high'].putpixel((0, 0), (1, 0, 0))
    assert replay.bind_observation(changed)[0] != first
    changed = deepcopy(observation); changed['images']['cam_high'] = Image.new('RGB', (14, 56))
    assert replay.bind_observation(changed)[0] != first


@pytest.mark.parametrize('marker,value', [
    ('BUNDLE_READY.json', {}), ('TRAINING_CANDIDATE.json', {}), ('wall_rlt_data.json', {}),
    ('manifest.json', {'kind': 'wall-rlt-stage1-inference'}),
    ('MANIFEST.json', {'status': 'SOURCE_PACKAGE_NOT_GPU_VALIDATED'}),
])
def test_reports_cannot_modify_immutable_artifact_inventory(tmp_path, marker, value):
    (tmp_path / marker).write_text(json.dumps(value))
    with pytest.raises(ValueError, match='outside'):
        replay._validate_output_location(tmp_path / 'reports' / 'replay.json')


def test_missing_episode_frame_fails_before_prediction(replay_data, tmp_path):
    contract, path, source = replay_data
    source.frame_indices = np.asarray([0, 1, 2])
    with fixture_server(contract) as (backend, endpoint):
        code, report = run_replay(path, endpoint, tmp_path / 'missing.json')
        assert backend.calls == 0
    assert code == 1 and 'exactly one usable observation' in report['error']
    assert not source.seen
