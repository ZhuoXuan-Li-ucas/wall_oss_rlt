"""Official Wall-OSS-0.5 EE-relative target mode (ar5_ee_rel6d_next_v1), CPU only."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from test_inference_artifact import checkpoint, refresh_metadata, write  # noqa: F401 (fixture)
from test_npz_pipeline import write_video
from wall_rlt.data.ar5 import (EE_REL6D_NEXT, JOINT_NEXT, aligned_targets, ee_absolute_window,
                               ee_relative_window, rot6d_to_matrix, rotvec_to_rot6d)
from wall_rlt.data.contract import DataContract
from wall_rlt.data.dataset import EpisodeWindowDataset
from wall_rlt.data.npz_source import NPZFrameSource
from wall_rlt.data.prepare_ar5 import EE_FIELDS, prepare_ar5, relative_windows
from wall_rlt.model import builder
from wall_rlt.model.ar5_adapter import (AR5_ACTION_LAYOUT, AR5_STATE_LAYOUT, EE_VARIANT,
                                       WALL_OSS05_ACTION_LAYOUT, WALL_OSS05_STATE_LAYOUT)

ROOT = Path(__file__).resolve().parents[2]
RIGHT_ARM = {'follow_right_ee_cartesian_pos_relative': 3, 'follow_right_ee_rotation_6D_relative': 6,
             'follow_right_gripper': 1}


def random_rows(rng, count, *, binary=True):
    pos = rng.normal(size=(count, 3)) * .2
    rot = rotvec_to_rot6d(Rotation.random(count, random_state=rng.integers(1 << 31)).as_rotvec())
    grip = rng.integers(0, 2, size=(count, 1)).astype(float) if binary else rng.random((count, 1))
    return np.concatenate((pos, rot, grip), axis=1)


def test_rot6d_is_first_two_rows_and_matches_wall_x_gram_schmidt():
    from wall_x._vendor.x2robot_utils.geometry import so3_to_matrix_batch_nb
    rng = np.random.default_rng(0)
    rotvec = Rotation.random(50, random_state=1).as_rotvec()
    matrix = Rotation.from_rotvec(rotvec).as_matrix()
    six = rotvec_to_rot6d(rotvec)
    np.testing.assert_allclose(six, matrix[:, :2, :].reshape(-1, 6), atol=1e-12)
    np.testing.assert_allclose(rot6d_to_matrix(six), matrix, atol=1e-12)
    noisy = six + rng.normal(size=six.shape) * .05   # network outputs are not orthonormal
    np.testing.assert_allclose(rot6d_to_matrix(noisy), so3_to_matrix_batch_nb(noisy), atol=1e-12)


def test_relative_window_equals_official_loader_and_inverts():
    from wall_x.data.backends.lerobot.loader import PreprocessedDataset
    rng = np.random.default_rng(1)
    official = SimpleNamespace(_dof_config=RIGHT_ARM)
    for _ in range(5):
        targets, state = random_rows(rng, 32), random_rows(rng, 1, binary=False)[0]
        ours = ee_relative_window(targets, state)
        theirs = PreprocessedDataset._to_relative_action(official, targets.astype(np.float32),
                                                         state.astype(np.float32))
        np.testing.assert_allclose(ours, theirs, atol=2e-6)
        back = ee_absolute_window(ours, state)
        np.testing.assert_allclose(back[:, :3], targets[:, :3], atol=1e-12)
        np.testing.assert_allclose(Rotation.from_rotvec(back[:, 3:6]).as_matrix(),
                                   rot6d_to_matrix(targets[:, 3:9]), atol=1e-9)
        np.testing.assert_array_equal(back[:, 6], targets[:, 9])


def test_relative_window_rejects_measured_opening_as_target():
    rng = np.random.default_rng(2)
    targets = random_rows(rng, 3)
    targets[1, 9] = .4
    with pytest.raises(ValueError, match='binary'):
        ee_relative_window(targets, random_rows(rng, 1)[0])


def test_aligned_ee_rows_use_measured_pose_opening_and_next_command():
    t = np.arange(4.)
    state = np.zeros((4, 8)); state[:, 7] = [.8, .7, .5, .3]
    action = np.zeros((4, 7)); action[:, 6] = [1, 1, 0, 0]
    ee = np.zeros((4, 6)); ee[:, 0] = t / 10; ee[:, 5] = t / 5
    targets, obs = aligned_targets(state, action, EE_REL6D_NEXT, ee_pose=ee)
    assert targets.shape == obs.shape == (3, 10) and targets.dtype == np.float32
    np.testing.assert_allclose(obs[:, 0], t[:-1] / 10)
    np.testing.assert_allclose(targets[:, 0], t[1:] / 10)
    np.testing.assert_allclose(obs[:, 9], state[:-1, 7])
    np.testing.assert_array_equal(targets[:, 9], action[1:, 6])
    np.testing.assert_allclose(targets[:, 3:9], rotvec_to_rot6d(ee[1:, 3:]), atol=1e-7)
    with pytest.raises(ValueError, match='requires the recorded ee_pose'):
        aligned_targets(state, action, EE_REL6D_NEXT)
    with pytest.raises(ValueError, match='does not use ee_pose'):
        aligned_targets(state, action, JOINT_NEXT, ee_pose=ee)


@pytest.fixture
def ee_dataset(tmp_path):
    root = tmp_path / 'recordings'; root.mkdir()
    audit = tmp_path / 'audit'; audit.mkdir()
    episodes, arrays = [], {}
    for eid, filename, length in ((17, 'episode_0002', 6), (91, 'episode_0009', 5)):
        t = np.arange(length, dtype=np.float64)
        state = np.zeros((length, 8)); state[:, 2] = .05 * t; state[:, 7] = np.linspace(.86, .2, length)
        action = np.zeros((length, 7)); action[:, 6] = (t < 2) | (t == 5)
        ee = np.zeros((length, 6))
        ee[:, 0], ee[:, 1], ee[:, 2] = .3 + .01 * t, -.1 + .005 * t, .4   # z constant on purpose
        ee[:, 3:] = Rotation.from_euler('xyz', np.stack((.1 * t, -.05 * t, .3 + .02 * t), 1)).as_rotvec()
        npz = root / f'{filename}.npz'
        np.savez(npz, state=state, action=action, ee_pose=ee, task=f'pick up object {eid}',
                 fps=30., camera_keys=['cam_high', 'cam_front'])
        videos = {camera: write_video(root / f'{filename}_{camera}.mp4', length, 20 + i * 30)
                  for i, camera in enumerate(('cam_high', 'cam_front'))}
        episodes.append({'episode_id': eid, 'task_group': 'fixture', 'npz': npz.name,
                         'npz_sha256': hashlib.sha256(npz.read_bytes()).hexdigest(),
                         'frames': length, 'videos': videos})
        arrays[eid] = (state, action, ee)
    write(audit / 'episodes.json', {'episodes': episodes, 'splits': {'train': [17], 'dev': [91], 'test': []},
                                    'excluded_groups': []})
    write(audit / 'audit.json', {'structural_status': 'PASS', 'semantic_status': 'PASS'})
    doc = tmp_path / 'format.md'; doc.write_text('fixture: measured flange pose + binary command')
    path = tmp_path / 'config' / 'ee.json'
    report = prepare_ar5(root, audit, doc, contract_path=path)
    return DataContract.load(path), arrays, report


def test_prepare_defaults_to_ee_contract_with_window_statistics(ee_dataset):
    contract, arrays, report = ee_dataset
    contract.require_trainable()
    assert (contract.target_mode, contract.model_adapter) == (EE_REL6D_NEXT, EE_VARIANT)
    assert contract.action_slots == contract.state_slots == list(range(10, 20))
    assert contract.field_conventions == ['delta'] * 9 + ['absolute']
    state, action, ee = arrays[17]
    targets, obs = aligned_targets(state, action, EE_REL6D_NEXT, ee_pose=ee)
    rows = np.concatenate([ee_relative_window(targets[s:], obs[s]) for s in range(len(targets))])
    assert report['train_action_rows'] == len(rows) == 5 * 6 // 2
    stats = json.loads(Path(contract.normalization_path).read_text())['norm_stats']
    np.testing.assert_allclose(relative_windows(targets, obs), rows)
    for key, values in (('action', rows), ('state', obs)):
        q01, q99 = np.quantile(values, .01, axis=0), np.quantile(values, .99, axis=0)
        np.testing.assert_allclose(stats[key]['q01'], q01)
        np.testing.assert_allclose(np.delete(stats[key]['q99'], 2), np.delete(q99, 2))
        assert stats[key]['q99'][2] == stats[key]['q01'][2] + 1.0   # constant z is not divided by ~0
    assert contract.provenance['degenerate_range_dims'] == {'action': [2], 'state': [2]}
    assert contract.provenance['locked_joint_indices_zero_based'] == [0, 1, 3, 4, 5, 6]


@pytest.mark.parametrize('change', [
    {'model_adapter': 'ar5_joint8_reinit_v1'}, {'action_slots': list(range(10))},
    {'state_slots': list(range(16, 26))}, {'field_conventions': ['delta'] * 10},
    {'alignment': 'same_frame'},
])
def test_ee_contract_requires_the_explicit_right_arm_layout(ee_dataset, change):
    contract, _, _ = ee_dataset
    with pytest.raises(ValueError, match='right-arm EE contract'):
        replace(contract, **change).require_trainable()


def processor_dir(tmp_path):
    from wall_rlt.workers.sft import PROCESSOR_FILES
    path = tmp_path / 'processor'; path.mkdir()
    for name in PROCESSOR_FILES:
        if (ROOT / 'models/wall-oss-0.5' / name).is_file():
            shutil.copyfile(ROOT / 'models/wall-oss-0.5' / name, path / name)
    return path


def test_source_window_and_collator_make_targets_relative_to_the_observation(ee_dataset, tmp_path):
    from wall_rlt.data.collator import WallCollator
    contract, arrays, _ = ee_dataset
    source = NPZFrameSource(contract)
    state, action, ee = arrays[17]
    targets, obs = aligned_targets(state, action, EE_REL6D_NEXT, ee_pose=ee)
    np.testing.assert_array_equal(source.actions[:5], targets)
    np.testing.assert_array_equal(source.states[:5], obs)
    sample = EpisodeWindowDataset(source, [17], horizon=contract.action_horizon)[1]
    valid = int(sample['time_mask'].sum())
    assert valid == 4
    collator = WallCollator(contract, processor_dir(tmp_path), image_size=56, max_length=2048)
    saved = collator.state_dict()
    converted = collator([sample])['wall_inputs']
    manual = dict(sample)
    manual['action'] = sample['action'].clone()
    manual['action'][:valid] = torch.from_numpy(ee_relative_window(targets[1:], obs[1])).float()
    collator.load_state_dict(saved)
    expected = collator([manual], convert_targets=False)['wall_inputs']
    torch.testing.assert_close(converted['action_chunk'], expected['action_chunk'], rtol=0, atol=0)
    mask = converted['dof_mask'][0].bool()
    assert mask[:valid, 10:20].all() and not mask[:, :10].any() and not mask[:, 20:].any()
    assert not mask[valid:].any()
    observed = collator.observation_batch([source.observation(1)])['wall_inputs']
    assert 'action_chunk' not in observed


@pytest.fixture
def ee_build_cfg(tmp_path):
    base = tmp_path / 'base'; base.mkdir()
    (base / 'config.json').write_text(json.dumps({'hidden_size': 8}))
    (base / 'model.safetensors').write_bytes(b'fixture; weight loader is replaced')
    (base / 'preprocessor_config.json').write_text('{}')
    (base / 'config.yml').write_text('model_type: qwen2_5\n')
    stats = {'train_episode_ids': [1], 'target_mode': EE_REL6D_NEXT,
             'norm_stats': {k: {'q01': [-.1] * 9 + [0.], 'q99': [.1] * 9 + [1.]} for k in ('state', 'action')}}
    norm = tmp_path / 'norm.json'; norm.write_text(json.dumps(stats))
    contract = DataContract(root=str(tmp_path), repo_id='fixture/ee', fps=30,
        camera_mapping={'cam_high': 'top', 'cam_front': 'wrist'}, splits={'train': [1], 'dev': [2], 'test': []},
        state_key='state', semantics_confirmed=True, target_mode=EE_REL6D_NEXT,
        alignment='observation_t_target_t_plus_1_drop_last',
        normalization_path=str(norm), normalization_sha256=hashlib.sha256(norm.read_bytes()).hexdigest(),
        **EE_FIELDS)
    cp = tmp_path / 'contract.json'; cp.write_text(json.dumps(asdict(contract)))
    return {'model_type': 'wall_rlt', 'precision': 'fp32', 'is_lora': False, 'initialization': 'new_transfer',
            'rlt': {'input_dim': 8, 'embed_dim': 8, 'num_heads': 2, 'num_layers': 1, 'prefix_seq_len': 8,
                    'action_token_id': 3, 'checkpoint_path': str(base), 'data_contract_path': str(cp),
                    'wall_train_config': str(base / 'config.yml')}}


def test_ee_builder_keeps_pretrained_projections_and_official_layout(ee_build_stack, ee_build_cfg):
    calls, layouts = ee_build_stack
    report = builder.validate_build_config(ee_build_cfg)
    assert report['model_adapter'] == EE_VARIANT
    policy = builder.build_wall_rlt(ee_build_cfg)
    assert len(calls) == 1 and layouts == [EE_VARIANT]
    proc = policy.wall.action_preprocessor
    assert torch.all(proc.w1.weight == .5) and torch.all(proc.action_proj_back.weight == .5)
    init = policy.build_report['adapter_initialization']
    assert init['reset_parameters'] == [] and init['initialization'] == 'pretrained_action_projections_kept'
    norm = proc.normalizer_action
    assert torch.equal(norm.min['fixture/ee'][10:20], torch.tensor([-.1] * 9 + [0.]))
    assert torch.equal(norm.delta['fixture/ee'][:10], torch.ones(10))


@pytest.fixture
def ee_build_stack(monkeypatch):
    from test_builder import TinyWall
    from wall_x._vendor.harrix.utils import train_config
    from wall_x.trainer import trainer_utils

    class EETinyWall(TinyWall):
        def __init__(self, config, processor=None, tokenizer_mixin=None):
            super().__init__(config, processor, tokenizer_mixin)
            self.action_preprocessor.dof_config = dict(WALL_OSS05_ACTION_LAYOUT)
            self.action_preprocessor.agent_pos_config = dict(WALL_OSS05_STATE_LAYOUT)

    class Tokenizer:
        def __len__(self): return 16
        def convert_tokens_to_ids(self, token): return 3
    layouts, calls = [], []
    config = SimpleNamespace(dof_config=dict(WALL_OSS05_ACTION_LAYOUT),
        agent_pos_config=dict(WALL_OSS05_STATE_LAYOUT), use_state_string_representation=True,
        proj_with_mask=True, action_hidden_size=4, initializer_range=.02, tie_word_embeddings=True,
        vocab_size=16, vision_config=SimpleNamespace())
    monkeypatch.setattr(builder, '_wall_components', lambda: (object, EETinyWall))
    monkeypatch.setattr(builder, '_runtime_train_config', lambda rlt, proc, variant: layouts.append(variant) or {})
    monkeypatch.setattr(builder, '_checkpoint_vocab', lambda _: 16)
    monkeypatch.setattr(train_config, 'build_model_config', lambda *_: config)
    monkeypatch.setattr(trainer_utils, 'load_wallx_processors',
                        lambda *a, **kw: {'processor': SimpleNamespace(tokenizer=Tokenizer())})
    monkeypatch.setattr(builder, 'reset_ar5_action_projections',
                        lambda *a, **kw: pytest.fail('EE variant must not reset projections'))

    def load(wall, path, cls):
        calls.append(path)
        return builder.load_base_state_strict(wall, {k: torch.full_like(v, .5) for k, v in wall.state_dict().items()})
    monkeypatch.setattr(builder, '_load_base_weights', load)
    return calls, layouts


def test_ee_layout_constants_equal_the_released_checkpoint_config():
    import yaml
    released = yaml.safe_load((ROOT / 'models/wall-oss-0.5/config.yml').read_text())
    assert list(released['dof_config'].items()) == list(WALL_OSS05_ACTION_LAYOUT.items())
    assert list(released['ar_dof_config'].items()) == list(WALL_OSS05_ACTION_LAYOUT.items())
    assert list(released['agent_pos_config'].items()) == list(WALL_OSS05_STATE_LAYOUT.items())
    rlt = SimpleNamespace(wall_train_config=str(ROOT / 'models/wall-oss-0.5/config.yml'),
                          checkpoint_path=str(ROOT / 'models/wall-oss-0.5'), action_horizon=32)
    train = builder._runtime_train_config(rlt, str(ROOT / 'models/wall-oss-0.5'), EE_VARIANT)
    assert list(train['dof_config'].items()) == list(WALL_OSS05_ACTION_LAYOUT.items())
    assert list(train['agent_pos_config'].items()) == list(WALL_OSS05_STATE_LAYOUT.items())
    assert train['use_state_string_representation'] is True
    joint = builder._runtime_train_config(rlt, str(ROOT / 'models/wall-oss-0.5'))
    assert list(joint['dof_config'].items()) == list(AR5_ACTION_LAYOUT.items())


def to_ee_checkpoint(root):
    """Turn the synthetic joint8 checkpoint fixture into the EE variant."""
    from wall_rlt.inference.artifact import _normalizer_values
    assets = root / 'wall_rlt_assets'
    value = json.loads((assets / 'data_contract.json').read_text())
    stats = {'action': {'q01': [-.05] * 3 + [-1.] * 6 + [0.], 'q99': [.05] * 3 + [1.] * 6 + [1.]},
             'state': {'q01': [.2, -.3, .1] + [-1.] * 6 + [0.], 'q99': [.6, .3, .5] + [1.] * 6 + [.86]}}
    write(assets / 'normalization.json', {'schema_version': 1, 'train_episode_ids': value['splits']['train'],
                                          'manifest_sha256': value['manifest_sha256'],
                                          'target_mode': EE_REL6D_NEXT, 'norm_stats': stats})
    value.update(EE_FIELDS, target_mode=EE_REL6D_NEXT, repo_id='fixture/ee',
                 normalization_sha256=hashlib.sha256((assets / 'normalization.json').read_bytes()).hexdigest())
    write(assets / 'data_contract.json', value)
    contract = DataContract.load(assets / 'data_contract.json')
    cfg = json.loads((assets / 'adapted_wall_config.json').read_text())
    cfg.update(dof_config=dict(WALL_OSS05_ACTION_LAYOUT), agent_pos_config=dict(WALL_OSS05_STATE_LAYOUT))
    write(assets / 'adapted_wall_config.json', cfg)
    path = root / 'model_state_dict/full_weights.pt'
    state = {k: v for k, v in torch.load(path, weights_only=True).items() if '.normalizer_' not in k}
    state.update({'wall.' + k: v for k, v in _normalizer_values(contract).items()})
    torch.save(state, path)
    refresh_metadata(root)
    return contract


def test_ee_artifact_exports_and_checks_its_own_layout(checkpoint, tmp_path):
    from wall_rlt.inference.artifact import export_inference_artifact, verify_inference_artifact
    contract = to_ee_checkpoint(checkpoint)
    manifest = export_inference_artifact(checkpoint, tmp_path / 'ee')
    assert verify_inference_artifact(tmp_path / 'ee') == manifest
    assert manifest['data_contract_hash'] == contract.fingerprint()
    cfgpath = checkpoint / 'wall_rlt_assets/adapted_wall_config.json'
    cfg = json.loads(cfgpath.read_text())
    cfg.update(dof_config=dict(AR5_ACTION_LAYOUT), agent_pos_config=dict(AR5_STATE_LAYOUT))
    write(cfgpath, cfg)
    refresh_metadata(checkpoint)
    with pytest.raises(ValueError, match='Wall-OSS-0.5 EE layout'):
        export_inference_artifact(checkpoint, tmp_path / 'wrong')


@pytest.fixture
def ee_runtime(checkpoint, tmp_path):
    from test_inference_runtime import make_runtime
    to_ee_checkpoint(checkpoint)
    return make_runtime(checkpoint, tmp_path, WALL_OSS05_ACTION_LAYOUT, WALL_OSS05_STATE_LAYOUT)


def ee_observation():
    from PIL import Image
    pose = [.4, -.1, .3] + rotvec_to_rot6d(np.array([[.1, -.2, .3]]))[0].tolist() + [.8]
    return {'instruction': 'pick up the object', 'state': pose,
            'images': {'cam_high': Image.new('RGB', (56, 56), (20, 30, 40)),
                       'cam_front': Image.new('RGB', (56, 56), (100, 120, 140))}}


def test_ee_runtime_returns_relative_chunk_and_absolute_flange_targets(ee_runtime):
    result = ee_runtime.infer(ee_observation(), seed=5)
    relative = np.asarray(result['actions'])
    absolute = np.asarray(result['absolute_targets'])
    assert relative.shape == (32, 10) and absolute.shape == (32, 7)
    np.testing.assert_allclose(absolute, ee_absolute_window(relative, np.asarray(ee_observation()['state'])),
                               atol=1e-6)
    meta = ee_runtime.metadata()
    assert meta['target_mode'] == EE_REL6D_NEXT and meta['state_dim'] == 10 and meta['action_dim'] == 10
    assert meta['absolute_target_dim'] == 7
    with pytest.raises(ValueError, match='state dimensions'):
        ee_runtime.infer({**ee_observation(), 'state': [0.] * 8})


def test_ee_runtime_through_real_http_protocol(ee_runtime):
    import threading
    from wall_rlt.inference.client import InferenceClient
    from wall_rlt.inference.server import create_server
    server = create_server(ee_runtime, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    thread.start()
    try:
        client = InferenceClient(f'http://127.0.0.1:{server.server_port}', timeout=30)
        assert client.health()['metadata']['state_dim'] == 10
        result = client.infer(ee_observation(), seed=4)
        assert len(result['actions'][0]) == 10 and len(result['absolute_targets'][0]) == 7
        assert result['robot_executed'] is False
        from wall_rlt.inference.client import RemoteInferenceError
        with pytest.raises(RemoteInferenceError):
            client.infer({**ee_observation(), 'state': [0.] * 8})
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_protocol_layouts_follow_the_backend_contract():
    from wall_rlt.inference.protocol import (InferenceRequest, ProtocolError, decode_request,
                                             make_request, make_response, validate_metadata,
                                             validate_response)
    request = make_request(ee_observation(), request_id='ee')
    assert len(decode_request(request, state_dim=10).observation['state']) == 10
    with pytest.raises(ProtocolError, match='length 8'):
        decode_request(request, state_dim=8)
    with pytest.raises(ProtocolError, match='length'):
        make_request({**ee_observation(), 'state': [0.] * 9}, request_id='bad')
    metadata = validate_metadata({'backend': 'b', 'data_contract_hash': 'a' * 64, 'action_horizon': 32,
                                  'action_dim': 10, 'action_units': ['m'] * 10, 'action_convention': 'mixed',
                                  'model_origin': 'synthetic_fixture', 'state_dim': 10,
                                  'absolute_target_dim': 7})
    result = {'actions': [[0.] * 10] * 32, 'absolute_targets': [[0.] * 7] * 32}
    response = make_response(result, InferenceRequest('ee', {}), metadata)
    assert validate_response(response, request_id='ee') == response
    with pytest.raises(ProtocolError, match='absolute EE targets'):
        make_response({'actions': result['actions']}, InferenceRequest('ee', {}), metadata)
    with pytest.raises(ProtocolError, match='length 10'):
        make_response({**result, 'actions': [[0.] * 8] * 32}, InferenceRequest('ee', {}), metadata)
    with pytest.raises(ProtocolError, match='every physical action unit'):
        validate_metadata({**metadata, 'action_units': ['m'] * 8})
