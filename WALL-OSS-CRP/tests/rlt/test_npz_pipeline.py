"""Exercise NPZ/video ingestion and data lineage using tiny local recordings."""
from dataclasses import replace
import hashlib
import json

import av
import numpy as np
import pytest
import torch

from wall_rlt.data.contract import DataContract
from wall_rlt.data.dataset import EpisodeWindowDataset
from wall_rlt.data.npz_source import NPZFrameSource


def write_json(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_video(path, count, color_offset):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=30)
        stream.width, stream.height = 32, 24
        stream.pix_fmt = 'yuv420p'
        stream.codec_context.gop_size = 1
        for i in range(count):
            rgb = np.full((24, 32, 3), color_offset + 30*i, dtype=np.uint8)
            rgb[:, :, 1] += 10
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {'path':str(path.name), 'bytes':path.stat().st_size,
                'frames':stream.frames, 'fps':float(stream.average_rate)}


@pytest.fixture
def recording(tmp_path):
    root = tmp_path/'recordings'
    root.mkdir()
    episodes, arrays = [], {}
    # Global episode IDs and local filename indices are intentionally unrelated
    # and non-contiguous; readers must use the frozen manifest paths.
    for eid, filename, length in ((17, 'episode_0002', 4), (91, 'episode_0009', 5)):
        state = np.arange(length*8, dtype=np.float32).reshape(length,8)/100 + eid/100
        state[:, 2] = .1218
        state[:, 4] = -1.6268
        state[:, 7] = np.linspace(1, .386, length)
        action = np.zeros((length,7), dtype=np.float32)
        action[:,6] = 1
        action[2:,6] = 0
        action[1:,:6] = np.arange((length-1)*6).reshape(length-1,6)/1000
        npz = root/f'{filename}.npz'
        np.savez(npz, state=state, action=action, task=f'pick up object {eid}',
                 fps=30., camera_keys=['cam_high','cam_front'])
        videos = {}
        for index, camera in enumerate(('cam_high','cam_front')):
            vp = root/f'{filename}_{camera}.mp4'
            videos[camera] = write_video(vp,length,20+index*30)
        episodes.append({'episode_id':eid,'task_group':'fixture','npz':npz.name,
                         'npz_sha256':hashlib.sha256(npz.read_bytes()).hexdigest(),
                         'frames':length,'videos':videos})
        arrays[eid] = (state,action)
    splits = {'train':[17], 'dev':[91], 'test':[]}
    manifest_path = tmp_path/'episodes.json'
    manifest = {'episodes':episodes,'splits':splits, 'excluded_groups':['mouse','tape']}
    manifest_sha = write_json(manifest_path, manifest)
    train_state, train_action = arrays[17]
    train_targets = np.concatenate((train_state[1:,:7],train_action[1:,6:7]),axis=1)
    norm = {'target_mode':'ar5_joint_next_v1', 'train_episode_ids':[17],
            'manifest_sha256':manifest_sha, 'norm_stats':{}}
    for key, values in (('action',train_targets), ('state',train_state[:-1])):
        norm['norm_stats'][key] = {'q01':np.quantile(values,.01,axis=0).tolist(),
                                  'q99':np.quantile(values,.99,axis=0).tolist()}
    norm_path = tmp_path/'normalization.json'
    norm_sha = write_json(norm_path,norm)
    contract = DataContract(
        root=str(root),repo_id='fixture/ar5',fps=30,
        action_names=[f'j{i}_target' for i in range(1,8)]+['gripper_command'],
        state_names=[f'j{i}' for i in range(1,8)]+['gripper_opening'],
        camera_mapping={'cam_high':'overhead','cam_front':'wrist'},splits=splits,
        state_key='state', action_horizon=3, action_slots=list(range(8)),
        state_slots=list(range(8)),action_units=['rad']*7+['binary'],
        state_units=['rad']*7+['ratio'],action_convention='absolute',
        coordinate_frame='AR5 joint coordinates',robot_type='AR5+OmniPicker',
        semantics_confirmed=True,purpose='synthetic',
        normalization_path=str(norm_path),normalization_sha256=norm_sha,
        source_format='npz_video',manifest_path=str(manifest_path),manifest_sha256=manifest_sha,
        target_mode='ar5_joint_next_v1',model_adapter='ar5_joint8_reinit_v1',
        alignment='observation_t_target_t_plus_1_drop_last')
    return contract, arrays, manifest, norm


def test_source_next_frame_targets_keep_current_observation(recording):
    contract, arrays, _, _ = recording
    source = NPZFrameSource(contract)
    assert source.episode_indices.tolist() == [17]*3+[91]*4
    assert source.frame_indices.tolist() == [0,1,2,0,1,2,3]
    for offset, eid in ((0,17),(3,91)):
        state, action = arrays[eid]
        n = len(state)-1
        np.testing.assert_array_equal(source.states[offset:offset+n],state[:-1])
        np.testing.assert_array_equal(source.actions[offset:offset+n,:7],state[1:,:7])
        np.testing.assert_array_equal(source.actions[offset:offset+n,7],action[1:,6])
        observation = source.observation(offset)
        torch.testing.assert_close(observation['state'],torch.from_numpy(state[0]))
        assert observation['instruction'] == f'pick up object {eid}'


def test_real_video_random_seek_matches_exact_sequential_rgb_frame(recording):
    contract, _, manifest, _ = recording
    source = NPZFrameSource(contract,cache_bytes=0)
    for index, eid, frame_index in ((5,91,2),(0,17,0),(6,91,3),(2,17,2)):
        entry = next(e for e in manifest['episodes'] if e['episode_id']==eid)
        observation = source.observation(index)
        for camera in contract.camera_mapping:
            with av.open(str(source.contract.root+'/'+entry['videos'][camera]['path'])) as container:
                expected = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)][frame_index]
            actual = observation['images'][camera].permute(1,2,0).numpy()
            np.testing.assert_array_equal(actual,expected)


def test_windows_never_borrow_actions_from_the_following_episode(recording):
    contract, arrays, _, _ = recording
    source = NPZFrameSource(contract)
    dataset = EpisodeWindowDataset(source,[17,91],horizon=3)
    assert len(dataset)==7
    tail = dataset[2]
    assert tail['episode_id']==17 and tail['frame_index']==2
    assert tail['time_mask'].tolist()==[True,False,False]
    torch.testing.assert_close(tail['action'][0,:7],torch.from_numpy(arrays[17][0][-1,:7]))
    assert not tail['action'][1:].count_nonzero()
    head = dataset[3]
    assert head['episode_id']==91 and head['frame_index']==0
    assert head['time_mask'].all()
    full = EpisodeWindowDataset(source,[17,91],horizon=3,complete_windows_only=True)
    assert [(full[i]['episode_id'],full[i]['frame_index']) for i in range(len(full))]==[(17,0),(91,0),(91,1)]


def test_requested_episode_subset_does_not_read_other_recordings(recording):
    contract, _, manifest, _ = recording
    from pathlib import Path
    (Path(contract.root)/manifest['episodes'][1]['npz']).write_bytes(b'invalid excluded source')
    assert len(NPZFrameSource(contract,episode_ids=[17]).actions)==3
    with pytest.raises(ValueError,match='recording changed after audit'):
        NPZFrameSource(contract)


def test_reader_cannot_load_an_episode_outside_frozen_splits(recording):
    from pathlib import Path
    contract, _, manifest, _ = recording
    manifest['splits']['dev'] = []
    sha = write_json(Path(contract.manifest_path),manifest)
    contract = replace(contract,splits=manifest['splits'],manifest_sha256=sha)
    assert len(NPZFrameSource(contract).actions) == 3
    with pytest.raises(ValueError,match='frozen dataset splits'):
        NPZFrameSource(contract,episode_ids=[91])


def test_prepare_stats_use_only_next_targets_from_train(recording,tmp_path):
    from pathlib import Path
    from wall_rlt.data.prepare_ar5 import prepare_ar5
    contract, arrays, _, _ = recording
    write_json(tmp_path/'audit.json',{'structural_status':'PASS','semantic_status':'PASS'})
    doc = tmp_path/'format.md'
    doc.write_text('Fixture semantics: measured next joints plus next gripper command.')
    destination = tmp_path/'config'/'data.json'
    report = prepare_ar5(contract.root,tmp_path,doc,contract_path=destination,
                         target_mode='ar5_joint_next_v1')
    assert report['train_observations'] == 3
    prepared = DataContract.load(destination)
    prepared.require_trainable()
    stats = json.loads(Path(prepared.normalization_path).read_text())['norm_stats']
    state, action = arrays[17]
    targets = np.concatenate([state[1:,:7],action[1:,6:7]],axis=1)
    np.testing.assert_allclose(stats['action']['q99'],np.quantile(targets,.99,axis=0))
    np.testing.assert_allclose(stats['state']['q01'],np.quantile(state[:-1],.01,axis=0))
    assert stats['state']['count'] == 3
    assert stats['action']['q01'][2] == stats['action']['q99'][2]


def test_train_contract_requires_synthetic_opt_in(recording):
    contract, _, _, _ = recording
    contract.require_trainable(allow_synthetic=True)
    with pytest.raises(ValueError,match='not a target training dataset'):
        contract.require_trainable()


@pytest.mark.parametrize('change,error', [
    ({'train_episode_ids':[17,91]}, 'exact train split'),
    ({'target_mode':'recorded'}, 'target representation'),
    ({'manifest_sha256':'0'*64}, 'source manifest'),
])
def test_normalization_rejects_wrong_training_lineage_even_with_valid_digest(recording,change,error):
    from pathlib import Path
    contract, _, _, norm = recording
    norm.update(change)
    sha = write_json(Path(contract.normalization_path),norm)
    contract = replace(contract,normalization_sha256=sha)
    with pytest.raises(ValueError,match=error):
        contract.require_trainable(allow_synthetic=True)


def test_normalization_content_tampering_is_detected(recording):
    from pathlib import Path
    contract, _, _, norm = recording
    norm['norm_stats']['action']['q01'][0] += .1
    write_json(Path(contract.normalization_path),norm)
    with pytest.raises(ValueError,match='normalization SHA256'):
        contract.require_trainable(allow_synthetic=True)


@pytest.mark.parametrize('change,error', [
    ({'splits':{'train':[91],'dev':[17],'test':[]}}, 'split differs'),
    ({'model_adapter':'unconfirmed'}, 'joint8 adaptation'),
    ({'alignment':'same_frame'}, 'joint8 adaptation'),
])
def test_contract_rejects_silent_split_or_representation_changes(recording,change,error):
    contract, _, _, _ = recording
    with pytest.raises(ValueError,match=error):
        replace(contract,**change).require_trainable(allow_synthetic=True)


@pytest.mark.parametrize('fault', ['excluded_group','semantic_errors','structural_errors'])
def test_manifest_cannot_admit_excluded_or_failed_episode(recording,fault):
    from pathlib import Path
    contract, _, manifest, _ = recording
    if fault=='excluded_group':
        manifest['episodes'][0]['task_group']='mouse'
    else:
        manifest['episodes'][0][fault]=['failed audit']
    sha = write_json(Path(contract.manifest_path),manifest)
    with pytest.raises(ValueError,match='excluded or failed episode'):
        replace(contract,manifest_sha256=sha).read_manifest()
