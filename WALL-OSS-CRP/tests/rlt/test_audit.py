"""Audit actual NPZ/video fixtures and keep excluded recordings unread."""
import json
from pathlib import Path

import av
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wall_rlt.data.audit import audit_npz


def make_episode(root, task, filename, *, offset=0, mutation=None):
    folder = root/task
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'dataset.json').write_text(json.dumps({'prompt':f'pick up the {task}',
                                                 'grip_force':64,'speed':128}))
    n = 5
    state = np.zeros((n,8), dtype=np.float32)
    state[:,0] = offset + np.arange(n)*.01
    state[:,2], state[:,4], state[:,7] = .1218, -1.6268, .8
    ee = np.zeros((n,6), dtype=np.float32)
    ee[:,0] = .3 + offset*.01 + np.arange(n)*.001
    ee[:,3:] = Rotation.from_euler('xyz',np.column_stack([
        .1+np.arange(n)*.001, np.full(n,.2), np.full(n,.3)])).as_rotvec()
    action = np.zeros((n,7), dtype=np.float32)
    action[:,6] = [1,1,1,0,0]
    action[1:,:3] = np.diff(ee[:,:3],axis=0)
    action[1:,3:6] = (Rotation.from_rotvec(ee[1:,3:]) *
                     Rotation.from_rotvec(ee[:-1,3:]).inv()).as_rotvec()
    timestamps = np.arange(n,dtype=np.float64)/30
    record = dict(state=state,action=action,ee_pose=ee,
                  timestamp=timestamps,t_cam_front=timestamps.copy(),t_cam_high=timestamps+.003,
                  fps=np.array(30.),held=np.zeros(n,dtype=bool),
                  camera_keys=np.array(['cam_high','cam_front']),task=np.array(f'pick up the {task}'))
    if mutation:
        mutation(record)
    np.savez(folder/f'{filename}.npz',**record)
    for camera in ('cam_high','cam_front'):
        with av.open(str(folder/f'{filename}_{camera}.mp4'),'w') as container:
            stream = container.add_stream('mpeg4',rate=30)
            stream.width,stream.height,stream.pix_fmt = 640,480,'yuv420p'
            for frame_id in range(n):
                pixels = np.full((480,640,3),frame_id*15,dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels,format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


def test_exclusions_precede_loading_ids_splits_and_statistics(tmp_path):
    root, output = tmp_path/'recordings', tmp_path/'report'
    for i, name in enumerate(('episode_0002','episode_0009','episode_0012','episode_0030')):
        make_episode(root,'banana',name,offset=i)
    for name in ('mouse','tape'):
        folder = root/name
        folder.mkdir()
        (folder/'episode_0000.npz').write_bytes(b'must never be parsed')
    derived = root/'banana'/'h100'
    derived.mkdir()
    (derived/'episode_0000.npz').write_bytes(b'not an original recording')
    report = audit_npz(root,output,exclude_groups=['tape','mouse','tape'])
    assert report['structural_status'] == report['semantic_status'] == 'PASS'
    assert report['excluded_groups'] == ['mouse','tape']
    assert report['groups'] == {'banana':4}
    manifest = json.loads((output/'episodes.json').read_text())
    assert manifest['excluded_groups'] == ['mouse','tape']
    entries = manifest['episodes']
    assert [e['episode_id'] for e in entries] == [0,1,2,3]
    assert [Path(e['npz']).name for e in entries] == [
        'episode_0002.npz','episode_0009.npz','episode_0012.npz','episode_0030.npz']
    selected = [i for split in manifest['splits'].values() for i in split]
    assert len(selected) == len(set(selected)) == 4
    norm = json.loads((output/'normalization.json').read_text())
    train_ids = manifest['splits']['train']
    assert norm['train_episode_ids'] == train_ids
    rows = []
    for eid in train_ids:
        with np.load(root/entries[eid]['npz']) as record:
            rows.append(record['state'])
    expected = np.quantile(np.concatenate(rows),.99,axis=0)
    np.testing.assert_allclose(norm['norm_stats']['state']['q99'],expected)
    assert not report['training_ready']


@pytest.mark.parametrize('mutation,fragment',[
    (lambda r:r['action'].__setitem__((2,6),.5),'binary'),
    (lambda r:r['held'].__setitem__(2,True),'held'),
    (lambda r:r.__setitem__('t_cam_front',r['timestamp']+.001),'timestamp'),
    (lambda r:r['action'].__setitem__((2,3),.2),'pose-delta'),
])
def test_semantic_error_does_not_become_training_episode(tmp_path,mutation,fragment):
    root, output = tmp_path/'recordings', tmp_path/'report'
    make_episode(root,'banana','episode_0000',mutation=mutation)
    report = audit_npz(root,output)
    assert report['structural_status'] == 'PASS'
    assert report['semantic_status'] == 'ISSUES'
    assert report['split_counts'] == {'train':0,'dev':0,'test':0}
    assert fragment in ' '.join(report['semantic_problems'][0]['errors'])


def test_structural_nonfinite_is_reported_without_nan_json(tmp_path):
    root, output = tmp_path/'recordings', tmp_path/'report'
    make_episode(root,'banana','episode_0000',mutation=lambda r:r['state'].__setitem__((1,0),np.nan))
    report = audit_npz(root,output)
    assert report['structural_status'] == 'ISSUES'
    assert report['semantic_problems'][0]['status'] == 'NOT_CHECKED'
    assert 'NaN' not in (output/'audit.json').read_text()


def test_fixed_dev_test_counts_per_task(tmp_path):
    root, output = tmp_path/'recordings', tmp_path/'report'
    for task in ('bolt','banana'):
        for i in range(6):
            make_episode(root,task,f'episode_{i:04d}',offset=i + (10 if task == 'bolt' else 0))
    report = audit_npz(root,output,dev_per_task=1,test_per_task=2)
    assert report['split_counts'] == {'train':6,'dev':2,'test':4}
    manifest = json.loads((output/'episodes.json').read_text())
    groups = {e['episode_id']:e['task_group'] for e in manifest['episodes']}
    for split, per_task in (('train',3),('dev',1),('test',2)):
        assert sorted(groups[i] for i in manifest['splits'][split]) == ['banana']*per_task + ['bolt']*per_task
    assert '1 dev and 2 test per task' in manifest['split_algorithm']
    with pytest.raises(ValueError,match='together'):
        audit_npz(root,tmp_path/'a',dev_per_task=1)
    with pytest.raises(ValueError,match='cannot hold'):
        audit_npz(root,tmp_path/'b',dev_per_task=3,test_per_task=3)


def test_output_must_not_modify_recording_root(tmp_path):
    with pytest.raises(ValueError,match='outside'):
        audit_npz(tmp_path,tmp_path/'generated')


def test_exclusions_are_directory_names(tmp_path):
    with pytest.raises(ValueError,match='immediate directory'):
        audit_npz(tmp_path/'data',tmp_path/'reports',exclude_groups=['../mouse'])
