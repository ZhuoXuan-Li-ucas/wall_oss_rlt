"""Read-only AR5 recording audit against the supplied dataset format document."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _validate_ar5_record(record, entry, dataset_metadata):
    """Check documented invariants, without claiming robot execution validation."""
    from .ar5 import validate_pose_deltas
    errors = []
    action, state = record['action'], record['state']
    n = len(action)
    if action.shape != (n, 7) or state.shape != (n, 8):
        errors.append('AR5 requires action [T,7] and state [T,8]')
    else:
        if not np.isin(action[:, 6], [0, 1]).all():
            errors.append('gripper command must be binary 0/1')
        if np.any((state[:, 7] < 0) | (state[:, 7] > 1)):
            errors.append('measured gripper opening outside [0,1]')
        # Numerical spread is recorded, not silently used to remove joint axes.
        entry['locked_joint_peak_to_peak_rad'] = {
            'j3':float(np.ptp(state[:, 2])), 'j5':float(np.ptp(state[:, 4]))}
    if entry['fps'] != 30:
        errors.append('AR5 recording nominal FPS must be 30')
    if entry['cameras'] != ['cam_high', 'cam_front']:
        errors.append('documented camera order is cam_high, cam_front')
    if not np.array_equal(record['timestamp'], record['t_cam_front']):
        errors.append('timestamp must equal the wrist camera timeline t_cam_front')
    if record['timestamp'][0] != 0:
        errors.append('episode timestamp must start at zero')
    if 'held' not in record or record['held'].shape != (n,) or record['held'].dtype != np.bool_:
        errors.append('held must be a boolean [T] array')
    elif record['held'].any():
        errors.append('OmniPicker held must be false on every frame')
    if entry['instruction'] != dataset_metadata.get('prompt'):
        errors.append('episode instruction differs from dataset.json prompt')
    if 'ee_pose' not in record:
        errors.append('missing ee_pose for pose-delta verification')
    else:
        try:
            entry['pose_delta_validation'] = validate_pose_deltas(record['ee_pose'], action)
        except ValueError as exc:
            errors.append(f'pose-delta verification: {exc}')
    return errors


def audit_npz(root, output, *, seed=20261029, exclude_groups=(), dev_per_task=None, test_per_task=None):
    """``dev_per_task``/``test_per_task`` (given together) fix the dev/test counts
    of every task and put the rest in train; otherwise 70/15/remainder percent."""
    import av
    if (dev_per_task is None) != (test_per_task is None):
        raise ValueError('give dev_per_task and test_per_task together')
    if dev_per_task is not None and (dev_per_task < 1 or test_per_task < 1):
        raise ValueError('dev_per_task and test_per_task must be at least 1')
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('audit outputs must be outside the original data root')
    output.mkdir(parents=True, exist_ok=True)
    excluded = sorted(set(exclude_groups))
    if any(not name or Path(name).name != name or name in {'.', '..'} for name in excluded):
        raise ValueError('excluded groups must be immediate directory names')
    episodes, problems, semantic_problems, groups = [], [], [], defaultdict(list)
    fingerprints = {}
    # Only direct episode files in immediate task directories. Ignore prior
    # masks, derived exports, QC folders, and historical validation datasets.
    # Check the name before inspecting the directory or any of its contents.
    files = sorted(p for task in root.iterdir() if task.name not in excluded and task.is_dir()
                   for p in task.glob('episode_*.npz'))
    for eid, path in enumerate(files):
        entry = {'episode_id':eid, 'task_group':path.parent.name,
                 'npz':str(path.relative_to(root)), 'npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        errors, semantic_errors = [], []
        semantic_checked = False
        try:
            with np.load(path, allow_pickle=False) as record:
                action, state = record['action'], record['state']
                n = len(action)
                if action.ndim != 2 or state.ndim != 2 or len(state) != n or n < 1:
                    raise ValueError('invalid action/state shape')
                if not np.isfinite(float(record['fps'])) or float(record['fps']) <= 0:
                    raise ValueError('invalid FPS')
                entry.update(frames=n, action_dim=action.shape[-1], state_dim=state.shape[-1],
                             fps=float(record['fps']), instruction=str(record['task'].item()),
                             cameras=record['camera_keys'].tolist(),
                             held_frames=int(record['held'].sum()) if 'held' in record else None)
                for key in ('action', 'state', 'ee_pose', 'timestamp'):
                    if key in record and not np.isfinite(record[key]).all():
                        raise ValueError(f'non-finite {key}')
                times = record['timestamp']
                if times.shape != (n,) or np.any(np.diff(times) <= 0):
                    raise ValueError('nonmonotonic or wrong-length timestamp')
                entry['duration_s'] = float(times[-1]-times[0])
                entry['action_min'] = action.min(axis=0).tolist()
                entry['action_max'] = action.max(axis=0).tolist()
                entry['camera_max_offset_s'] = {}
                for camera in entry['cameras']:
                    ts = record[f't_{camera}']
                    if ts.shape != (n,) or not np.isfinite(ts).all() or np.any(np.diff(ts) < 0):
                        errors.append(f'invalid camera timeline: {camera}')
                    else:
                        entry['camera_max_offset_s'][camera] = float(np.max(np.abs(ts-times)))
                # Identical low-dimensional sequences cannot straddle splits.
                h = hashlib.sha256(action.tobytes()+state.tobytes()).hexdigest()
                entry['trajectory_sha256'] = h
                if h in fingerprints:
                    errors.append(f'duplicate trajectory: episode {fingerprints[h]}')
                fingerprints[h] = eid
                metadata_path = path.parent/'dataset.json'
                dataset_metadata = json.loads(metadata_path.read_text())
                entry['dataset_metadata_sha256'] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
                entry['gripper_settings_registers'] = {
                    key:dataset_metadata.get(key) for key in ('grip_force','speed')}
                semantic_errors = _validate_ar5_record(record, entry, dataset_metadata)
                semantic_checked = True
            entry['videos'] = {}
            for camera in entry['cameras']:
                vp = path.with_name(f'{path.stem}_{camera}.mp4')
                with av.open(str(vp)) as container:
                    stream = container.streams.video[0]
                    video = {'path':str(vp.relative_to(root)), 'bytes':vp.stat().st_size,
                             'frames':stream.frames, 'width':stream.width, 'height':stream.height,
                             'fps':float(stream.average_rate), 'time_base':str(stream.time_base)}
                    entry['videos'][camera] = video
                    if stream.frames != n or abs(float(stream.average_rate)-entry['fps']) > 1e-6:
                        errors.append(f'video length/FPS mismatch: {camera}')
                    if (stream.width,stream.height) != (640,480):
                        semantic_errors.append(f'documented camera resolution is 640x480: {camera}')
                    if next(container.decode(stream), None) is None:
                        errors.append(f'undecodable first video frame: {camera}')
        except (OSError, ValueError, KeyError, IndexError) as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
        entry['structural_errors'] = errors
        entry['semantic_errors'] = semantic_errors
        entry['semantic_status'] = ('ISSUES' if semantic_errors else 'PASS') if semantic_checked else 'NOT_CHECKED'
        if errors:
            problems.append({'episode_id':eid, 'file':entry['npz'], 'errors':errors})
        if semantic_errors or not semantic_checked:
            semantic_problems.append({'episode_id':eid, 'file':entry['npz'],
                                      'status':entry['semantic_status'], 'errors':semantic_errors})
        if not errors and not semantic_errors and semantic_checked:
            groups[entry['task_group']].append(eid)
        episodes.append(entry)
    if not files:
        raise ValueError(f'no task/episode_*.npz files under {root}')
    splits = {k:[] for k in ('train','dev','test')}
    rng = random.Random(seed)
    for task, ids in sorted(groups.items()):
        rng.shuffle(ids)
        if dev_per_task is None:
            ntrain, ndev = int(len(ids)*.7), int(len(ids)*.15)
        else:
            ndev, ntrain = dev_per_task, len(ids)-dev_per_task-test_per_task
            if ntrain < 1:
                raise ValueError(f'task {task}: {len(ids)} episodes cannot hold {dev_per_task} dev + {test_per_task} test')
        for key, subset in zip(splits, (ids[:ntrain],ids[ntrain:ntrain+ndev],ids[ntrain+ndev:])):
            splits[key].extend(subset)
    splits = {k:sorted(v) for k,v in splits.items()}
    algorithm = ('task-stratified shuffle, 70/15/remainder percent, integer floor' if dev_per_task is None else
                 f'task-stratified shuffle, {dev_per_task} dev and {test_per_task} test per task, remainder train')
    manifest = {'schema_version':1, 'source_format':'npz_video',
                'excluded_groups':excluded,
                'split_algorithm':algorithm,
                'seed':seed, 'splits':splits, 'episodes':episodes}
    manifest_path = output/'episodes.json'
    write_json(manifest_path, manifest)
    # Statistics are strictly train-only and are descriptive until semantics and
    # alignment are confirmed; do not install guessed model slot maps.
    cols = {'action':[], 'state':[]}
    for eid in splits['train']:
        with np.load(root/episodes[eid]['npz'], allow_pickle=False) as record:
            for key in cols:
                cols[key].append(record[key])
    norm = {'schema_version':1, 'train_episode_ids':splits['train'], 'norm_stats':{}}
    for key, arrays in cols.items():
        if arrays:
            values = np.concatenate(arrays).astype(np.float64)
            norm['norm_stats'][key] = {'q01':np.quantile(values,.01,axis=0).tolist(),
                                      'q99':np.quantile(values,.99,axis=0).tolist()}
    norm_path = output/'normalization.json'
    write_json(norm_path,norm)
    valid = [e for e in episodes if not e['structural_errors'] and e['semantic_status'] == 'PASS']
    first = valid[0] if valid else episodes[0]
    contract = {'root':str(root), 'repo_id':'local/top_error', 'source_format':'npz_video',
                'fps':first.get('fps',30), 'action_key':'action','state_key':'state',
                'action_names':[f'unconfirmed_action_{i}' for i in range(first.get('action_dim',7))],
                'state_names':[f'unconfirmed_state_{i}' for i in range(first.get('state_dim',8))],
                'camera_mapping':{k:k for k in first.get('cameras',[])},
                'splits':splits, 'manifest_path':'episodes.json',
                'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'normalization_path':'normalization.json',
                'normalization_sha256':hashlib.sha256(norm_path.read_bytes()).hexdigest(),
                'semantics_confirmed':False, 'purpose':'inspection'}
    write_json(output/'data_contract.inspection.json',contract)
    report = {'root':str(root), 'task':'grasping', 'scope':'original NPZ and MP4 only',
              'structural_status':'PASS' if not problems else 'ISSUES',
              'semantic_status':'PASS' if not semantic_problems else 'ISSUES',
              'semantic_check_scope':'documented shapes, binary command, wrist timeline, held, prompt, camera layout and SO(3) previous-to-current pose delta',
              'excluded_groups':excluded,
              'training_ready':False, 'episodes':len(episodes),
              'frames':sum(e.get('frames',0) for e in episodes),
              'groups':dict(Counter(e['task_group'] for e in episodes)),
              'split_counts':{k:len(v) for k,v in splits.items()},
              'split_status':'provisional stratified episode split; target representation and model contract are separate',
              'action_dims':sorted({e.get('action_dim',0) for e in episodes}),
              'state_dims':sorted({e.get('state_dim',0) for e in episodes}),
              'video_check_scope':'container frame counts/FPS and first-frame decode, not full-stream decode/hash',
              'unconfirmed':['model slot mapping', 'episode success/quality labels'],
              'problems':problems, 'semantic_problems':semantic_problems}
    write_json(output/'audit.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--exclude-group', action='append', default=[],
                        help='Immediate task directory to ignore before reading any recordings; repeatable')
    parser.add_argument('--dev-per-task', type=int, default=None,
                        help='Fixed dev episodes per task (with --test-per-task); default 15 percent')
    parser.add_argument('--test-per-task', type=int, default=None,
                        help='Fixed test episodes per task (with --dev-per-task); default remainder after 70/15 percent')
    args = parser.parse_args()
    print(json.dumps(audit_npz(args.root,args.output,seed=args.seed,exclude_groups=args.exclude_group,
                               dev_per_task=args.dev_per_task,test_per_task=args.test_per_task),
                     ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
