"""Freeze an audited AR5 dataset and train-only statistics.

2026-10-31: recordings collected before the joint unlock (j3/j5 locked) are void.
Locked joints are therefore measured from the train split instead of assumed,
required group exclusions are explicit arguments, and the collection settings
written by wall_collect.py (``wall_collection.json``) are carried into provenance.

2026-11-02: the default target is the official Wall-OSS-0.5 EE layout
(``ar5_ee_rel6d_next_v1``, right-arm slots 10..19, pretrained projections kept).
Its action statistics cover every valid relative window step k=0..31, computed
with the same function the collator applies. ``ar5_joint_next_v1`` is unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .ar5 import EE_REL6D_NEXT, JOINT_NEXT, aligned_targets, ee_relative_window
from .audit import write_json
from .contract import DataContract
from .npz_source import contained_path


COLLECTION_RECORD = 'wall_collection.json'
HORIZON = 32  # DataContract.action_horizon; WALL flow horizon
MIN_RANGE = 1e-9
ROT6D = [f'r{i}{j}' for i in range(2) for j in range(3)]
EE_FIELDS = {
    'action_names': ['flange_dx', 'flange_dy', 'flange_dz'] + [f'rot6d_rel_{n}' for n in ROT6D]
                    + ['gripper_next_command'],
    'state_names': ['flange_x', 'flange_y', 'flange_z'] + [f'rot6d_{n}' for n in ROT6D]
                   + ['gripper_measured_opening'],
    'action_units': ['m'] * 3 + ['rotation_matrix_element'] * 6 + ['binary_0_close_1_open'],
    'state_units': ['m'] * 3 + ['rotation_matrix_element'] * 6 + ['opening_fraction_position_div_210'],
    'action_slots': list(range(10, 20)), 'state_slots': list(range(10, 20)),
    'action_convention': 'mixed', 'field_conventions': ['delta'] * 9 + ['absolute'],
    'coordinate_frame': 'AR5_base_frame_flange_pose',
    'model_adapter': 'wall_oss05_right_arm_ee_v1',
}
JOINT_FIELDS = {
    'action_names': [f'j{i}_next_measured' for i in range(1,8)]+['gripper_next_command'],
    'state_names': [f'j{i}_measured' for i in range(1,8)]+['gripper_measured_opening'],
    'action_units': ['rad']*7+['binary_0_close_1_open'],
    'state_units': ['rad']*7+['opening_fraction_position_div_210'],
    'action_slots': list(range(8)), 'state_slots': list(range(8)),
    'action_convention': 'absolute',
    'coordinate_frame': 'AR5_controller_joint_axes_1_to_7',
    'model_adapter': 'ar5_joint8_reinit_v1',
}
DEFAULT_REPO_IDS = {JOINT_NEXT: 'local/top_error_ar5_joint_next_v1', EE_REL6D_NEXT: 'local/ar5_ee_rel6d_next_v1'}


def relative_windows(targets, observations, horizon=HORIZON):
    """Every valid step of every training window, as the collator sees it."""
    n = len(targets)
    return np.concatenate([ee_relative_window(targets[t:min(n, t + horizon)], observations[t])
                           for t in range(n)])


def prepare_ar5(root, audit_dir, document, *, contract_path, required_excluded=(),
                repo_id=None, locked_tol_rad=1e-3, target_mode=EE_REL6D_NEXT):
    if target_mode not in DEFAULT_REPO_IDS:
        raise ValueError(f'unsupported target mode: {target_mode}')
    repo_id = repo_id or DEFAULT_REPO_IDS[target_mode]
    ee = target_mode == EE_REL6D_NEXT
    root, audit_dir, document = Path(root).resolve(), Path(audit_dir).resolve(), Path(document).resolve()
    contract_path = Path(contract_path).resolve()
    if contract_path.is_relative_to(root) or audit_dir.is_relative_to(root):
        raise ValueError('derived artifacts must be outside original recordings')
    report = json.loads((audit_dir/'audit.json').read_text())
    if report['structural_status'] != 'PASS' or report.get('semantic_status') != 'PASS':
        raise ValueError('all selected episodes must pass structural and documented-semantic audit')
    mp = audit_dir/'episodes.json'
    manifest_bytes = mp.read_bytes()
    manifest = json.loads(manifest_bytes)
    excluded = sorted(manifest.get('excluded_groups',[]))
    missing = sorted(set(required_excluded)-set(excluded))
    if missing:
        raise ValueError(f'groups {missing} must be excluded; regenerate the manifest with exclusions')
    if any(e['task_group'] in excluded for e in manifest['episodes']):
        raise ValueError('excluded recordings present in manifest')
    splits = manifest['splits']
    train_ids = set(splits['train'])
    actions, states, joints = [], [], []
    for entry in manifest['episodes']:
        if entry['episode_id'] not in train_ids:
            continue
        path = contained_path(root,entry['npz'])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['npz_sha256']:
            raise ValueError('recording changed after audit')
        with np.load(path,allow_pickle=False) as record:
            target, obs = aligned_targets(record['state'],record['action'],target_mode,
                                          ee_pose=record['ee_pose'] if ee else None)
            actions.append(relative_windows(target, obs) if ee else target); states.append(obs)
            joints.append(np.asarray(record['state'][:-1, :7] if ee else obs[:, :7], dtype=np.float64))
    if not actions:
        raise ValueError('empty training split')
    stats = {}
    for key, arrays in (('action',actions),('state',states)):
        values = np.concatenate(arrays).astype(np.float64)
        stats[key] = {'q01':np.quantile(values,.01,axis=0).tolist(),
                      'q99':np.quantile(values,.99,axis=0).tolist(),
                      'min':values.min(axis=0).tolist(), 'max':values.max(axis=0).tolist(),
                      'count':len(values)}
    degenerate = {}
    if ee:
        # wall-x treats a zero range as 1 when normalizing; make that explicit
        # so normalization and unnormalization agree.
        for key in ('action', 'state'):
            lo, hi = stats[key]['q01'], stats[key]['q99']
            degenerate[key] = [i for i in range(len(lo)) if hi[i] - lo[i] <= MIN_RANGE]
            for i in degenerate[key]:
                hi[i] = lo[i] + 1.0
    # A joint counts as locked only if it never moved in the train split.
    joint_values = np.concatenate(joints)
    joint_ptp = joint_values.max(axis=0)-joint_values.min(axis=0)
    locked = [int(i) for i in np.flatnonzero(joint_ptp < locked_tol_rad)]
    collection = {}
    record_path = root/COLLECTION_RECORD
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        collection = {'collection_settings':record.get('settings'),
                      'collection_fixed_args':record.get('fixed_args'),
                      'collection_record_sha256':hashlib.sha256(record_path.read_bytes()).hexdigest()}
    mhash = hashlib.sha256(manifest_bytes).hexdigest()
    norm_path = audit_dir/f'normalization.{target_mode}.json'
    write_json(norm_path, {'schema_version':1,'train_episode_ids':splits['train'],
                          'manifest_sha256':mhash,'target_mode':target_mode,
                          'norm_stats':stats})
    fields = EE_FIELDS if ee else JOINT_FIELDS
    if ee:
        semantics = {
            'target_source':'next-frame measured flange pose (npz ee_pose, base frame) relative to the '
                            'observation pose of each window, as wall-x LerobotDataset._to_relative_action; '
                            'next-frame binary gripper command (absolute)',
            'rotation_6d':'first two rows of the rotation matrix, row-major; R_rel = R_abs @ R_obs^T',
            'model_slots':'Wall-OSS-0.5 follow_right_* slots 10..19 (pos 10..12, 6D 13..18, gripper 19)',
            'action_statistics':f'all valid relative window steps k=0..{HORIZON-1} of the train split',
            'degenerate_range_dims':degenerate,
            'model_adaptation':'official Wall-OSS-0.5 26-slot layout and pretrained action projections; '
                               'train-only target normalizers; state string input'}
    else:
        semantics = {
            'target_source':'next-frame measured joint angles, not saved arm commands; next-frame binary gripper command',
            'model_adaptation':'redefine first 8 of 26 slots; reset action_preprocessor.w1 and action_preprocessor.action_proj_back before fine-tuning; state string input'}
    contract = DataContract(
        root=str(root), repo_id=repo_id, fps=30.0,
        source_format='npz_video', action_key='action',state_key='state',
        camera_mapping={'cam_high':'top_view','cam_front':'wrist_view'},
        splits=splits, robot_type='AR5_7axis_OmniPicker',semantics_confirmed=True,purpose='target',
        alignment='observation_t_target_t_plus_1_drop_last', target_mode=target_mode,
        **fields,
        manifest_path=str(mp),manifest_sha256=mhash,
        normalization_path=str(norm_path),normalization_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        provenance={'format_document':document.name,'format_document_sha256':hashlib.sha256(document.read_bytes()).hexdigest(),
                    'collection_host':'user-confirmed intended Ubuntu inference validation host',
                    'collection_robot':'user-confirmed intended inference validation robot',
                    'excluded_groups':excluded,
                    **{k:v for k,v in semantics.items() if k != 'model_adaptation'},
                    'locked_joint_indices_zero_based':locked,
                    'locked_joint_rule':f'train-split state peak-to-peak < {locked_tol_rad} rad',
                    'train_joint_peak_to_peak_rad':[round(float(x),6) for x in joint_ptp],
                    **collection,
                    'model_adaptation':semantics['model_adaptation'],
                    'dataset_rewards':'not supplied; not an RL replay dataset'})
    contract.require_trainable()
    # Artifact references are portable within the project; root is overridden
    # explicitly when the existing disk is mounted at another host path.
    import os
    value = asdict(contract)
    for key in ('manifest_path','normalization_path'):
        value[key] = os.path.relpath(value[key],contract_path.parent)
    contract_path.parent.mkdir(parents=True,exist_ok=True)
    write_json(contract_path,value)
    return {'contract_path':str(contract_path),'contract_hash':contract.fingerprint(),
            'target_mode':contract.target_mode,'model_adapter':contract.model_adapter,
            'episodes':len(manifest['episodes']),'split_counts':{k:len(v) for k,v in splits.items()},
            'train_observations':stats['state']['count'],
            'train_action_rows':stats['action']['count'],
            'locked_joint_indices_zero_based':locked,
            'collection_record':bool(collection),
            'raw_frames':sum(e['frames'] for e in manifest['episodes']),
            'usable_observations':sum(e['frames']-1 for e in manifest['episodes']),
            'model_training_status':'NOT_RUN'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--audit-dir',required=True)
    parser.add_argument('--document',required=True)
    parser.add_argument('--contract-path',required=True)
    parser.add_argument('--require-excluded',action='append',default=[],
                        help='Task group the manifest must have excluded; repeatable (old top_error flow: mouse, tape)')
    parser.add_argument('--target-mode',choices=sorted(DEFAULT_REPO_IDS),default=EE_REL6D_NEXT,
                        help='ar5_ee_rel6d_next_v1: official Wall-OSS-0.5 EE layout (default); '
                             'ar5_joint_next_v1: joint8 reinit variant')
    parser.add_argument('--repo-id',default=None,help=f'default per target mode: {DEFAULT_REPO_IDS}')
    parser.add_argument('--locked-tol-rad',type=float,default=1e-3,
                        help='A joint whose train-split peak-to-peak is below this is recorded as locked')
    args = parser.parse_args()
    print(json.dumps(prepare_ar5(args.root,args.audit_dir,args.document,contract_path=args.contract_path,
                                 required_excluded=args.require_excluded,repo_id=args.repo_id,
                                 locked_tol_rad=args.locked_tol_rad,target_mode=args.target_mode),
                     ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
