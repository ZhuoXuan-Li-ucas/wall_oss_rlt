"""CPU checks on audited recorded data using WALL's official processor only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from wall_rlt.model.prefix import observation_prefix_mask
from .ar5 import EE_REL6D_NEXT
from .audit import write_json
from .collator import WallCollator
from .contract import DataContract
from .dataset import EpisodeWindowDataset
from .npz_source import NPZFrameSource


def expected_rows(raw, target_mode):
    """Absolute ``(targets, observations)`` straight from the NPZ, independent of ar5.py.

    Joint mode: measured j1..j7 (+ command / measured opening). EE mode: flange
    position and the first two rotation-matrix rows of ``ee_pose`` via scipy.
    """
    state, command = raw['state'].astype(np.float64), raw['action'][:, 6:7].astype(np.float64)
    if target_mode != EE_REL6D_NEXT:
        return np.concatenate((state[1:, :7], command[1:]), axis=1), state[:-1]
    from scipy.spatial.transform import Rotation
    ee = raw['ee_pose'].astype(np.float64)
    pose = np.concatenate((ee[:, :3], Rotation.from_rotvec(ee[:, 3:]).as_matrix()[:, :2, :].reshape(-1, 6)), axis=1)
    return (np.concatenate((pose[1:], command[1:]), axis=1),
            np.concatenate((pose[:-1], state[:-1, 7:8]), axis=1))


def check_data(contract_path, processor_path):
    torch.set_num_threads(1)
    contract = DataContract.load(contract_path)
    contract.require_trainable()
    manifest = contract.read_manifest()
    all_ids = [e['episode_id'] for e in manifest['episodes']]
    source = NPZFrameSource(contract,episode_ids=all_ids)
    dataset = EpisodeWindowDataset(source,all_ids,contract.action_horizon)
    expected = sum(e['frames']-1 for e in manifest['episodes'])
    if len(dataset) != expected:
        raise AssertionError('unexpected next-frame dataset length')
    collator = WallCollator(contract,processor_path,seed=20261029)
    records = []
    seen = set()
    for start, stop, entry, _ in source.episodes:
        group = entry['task_group']
        if group in seen:
            continue
        seen.add(group)
        # Consecutive video decoding in the reader is independent of actions.
        # Check both the first and last supervised frame of each task sample.
        first, tail = dataset[start],dataset[stop-1]
        with np.load(Path(contract.root)/entry['npz'],allow_pickle=False) as raw:
            targets, observations = expected_rows(raw, contract.target_mode)
        # float32 rows; EE rows go through float64 rotation math first.
        for got, want in ((first['action'][0], targets[0]), (tail['action'][0], targets[-1]),
                          (first['state'], observations[0]), (tail['state'], observations[-1])):
            np.testing.assert_allclose(got.numpy(), want.astype(np.float32), rtol=0, atol=1e-6)
        if tail['time_mask'].sum() != 1:
            raise AssertionError('last supervised frame must have one valid action')
        saved = collator.state_dict()
        batch = collator([first,tail])
        wall = batch['wall_inputs']
        if wall['action_chunk'].shape != (2,contract.action_horizon,26):
            raise AssertionError('wrong model action shape')
        if not torch.isfinite(wall['action_chunk']).all() or not torch.isfinite(wall['proprioception']).all():
            raise AssertionError('nonfinite normalization, including constant joints')
        width = len(contract.action_slots)
        if wall['dof_mask'][1].sum() != width or wall['dof_mask'][0].sum() != width*contract.action_horizon:
            raise AssertionError('dimension/time mask mismatch')
        prefix = observation_prefix_mask(wall['input_ids'],wall['attention_mask'],collator.action_token_id)
        changed = copy.deepcopy(tail)
        changed['action'][1:] = float('nan')
        collator.load_state_dict(saved)
        other = collator([first,changed])['wall_inputs']
        for key in ('action_chunk','dof_mask','input_ids','sample_time'):
            torch.testing.assert_close(wall[key],other[key],rtol=0,atol=0)
        obs_batch = collator.observation_batch([first,tail])['wall_inputs']
        if 'action_chunk' in obs_batch:
            raise AssertionError('observation-only batch contains action targets')
        torch.testing.assert_close(wall['input_ids'],obs_batch['input_ids'],rtol=0,atol=0)
        records.append({'group':group,'episode':entry['npz'], 'checked_frame_indices':[0,entry['frames']-2],
                        'rgb_shapes':{key:list(value.shape) for key,value in first['images'].items()},
                        'action_shape':list(wall['action_chunk'].shape),
                        'prefix_lengths':prefix.sum(dim=1).tolist(),
                        'valid_actions':wall['flow_loss_mask'].sum(dim=1).tolist()})
    return {'status':'PASS','scope':'CPU recorded data and official WALL processor; no model weights/forward',
            'contract_hash':contract.fingerprint(),'episodes':len(all_ids),
            'usable_observations':len(dataset),'dropped_final_observations':len(all_ids),
            'split_counts':{k:len(v) for k,v in contract.splits.items()},
            'excluded_groups':manifest['excluded_groups'],
            'checks':['all NPZ hashes and target dimensions','per-episode next-frame alignment',
                      'representative first/tail RGB seek in each task','official WALL preprocessing',
                      'tail time/dimension masks','constant-joint finite normalization',
                      'invalid targets do not alter batch','collator RNG restore',
                      'action targets do not alter observation prompt'],
            'samples':records,'model_training':'NOT_RUN','robot_inference':'DEFERRED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True)
    parser.add_argument('--processor',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    report = check_data(args.contract,args.processor)
    write_json(args.output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
