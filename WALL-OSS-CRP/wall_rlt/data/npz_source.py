"""Read existing NPZ + MP4 episodes without rewriting recordings or geometry."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path

import numpy as np
import torch


def contained_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('manifest path escapes the data root')
    return path


class NPZFrameSource:
    """Low dimensional columns in RAM; a bounded RGB cache for sequential windows.

    Camera frame i is the already aligned frame i in the recorded MP4. We never
    resample raw_arm_* or invent missing frames. The contract must separately
    establish what the collector's alignment and action labels mean.
    """

    def __init__(self, contract, *, episode_ids=None, cache_bytes=64 * 1024**2):
        self.contract = contract
        allowed = {eid for ids in contract.splits.values() for eid in ids}
        selected = set(episode_ids) if episode_ids is not None else allowed
        if not selected or not selected.issubset(allowed):
            raise ValueError('requested episodes must belong to the frozen dataset splits')
        self.episodes = []
        actions, states, ids, frames = [], [], [], []
        offset = 0
        self._cache, self._cache_bytes, self.cache_limit = OrderedDict(), 0, cache_bytes
        for entry in contract.read_manifest()['episodes']:
            eid = entry['episode_id']
            if eid not in selected:
                continue
            path = contained_path(contract.root, entry['npz'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['npz_sha256']:
                raise ValueError(f'recording changed after audit: {entry["npz"]}')
            with np.load(path, allow_pickle=False) as record:
                action = np.asarray(record[contract.action_key], dtype=np.float32)
                state = np.asarray(record[contract.state_key], dtype=np.float32)
                recorded_count = len(action)
                if recorded_count != entry['frames']:
                    raise ValueError('recorded length differs from manifest')
                if contract.target_mode != 'recorded':
                    from .ar5 import EE_REL6D_NEXT, aligned_targets
                    ee_pose = (np.asarray(record['ee_pose'], dtype=np.float64)
                               if contract.target_mode == EE_REL6D_NEXT else None)
                    action, state = aligned_targets(state, action, contract.target_mode, ee_pose=ee_pose)
                count = len(action)
                if action.shape != (count, len(contract.action_names)) or state.shape != (count, len(contract.state_names)):
                    raise ValueError('recording dimensions disagree with contract')
                if not count or not np.isfinite(action).all() or not np.isfinite(state).all():
                    raise ValueError('invalid recording length or non-finite values')
                if abs(float(record['fps']) - contract.fps) > 1e-6:
                    raise ValueError('recording FPS disagrees with contract')
                if not set(contract.camera_mapping).issubset(record['camera_keys'].tolist()):
                    raise ValueError('recording lacks a contract camera')
                instruction = str(record['task'].item())
            for camera in contract.camera_mapping:
                video = entry['videos'][camera]
                vp = contained_path(contract.root, video['path'])
                if video['frames'] != recorded_count or vp.stat().st_size != video['bytes']:
                    raise ValueError('video length/size differs from audited recording')
            self.episodes.append((offset, offset + count, entry, instruction))
            actions.append(action); states.append(state)
            ids.extend([eid] * count); frames.extend(range(count))
            offset += count
        if not actions:
            raise ValueError('no selected recordings')
        self.actions, self.states = np.concatenate(actions), np.concatenate(states)
        self.episode_indices = np.asarray(ids, dtype=np.int64)
        self.frame_indices = np.asarray(frames, dtype=np.int64)
        self._stops = np.asarray([e[1] for e in self.episodes])

    def _frame(self, entry, camera, frame_index):
        import av
        video = entry['videos'][camera]
        key = (entry['episode_id'], camera, frame_index)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        path = contained_path(self.contract.root, video['path'])
        # Seek by PTS for CFR recordings. Require the exact frame; do not use an
        # arbitrary nearest frame if timestamps or codec indexing differ.
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'NONE'
            start = stream.start_time or 0
            target = start + round(frame_index / float(stream.average_rate) / float(stream.time_base))
            container.seek(target, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError(f'video frame has no timestamp: {path}')
                if frame.pts < target:
                    continue
                if frame.pts != target:
                    raise ValueError(f'no exact recorded video frame {frame_index}: {path}')
                result = frame.to_ndarray(format='rgb24')
                if result.nbytes <= self.cache_limit:
                    while self._cache and self._cache_bytes + result.nbytes > self.cache_limit:
                        _, old = self._cache.popitem(last=False)
                        self._cache_bytes -= old.nbytes
                    self._cache[key] = result
                    self._cache_bytes += result.nbytes
                return result.copy()
        raise ValueError(f'video ended before frame {frame_index}: {path}')

    def observation(self, index):
        if not 0 <= index < len(self.actions):
            raise IndexError(index)
        start, _, entry, instruction = self.episodes[int(np.searchsorted(self._stops, index, side='right'))]
        images = {key: torch.from_numpy(self._frame(entry,key,index-start)).permute(2,0,1)
                  for key in self.contract.camera_mapping}
        return {'images': images, 'state': torch.from_numpy(self.states[index].copy()),
                'instruction': instruction}


def build_source(contract, *, episode_ids=None):
    if contract.source_format == 'npz_video':
        return NPZFrameSource(contract, episode_ids=episode_ids)
    if contract.source_format == 'lerobot':
        from .dataset import LeRobotFrameSource
        return LeRobotFrameSource(contract)
    raise ValueError('in-memory fixtures cannot be used as a training source')
