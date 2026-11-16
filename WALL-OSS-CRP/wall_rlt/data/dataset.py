"""Episode-safe windows over existing LeRobot datasets or in-memory fixtures."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeWindowDataset(Dataset):
    """The source supplies cheap action columns and decodes only the current image."""
    def __init__(self, source, episode_ids, horizon=32, *, complete_windows_only=False):
        if horizon < 1 or not episode_ids or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("need a positive horizon and unique nonempty episode IDs")
        self.source, self.horizon = source, horizon
        ids = np.asarray(source.episode_indices, dtype=np.int64)
        frames = np.asarray(source.frame_indices, dtype=np.int64)
        if ids.ndim != 1 or ids.shape != frames.shape or len(ids) != len(source.actions):
            raise ValueError("source episode/frame/action rows are misaligned")
        allowed = set(episode_ids)
        if not allowed.issubset(set(ids.tolist())):
            raise ValueError("requested episode is absent from dataset")
        self.windows = []
        visited = set()
        start = 0
        while start < len(ids):
            ep = int(ids[start])
            stop = start + 1
            while stop < len(ids) and ids[stop] == ep:
                stop += 1
            if ep in visited or not np.array_equal(frames[start:stop],np.arange(stop-start)):
                raise ValueError("episodes must be contiguous and frame_index must start at zero")
            visited.add(ep)
            if ep in allowed:
                for index in range(start, stop):
                    if not complete_windows_only or stop-index >= horizon:
                        self.windows.append((index,min(stop,index+horizon)))
            start = stop
        if not self.windows:
            raise ValueError("selected episodes contain no usable windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, item):
        start, end = self.windows[item]
        row = dict(self.source.observation(start))
        actions = torch.as_tensor(np.array(self.source.actions[start:end],copy=True),dtype=torch.float32)
        if actions.ndim != 2 or not torch.isfinite(actions).all():
            raise ValueError("source actions must be finite [time, action_dim]")
        padded = actions.new_zeros((self.horizon,actions.shape[-1]))
        padded[:end-start] = actions
        row['action'] = padded
        row['time_mask'] = torch.arange(self.horizon) < end-start
        row['episode_id'] = int(self.source.episode_indices[start])
        row['frame_index'] = int(self.source.frame_indices[start])
        return row


class LeRobotFrameSource:
    """Local dataset reader, with no implicit download, split, or geometry conversion."""
    def __init__(self, contract):
        from pathlib import Path
        root = Path(contract.root)
        if not (root/'meta/info.json').is_file():
            raise FileNotFoundError(f"local LeRobot metadata missing: {root}")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.dataset = LeRobotDataset(contract.repo_id, root=root, video_backend='pyav',
                                      download_videos=False)
        self.contract = contract
        hf = self.dataset.hf_dataset
        self.episode_indices = np.asarray(hf['episode_index'],dtype=np.int64)
        self.frame_indices = np.asarray(hf['frame_index'],dtype=np.int64)
        self.actions = np.asarray(hf[contract.action_key],dtype=np.float32)
        self.states = np.asarray(hf[contract.state_key],dtype=np.float32)
        if self.actions.shape != (len(hf),len(contract.action_names)):
            raise ValueError("dataset action shape disagrees with contract")
        if self.states.shape != (len(hf),len(contract.state_names)):
            raise ValueError("dataset state shape disagrees with contract")
        if abs(float(self.dataset.fps)-contract.fps) > 1e-6:
            raise ValueError("dataset fps disagrees with contract")
        missing = set(contract.camera_mapping)-set(self.dataset.meta.camera_keys)
        if missing:
            raise ValueError(f"dataset missing cameras: {sorted(missing)}")

    def observation(self,index):
        row = self.dataset[index]
        return {'images': {k:row[k] for k in self.contract.camera_mapping},
                'state':row[self.contract.state_key], 'instruction':row['task']}
