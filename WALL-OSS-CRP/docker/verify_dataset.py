"""Read first, middle and last real samples, including both camera videos."""
import argparse
import json
from pathlib import Path
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ds = LeRobotDataset(args.root.name, root=args.root, video_backend='pyav')
assert len(ds) > 0
rows = []
for index in sorted({0, len(ds) // 2, len(ds) - 1}):
    item = ds[index]
    fields = {}
    for name, value in item.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), (index, name)
            fields[name] = {'shape': list(value.shape), 'dtype': str(value.dtype)}
    assert 'action' in fields and 'observation.state' in fields
    assert len([key for key in fields if key.startswith('observation.images.')]) >= 2
    rows.append({'index': index, 'fields': fields})
report = {'status': 'PASS', 'root': str(args.root), 'episodes': ds.num_episodes,
          'frames': ds.num_frames, 'samples': rows,
          'scope': 'historical smoke dataset; not AR5 demonstrations or a WALL training batch'}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
