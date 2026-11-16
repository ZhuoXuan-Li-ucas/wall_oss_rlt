#!/usr/bin/env python3
"""Offline relocation checks; does not initialize CUDA or load model weights."""
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
if mode == '--data':
    expected = json.loads((root / 'migration/data-files.json').read_text())
    data = root / 'data/top_error'
    for name, spec in expected.items():
        path = data / name
        assert path.is_file() and path.stat().st_size == spec['bytes'], name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b''):
                digest.update(block)
        assert digest.hexdigest() == spec['sha256'], name
    sys.path.insert(0, str(root))
    from wall_rlt.data.contract import DataContract
    from wall_rlt.data.npz_source import NPZFrameSource
    contract = DataContract.load(root / 'configs/rlt/top_error_ar5_joint_next_v1.json')
    contract.require_trainable()
    # Read actual first-episode numeric records and decode both camera frames.
    source = NPZFrameSource(contract, episode_ids=[0])
    first = source.observation(0)
    print(json.dumps({'status':'PASS', 'files_sha256_checked': len(expected),
                     'episodes':len(contract.read_manifest()['episodes']),
                     'first_sample_keys':sorted(first), 'data_root': str(data)}, indent=2))
else:
    expected_name = 'source-env-versions.json' if mode == 'env' else 'source-rlt-versions.json'
    expected = json.loads((root / 'migration' / expected_name).read_text())
    actual = dict(sorted((d.metadata['Name'], d.version) for d in metadata.distributions()))
    assert actual == expected, {'missing':set(expected)-set(actual),
        'extra':set(actual)-set(expected), 'changed':{k:[v,actual.get(k)] for k,v in expected.items() if actual.get(k)!=v}}
    assert Path(sys.prefix).resolve() == root / mode, sys.prefix
    paths = {}
    for name in ['torch', 'numpy', 'transformers', 'wall_x', 'dmuon', 'lerobot']:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(root), (name, str(path))
        paths[name] = str(path)
    import torch
    assert torch.isfinite(torch.ones(3, 3) @ torch.ones(3, 3)).all()
    for path in sys.path:
        assert not path.startswith('/home/pangk/workspaces/WALL-OSS-CRP'), path
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    expected_issues = (root / 'migration/source-pip-check.txt').read_text().splitlines()
    if mode == 'env':
        assert sorted(check.stdout.splitlines()) == sorted(expected_issues), check.stdout
    print(json.dumps({'status': 'PASS', 'environment': mode, 'versions_equal_to_source': True,
        'distributions':len(actual), 'modules':paths, 'torch_cuda':torch.version.cuda,
        'cpu_matmul':'PASS', 'pip_check_exit_code':check.returncode,
        'pip_check_issues':check.stdout.splitlines(), 'gpu_validation':'NOT_RUN'}, indent=2))
