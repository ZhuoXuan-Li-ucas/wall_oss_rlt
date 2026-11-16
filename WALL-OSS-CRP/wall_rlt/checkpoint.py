"""Atomic local checkpoints for unsharded training and CPU recovery checks.

Distributed FSDP state is saved by RLinf's model manager and the project worker.
Only load trusted local training artifacts: optimizer/RNG/loader state uses
PyTorch's pickle format. SHA256 detects incomplete/corrupted files, not trust.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch


def rng_state():
    return {'python':random.getstate(), 'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('checkpoint CUDA RNG topology differs from this process')
        torch.cuda.set_rng_state_all(state['cuda'])


def _json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)


def _digest(path):
    digest = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def save_training_checkpoint(path, model, optimizer, scheduler, *, step, metadata,
                             loader=None, collator=None):
    """Save at an optimizer-step boundary; reject overwrite and live gradients.

    metadata must bind the source/config/data contract used by the caller.
    Loader/collator participation must match on restore. This function is not
    suitable for sharded models; use WallRLTSftWorker for FSDP training.
    """
    if type(step) is not int or step < 0:
        raise ValueError('step must be a nonnegative integer')
    if not metadata or not metadata.get('contract_hash'):
        raise ValueError('checkpoint metadata must bind a data contract hash')
    _json(metadata)
    if any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters()):
        raise ValueError('save only after optimizer.step and zero_grad')
    if any('FullySharded' in type(m).__name__ or hasattr(m,'_fsdp_state') for m in model.modules()):
        raise ValueError('sharded checkpoints must use the RLinf worker')
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{path.name}.',dir=path.parent))
    try:
        state = {'schema_version':1,'step':step,'metadata':metadata,
                 'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'scheduler':scheduler.state_dict() if scheduler is not None else None,
                 'loader':loader.state_dict() if loader is not None else None,
                 'collator':collator.state_dict() if collator is not None else None,
                 'rng':rng_state()}
        payload = temporary/'training.pt'
        torch.save(state,payload)
        with payload.open('rb') as stream:
            os.fsync(stream.fileno())
        manifest = {'schema_version':1,'step':step,'metadata':metadata,
                    'files':{'training.pt':_digest(payload)}}
        with (temporary/'manifest.json').open('w') as stream:
            stream.write(_json(manifest)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        temporary.rename(path)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise
    return manifest


def load_training_checkpoint(path, model, optimizer, scheduler, *, expected_metadata,
                             loader=None, collator=None):
    """Restore complete state and return step and the recreated loader iterator.

    RNG is restored *after* iterator creation: constructing a DataLoader
    iterator can consume torch RNG even with num_workers=0.
    """
    path = Path(path).resolve()
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('files',{}).keys() != {'training.pt'}:
        raise ValueError('invalid checkpoint manifest')
    if _json(manifest['metadata']) != _json(expected_metadata):
        raise ValueError('checkpoint metadata/config/data contract mismatch')
    if _digest(path/'training.pt') != manifest['files']['training.pt']:
        raise ValueError('checkpoint payload SHA256 mismatch')
    state = torch.load(path/'training.pt',map_location='cpu',weights_only=False)
    if (state['schema_version'] != 1 or state['step'] != manifest['step'] or
            _json(state['metadata']) != _json(expected_metadata)):
        raise ValueError('checkpoint payload disagrees with manifest')
    for key,component in (('loader',loader),('collator',collator),('scheduler',scheduler)):
        if (state[key] is None) != (component is None):
            raise ValueError(f'checkpoint {key} participation mismatch')
    current = model.state_dict()
    if current.keys() != state['model'].keys() or any(
            current[k].shape != state['model'][k].shape for k in current):
        raise ValueError('checkpoint model architecture mismatch')
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    if loader is not None:
        loader.load_state_dict(state['loader'])
    iterator = iter(loader) if loader is not None else None
    if collator is not None:
        collator.load_state_dict(state['collator'])
    restore_rng(state['rng'])
    if hasattr(model,'_rlt_requires_checkpoint_restore'):
        model._rlt_requires_checkpoint_restore = False
    return {'step':state['step'],'iterator':iterator,'metadata':state['metadata']}
