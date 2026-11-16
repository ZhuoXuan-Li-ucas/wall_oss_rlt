import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from wall_rlt.checkpoint import save_training_checkpoint,load_training_checkpoint
from checkpoint_case import METADATA, components, continuation


def test_fresh_process_restores_next_batch_both_losses_and_optimizer_update(tmp_path):
    checkpoint = tmp_path/'step1'
    expected = continuation(checkpoint,resume=False)
    target = tmp_path/'resumed.pt'
    script = Path(__file__).with_name('checkpoint_case.py')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONPATH=str(Path.cwd()))
    subprocess.run([sys.executable,str(script),str(checkpoint),str(target)],env=env,check=True,
                   capture_output=True,text=True,timeout=60)
    actual = torch.load(target,weights_only=True)
    assert actual['lr'] == expected['lr']
    for key in ('batch','losses','weights'):
        assert actual[key].keys() == expected[key].keys()
        for name,value in expected[key].items():
            torch.testing.assert_close(actual[key][name],value,rtol=0,atol=0)


def test_mismatched_contract_rejected_before_weights_change(tmp_path):
    model,opt,scheduler,_,_ = components()
    path=tmp_path/'ckpt'
    save_training_checkpoint(path,model,opt,scheduler,step=0,metadata=METADATA)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError,match='metadata/config/data'):
        load_training_checkpoint(path,model,opt,scheduler,expected_metadata={'contract_hash':'wrong'})
    for k,v in before.items():
        torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


def test_corrupted_payload_and_overwrite_are_rejected(tmp_path):
    model,opt,scheduler,_,_ = components()
    path=tmp_path/'ckpt'
    save_training_checkpoint(path,model,opt,scheduler,step=0,metadata=METADATA)
    with pytest.raises(FileExistsError):
        save_training_checkpoint(path,model,opt,scheduler,step=0,metadata=METADATA)
    with (path/'training.pt').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError,match='SHA256'):
        load_training_checkpoint(path,model,opt,scheduler,expected_metadata=METADATA)


def test_unfinished_optimizer_step_is_not_saved(tmp_path):
    model,opt,scheduler,loader,_ = components()
    model(data=next(iter(loader)))['loss'].backward()
    with pytest.raises(ValueError,match='zero_grad'):
        save_training_checkpoint(tmp_path/'bad',model,opt,scheduler,step=1,metadata=METADATA)
    assert not (tmp_path/'bad').exists()
