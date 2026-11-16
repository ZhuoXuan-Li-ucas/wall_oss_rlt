"""CPU contracts for worker loss accounting and resumable data state, not FSDP."""
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'.cache/rlinf-rlt'))
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker
from wall_rlt.workers.sft import WallRLTSftWorker,loss_weights,to_device,validate_training_config,select_eval_dataset
from wall_rlt.config import WallRLTConfig


class FixtureData(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self,index):
        return index


class FixtureCollator:
    def __init__(self):
        self.native = SimpleNamespace(np_rng=np.random.default_rng(42))

    def state_dict(self):
        return copy.deepcopy(self.native.np_rng.bit_generator.state)

    def load_state_dict(self,state):
        self.native.np_rng.bit_generator.state = copy.deepcopy(state)

    def __call__(self,items):
        index = items[0]
        valid = 1+(index%2)
        prefix = 1+(index%3)
        return {'contract_hash':'fixture','metadata':[{'index':index}],
                'wall_inputs':{'input_ids':torch.tensor([[1]*prefix+[9,9]]),
                               'attention_mask':torch.ones(1,prefix+2,dtype=torch.long),
                               'flow_loss_mask':torch.tensor([[1,valid-1]]),
                               'dof_mask':torch.ones(1,2,2),
                               'index':index,'noise':self.native.np_rng.random()}}


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self,forward_type,data):
        assert forward_type == 'sft'
        i = data['wall_inputs']['index']
        vla = self.weight*(i+1)
        rlt = self.weight*(10-i)
        return {'loss':rlt+2*vla,'vla_loss':vla,'rlt_loss':rlt}


def make_worker():
    worker = object.__new__(WallRLTSftWorker)
    worker.cfg = OmegaConf.create({'actor':{'seed':11},'data':{}})
    worker.rlt_config = WallRLTConfig(input_dim=4,embed_dim=4,num_heads=1,
        num_layers=1,prefix_seq_len=8,rlt_alpha=2,action_horizon=2,model_action_dim=2,action_token_id=9)
    worker.contract = SimpleNamespace(fingerprint=lambda:'fixture')
    worker.model = FixtureModel()
    worker.device = torch.device('cpu')
    worker.amp_context = nullcontext()
    worker.worker_timer = nullcontext
    worker._rank=0; worker._world_size=1
    worker.micro_batch_size=1; worker.global_batch_size=2; worker.eval_batch_size=1
    worker._data_epoch=0; worker._data_iter_offset=0; worker.global_step=0
    data = FixtureData()
    sampler = DistributedSampler(data,num_replicas=1,rank=0,seed=11,shuffle=True,drop_last=True)
    worker.data_loader = StatefulDataLoader(data,batch_size=1,sampler=sampler,
        collate_fn=FixtureCollator(),num_workers=0,generator=torch.Generator().manual_seed(11))
    worker.data_iter = iter(worker.data_loader)
    worker.eval_data_loader = StatefulDataLoader(data,batch_size=1,shuffle=False,
        collate_fn=FixtureCollator(),num_workers=0,generator=torch.Generator().manual_seed(111))
    return worker


def test_eval_uses_separate_valid_element_denominators_and_restores_rng():
    worker = make_worker()
    global_state = torch.get_rng_state().clone()
    collator_state = worker.eval_data_loader.collate_fn.state_dict()
    result = worker.run_eval()
    expected_vla = sum((i+1)*(1+i%2)*2 for i in range(6))/sum((1+i%2)*2 for i in range(6))
    expected_rlt = sum((10-i)*(1+i%3)*4 for i in range(6))/sum((1+i%3)*4 for i in range(6))
    assert result['vla_loss'] == pytest.approx(expected_vla)
    assert result['rlt_loss'] == pytest.approx(expected_rlt)
    assert result['loss'] == pytest.approx(expected_rlt+2*expected_vla)
    assert result['samples']==6 and result['action_elements']==18 and result['prefix_elements']==48
    assert 'accuracy' not in result
    torch.testing.assert_close(global_state,torch.get_rng_state())
    assert worker.eval_data_loader.collate_fn.state_dict()==collator_state
    assert worker.model.training
    assert worker.run_eval()==result


def test_eval_multirank_is_explicitly_rejected_before_forward():
    worker = make_worker(); worker._world_size=2
    with pytest.raises(ValueError,match='world_size=1'):
        worker.run_eval()


def test_training_preserves_gradients_and_checks_batch_contract():
    worker = make_worker()
    batch = next(worker.data_iter)
    loss,metrics = worker.get_train_model_output(batch)
    assert loss.requires_grad and all(isinstance(x,float) for x in metrics.values())
    loss.backward()
    assert worker.model.weight.grad != 0
    batch['contract_hash']='wrong'
    with pytest.raises(ValueError,match='contract mismatch'):
        worker.get_train_model_output(batch)


def test_loss_mask_rejects_zero_elements():
    worker = make_worker(); batch=next(worker.data_iter)
    batch['wall_inputs']['dof_mask'].zero_()
    with pytest.raises(ValueError,match='zero valid'):
        loss_weights(batch,worker.rlt_config)


def test_to_device_keeps_non_tensor_metadata_and_nested_structure():
    value={'values':(torch.ones(1),[torch.zeros(2),'text']),'source':'fixture'}
    moved=to_device(value,'cpu')
    assert isinstance(moved['values'],tuple) and moved['source']=='fixture'
    assert moved['values'][1][1]=='text'


@pytest.fixture
def simulated_upstream(monkeypatch):
    # Test our sidecar ordering against the same StatefulDataLoader API used by
    # RLinf, without claiming distributed optimizer serialization was exercised.
    calls=[]
    def save(self,path,step=0):
        root=Path(path);root.mkdir(exist_ok=True)
        torch.save(self.data_loader.state_dict(),root/'data.pt')
        torch.save(torch.get_rng_state(),root/'rng.pt')
        calls.append('save')
    def restore(self,path):
        root=Path(path)
        self.data_loader.load_state_dict(torch.load(root/'data.pt',weights_only=False))
        self.data_iter=iter(self.data_loader)
        torch.set_rng_state(torch.load(root/'rng.pt',weights_only=False))
        calls.append(('load',self.data_loader.sampler.epoch))
    monkeypatch.setattr(FSDPVlaSftWorker,'save_checkpoint',save)
    monkeypatch.setattr(FSDPVlaSftWorker,'load_checkpoint',restore)
    monkeypatch.setattr(torch.distributed,'all_gather_object',lambda values,state:values.__setitem__(0,state))
    monkeypatch.setattr(torch.distributed,'barrier',lambda:None)
    return calls


def test_checkpoint_restores_sampler_epoch_next_batch_and_collator_rng(tmp_path,simulated_upstream):
    worker=make_worker()
    worker._data_epoch=3
    worker.data_loader.sampler.set_epoch(3)
    worker.data_iter=iter(worker.data_loader)
    next(worker.data_iter);next(worker.data_iter)
    worker._data_iter_offset=2
    worker.save_checkpoint(str(tmp_path),step=27)
    expected=next(worker.data_iter)
    restored=make_worker()
    restored.model._rlt_requires_checkpoint_restore=True
    restored.load_checkpoint(str(tmp_path))
    actual=next(restored.data_iter)
    assert actual['metadata']==expected['metadata']
    assert actual['wall_inputs']['noise']==expected['wall_inputs']['noise']
    assert restored._data_epoch==3 and restored._data_iter_offset==2 and restored.global_step==27
    assert restored.model._rlt_requires_checkpoint_restore is False
    assert simulated_upstream[-1]==('load',3)


@pytest.mark.parametrize('fault',['batch','digest','missing_rng','optimizer','backend','asset'])
def test_resume_rejects_incompatible_or_incomplete_state_before_model_load(tmp_path,simulated_upstream,fault):
    worker=make_worker();worker.save_checkpoint(str(tmp_path),step=4)
    restored=make_worker()
    if fault=='batch':
        restored.global_batch_size=4
    elif fault=='digest':
        with (tmp_path/'wall_rlt_data.pt').open('ab') as f:
            f.write(b'bad')
    elif fault=='optimizer':
        restored.cfg.actor.optim={'lr':0.1}
    elif fault=='backend':
        restored.cfg.actor.model={'wall_attention':'eager'}
    elif fault=='asset':
        (tmp_path/'wall_rlt_assets'/'rlt_config.json').write_text('{}')
    else:
        (tmp_path/'rng.pt').unlink()
    with pytest.raises(ValueError):
        restored.load_checkpoint(str(tmp_path))
    assert simulated_upstream==['save']


def test_supplied_yaml_validates_without_starting_cluster():
    from wall_rlt.rlinf_ext import register
    from rlinf.config import validate_sft_cfg,validate_fsdp_cfg
    register()
    cfg=OmegaConf.load(PROJECT/'configs/rlt/stage1.yaml')
    cfg=validate_sft_cfg(cfg); cfg.actor=validate_fsdp_cfg(cfg.actor)
    rlt,contract=validate_training_config(cfg,world_size=1)
    assert rlt.action_horizon==32
    assert len(contract.splits['train'])==245
    assert cfg.actor.fsdp_config.mixed_precision_rules.rlt_features=='fp32'


@pytest.mark.parametrize('setting,value,error',[
    ('data.num_workers',1,'num_workers=0'),
    ('actor.model.initialization','resume','set together'),
    ('actor.model.rlt.action_horizon',31,'dimensions disagree'),
])
def test_config_rejects_unsafe_resume_or_data_changes(setting,value,error):
    cfg=OmegaConf.load(PROJECT/'configs/rlt/stage1.yaml')
    OmegaConf.update(cfg,setting,value)
    with pytest.raises(ValueError,match=error):
        validate_training_config(cfg,world_size=1)


def test_cli_static_preflight_requires_the_patched_model_method():
    import importlib.util
    spec=importlib.util.spec_from_file_location('rlt_training_cli',PROJECT/'scripts/train_wall_rlt.py')
    cli=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    report=cli.validate_wall_source(PROJECT/'.cache/wall-x-rlt')
    assert len(report['sha256'])==64
    with pytest.raises(ValueError,match='lacks the RLT'):
        cli.validate_wall_source(PROJECT/'wall-x')


def test_eval_subset_is_fixed_unique_and_covers_whole_dev_range():
    data=FixtureData()
    subset,report=select_eval_dataset(data,4)
    other,_=select_eval_dataset(data,4)
    assert subset.indices==other.indices
    assert len(subset.indices)==len(set(subset.indices))==4
    assert subset.indices[0]==0 and subset.indices[-1]==len(data)-1
    assert report['full_windows']==6 and report['evaluated_windows']==4
    assert 'smoke_subset' in report['subset_strategy']
    loader=StatefulDataLoader(subset,batch_size=3,drop_last=False)
    values=torch.cat(list(loader)).tolist()
    assert values==subset.indices  # Partial final batch is kept, without duplication.


@pytest.mark.parametrize('maximum',[None,6,100])
def test_eval_limit_cannot_invent_additional_dev_rows(maximum):
    data=FixtureData()
    result,report=select_eval_dataset(data,maximum)
    assert result is data and report['evaluated_windows']==6
    assert report['subset_strategy']=='all_dev_windows'


@pytest.mark.parametrize('maximum',[0,-1,True,1.5])
def test_eval_limit_rejects_invalid_values(maximum):
    with pytest.raises(ValueError,match='positive integer'):
        select_eval_dataset(FixtureData(),maximum)
