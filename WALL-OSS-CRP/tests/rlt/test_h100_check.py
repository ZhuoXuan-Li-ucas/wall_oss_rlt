import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf


ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('rlt_h100_check',ROOT/'scripts/check_wall_rlt_h100.py')
check=importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.fixture(autouse=True)
def restore_cli_process_state(monkeypatch):
    monkeypatch.setattr(check.sys,'path',list(check.sys.path))
    for key in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','PYTHONPATH','RLINF_EXT_MODULE','HF_HUB_OFFLINE'):
        monkeypatch.setenv(key,os.environ.get(key,''))


def records():
    return [{'index':3,'name':'NVIDIA H100 80GB HBM3','uuid':'GPU-allocated',
             'total_memory_mib':81920,'free_memory_mib':70000,'driver_version':'570.86.10'},
            {'index':0,'name':'NVIDIA GeForce RTX 3060 Ti','uuid':'GPU-local',
             'total_memory_mib':8192,'free_memory_mib':8192,'driver_version':'580.0'}]


@pytest.mark.parametrize('selector',['3','GPU-allocated'])
def test_selects_exact_allocated_h100_without_auto_selection(selector):
    result=check.select_h100(records(),selector)
    assert result['uuid']=='GPU-allocated'
    assert result['index']==3


def two_h100s():
    first=records()[0]
    return [first,dict(first,index=5,uuid='GPU-allocated-two')]


def test_multiple_selections_must_match_world_size_and_be_distinct():
    assert [gpu['index'] for gpu in check.select_h100s(two_h100s(),['3','GPU-allocated-two'],2)]==[3,5]
    with pytest.raises(ValueError,match='different'):
        check.select_h100s(two_h100s(),['3','GPU-allocated'],2)
    with pytest.raises(ValueError,match='equal --world-size'):
        check.select_h100s(two_h100s(),['3'],2)
    assert check.select_h100s(two_h100s(),[],2)==[]


@pytest.mark.parametrize('selector',[None,'','0','GPU-local','4','3,0','GPU-alloc'])
def test_rejects_missing_other_or_ambiguous_gpu(selector):
    with pytest.raises(ValueError): check.select_h100(records(),selector)


def test_missing_nvidia_smi_has_readable_error(monkeypatch):
    def missing(*args,**kwargs): raise FileNotFoundError('not installed')
    monkeypatch.setattr(check.subprocess,'run',missing)
    with pytest.raises(RuntimeError,match='nvidia-smi is unavailable'):
        check.query_gpus()


def test_gpu_inventory_is_read_only_and_parses_numeric_units(monkeypatch):
    def run(command,**kwargs):
        assert command==['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free,driver_version',
                          '--format=csv,noheader,nounits']
        return SimpleNamespace(stdout='3, NVIDIA H100 80GB HBM3, GPU-allocated, 81920, 70000, 570.86.10\n')
    monkeypatch.setattr(check.subprocess,'run',run)
    assert check.query_gpus()==records()[:1]


def fake_torch(uuid='GPU-allocated',name='NVIDIA H100',count=1):
    return SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda:True,device_count=lambda:count,is_bf16_supported=lambda:True,
        mem_get_info=lambda _:(70*1024**3,80*1024**3),
        get_device_properties=lambda _:SimpleNamespace(name=name,uuid=uuid,total_memory=80*1024**3)))


def test_worker_revalidates_the_same_uuid_before_model_allocation():
    report=check.verify_runtime_gpu(fake_torch(),records()[0])
    assert report['uuid']=='GPU-allocated'
    with pytest.raises(RuntimeError,match='UUID differs'):
        check.verify_runtime_gpu(fake_torch(uuid='GPU-another'),records()[0])
    with pytest.raises(RuntimeError,match='non-H100'):
        check.verify_runtime_gpu(fake_torch(name='RTX 3060 Ti'),records()[0])
    with pytest.raises(RuntimeError,match='exactly one'):
        check.verify_runtime_gpu(fake_torch(count=2),records()[0])
    with pytest.raises(RuntimeError,match='free-memory floor'):
        check.verify_runtime_gpu(fake_torch(),records()[0],minimum_free_bytes=75*1024**3)


def test_memory_admission_does_not_mistake_storage_lower_bound_for_fit():
    estimate={'persistent_per_rank_bytes':73*1024**3,
              'bf16_whole_model_working_copy_bytes_excluded':9*1024**3}
    result=check.memory_gate(estimate,records()[0])
    assert result['minimum_free_gib']==86
    assert result['admission_passed'] is False
    assert result['capacity_fit_verified'] is False
    assert check.memory_gate(estimate)['admission_passed'] is None


def test_single_step_overrides_do_not_mutate_training_baseline(tmp_path):
    baseline=OmegaConf.load(ROOT/'configs/rlt/stage1.yaml')
    before=OmegaConf.to_container(baseline,resolve=True)
    candidate,changes=check.prepare_single_step_config(baseline,records()[0],tmp_path)
    assert OmegaConf.to_container(baseline,resolve=True)==before
    assert candidate.actor.global_batch_size==candidate.actor.micro_batch_size==1
    assert candidate.actor.optim.lr_warmup_steps==0
    assert candidate.cluster.component_placement.actor==3
    assert candidate.data.val_data_paths is None
    assert candidate.runner.max_steps==1
    assert candidate.actor.model.rlt==baseline.actor.model.rlt
    assert candidate.actor.fsdp_config==baseline.actor.fsdp_config
    assert changes['actor.optim.lr_warmup_steps']==0


@pytest.mark.parametrize('key,value',[
    ('actor.model.initialization','resume'),
    ('actor.fsdp_config.sharding_strategy','full_shard'),
    ('actor.fsdp_config.use_orig_params',False),
    ('actor.fsdp_config.grad_scaler.enabled',True),
    ('actor.model.vision_attention','eager'),
])
def test_incompatible_probe_configuration_rejected(key,value,tmp_path):
    cfg=OmegaConf.load(ROOT/'configs/rlt/stage1.yaml')
    OmegaConf.update(cfg,key,value)
    with pytest.raises(ValueError): check.prepare_single_step_config(cfg,records()[0],tmp_path)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wall=nn.Linear(2,3)
        self.rlt_module=nn.Linear(3,1)

    def forward(self,x): return self.rlt_module(self.wall(x))


def test_actual_optimizer_probes_show_finite_gradients_and_parameter_change():
    torch.manual_seed(1)
    model=SmallModel()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    model(torch.ones(2,2)).square().mean().backward()
    probes=check.gradient_probes(model)
    optimizer.step()
    result=check.finish_probes(probes)
    assert set(result)=={'wall','rlt'}
    assert all(group['gradient_l2']>0 and group['absolute_change']>0 for group in result.values())
    assert all('parameter' not in group['parameter_change_probe'] for group in result.values())


def test_zero_lr_or_nonfinite_gradients_cannot_be_reported_as_update_pass():
    model=SmallModel()
    model(torch.ones(2,2)).square().mean().backward()
    probes=check.gradient_probes(model)
    with pytest.raises(ValueError,match='did not change'):
        check.finish_probes(probes)
    model.wall.weight.grad[0,0]=float('nan')
    with pytest.raises(ValueError,match='non-finite'):
        check.gradient_probes(model)


def test_missing_run_gpu_rejected_before_creating_output(tmp_path,monkeypatch):
    path=tmp_path/'new'
    monkeypatch.setattr(check.sys,'argv',['check','--run-single-step','--output',str(path)])
    with pytest.raises(SystemExit) as exc: check.main()
    assert exc.value.code==2
    assert not path.exists()


def test_multirank_runtime_probes_rejected_before_creating_output(tmp_path,monkeypatch):
    path=tmp_path/'new'
    monkeypatch.setattr(check.sys,'argv',['check','--run-single-step','--world-size','2',
                        '--gpu','3','--gpu','5','--output',str(path)])
    with pytest.raises(SystemExit) as exc: check.main()
    assert exc.value.code==2
    assert not path.exists()


def test_existing_output_is_never_overwritten(tmp_path,monkeypatch):
    marker=tmp_path/'untouched';marker.write_text('old result')
    monkeypatch.setattr(check.sys,'argv',['check','--output',str(tmp_path)])
    with pytest.raises(FileExistsError): check.main()
    assert marker.read_text()=='old result'


def test_preflight_failure_saves_report_without_launch(monkeypatch,tmp_path):
    def missing(): raise RuntimeError('nvidia-smi is unavailable')
    def forbidden(*args): raise AssertionError('Ray/model execution must not happen')
    monkeypatch.setattr(check,'query_gpus',missing)
    monkeypatch.setattr(check,'launch_one_update',forbidden)
    output=tmp_path/'new'
    monkeypatch.setattr(check.sys,'argv',['check','--output',str(output)])
    with pytest.raises(SystemExit) as exc: check.main()
    assert exc.value.code==1
    report=json.loads((output/'report.json').read_text())
    assert report['status']=='FAIL'
    assert report['weights_constructed'] is report['ray_started'] is False


def fake_config_preflight(monkeypatch):
    import wall_rlt.workers.sft as worker
    import wall_rlt.model.builder as builder
    import wall_rlt.resources as resources
    contract=SimpleNamespace(fingerprint=lambda:'fixture-contract')
    monkeypatch.setattr(worker,'validate_training_config',lambda cfg,world_size:
                        (SimpleNamespace(checkpoint_path='fixture'),contract))
    monkeypatch.setattr(worker,'source_hashes',lambda:{'fixture.py':'fixture-hash'})
    monkeypatch.setattr(builder,'validate_build_config',lambda cfg:{'weights_loaded':False})
    monkeypatch.setattr(resources,'estimate_training_memory',lambda *args,**kwargs:
                        {'persistent_per_rank_bytes':73*1024**3,
                         'bf16_whole_model_working_copy_bytes_excluded':9*1024**3})


def test_default_preflight_cannot_launch_ray_or_allocate_model(monkeypatch,tmp_path):
    fake_config_preflight(monkeypatch)
    monkeypatch.setattr(check,'query_gpus',records)
    def forbidden(*args): raise AssertionError('execution must remain disabled')
    monkeypatch.setattr(check,'launch_one_update',forbidden)
    output=tmp_path/'preflight'
    monkeypatch.setattr(check.sys,'argv',['check','--output',str(output)])
    check.main()
    report=json.loads((output/'report.json').read_text())
    assert report['status']=='PREFLIGHT_PASS'
    assert report['execution_readiness']=='NO_GPU_SELECTED'
    assert report['weights_constructed'] is report['ray_started'] is False
    assert check.os.environ['CUDA_VISIBLE_DEVICES']==''


def test_explicit_run_rejects_insufficient_budget_before_launch(monkeypatch,tmp_path):
    fake_config_preflight(monkeypatch)
    monkeypatch.setattr(check,'query_gpus',records)
    def forbidden(*args): raise AssertionError('insufficient GPU must not allocate model')
    monkeypatch.setattr(check,'launch_one_update',forbidden)
    output=tmp_path/'under-budget'
    monkeypatch.setattr(check.sys,'argv',['check','--gpu','GPU-allocated','--run-single-step','--output',str(output)])
    with pytest.raises(SystemExit) as exc: check.main()
    assert exc.value.code==1
    report=json.loads((output/'report.json').read_text())
    assert report['status']=='FAIL'
    assert report['memory_gate']['admission_passed'] is False
    assert report['weights_constructed'] is report['ray_started'] is False
    assert check.os.environ['CUDA_VISIBLE_DEVICES']=='3'


def test_multirank_preflight_uses_sharded_budget_and_emits_unexecuted_plan(monkeypatch,tmp_path):
    fake_config_preflight(monkeypatch)
    import wall_rlt.resources as resources
    def estimate(*args,world_size,sharding_strategy):
        assert world_size==2
        assert sharding_strategy=='full_shard'
        return {'persistent_per_rank_bytes':37*1024**3,
                'bf16_whole_model_working_copy_bytes_excluded':9*1024**3}
    monkeypatch.setattr(resources,'estimate_training_memory',estimate)
    monkeypatch.setattr(check,'query_gpus',two_h100s)
    def forbidden(*args): raise AssertionError('preflight must not launch Ray/model')
    monkeypatch.setattr(check,'launch_one_update',forbidden)
    output=tmp_path/'two-gpu'
    monkeypatch.setattr(check.sys,'argv',['check','--config',str(ROOT/'configs/rlt/stage1_2gpu.yaml'),
                        '--world-size','2','--gpu','3','--gpu','GPU-allocated-two','--output',str(output)])
    check.main()
    report=json.loads((output/'report.json').read_text())
    assert report['status']=='PREFLIGHT_PASS'
    assert report['h100_runtime']=='NOT_RUN'
    assert report['memory_gate']['minimum_free_gib']==50
    assert report['memory_gate']['admission_passed'] is True
    assert report['memory_gate']['capacity_fit_verified'] is False
    assert [item['uuid'] for item in report['memory_gate']['per_gpu']]==['GPU-allocated','GPU-allocated-two']
    plan=report['multirank_launch_plan']
    assert plan['status']=='NOT_RUN'
    assert plan['environment']['CUDA_VISIBLE_DEVICES']=='3,5'
    assert [plan['argv'][i+1] for i,value in enumerate(plan['argv']) if value=='--gpu']==['GPU-allocated','GPU-allocated-two']
    assert 'actor.optim.lr_warmup_steps=0' in plan['argv']
    assert report['ray_started'] is report['weights_constructed'] is False
    assert check.os.environ['CUDA_VISIBLE_DEVICES']==''
