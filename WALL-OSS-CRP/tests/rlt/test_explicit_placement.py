from types import SimpleNamespace
from omegaconf import OmegaConf
import pytest
from wall_rlt.placement import bind_allocated_gpus,verify_worker_allocation


def fixture():
    cfg=OmegaConf.create({'cluster':{'num_nodes':1,'component_placement':{'actor':'0-1'}}})
    gpus=[{'index':3,'uuid':'GPU-three','name':'NVIDIA H100'},
          {'index':5,'uuid':'GPU-five','name':'NVIDIA H100'}]
    return cfg,gpus


def test_physical_devices_replace_logical_zero_one_placeholders():
    cfg,gpus=fixture()
    chosen=bind_allocated_gpus(cfg,gpus,['GPU-five','3'],2)
    assert cfg.cluster.component_placement.actor=='3,5'
    assert [g['index'] for g in chosen]==[3,5]
    torch=SimpleNamespace(cuda=SimpleNamespace(current_device=lambda:0,
          get_device_properties=lambda n:SimpleNamespace(uuid='five',name='NVIDIA H100')))
    verify_worker_allocation(cfg,1,torch)
    with pytest.raises(RuntimeError,match='refusing model'):
        verify_worker_allocation(cfg,0,torch)


@pytest.mark.parametrize('selectors,size',[(['3'],2),(['3','GPU-three'],2),(['0'],1)])
def test_unallocated_duplicate_or_wrong_count_is_rejected(selectors,size):
    cfg,gpus=fixture()
    with pytest.raises(ValueError):bind_allocated_gpus(cfg,gpus,selectors,size)


def test_main_binds_physical_gpu_order_before_first_torch_import():
    """A fresh process intercepts torch before CUDA can observe stale settings."""
    from pathlib import Path
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[2]
    script=r'''
import importlib.abc, os, runpy, sys, types
os.environ['CUDA_DEVICE_ORDER']='FASTEST_FIRST'
os.environ['CUDA_VISIBLE_DEVICES']='0'
query=types.ModuleType('check_wall_rlt_h100')
query.query_gpus=lambda:[{'index':3,'uuid':'GPU-three','name':'NVIDIA H100'},
                         {'index':5,'uuid':'GPU-five','name':'NVIDIA H100'}]
sys.modules['check_wall_rlt_h100']=query
class BoundBeforeTorch(Exception):pass
class Probe(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='torch':
            assert os.environ['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
            assert os.environ['CUDA_VISIBLE_DEVICES']=='3,5'
            raise BoundBeforeTorch()
assert 'torch' not in sys.modules
sys.meta_path.insert(0,Probe())
sys.argv=['scripts/train_wall_rlt.py','--validate-only','--world-size','2',
          '--config','configs/rlt/stage1_2gpu.yaml','--gpu','5','--gpu','3']
try:runpy.run_path('scripts/train_wall_rlt.py',run_name='__main__')
except BoundBeforeTorch:print('bound-before-torch')
else:raise AssertionError('the torch import boundary was not reached')
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=root,
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='bound-before-torch'
