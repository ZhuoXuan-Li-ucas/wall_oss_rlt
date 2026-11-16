import os
from pathlib import Path
import subprocess
import sys


def test_rlinf_extension_registers_in_fresh_worker_process():
    root=Path(__file__).resolve().parents[2]
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',
               RLINF_EXT_MODULE='wall_rlt.rlinf_ext',
               PYTHONPATH=os.pathsep.join([str(root/'.cache/rlinf-rlt'),str(root)]))
    code='''
import sys
from rlinf.scheduler.cluster.utils import load_user_extension_module
from rlinf.models import _MODEL_REGISTRY, get_model
from rlinf.config import SupportedModel, EMBODIED_MODEL
from omegaconf import OmegaConf
load_user_extension_module()
load_user_extension_module()
assert 'wall_rlt' in _MODEL_REGISTRY
assert SupportedModel('wall_rlt') in EMBODIED_MODEL
assert not any(k.startswith('wall_x') for k in sys.modules)
assert 'wall_rlt.model.builder' not in sys.modules
try:
    get_model(OmegaConf.create({'model_type':'wall_rlt','precision':'fp32','is_lora':False,'rlt':{}}))
except ValueError as exc:
    assert 'explicit local path' in str(exc), str(exc)
else:
    raise AssertionError('registry did not dispatch to strict builder')
print('fresh worker registry and factory dispatch PASS')
'''
    result=subprocess.run([sys.executable,'-c',code],env=env,cwd=root,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'registry and factory dispatch PASS' in result.stdout
