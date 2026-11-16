"""Real tiny WALL/encoder inference, explicitly synthetic and CPU only."""
import json
import hashlib
import importlib
from pathlib import Path
import shutil

import pytest
import torch
from PIL import Image

from test_inference_artifact import checkpoint, refresh_metadata, write
from wall_rlt.config import WallRLTConfig
from wall_rlt.data.contract import DataContract
from wall_rlt.data.collator import WallCollator
from wall_rlt.model.ar5_adapter import AR5_ACTION_LAYOUT, AR5_STATE_LAYOUT
from wall_rlt.model.builder import build_target_normalizers, _checkpoint_vocab
from wall_rlt.model.policy import WallRLTPolicy
from wall_rlt.inference.artifact import export_inference_artifact
from wall_rlt.inference.runtime import (WallRLTInferenceBackend, _load_modules,
                                       load_backend, MODEL_SOURCES, verify_model_sources)

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime(checkpoint,tmp_path):
    return make_runtime(checkpoint,tmp_path)


def make_runtime(checkpoint,tmp_path,action_layout=AR5_ACTION_LAYOUT,state_layout=AR5_STATE_LAYOUT):
    from wall_x.model.qact.qwen2_5 import Qwen2_5_VLConfig, Qwen2_5_VLMoEForAction
    torch.set_num_threads(1)
    assets=checkpoint/'wall_rlt_assets'
    processor=assets/'processor'
    for path in processor.iterdir(): path.unlink()
    from wall_rlt.workers.sft import PROCESSOR_FILES
    for name in PROCESSOR_FILES:
        path=ROOT/'models/wall-oss-0.5'/name
        if path.is_file(): shutil.copyfile(path,processor/name)
    contract=DataContract.load(assets/'data_contract.json')
    collator=WallCollator(contract,processor,image_size=56,max_length=2048)
    c=json.loads((ROOT/'models/wall-oss-0.5/config.json').read_text())
    c.update(vocab_size=_checkpoint_vocab(ROOT/'models/wall-oss-0.5'),hidden_size=32,
             action_hidden_size=16,state_hidden_size=32,intermediate_size=64,
             num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=2,
             max_position_embeddings=2048,dim_inputs=[32,16],
             experts=[{'hidden_size':32,'intermediate_size':64,'hidden_act':'silu'},
                      {'hidden_size':16,'intermediate_size':32,'hidden_act':'silu'}],
             dof_config=dict(action_layout),agent_pos_config=dict(state_layout),
             use_state_string_representation=True,proj_with_mask=True,mot_opt=True,
             ar_loss_weight=0.,flow_loss_weight=1.,_attn_implementation='sdpa',
             rope_scaling={'type':'mrope','mrope_section':[1,1,2]},use_cache=False,torch_dtype='float32')
    c['vision_config'].update(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
        out_hidden_size=32,patch_size=14,spatial_patch_size=14,spatial_merge_size=2,
        window_size=56,fullatt_block_indexes=[0],_attn_implementation='eager')
    rlt=WallRLTConfig(input_dim=32,embed_dim=16,prefix_seq_len=1024,num_layers=1,
        num_heads=2,mlp_ratio=2,action_horizon=32,model_action_dim=26,
        action_token_id=collator.action_token_id)
    rlt.save(assets/'rlt_config.json')
    wall=Qwen2_5_VLMoEForAction(Qwen2_5_VLConfig(**c),collator.native.processor,None)
    wall.set_normalizer(*build_target_normalizers(contract))
    policy=WallRLTPolicy(wall,rlt)
    write(assets/'adapted_wall_config.json',wall.config.to_dict())
    torch.save(policy.state_dict(),checkpoint/'model_state_dict/full_weights.pt')
    refresh_metadata(checkpoint)
    output=tmp_path/'inference'
    manifest=export_inference_artifact(checkpoint,output)
    with pytest.raises(ValueError,match='synthetic'):
        load_backend(output,device='cpu',vision_attention='eager')
    new_contract=DataContract.load(output/'assets/data_contract.json')
    new_collator=WallCollator(new_contract,output/'assets/processor',image_size=56,max_length=2048)
    restored,encoder=_load_modules(output,new_contract,rlt,new_collator,torch.device('cpu'),'eager')
    assert not any(p.is_meta for m in (restored,encoder) for p in m.parameters())
    assert not any(b.is_meta for m in (restored,encoder) for b in m.buffers())
    assert restored.model.embed_tokens.weight.dtype==torch.bfloat16
    assert restored.action_preprocessor.w1.weight.dtype==torch.float32
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    assert all(p.dtype==torch.float32 for p in encoder.parameters())
    return WallRLTInferenceBackend(wall=restored,encoder=encoder,collator=new_collator,
        contract=new_contract,config=rlt,manifest={**manifest,'model_origin':'synthetic_fixture'},
        device='cpu',inference_steps=2)


def observation():
    return {'instruction':'pick up the object','state':[0.]*8,
        'images':{'cam_high':Image.new('RGB',(56,56),(20,30,40)),
                  'cam_front':Image.new('RGB',(56,56),(100,120,140))}}


def test_unvalidated_mixed_precision_path_is_rejected_before_allocation(tmp_path):
    assets=tmp_path/'assets'; assets.mkdir()
    c=json.loads((ROOT/'models/wall-oss-0.5/config.json').read_text())
    c['mot_opt']=False
    write(assets/'adapted_wall_config.json',c)
    with pytest.raises(ValueError,match='mot_opt=True'):
        _load_modules(tmp_path,None,None,None,torch.device('cpu'),'eager')


@pytest.mark.parametrize('alteration',['missing','different'])
def test_recorded_training_model_sources_must_match_installed_code(tmp_path,alteration):
    recorded={}
    for package,names in MODEL_SOURCES.items():
        root=Path(importlib.import_module(package).__file__).resolve().parent
        for name in names:
            recorded[f'{package}/{name}']=hashlib.sha256((root/name).read_bytes()).hexdigest()
    recorded['wall_rlt/workers/sft.py']='unrelated_training_only_version'
    write(tmp_path/'source_hashes.json',recorded)
    assert len(verify_model_sources(tmp_path))==64
    if alteration=='missing':
        recorded.pop('wall_rlt/data/collator.py')
    else:
        recorded['wall_rlt/data/collator.py']='0'*64
    write(tmp_path/'source_hashes.json',recorded)
    with pytest.raises(ValueError,match='data/collator.py'):
        verify_model_sources(tmp_path)


def test_real_tiny_export_meta_restore_and_seeded_visual_inference(runtime):
    before=torch.get_rng_state().clone()
    a=runtime.infer(observation(),return_features=True,seed=7)
    assert torch.equal(torch.get_rng_state(),before)
    b=runtime.infer(observation(),return_features=True,seed=7)
    assert a==b
    assert torch.tensor(a['actions']).shape==(32,8)
    assert len(a['z_rl'])==16 and torch.isfinite(torch.tensor(a['z_rl'])).all()
    assert not runtime.wall.training and not runtime.encoder.training
    assert all(not p.requires_grad for m in (runtime.wall,runtime.encoder) for p in m.parameters())
    assert not any('decoder' in name for name,_ in runtime.encoder.named_modules())
    c=runtime.infer(observation(),return_features=False,seed=7)
    assert c['actions']==a['actions'] and 'z_rl' not in c
    changed=observation();changed['images']['cam_high']=Image.new('RGB',(56,56),(220,210,190))
    d=runtime.infer(changed,return_features=True,seed=7)
    assert not torch.allclose(torch.tensor(a['z_rl']),torch.tensor(d['z_rl']))
    assert runtime.metadata()['action_source']=='WALL_reference_prediction'


def test_observation_runtime_rejects_targets_and_wrong_camera_keys(runtime):
    obs=observation();obs['action']=[0.]*8
    with pytest.raises(ValueError,match='only current'):
        runtime.infer(obs)
    obs=observation();obs['images']['cam_wrist']=obs['images'].pop('cam_front')
    with pytest.raises(ValueError,match='camera keys'):
        runtime.infer(obs)


def test_tiny_model_through_real_http_protocol(runtime):
    import threading
    from wall_rlt.inference.server import create_server
    from wall_rlt.inference.client import InferenceClient
    server=create_server(runtime,port=0)
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
    thread.start()
    try:
        client=InferenceClient(f'http://127.0.0.1:{server.server_port}',timeout=30)
        result=client.infer(observation(),return_features=True,seed=4)
        assert result['robot_executed'] is False and len(result['z_rl'])==16
        assert len(result['actions'])==32 and len(result['actions'][0])==8
        assert result['model_origin']=='synthetic_fixture'
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
