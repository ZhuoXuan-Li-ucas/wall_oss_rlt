"""Prefix helpers and actual tiny WALL CPU integration, without pretrained weights.

The integration subprocess uses 2 decoder layers, hidden sizes 32/16, language
SDPA and vision eager, synthetic text/RGB/action inputs, and an identity normalizer. It exercises both
ordinary and MoE-permuted paths; this is not H100/FlashAttention validation.
"""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace, MethodType

import pytest
import torch

from wall_rlt.model.prefix import observation_prefix_mask, pack_prefix

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'wall_x/model/qact/qwen2_5/modeling_qwen2_5_vl_act.py'


def test_prefix_gathers_observable_tokens_in_order_and_ignores_bad_padding():
    ids=torch.tensor([[0,1,2,9,9],[3,4,5,9,9]])
    attention=torch.tensor([[0,1,1,1,1],[1,1,1,1,1]])
    mask=observation_prefix_mask(ids,attention,9)
    hidden=torch.arange(30,dtype=torch.float32).reshape(2,5,3)
    hidden[0,0]=float('nan'); hidden[:,3:]=float('inf')
    h,m=pack_prefix(hidden,mask)
    assert m.tolist()==[[True,True,False],[True,True,True]]
    torch.testing.assert_close(h[0,:2],hidden[0,1:3])
    assert h[0,2].eq(0).all() and torch.isfinite(h).all()


@pytest.mark.parametrize('ids,attention',[
    ([[1,2,3]],[[1,1,1]]), # no action
    ([[9,1,2]],[[1,1,1]]), # no observable prefix
    ([[1,2,9]],[[1,1,0]]), # only padding action
    ([[1,2,9]],[[1,2,1]]), # nonbinary attention
])
def test_invalid_prefix_fails(ids,attention):
    with pytest.raises(ValueError):
        observation_prefix_mask(torch.tensor(ids),torch.tensor(attention),9)


def test_prefix_overflow_is_not_silently_truncated():
    with pytest.raises(ValueError,match='exceeds'):
        pack_prefix(torch.zeros(1,3,4),torch.ones(1,3,dtype=torch.bool),max_length=2)


@pytest.mark.parametrize('mot_opt',[False,True])
def test_actual_tiny_wall_prefix_and_flow_paths(mot_opt):
    assert (ROOT/'.cache/wall-x-rlt'/SOURCE).is_file(), 'prepare isolated WALL source first'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',
             PYTHONPATH=os.pathsep.join([str(ROOT/'.cache/wall-x-rlt'),str(ROOT)]))
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--tiny-wall-check',str(int(mot_opt))],
                          cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
    assert result.returncode==0, result.stdout+'\n'+result.stderr
    report=json.loads(result.stdout.strip().splitlines()[-1])
    assert report['status']=='PASS' and report['mot_opt']==mot_opt


def tiny_wall(mot_opt):
    from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction,Qwen2_5_VLConfig
    c=json.loads((ROOT/'models/wall-oss-0.5/config.json').read_text())
    c.update(vocab_size=32,hidden_size=32,action_hidden_size=16,state_hidden_size=32,
             intermediate_size=64,num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=2,
             max_position_embeddings=128,dim_inputs=[32,16],
             experts=[{'hidden_size':32,'intermediate_size':64,'hidden_act':'silu'},
                      {'hidden_size':16,'intermediate_size':32,'hidden_act':'silu'}],
             dof_config={'arm':2},agent_pos_config={'arm':2},use_state_string_representation=True,
             proj_with_mask=True,mot_opt=mot_opt,ar_loss_weight=0.,flow_loss_weight=1.,
             _attn_implementation='sdpa',image_token_id=24,video_token_id=25,
             vision_start_token_id=26,vision_end_token_id=27,pad_token_id=0,bos_token_id=1,eos_token_id=2,
             rope_scaling={'type':'mrope','mrope_section':[1,1,2]},use_cache=False,torch_dtype='float32')
    c['vision_config'].update(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
                             out_hidden_size=32,patch_size=2,spatial_patch_size=2,spatial_merge_size=2,
                             window_size=4,fullatt_block_indexes=[0],_attn_implementation='eager')
    tokenizer=SimpleNamespace(pad_token_id=0,convert_tokens_to_ids=lambda name:30 if name=='<|action|>' else 29)
    model=Qwen2_5_VLMoEForAction(Qwen2_5_VLConfig(**c),processor=SimpleNamespace(tokenizer=tokenizer))
    class IdentityNormalizer:
        delta={'fixture':torch.ones(2)}
        def unnormalize_data(self,x,names,dof_mask=None): return x
        def normalize_data(self,x,names): return x
    model.action_preprocessor.normalizer_action=IdentityNormalizer()
    return model


def tiny_batch():
    ids=torch.tensor([[0,3,4,30,30],[5,6,7,30,30]])
    mask=ids.ne(0)
    return {'input_ids':ids,'attention_mask':mask,
            'position_ids':torch.arange(5).reshape(1,1,5).expand(3,2,5),
            'moe_token_types':ids.eq(30),'labels':torch.full_like(ids,-100),
            'action_chunk':torch.randn(2,2,2),'sample_time':torch.tensor([.3,.7]),
            'dof_mask':torch.ones(2,2,2),'flow_loss_mask':torch.ones(2,2),
            'dataset_names':['fixture','fixture'],
            'positional_masks':{'padding_positions':~mask,'valid_flow_action_positions':ids.eq(30)}}


def original_method(name):
    """Run the unmodified pinned method body on the same tiny model/weights."""
    import wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act as module
    tree=ast.parse((ROOT/'wall-x'/SOURCE).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen2_5_VLMoEForAction')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace=dict(vars(module))
    exec(compile(ast.Module(body=[method],type_ignores=[]),f'<pinned-original-{name}>','exec'),namespace)
    return namespace[name]


def exercise_tiny_wall(mot_opt):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model=tiny_wall(mot_opt)
    batch=tiny_batch()
    torch.manual_seed(13)
    train=model(mode='train',return_rlt_prefix=True,flow_loss_only=True,**batch)
    assert train.rlt_prefix_hidden_states.shape==(2,3,32) # prefix, not action expert width 16
    assert train.rlt_prefix_mask.tolist()==[[True,True,False],[True,True,True]]
    assert not train.rlt_prefix_hidden_states.requires_grad
    assert train.logits is None and train.cross_entropy_loss is None
    train.flow_loss.backward()
    assert model.action_preprocessor.w1.weight.grad.abs().sum()>0
    assert model.model.embed_tokens.weight.grad.abs().sum()>0
    # Noise, time and demonstration actions may alter flow loss but must never
    # affect already-observed prefix tokens, including the permuted MoE path.
    changed=copy.deepcopy(batch)
    changed['action_chunk']=torch.randn_like(batch['action_chunk'])*17
    changed['sample_time']=torch.tensor([.9,.1])
    torch.manual_seed(99)
    changed_out=model(mode='train',return_rlt_prefix=True,flow_loss_only=True,**changed)
    torch.testing.assert_close(train.rlt_prefix_hidden_states,changed_out.rlt_prefix_hidden_states,rtol=0,atol=0)
    # True opt-out regression: call the original upstream train method body.
    torch.manual_seed(42)
    old=original_method('train_step_forward')(model,**batch)
    torch.manual_seed(42)
    disabled=model.train_step_forward(**batch)
    torch.testing.assert_close(old.flow_loss,disabled.flow_loss,rtol=0,atol=0)
    assert disabled.rlt_prefix_hidden_states is None
    # Eval loss-only bypasses the vocabulary projection entirely.
    model.eval()
    def forbidden_lm(*args): raise AssertionError('loss-only path invoked lm_head')
    hook=model.lm_head.register_forward_pre_hook(forbidden_lm)
    try:
        validation=model(mode='train',return_rlt_prefix=True,flow_loss_only=True,**batch)
    finally:
        hook.remove()
    assert validation.logits is None and torch.isfinite(validation.flow_loss)
    # Changing padded action values cannot change a valid loss or prefix.
    padded=copy.deepcopy(batch)
    padded['dof_mask'][0,1]=0; padded['flow_loss_mask'][0,1]=0
    padded['positional_masks']['valid_flow_action_positions'][0,-1]=False
    altered=copy.deepcopy(padded); altered['action_chunk'][0,1]=1e4
    torch.manual_seed(24)
    out1=model(mode='train',return_rlt_prefix=True,flow_loss_only=True,**padded)
    torch.manual_seed(24)
    out2=model(mode='train',return_rlt_prefix=True,flow_loss_only=True,**altered)
    torch.testing.assert_close(out1.flow_loss,out2.flow_loss,rtol=0,atol=0)
    torch.testing.assert_close(out1.rlt_prefix_hidden_states,out2.rlt_prefix_hidden_states,rtol=0,atol=0)
    inference={k:v for k,v in batch.items() if k not in {'action_chunk','sample_time','flow_loss_mask'}}
    torch.manual_seed(33)
    generated=model.generate_flow_action(action_horizon=2,action_dim=2,num_inference_timesteps=2,
                                         return_rlt_prefix=True,**inference)
    torch.testing.assert_close(validation.rlt_prefix_hidden_states,generated['rlt_prefix_hidden_states'],rtol=1e-6,atol=1e-6)
    torch.manual_seed(33)
    default=model.generate_flow_action(action_horizon=2,action_dim=2,num_inference_timesteps=2,**inference)
    torch.testing.assert_close(default['predict_action'],generated['predict_action'],rtol=0,atol=0)
    assert set(default)=={'predict_action'}
    # Original inference entry/prefill/finalizer, same model and random noise.
    replaced={name:getattr(model,name) for name in ('generate_flow_action','_generate_flow_action_vanilla','_finalize_flow_action_output')}
    try:
        for name in replaced:
            setattr(model,name,MethodType(original_method(name),model))
        torch.manual_seed(33)
        baseline=model.generate_flow_action(action_horizon=2,action_dim=2,num_inference_timesteps=2,**inference)
    finally:
        for name,method in replaced.items(): setattr(model,name,method)
    torch.testing.assert_close(default['predict_action'],baseline['predict_action'],rtol=0,atol=0)
    # The exact normalizer used by the builder preserves model width even when
    # only 8 of 26 dimensions are active. Decode the slot map exactly once.
    from wall_x.model.core.action.normalizer import Normalizer,unnormalize_data_with_virtual_tail
    from wall_x.data.backends.lerobot.utils import NormStats
    from wall_rlt.data.transforms import SlotMap
    minimum=torch.zeros(26); maximum=torch.ones(26)
    normalizer=Normalizer.from_lerobot_norm_stats(NormStats(minimum,maximum,maximum-minimum),'fixture')
    values=torch.zeros(2,2,26)
    dof=torch.zeros_like(values); dof[...,:8]=1
    physical=unnormalize_data_with_virtual_tail(normalizer,values,['fixture']*2,dof)
    assert physical.shape==(2,2,26)
    torch.testing.assert_close(SlotMap(list(range(8)),26).decode(physical),torch.full((2,2,8),.5))
    # Exercise the actual wrapper and upstream RLT on the real tiny WALL, too.
    from wall_rlt.model.policy import WallRLTPolicy
    from wall_rlt.config import WallRLTConfig
    wrapper=WallRLTPolicy(model,WallRLTConfig(input_dim=32,embed_dim=16,num_layers=1,num_heads=2,
                          mlp_ratio=2,prefix_seq_len=8,action_horizon=2,model_action_dim=2,action_token_id=30))
    wrapper.zero_grad(set_to_none=True)
    losses=wrapper(data={'wall_inputs':batch})
    losses['rlt_loss'].backward()
    assert all(p.grad is None for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in wrapper.rlt_module.encoder.parameters())
    # Real vision module and official image patchification, with two 4x4 RGB
    # images. Upstream vision SDPA lacks the max_seqlen kwarg accepted by its
    # caller, so the CPU vision backend is explicitly eager, not patched here.
    import numpy as np
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    image_processor=Qwen2VLImageProcessor(min_pixels=16,max_pixels=16,patch_size=2,
                                        temporal_patch_size=2,merge_size=2)
    rgb=[Image.fromarray(np.full((4,4,3),color,dtype=np.uint8)) for color in (32,224)]
    images=image_processor(images=rgb,return_tensors='pt')
    ids=torch.tensor([[26,24,27,26,24,27,3,30,30]])
    visual_batch={'input_ids':ids,'attention_mask':torch.ones_like(ids,dtype=torch.bool),
                  'position_ids':torch.arange(9).reshape(1,1,9).expand(3,1,9),
                  'moe_token_types':ids.eq(30),'labels':torch.full_like(ids,-100),
                  'action_chunk':torch.randn(1,2,2),'sample_time':torch.tensor([.4]),
                  'dof_mask':torch.ones(1,2,2),'dataset_names':['fixture'],
                  'positional_masks':{'padding_positions':torch.zeros_like(ids,dtype=torch.bool),
                                      'valid_flow_action_positions':ids.eq(30)},**dict(images)}
    visual_train=model(mode='train',return_rlt_prefix=True,flow_loss_only=True,**visual_batch)
    assert visual_train.rlt_prefix_hidden_states.shape==(1,7,32)
    visual_train.flow_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.visual.parameters())
    visual_inference={k:v for k,v in visual_batch.items() if k not in {'action_chunk','sample_time'}}
    visual_generated=model.generate_flow_action(action_horizon=2,action_dim=2,num_inference_timesteps=2,
                                                return_rlt_prefix=True,**visual_inference)
    torch.testing.assert_close(visual_train.rlt_prefix_hidden_states,visual_generated['rlt_prefix_hidden_states'],
                               rtol=1e-6,atol=1e-6)
    changed_visual=copy.deepcopy(visual_batch)
    changed_visual['pixel_values']=-visual_batch['pixel_values']
    changed_visual_out=model(mode='train',return_rlt_prefix=True,flow_loss_only=True,**changed_visual)
    assert not torch.allclose(visual_train.rlt_prefix_hidden_states,changed_visual_out.rlt_prefix_hidden_states)
    return {'status':'PASS','scope':'actual tiny WALL random weights CPU language SDPA + vision eager; synthetic RGB',
            'mot_opt':mot_opt,'parameters':sum(p.numel() for p in model.parameters()),
            'checks':['backward','prefix isolation from target/time/noise','prefix width and padding',
                      'original training method regression','eval loss-only vocabulary bypass',
                      'invalid action padding isolation','training/prefill prefix parity',
                      'original flow generation regression','normalizer width before slot decode',
                      'actual WALL wrapper and RLT gradient isolation','two RGB images through official processor and vision encoder',
                      'vision gradient and image-sensitive prefix','visual training/prefill prefix parity']}


if __name__=='__main__' and sys.argv[1:2]==['--tiny-wall-check']:
    print(json.dumps(exercise_tiny_wall(bool(int(sys.argv[2])))))
