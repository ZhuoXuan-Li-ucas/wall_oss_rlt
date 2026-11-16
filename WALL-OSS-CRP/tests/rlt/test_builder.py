from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wall_rlt.data.contract import DataContract
from wall_rlt.model import builder
from wall_rlt.model.ar5_adapter import AR5_ACTION_LAYOUT,AR5_STATE_LAYOUT


@pytest.fixture
def build_cfg(tmp_path):
    base=tmp_path/'base'; base.mkdir()
    (base/'config.json').write_text(json.dumps({'hidden_size':8}))
    (base/'model.safetensors').write_bytes(b'fixture; weight loader is replaced in orchestration tests')
    (base/'preprocessor_config.json').write_text('{}')
    (base/'config.yml').write_text('model_type: qwen2_5\n')
    stats={'train_episode_ids':[1], 'target_mode':'ar5_joint_next_v1',
           'norm_stats':{k:{'q01':[0.0]*8,'q99':[1.,2.,0.,4.,0.,6.,7.,1.]} for k in ('state','action')}}
    norm=tmp_path/'norm.json'; norm.write_text(json.dumps(stats))
    contract=DataContract(root=str(tmp_path),repo_id='fixture/ar5',fps=30,
        action_names=[f'q{i}' for i in range(7)]+['command'],state_names=[f'q{i}' for i in range(7)]+['opening'],
        camera_mapping={'cam_high':'top','cam_front':'wrist'},splits={'train':[1],'dev':[2],'test':[]},
        state_key='state',action_slots=list(range(8)),state_slots=list(range(8)),
        action_units=['rad']*7+['binary'],state_units=['rad']*7+['ratio'],
        action_convention='absolute',coordinate_frame='AR5 axes',semantics_confirmed=True,
        normalization_path=str(norm),normalization_sha256=hashlib.sha256(norm.read_bytes()).hexdigest(),
        target_mode='ar5_joint_next_v1',model_adapter='ar5_joint8_reinit_v1',
        alignment='observation_t_target_t_plus_1_drop_last')
    cp=tmp_path/'contract.json';cp.write_text(json.dumps(asdict(contract)))
    return {'model_type':'wall_rlt','precision':'fp32','is_lora':False,'initialization':'new_transfer',
            'rlt':{'input_dim':8,'embed_dim':8,'num_heads':2,'num_layers':1,'prefix_seq_len':8,
                   'action_token_id':3,'checkpoint_path':str(base),'data_contract_path':str(cp),
                   'wall_train_config':str(base/'config.yml')}}


def test_validation_does_not_import_model_or_load_weights(build_cfg,monkeypatch):
    def forbidden():
        raise AssertionError('validation attempted model construction')
    monkeypatch.setattr(builder,'_wall_components',forbidden)
    report=builder.validate_build_config(build_cfg)
    assert report['weights_loaded'] is False
    assert report['contract_hash']


@pytest.mark.parametrize('key,value,error',[
    ('precision','bf16','FP32'),('is_lora',True,'LoRA'),
    ('initialization','restart','initialization'),('wall_attention','eager','attention'),
    ('vision_attention','sdpa','max_seqlen'),
    ('num_action_chunks',31,'conflicts'),('action_dim',8,'conflicts'),
    ('model_path','/not-the-base','conflicts'),('enable_cuda_graph',True,'captured'),
])
def test_conflicting_options_fail_early(build_cfg,key,value,error):
    build_cfg[key]=value
    with pytest.raises(ValueError,match=error): builder.validate_build_config(build_cfg)


def test_prefix_dim_and_unknown_rlt_keys_rejected(build_cfg):
    build_cfg['rlt']['input_dim']=16
    with pytest.raises(ValueError,match='hidden_size'):builder.validate_build_config(build_cfg)
    build_cfg['rlt']['typo_dim']=8
    with pytest.raises(ValueError,match='unknown'):builder.validate_build_config(build_cfg)


class TinyWall(nn.Module):
    def __init__(self,config,processor=None,tokenizer_mixin=None):
        super().__init__();self.config=config
        self.model=nn.Module();self.model.embed_tokens=nn.Embedding(config.vocab_size,8)
        self.lm_head=nn.Linear(8,config.vocab_size,bias=False)
        self.lm_head.weight=self.model.embed_tokens.weight
        proc=nn.Module();self.action_preprocessor=proc
        proc.action_dim=proc.propri_dim=26
        proc.action_hidden_size=4
        proc.dof_config=dict(AR5_ACTION_LAYOUT);proc.agent_pos_config=dict(AR5_STATE_LAYOUT)
        proc.w1=nn.Linear(52,4,bias=False);proc.action_proj_back=nn.Linear(4,26,bias=False)

    def train_step_forward(self,return_rlt_prefix=False,flow_loss_only=False):pass

    def set_normalizer(self,action,state):
        self.action_preprocessor.normalizer_action=action
        self.action_preprocessor.normalizer_propri=state


def config_fixture():
    return SimpleNamespace(dof_config=dict(AR5_ACTION_LAYOUT),agent_pos_config=dict(AR5_STATE_LAYOUT),
        use_state_string_representation=True,proj_with_mask=True,action_hidden_size=4,
        initializer_range=.02,tie_word_embeddings=True,vocab_size=16,vision_config=SimpleNamespace())


@pytest.fixture
def mock_wall_stack(monkeypatch):
    from wall_x._vendor.harrix.utils import train_config
    from wall_x.trainer import trainer_utils
    class Tokenizer:
        def __len__(self):return 16
        def convert_tokens_to_ids(self,token):return 3
    monkeypatch.setattr(builder,'_wall_components',lambda:(object,TinyWall))
    monkeypatch.setattr(builder,'_runtime_train_config',lambda *_:{})
    monkeypatch.setattr(builder,'_checkpoint_vocab',lambda _:16)
    monkeypatch.setattr(train_config,'build_model_config',lambda *_:config_fixture())
    monkeypatch.setattr(trainer_utils,'load_wallx_processors',lambda *a,**kw:{'processor':SimpleNamespace(tokenizer=Tokenizer())})
    calls=[]
    def load(wall,path,cls):
        state={k:torch.full_like(v,0.5) for k,v in wall.state_dict().items()}
        calls.append(path)
        return builder.load_base_state_strict(wall,state)
    monkeypatch.setattr(builder,'_load_base_weights',load)
    return calls


def test_new_transfer_loads_backbone_resets_only_action_and_attaches_target_stats(build_cfg,mock_wall_stack):
    policy=builder.build_wall_rlt(build_cfg)
    assert len(mock_wall_stack)==1
    assert policy.build_report['adapter_initialization']['training_required']
    assert torch.all(policy.wall.model.embed_tokens.weight==.5)
    assert not torch.all(policy.wall.action_preprocessor.w1.weight==.5)
    assert not policy._rlt_requires_checkpoint_restore
    assert all(p.dtype==torch.float32 for p in policy.rlt_module.parameters())
    norm=policy.wall.action_preprocessor.normalizer_action
    assert list(norm.min.keys())==['fixture/ar5']
    assert norm.min['fixture/ar5'].shape==(26,)
    assert norm.delta['fixture/ar5'][2]==0
    physical=torch.zeros(1,1,26)
    assert torch.isfinite(norm.normalize_data(physical,['fixture/ar5'])).all()


def test_resume_never_loads_base_or_reinitializes_action(build_cfg,mock_wall_stack,monkeypatch):
    build_cfg['initialization']='resume'
    def forbidden(*a,**kw):raise AssertionError('resume must not reset projections')
    monkeypatch.setattr(builder,'reset_ar5_action_projections',forbidden)
    policy=builder.build_wall_rlt(build_cfg)
    assert mock_wall_stack==[]
    assert policy._rlt_requires_checkpoint_restore
    assert policy.build_report['requires_checkpoint_restore']


def test_strict_loading_ignores_only_old_stats_and_restores_tied_weight():
    wall=TinyWall(config_fixture())
    state={k:torch.full_like(v,.2) for k,v in wall.state_dict().items() if k!='lm_head.weight'}
    state['action_preprocessor.normalizer_action.min.old_dataset']=torch.zeros(26)
    report=builder.load_base_state_strict(wall,state)
    assert report['discarded_base_normalizer_keys']==1
    assert torch.all(wall.lm_head.weight==.2)


@pytest.mark.parametrize('fault',['missing','extra','wrong_shape'])
def test_bad_checkpoint_is_rejected_before_any_weight_changes(fault):
    wall=TinyWall(config_fixture())
    before={k:v.clone() for k,v in wall.state_dict().items()}
    state={k:torch.full_like(v,.5) for k,v in before.items()}
    if fault=='missing':state.pop('action_preprocessor.w1.weight')
    elif fault=='extra':state['unrelated.weight']=torch.ones(1)
    else:state['action_preprocessor.w1.weight']=torch.ones(4,51)
    with pytest.raises(ValueError,match='checkpoint'):builder.load_base_state_strict(wall,state)
    for key,value in wall.state_dict().items():torch.testing.assert_close(value,before[key])


def _padded_mlp_wall():
    from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl import Qwen2_5_VLMLP
    class Wall(nn.Module):
        def __init__(self):
            super().__init__()
            self.config=SimpleNamespace(tie_word_embeddings=False)
            self.mlp=Qwen2_5_VLMLP(SimpleNamespace(hidden_size=8,intermediate_size=100,hidden_act='silu'),bias=True)
    return Wall()


def _mlp_state(width):
    return {'mlp.gate_up_proj.weight':torch.randn(2*width,8),'mlp.gate_up_proj.bias':torch.randn(2*width),
            'mlp.down_proj.weight':torch.randn(8,width),'mlp.down_proj.bias':torch.randn(8)}


def test_released_mlp_width_is_zero_padded_exactly():
    # Released WALL stores vision MLPs at 3420; wall-x pads them to 3456 on load.
    torch.manual_seed(0)
    wall=_padded_mlp_wall()
    assert wall.state_dict()['mlp.down_proj.weight'].shape==(8,128)
    state=_mlp_state(100)
    assert builder.load_base_state_strict(wall,state)['zero_padded_mlp_keys']==3
    x=torch.randn(5,8)
    gate,up=(x@state['mlp.gate_up_proj.weight'].T+state['mlp.gate_up_proj.bias']).split(100,-1)
    reference=(torch.nn.functional.silu(gate)*up)@state['mlp.down_proj.weight'].T+state['mlp.down_proj.bias']
    torch.testing.assert_close(wall.mlp(x),reference)


def test_other_mlp_width_is_rejected_before_any_weight_changes():
    torch.manual_seed(0)
    wall=_padded_mlp_wall()
    before={k:v.clone() for k,v in wall.state_dict().items()}
    with pytest.raises(ValueError,match='shapes differ'):builder.load_base_state_strict(wall,_mlp_state(101))
    for key,value in wall.state_dict().items():torch.testing.assert_close(value,before[key])


def test_unpatched_wall_is_rejected():
    class Unpatched:
        def train_step_forward(self,**kwargs):pass
    with pytest.raises(RuntimeError,match='patch missing'):builder._require_prefix_patch(Unpatched)
