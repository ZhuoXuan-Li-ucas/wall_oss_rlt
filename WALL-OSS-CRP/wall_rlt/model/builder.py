"""Build WALL + RLT for RLinf without mutating the base checkpoint or source.

``cfg`` is actor.model. ``cfg.rlt`` contains strict WallRLTConfig fields;
processor_path defaults to rlt.checkpoint_path. initialization is new_transfer
or resume. Resume constructs the same architecture but requires the worker to
restore its full checkpoint before any forward; it never resets projections.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path

import torch

from wall_rlt.config import WallRLTConfig
from wall_rlt.data.contract import DataContract
from wall_rlt.data.transforms import SlotMap
from .ar5_adapter import (AR5_VARIANT, EE_VARIANT, reset_ar5_action_projections,
                          validate_ar5_model_schema, variant_layouts)

# Data target -> model adaptation. Only these explicit pairs are buildable.
SUPPORTED_ADAPTATIONS = {
    ('ar5_joint_next_v1', AR5_VARIANT),
    ('ar5_ee_rel6d_next_v1', EE_VARIANT),
}


def _plain(value):
    from omegaconf import OmegaConf
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value,resolve=True)
    return dict(value)


def _parse_build_config(cfg, torch_dtype=torch.float32):
    cfg = _plain(cfg)
    if cfg.get('model_type') != 'wall_rlt':
        raise ValueError('builder requires model_type=wall_rlt')
    if torch_dtype not in (None,torch.float32) or cfg.get('precision','fp32') not in ('fp32','float32',32):
        raise ValueError('WALL-RLT uses FP32 master parameters; set FSDP mixed precision separately')
    if cfg.get('is_lora',False):
        raise ValueError('the AR5 joint8 baseline does not enable LoRA')
    if cfg.get('enable_torch_compile',False) or cfg.get('enable_cuda_graph',False):
        raise ValueError('compiled/captured training is not supported by this builder')
    rlt = WallRLTConfig.from_dict(_plain(cfg.get('rlt',{})))
    for key in ('checkpoint_path','data_contract_path','wall_train_config'):
        value = getattr(rlt,key)
        if not value:
            raise ValueError(f'rlt.{key} must be an explicit local path')
    rlt = replace(rlt,**{key:str(Path(getattr(rlt,key)).resolve()) for key in
                         ('checkpoint_path','data_contract_path','wall_train_config')})
    checkpoint = Path(rlt.checkpoint_path)
    if not (checkpoint/'config.json').is_file() or not Path(rlt.wall_train_config).is_file():
        raise FileNotFoundError('local base config.json and WALL training YAML are required')
    if (checkpoint/'rlt_config.json').exists():
        raise ValueError('checkpoint_path must name the base WALL artifact, not an already adapted export')
    mode = cfg.get('initialization','new_transfer')
    if mode not in ('new_transfer','resume'):
        raise ValueError('initialization must be new_transfer or resume')
    weight_path = checkpoint/'model.safetensors'
    if not weight_path.is_file():
        raise FileNotFoundError('builder requires an explicit base model.safetensors checkpoint')
    processor = str(Path(cfg.get('processor_path') or checkpoint).resolve())
    if not (Path(processor)/'preprocessor_config.json').is_file():
        raise FileNotFoundError('local processor preprocessor_config.json missing')
    contract = DataContract.load(rlt.data_contract_path)
    contract.require_trainable()
    if (contract.target_mode, contract.model_adapter) not in SUPPORTED_ADAPTATIONS:
        raise ValueError('this builder requires the explicit AR5 joint8 or Wall-OSS-0.5 EE adaptation contract')
    if rlt.action_horizon != contract.action_horizon or rlt.model_action_dim != contract.model_action_dim:
        raise ValueError('RLT horizon/action width disagrees with data contract')
    base = json.loads((checkpoint/'config.json').read_text())
    if base.get('hidden_size') != rlt.input_dim:
        raise ValueError('RLT input_dim must equal the WALL observable-prefix hidden_size')
    if cfg.get('model_path') and Path(cfg['model_path']).resolve() != checkpoint:
        raise ValueError('model_path conflicts with rlt.checkpoint_path')
    for key in ('action_dim','num_action_chunks'):
        if key in cfg and cfg[key] != (rlt.model_action_dim if key=='action_dim' else rlt.action_horizon):
            raise ValueError(f'{key} conflicts with the RLT model interface')
    seed = cfg.get('initialization_seed',0)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('initialization_seed must be an integer in [0,2**63)')
    wall_attn, vision_attn = cfg.get('wall_attention','sdpa'),cfg.get('vision_attention','flash_attention_2')
    if vision_attn == 'sdpa':
        raise ValueError('vision sdpa is incompatible with pinned WALL VisionBlock max_seqlen; use eager or flash_attention_2')
    if wall_attn != 'sdpa' or vision_attn not in ('eager','flash_attention_2'):
        raise ValueError('supported attention: WALL sdpa; vision eager or flash_attention_2')
    return cfg,rlt,contract,processor,mode,seed,wall_attn,vision_attn


def validate_build_config(cfg, torch_dtype=torch.float32):
    """Read only configuration/statistics; allocate no model and read no weights."""
    _,rlt,contract,processor,mode,seed,wall_attn,vision_attn = _parse_build_config(cfg,torch_dtype)
    return {'model_type':'wall_rlt','initialization':mode,'rlt':asdict(rlt),
            'processor_path':processor,'contract_hash':contract.fingerprint(),
            'model_adapter':contract.model_adapter,'initialization_seed':seed,
            'wall_attention':wall_attn,'vision_attention':vision_attn,
            'master_parameter_dtype':'float32','rlt_dtype':'float32',
            'requires_checkpoint_restore':mode=='resume','weights_loaded':False}


def build_target_normalizers(contract):
    """Use the official normalizer with the same slots/quantiles as the collator."""
    from wall_x.data.backends.lerobot.utils import NormStats
    from wall_x.model.core.action.normalizer import Normalizer
    raw = json.loads(Path(contract.normalization_path).read_text())['norm_stats']
    result=[]
    for key,slots,width in ((contract.action_key,contract.action_slots,contract.model_action_dim),
                            (contract.state_key,contract.state_slots,contract.model_state_dim)):
        mapping = SlotMap(slots,width)
        lo,_ = mapping.encode(torch.tensor(raw[key]['q01'],dtype=torch.float32))
        hi,valid = mapping.encode(torch.tensor(raw[key]['q99'],dtype=torch.float32))
        hi = torch.where(valid,hi,torch.ones_like(hi))
        result.append(Normalizer.from_lerobot_norm_stats(NormStats(lo,hi,hi-lo),contract.repo_id))
    return tuple(result)


def _wall_components():
    from wall_x.model.qact.qwen2_5 import Qwen2_5_VLConfig, Qwen2_5_VLMoEForAction
    return Qwen2_5_VLConfig,Qwen2_5_VLMoEForAction


def _require_prefix_patch(model_class):
    parameters = inspect.signature(model_class.train_step_forward).parameters
    if not {'return_rlt_prefix','flow_loss_only'}.issubset(parameters):
        raise RuntimeError('WALL RLT patch missing; put the isolated patched WALL source first in PYTHONPATH')


def _runtime_train_config(rlt,processor,variant=AR5_VARIANT):
    from wall_x._vendor.harrix.utils.train_config import (
        load_train_config_with_ckpt_overlay, normalize_train_config_for_inference,
        strip_action_tokenizer_fields)
    train = normalize_train_config_for_inference(
        load_train_config_with_ckpt_overlay(rlt.wall_train_config,rlt.checkpoint_path),rlt.wall_train_config)
    strip_action_tokenizer_fields(train)
    action_layout,state_layout = variant_layouts(variant)
    train.update(processor_path=processor,dof_config=dict(action_layout),
                 agent_pos_config=dict(state_layout),ar_dof_config=dict(action_layout),
                 ar_loss_weight=0.0,flow_loss_weight=1.0,use_state_string_representation=True,state_bins=512)
    train.setdefault('data',{}).update(use_state_string_representation=True,state_bins=512,
                                      action_horizon_flow=rlt.action_horizon,generate_subtask_ratio=0.0)
    return train


def _checkpoint_vocab(checkpoint):
    from safetensors import safe_open
    # Metadata access only; the embedding matrix is not loaded here.
    with safe_open(str(Path(checkpoint)/'model.safetensors'),framework='pt',device='cpu') as handle:
        return handle.get_slice('model.embed_tokens.weight').get_shape()[0]


def _load_base_weights(wall,checkpoint,model_class):
    from wall_x._vendor.harrix.utils.ckpt_load import load_state_dict
    state = load_state_dict(checkpoint,model_class)
    return load_base_state_strict(wall,state)


def _zero_padded_mlp_shapes(wall):
    """Checkpoint shapes of MLPs that wall-x zero-pads on load (vision 3420 -> 3456).

    Its _save_to_state_dict crop runs before the child Linear keys exist, so
    wall.state_dict() reports the padded width while released weights do not.
    """
    shapes = {}
    for name,module in wall.named_modules():
        width = getattr(module,'padded_intermediate_size',None)
        if width is None or width == module.intermediate_size:
            continue
        inner,hidden = module.intermediate_size,module.hidden_size
        shapes[f'{name}.gate_up_proj.weight'] = (2*inner,hidden)
        shapes[f'{name}.down_proj.weight'] = (hidden,inner)
        if module.gate_up_proj.bias is not None:
            shapes[f'{name}.gate_up_proj.bias'] = (2*inner,)
    return shapes


def _require_zero_padding(key,loaded,source):
    source = source.to(loaded.dtype)
    if key.endswith('down_proj.weight'):
        inner = source.shape[1]
        kept,tail = [(loaded[:,:inner],source)],[loaded[:,inner:]]
    else:
        inner,width = source.shape[0]//2,loaded.shape[0]//2
        kept = [(loaded[:inner],source[:inner]),(loaded[width:width+inner],source[inner:])]
        tail = [loaded[inner:width],loaded[width+inner:]]
    if not all(torch.equal(a,b) for a,b in kept) or any(t.any() for t in tail):
        raise ValueError(f'zero-padded base checkpoint tensor loaded incorrectly: {key}')


def load_base_state_strict(wall,state):
    """Discard only obsolete base dataset stats; never crop/pad learned weights.

    The one exception is wall-x's own MLP zero-padding, verified after loading.
    """
    state = dict(state)
    obsolete = [k for k in state if k.startswith((
        'action_preprocessor.normalizer_action.','action_preprocessor.normalizer_propri.'))]
    for key in obsolete:
        state.pop(key)
    # HF safetensors omits the tied lm_head copy. Explicitly reconstruct that
    # alias only when both the config and the actual parameters are tied.
    tied = getattr(wall.config,'tie_word_embeddings',False)
    if tied and 'lm_head.weight' not in state and 'model.embed_tokens.weight' in state:
        if wall.lm_head.weight is not wall.model.embed_tokens.weight:
            raise ValueError('checkpoint omits lm_head but model embedding weights are not tied')
        state['lm_head.weight'] = state['model.embed_tokens.weight']
    expected = wall.state_dict()
    if set(state) != set(expected):
        raise ValueError(f'base checkpoint keys differ: missing={sorted(set(expected)-set(state))}, '
                         f'unexpected={sorted(set(state)-set(expected))}')
    padded = {k for k,shape in _zero_padded_mlp_shapes(wall).items() if tuple(state[k].shape) == shape}
    mismatched = [k for k in expected if k not in padded and state[k].shape != expected[k].shape]
    if mismatched:
        raise ValueError(f'base checkpoint tensor shapes differ: {mismatched}')
    wall.load_state_dict(state,strict=True)
    loaded = wall.state_dict()
    for key in sorted(padded):
        _require_zero_padding(key,loaded[key],state[key])
    return {'loaded_parameter_keys':len(state),'discarded_base_normalizer_keys':len(obsolete),
            'zero_padded_mlp_keys':len(padded)}


def build_wall_rlt(cfg,torch_dtype=torch.float32):
    """RLinf model factory. Full-size calls belong on the allocated training host."""
    _,rlt,contract,processor_path,mode,seed,wall_attn,vision_attn = _parse_build_config(cfg,torch_dtype)
    report = validate_build_config(cfg,torch_dtype)
    ConfigClass,ModelClass = _wall_components()
    _require_prefix_patch(ModelClass)
    from wall_x._vendor.harrix.utils.train_config import build_model_config
    from wall_x.trainer.trainer_utils import load_wallx_processors
    train = _runtime_train_config(rlt,processor_path,contract.model_adapter)
    model_config = build_model_config(ConfigClass,rlt.checkpoint_path,train)
    model_config._attn_implementation = wall_attn
    model_config.vision_config._attn_implementation = vision_attn
    model_config.proj_with_mask = True
    model_config.use_lora = False
    model_config.use_cache = False
    model_config.ar_loss_weight = 0.0
    model_config.flow_loss_weight = 1.0
    model_config.vocab_size = _checkpoint_vocab(rlt.checkpoint_path)
    norm_action,norm_state = build_target_normalizers(contract)
    procs = load_wallx_processors(train,normalizer=norm_action,device='cpu')
    processor = procs['processor']
    if procs.get('tokenizer_mixin') is not None:
        raise ValueError('joint8 RLT requires flow-only processor without an action tokenizer')
    if len(processor.tokenizer) > model_config.vocab_size:
        raise ValueError('processor vocabulary exceeds the frozen checkpoint embedding matrix')
    action_token = processor.tokenizer.convert_tokens_to_ids('<|action|>')
    if action_token != rlt.action_token_id:
        raise ValueError('processor action token differs from RLT config')
    wall = ModelClass(model_config,processor,None)
    wall.float()
    validate_ar5_model_schema(wall,contract.model_adapter)
    if mode == 'new_transfer':
        report['base_loading'] = _load_base_weights(wall,rlt.checkpoint_path,ModelClass)
        if contract.model_adapter == AR5_VARIANT:
            report['adapter_initialization'] = reset_ar5_action_projections(wall,seed=seed)
        else:
            # Same slot semantics as pretraining: keep the learned projections.
            report['adapter_initialization'] = {
                'variant':contract.model_adapter,'reset_parameters':[],
                'initialization':'pretrained_action_projections_kept',
                'target_normalizers_required':True,'training_required':True}
        report['weights_loaded'] = True
    wall.set_normalizer(norm_action,norm_state)
    from .policy import WallRLTPolicy
    policy = WallRLTPolicy(wall,rlt)
    policy.data_contract = contract
    policy._rlt_requires_checkpoint_restore = mode == 'resume'
    source_path = Path(inspect.getfile(ModelClass)).resolve()
    report['wall_model_source'] = str(source_path)
    report['wall_model_source_sha256'] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    policy.build_report = report
    policy.train()
    return policy
