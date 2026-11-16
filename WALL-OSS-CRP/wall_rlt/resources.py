"""Static training-state memory estimates; never allocate weights or use a GPU."""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct


def estimate_training_memory(checkpoint_path, rlt_config, *, world_size=1,
                             sharding_strategy='no_shard'):
    """Estimate FP32 parameter/gradient/Adam storage from headers and meta RLT.

    This is a state-storage lower bound, not a fit guarantee or measured peak.
    The optional BF16 whole-model copy is reported separately; actual FSDP
    materialization depends on wrapping and collectives. GPU allocation must be
    explicitly established elsewhere before using any training entry point.
    """
    import torch
    from wall_rlt._vendor.rlt_token_transformer import RLTTokenTransformer
    from wall_rlt.config import WallRLTConfig
    if type(world_size) is not int or world_size < 1:
        raise ValueError('world_size must be a positive integer')
    if sharding_strategy not in {'no_shard','full_shard'}:
        raise ValueError('static estimator supports only no_shard/full_shard')
    if not isinstance(rlt_config,WallRLTConfig):
        rlt_config=WallRLTConfig.from_dict(dict(rlt_config))
    checkpoint=Path(checkpoint_path)
    weight_file=checkpoint/'model.safetensors' if checkpoint.is_dir() else checkpoint
    with weight_file.open('rb') as stream:
        prefix=stream.read(8)
        if len(prefix)!=8:
            raise ValueError('truncated safetensors header length')
        size=struct.unpack('<Q',prefix)[0]
        if size<2 or size>64*1024**2:
            raise ValueError('unexpected safetensors header size')
        raw=stream.read(size)
        if len(raw)!=size:
            raise ValueError('truncated safetensors header')
    header=json.loads(raw)
    excluded_prefixes=('action_preprocessor.normalizer_action.',
                       'action_preprocessor.normalizer_propri.')
    shapes={key:value['shape'] for key,value in header.items()
            if key!='__metadata__' and not key.startswith(excluded_prefixes)}
    if not shapes or any(any(type(d) is not int or d<0 for d in shape) for shape in shapes.values()):
        raise ValueError('invalid parameter shapes in safetensors header')
    wall_count=sum(math.prod(shape) for shape in shapes.values())
    with torch.device('meta'):
        rlt=RLTTokenTransformer(**rlt_config.token_kwargs())
    rlt_count=sum(parameter.numel() for parameter in rlt.parameters())
    total=wall_count+rlt_count
    components={key:total*4 for key in ('fp32_parameters','fp32_gradients',
                                       'adam_first_moment','adam_second_moment')}
    divisor=world_size if sharding_strategy=='full_shard' else 1
    per_rank={key:math.ceil(size/divisor) for key,size in components.items()}
    return {'schema_version':1,'scope':'static state-storage lower bound; no GPU execution',
            'weight_file':str(weight_file.resolve()),'checkpoint_header_bytes_read':size+8,
            'wall_serialized_scalars_excluding_normalizers':wall_count,
            'rlt_parameter_count':rlt_count,'total_scalar_count':total,
            'components_bytes_unsharded':components,
            'unsharded_state_bytes':sum(components.values()),
            'unsharded_state_gib':sum(components.values())/1024**3,
            'world_size':world_size,'sharding_strategy':sharding_strategy,
            'components_bytes_per_rank':per_rank,
            'persistent_per_rank_bytes':sum(per_rank.values()),
            'persistent_per_rank_gib':sum(per_rank.values())/1024**3,
            'bf16_whole_model_working_copy_bytes_excluded':total*2,
            'bf16_whole_model_working_copy_gib_excluded':total*2/1024**3,
            'assumptions':['all serialized WALL non-normalizer scalars are trainable',
                           'FP32 master parameters and gradients; two FP32 Adam states',
                           'full_shard divides persistent states evenly; no_shard replicates',
                           'allocator and FSDP implementation may use additional storage'],
            'not_included':['BF16 working parameters','activations','Adam update temporaries',
                            'FSDP collective and flattened-parameter buffers','CUDA workspaces',
                            'allocator fragmentation/reserve','other processes'],
            'measured_gpu_peak_bytes':None,'capacity_fit_verified':False}
