"""Explicit hardware placement; pinned RLinf uses physical ranks, not CVD remaps."""
from __future__ import annotations


def bind_allocated_gpus(cfg, records, selectors, world_size):
    from omegaconf import OmegaConf
    if cfg.cluster.num_nodes != 1:
        raise ValueError('explicit hardware binding currently supports one node')
    if len(selectors)!=world_size or world_size<1:
        raise ValueError('number of explicit --gpu selections must equal --world-size')
    selected=[]
    for selector in selectors:
        candidates=[r for r in records if str(r['index'])==str(selector) or r['uuid']==selector]
        if len(candidates)!=1 or 'H100' not in candidates[0]['name'].upper():
            raise ValueError('each explicit GPU must resolve uniquely to an H100')
        selected.append(dict(candidates[0]))
    if len({r['uuid'] for r in selected})!=world_size:
        raise ValueError('the same GPU was selected more than once')
    selected.sort(key=lambda r:r['index']) # RLinf flexible placement uses this order.
    ranks=','.join(str(r['index']) for r in selected)
    OmegaConf.update(cfg,'cluster.component_placement',{'actor':ranks},merge=False)
    OmegaConf.update(cfg,'rlt_allocated_gpus',selected,merge=False)
    return selected


def verify_worker_allocation(cfg, rank, torch_module):
    if not cfg.get('rlt_allocated_gpus'):
        return # Special diagnostic workers provide their own strict check.
    selected=cfg.rlt_allocated_gpus[rank]
    device=torch_module.cuda.current_device()
    props=torch_module.cuda.get_device_properties(device)
    actual=str(getattr(props,'uuid','')).lower().removeprefix('gpu-')
    expected=str(selected.uuid).lower().removeprefix('gpu-')
    if not actual or actual!=expected or 'H100' not in props.name.upper():
        raise RuntimeError('worker GPU UUID/name differs from allocated device; refusing model construction')
