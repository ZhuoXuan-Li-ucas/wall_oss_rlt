"""RLinf worker extension: RLINF_EXT_MODULE=wall_rlt.rlinf_ext."""
from __future__ import annotations


def _build(cfg,torch_dtype):
    # Registration must not import WALL, load weights, or allocate RLT modules.
    from wall_rlt.model.builder import build_wall_rlt
    return build_wall_rlt(cfg,torch_dtype)


def register():
    from rlinf.models import _MODEL_REGISTRY, register_model
    existing = _MODEL_REGISTRY.get('wall_rlt')
    if existing is _build:
        return
    if existing is not None:
        raise RuntimeError('wall_rlt is already registered by another extension')
    register_model('wall_rlt',_build,category='embodied')
