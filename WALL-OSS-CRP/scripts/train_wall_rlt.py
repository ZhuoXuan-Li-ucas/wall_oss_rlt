#!/usr/bin/env python3
"""Validate or launch the WALL-RLT SFT worker through RLinf's official runner."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys


def validate_wall_source(root):
    """Check the optional training interface without importing/allocating WALL."""
    source = Path(root)/'wall_x/model/qact/qwen2_5/modeling_qwen2_5_vl_act.py'
    raw = source.read_bytes()
    tree = ast.parse(raw,filename=str(source))
    model = next((node for node in tree.body if isinstance(node,ast.ClassDef)
                  and node.name=='Qwen2_5_VLMoEForAction'),None)
    forward = next((node for node in model.body if isinstance(node,ast.FunctionDef)
                    and node.name=='train_step_forward'),None) if model else None
    names = {arg.arg for arg in forward.args.args+forward.args.kwonlyargs} if forward else set()
    if not {'return_rlt_prefix','flow_loss_only'}.issubset(names):
        raise ValueError('selected WALL source lacks the RLT prefix/loss-only patch; use the isolated patched checkout')
    return {'source':str(source.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),
            'interface':'Qwen2_5_VLMoEForAction.train_step_forward(return_rlt_prefix, flow_loss_only)',
            'scope':'static interface check; numerical model verification is separate'}


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(project/'configs/rlt/stage1.yaml'))
    parser.add_argument('--rlinf-root',default=str(project/'.cache/rlinf-rlt'))
    parser.add_argument('--wall-source',default=str(project/'.cache/wall-x-rlt'))
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--world-size',type=int,default=1,
                        help='Expected allocated actor GPU count; checked without Ray during preflight')
    parser.add_argument('--gpu',action='append',default=[],
                        help='Explicitly allocated H100 physical index/UUID; repeat for multiple GPUs')
    parser.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')
    args = parser.parse_args()
    roots = [project,Path(args.wall_source).resolve(),Path(args.rlinf_root).resolve()]
    for root,package in ((roots[1],'wall_x'),(roots[2],'rlinf')):
        if not (root/package/'__init__.py').is_file():
            parser.error(f'missing isolated dependency {root/package}')
    source_report = validate_wall_source(roots[1])
    sys.path[:0] = [str(path) for path in roots]
    os.environ['PYTHONPATH'] = os.pathsep.join([str(path) for path in roots]+[os.environ.get('PYTHONPATH','')])
    os.environ['RLINF_EXT_MODULE'] = 'wall_rlt.rlinf_ext'
    from omegaconf import OmegaConf
    cfg = OmegaConf.merge(OmegaConf.load(args.config),OmegaConf.from_dotlist(args.set))
    allocated = None
    if args.gpu:
        from check_wall_rlt_h100 import query_gpus
        from wall_rlt.placement import bind_allocated_gpus
        allocated = bind_allocated_gpus(cfg,query_gpus(),args.gpu,args.world_size)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g['index']) for g in allocated)
    elif not args.validate_only:
        parser.error('training requires explicit allocated --gpu index/UUID selections; CVD alone is not an RLinf placement contract')
    # Bind physical device order before any RLinf/worker import can import torch
    # or initialize CUDA. Actor placement and UUID checks use the same inventory.
    from wall_rlt.rlinf_ext import register
    from wall_rlt.workers.sft import validate_training_config
    from rlinf.config import validate_sft_cfg,validate_fsdp_cfg
    register()
    # No Cluster() in validation: RLinf validate_cfg starts Ray, so use its
    # pure SFT/FSDP validators here and full validate_cfg only at actual launch.
    cfg = validate_sft_cfg(cfg)
    cfg.actor = validate_fsdp_cfg(cfg.actor)
    rlt,contract = validate_training_config(cfg,world_size=args.world_size)
    if args.validate_only:
        report = {'status':'PASS','scope':'configuration, source availability and dataset contract only',
                  'contract_hash':contract.fingerprint(),
                  'model_type':cfg.actor.model.model_type,
                  'model_initialization':cfg.actor.model.initialization,
                  'train_episodes':len(contract.splits['train']),
                  'dev_episodes':len(contract.splits['dev']),
                  'expected_world_size':args.world_size,
                  'allocated_gpus':allocated,
                  'eval_max_samples':cfg.data.get('eval_max_samples'),
                  'eval_scope':'fixed uniform dev smoke subset' if cfg.data.get('eval_max_samples') is not None else 'full dev',
                  'weights_constructed':False,'ray_started':False,'gpu_training':'NOT_RUN'}
        report['wall_source'] = source_report
        # Builder provides the optional no-allocation preflight independently.
        from wall_rlt.model.builder import validate_build_config
        report['builder'] = validate_build_config(cfg.actor.model)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return
    import torch.multiprocessing as mp
    from rlinf.config import validate_cfg
    from rlinf.runners.sft_runner import SFTRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from wall_rlt.workers.sft import WallRLTSftWorker
    mp.set_start_method('spawn',force=True)
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg,cluster)
    if placement.get_world_size('actor') != args.world_size:
        raise ValueError('actual actor placement differs from --world-size; use only allocated GPUs')
    validate_training_config(cfg,world_size=placement.get_world_size('actor'))
    actor = WallRLTSftWorker.create_group(cfg).launch(cluster,name=cfg.actor.group_name,
            placement_strategy=placement.get_strategy('actor'))
    runner = SFTRunner(cfg=cfg,actor=actor)
    runner.init_workers()
    runner.run()


if __name__ == '__main__':
    main()
