#!/usr/bin/env python3
"""H100 preflight, or an explicitly requested single optimizer update.

Default mode reads configuration and nvidia-smi metadata only. It never starts
Ray, loads model weights, or launches a CUDA kernel. --run-single-step requires
an explicitly allocated --gpu index/UUID, verifies that exact H100 UUID again
inside the worker, and reuses the project RLinf/FSDP training loop.
Repeat --gpu with --world-size N for multi-GPU memory preflight. Runtime probes
currently support one GPU only; multi-rank FSDP uses the main training entry.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]


def query_gpus():
    command = ['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free,driver_version',
               '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command,capture_output=True,text=True,timeout=15,check=True)
    except FileNotFoundError as exc:
        raise RuntimeError('nvidia-smi is unavailable; install/use the target NVIDIA host environment before GPU validation') from exc
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
        raise RuntimeError('nvidia-smi could not query the target GPUs; no training was started') from exc
    records = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 6:
            raise ValueError('unexpected nvidia-smi output; refusing to infer a GPU')
        index,name,uuid,total,free,driver = [value.strip() for value in row]
        records.append({'index':int(index),'name':name,'uuid':uuid,
                        'total_memory_mib':int(total),'free_memory_mib':int(free),
                        'driver_version':driver})
    return records


def select_h100(records,selector):
    if not selector or ',' in selector:
        raise ValueError('specify exactly one allocated --gpu index or full UUID')
    matches = [gpu for gpu in records if str(gpu['index']) == selector or gpu['uuid'] == selector]
    if len(matches) != 1:
        raise ValueError('selected GPU index/UUID was not found uniquely; no automatic GPU selection')
    selected = matches[0]
    if 'H100' not in selected['name'].upper() or not selected['uuid'].startswith('GPU-'):
        raise ValueError('this check requires the explicitly selected full H100 GPU; no other GPU or MIG fallback')
    return dict(selected,requested_selector=selector)


def normalized_uuid(value):
    return str(value).lower().removeprefix('gpu-')


def select_h100s(records,selectors,world_size):
    selected = [select_h100(records,value) for value in selectors]
    if len({gpu['uuid'] for gpu in selected}) != len(selected):
        raise ValueError('each --gpu must refer to a different explicitly allocated H100')
    if selected and len(selected) != world_size:
        raise ValueError('the number of explicit --gpu selections must equal --world-size')
    return selected


def memory_gate(estimate,selected=None):
    """Conservative admission floor, never a promise that activations will fit."""
    reserve = 4*1024**3
    required = (estimate['persistent_per_rank_bytes']+
                estimate['bf16_whole_model_working_copy_bytes_excluded']+reserve)
    available = selected['free_memory_mib']*1024**2 if selected else None
    return {'minimum_free_bytes':required,'minimum_free_gib':required/1024**3,
            'available_bytes':available,'workspace_reserve_bytes':reserve,
            'scope':'persistent states + conservative full BF16 working copy + 4 GiB reserve; activation peak still unmeasured',
            'admission_passed':available>=required if available is not None else None,
            'capacity_fit_verified':False}


def verify_runtime_gpu(torch_module,selected,*,minimum_free_bytes=0):
    """Run in the worker before constructing a full model, never in preflight."""
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise RuntimeError('single-step worker must see exactly one CUDA GPU')
    props = torch_module.cuda.get_device_properties(0)
    if 'H100' not in props.name.upper():
        raise RuntimeError('worker was assigned a non-H100 device; model construction refused')
    actual_uuid = getattr(props,'uuid',None)
    if actual_uuid is None or normalized_uuid(actual_uuid) != normalized_uuid(selected['uuid']):
        raise RuntimeError('worker GPU UUID differs from the explicitly allocated GPU; model construction refused')
    if not torch_module.cuda.is_bf16_supported():
        raise RuntimeError('selected H100 runtime does not report BF16 support')
    free,total = torch_module.cuda.mem_get_info(0)
    if free < minimum_free_bytes:
        raise RuntimeError('selected H100 no longer has the required conservative free-memory floor; model construction refused')
    return {'name':props.name,'uuid':str(actual_uuid),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'total_memory_bytes':props.total_memory,'free_memory_bytes_before_model':free,
            'bf16_supported':True}


def prepare_single_step_config(cfg,selected,output):
    """Clone the training candidate; document every smoke-only override."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
    if cfg.actor.model.initialization != 'new_transfer' or cfg.runner.get('resume_dir'):
        raise ValueError('first-update validation uses new_transfer; checkpoint resume is a separate check')
    if (cfg.actor.fsdp_config.strategy != 'fsdp' or
            cfg.actor.fsdp_config.sharding_strategy != 'no_shard' or
            not cfg.actor.fsdp_config.use_orig_params):
        raise ValueError('single-step probes require single-rank FSDP1 no_shard with use_orig_params')
    if cfg.actor.fsdp_config.grad_scaler.get('enabled',False):
        raise ValueError('single-step gradient measurements require the BF16 configuration without scaling')
    if cfg.actor.model.vision_attention != 'flash_attention_2':
        raise ValueError('H100 validation must exercise the candidate FlashAttention vision path')
    # RLinf enumerates NVML hardware ranks and rewrites each worker CVD; these
    # are physical indices, not ordinals within the driver's CVD list.
    changes = {'cluster.num_nodes':1,'cluster.component_placement':{'actor':selected['index']},
               'actor.micro_batch_size':1,'actor.global_batch_size':1,
               'data.val_data_paths':None,'runner.max_steps':1,'runner.max_epochs':-1,
               'runner.val_check_interval':-1,'runner.save_interval':-1,
               'runner.logger.log_path':str(output/'runner'),'runner.logger.logger_backends':[],
               'actor.optim.lr_warmup_steps':0,'actor.optim.total_training_steps':1}
    for key,value in changes.items():
        OmegaConf.update(cfg,key,value,merge=False)
    cfg.h100_check = {'gpu':selected,'output':str(output)}
    return cfg,changes


def multirank_launch_plan(args,selected,output):
    """A reviewable invocation, never executed by this diagnostic entry."""
    indices = ','.join(str(gpu['index']) for gpu in selected)
    argv = [sys.executable,str(PROJECT/'scripts/train_wall_rlt.py'),
            '--config',str(Path(args.config).resolve()),'--world-size',str(args.world_size),
            '--wall-source',str(Path(args.wall_source).resolve()),
            '--rlinf-root',str(Path(args.rlinf_root).resolve())]
    for gpu in selected:
        argv.extend(['--gpu',gpu['uuid']])
    overrides = list(args.set)+['runner.max_steps=1','actor.optim.lr_warmup_steps=0',
                'runner.val_check_interval=-1','runner.save_interval=-1',
                'data.val_data_paths=null',f'runner.logger.log_path={output / "multirank-runner"}']
    for value in overrides:
        argv.extend(['--set',value])
    return {'status':'NOT_RUN','argv':argv,
            'environment':{'CUDA_VISIBLE_DEVICES':indices,'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
                           'HF_HUB_OFFLINE':'1'},
            'allocated_gpu_uuids':[gpu['uuid'] for gpu in selected],
            'scope':'main RLinf/FSDP training loop, one optimizer step; separate command only',
            'warmup_override_reason':'warmup=0 is a single-step smoke override so the first update has nonzero LR',
            'limitations':['not executed or validated by this preflight',
                           'main runner may write its final-step checkpoint despite save_interval=-1',
                           'this invocation does not include single-rank gradient/coordinate probes',
                           'not 100-step, checkpoint restore, or robot inference acceptance']}


def gradient_probes(model):
    """Measure both trainable groups; retain one maximum-gradient coordinate each.

    Parameter-change evidence is explicitly a representative coordinate per
    group, not a full weight digest or proof that every parameter changed.
    """
    import torch
    totals = {key:{'squared_norm':0.,'parameters_with_grad':0,'nonzero_parameters':0,
                   'max_abs_gradient':0.,'probe':None} for key in ('wall','rlt')}
    for name,param in model.named_parameters():
        if param.grad is None:
            continue
        parts = name.split('.')
        group = 'rlt' if 'rlt_module' in parts else 'wall' if 'wall' in parts else None
        if group is None:
            raise ValueError(f'unclassified trainable gradient parameter: {name}')
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            raise ValueError(f'non-finite gradient in {name}')
        item = totals[group]
        item['parameters_with_grad'] += 1
        norm = float(torch.linalg.vector_norm(grad.float()))
        item['squared_norm'] += norm*norm
        maximum = float(grad.abs().max()) if grad.numel() else 0.
        if maximum > 0:
            item['nonzero_parameters'] += 1
        if maximum > item['max_abs_gradient']:
            index = int(grad.abs().reshape(-1).argmax())
            item['max_abs_gradient'] = maximum
            item['probe'] = {'name':name,'index':index,'before':float(param.detach().reshape(-1)[index]),
                             'parameter':param}
    for group,item in totals.items():
        if item['probe'] is None or not math.isfinite(item['squared_norm']) or item['squared_norm'] <= 0:
            raise ValueError(f'{group} did not receive a finite nonzero gradient')
    return totals


def finish_probes(probes):
    result = {}
    for group,item in probes.items():
        probe = item['probe']
        after = float(probe['parameter'].detach().reshape(-1)[probe['index']])
        if not math.isfinite(after) or after == probe['before']:
            raise ValueError(f'{group} representative parameter did not change after optimizer.step')
        result[group] = {'gradient_l2':math.sqrt(item['squared_norm']),
                         'parameters_with_grad':item['parameters_with_grad'],
                         'parameters_with_nonzero_grad':item['nonzero_parameters'],
                         'max_abs_gradient':item['max_abs_gradient'],
                         'parameter_change_probe':{key:probe[key] for key in ('name','index','before')},
                         'after':after,'absolute_change':abs(after-probe['before'])}
    return result


def make_check_worker(base_class):
    """Use the existing worker implementation, augmenting only observations."""
    class H100SingleStepWorker(base_class):
        def init_worker(self):
            import torch
            self._h100_device = verify_runtime_gpu(torch,self.cfg.h100_check.gpu,
                minimum_free_bytes=int(self.cfg.h100_check.memory_gate.minimum_free_bytes))
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            super().init_worker()
            torch.cuda.synchronize()
            self._h100_load_seconds = time.perf_counter()-started
            self._h100_loss_records = []
            self._h100_input_records = []
            self._h100_updates = None
            for module in self.model.modules():
                if 'build_report' in module.__dict__:
                    self._h100_build_report = module.build_report
                    break
            else:
                raise RuntimeError('constructed worker lacks the project model build report')

        def get_train_model_output(self,batch):
            loss,metrics = super().get_train_model_output(batch)
            self._h100_loss_records.append(metrics)
            self._h100_input_records.append({'contract_hash':batch['contract_hash'],
                                             'samples':batch['metadata']})
            return loss,metrics

        def optimizer_step(self):
            probes = gradient_probes(self.model)
            result = super().optimizer_step()
            self._h100_updates = finish_probes(probes)
            return result

        def run_h100_single_step(self):
            import torch
            started = time.perf_counter()
            metrics = super().run_training()
            torch.cuda.synchronize()
            if self._h100_updates is None or len(self._h100_loss_records) != 1:
                raise RuntimeError('single-step worker did not execute exactly one microbatch/update')
            return {'status':'PASS','scope':'one target-data optimizer update through the actual single-rank RLinf FSDP worker',
                    'device':self._h100_device,'load_seconds':self._h100_load_seconds,
                    'step_seconds':time.perf_counter()-started,'metrics':metrics,
                    'losses':self._h100_loss_records,'inputs':self._h100_input_records,
                    'groups':self._h100_updates,'build_report':self._h100_build_report,
                    'data_signature':self._data_signature(),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'parameter_update_scope':'one maximum-gradient coordinate checked per WALL/RLT group',
                    'optimizer_updates':1,'checkpoint_saved':False,
                    'not_validated':['100-step training','checkpoint save/restore','multi-rank FSDP','robot inference']}
    return H100SingleStepWorker


def launch_one_update(cfg):
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from wall_rlt.workers.sft import WallRLTSftWorker
    import ray
    import torch.multiprocessing as mp
    mp.set_start_method('spawn',force=True)
    # validate_cfg creates a Cluster; only reached after explicit run selection.
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg,cluster)
    if placement.get_world_size('actor') != 1:
        raise ValueError('single-step check requires exactly one worker')
    worker = make_check_worker(WallRLTSftWorker)
    actor = worker.create_group(cfg).launch(cluster,name='WallRLTH100SingleStep',
                                           placement_strategy=placement.get_strategy('actor'))
    try:
        actor.init_worker().wait()
        reports = actor.run_h100_single_step().wait()
        if len(reports) != 1:
            raise RuntimeError('expected exactly one worker report')
        return reports[0]
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(PROJECT/'configs/rlt/stage1.yaml'))
    parser.add_argument('--wall-source',default=str(PROJECT/'.cache/wall-x-rlt'))
    parser.add_argument('--rlinf-root',default=str(PROJECT/'.cache/rlinf-rlt'))
    parser.add_argument('--gpu',action='append',default=[],
                        help='Explicitly allocated physical GPU index or full GPU UUID; repeat for multi-GPU preflight')
    parser.add_argument('--world-size',type=int,default=1,
                        help='Expected actor GPU count for config validation and per-rank memory estimate')
    parser.add_argument('--run-single-step',action='store_true')
    parser.add_argument('--output',default='logs/rlt-h100-check-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')
    args = parser.parse_args()
    if args.run_single_step and not args.gpu:
        parser.error('--run-single-step requires --gpu for an explicitly allocated H100')
    if args.world_size < 1:
        parser.error('--world-size must be a positive integer')
    if args.run_single_step and (args.world_size != 1 or len(args.gpu) != 1):
        parser.error('--run-single-step currently supports one GPU only; use multi-GPU preflight and the main FSDP training entry')
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report = {'status':'RUNNING','requested_run':args.run_single_step,'weights_constructed':False,
              'ray_started':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'config_path':str(Path(args.config).resolve()),'working_directory':str(Path.cwd())}
    try:
        records = query_gpus()
        report['gpu_inventory'] = records
        selections = select_h100s(records,args.gpu,args.world_size)
        selected = selections[0] if len(selections) == 1 else None
        report['selected_gpu'] = selected
        report['selected_gpus'] = selections
        report['expected_world_size'] = args.world_size
        # Pinned RLinf currently accepts numeric CUDA_VISIBLE_DEVICES only.
        # Bind its index to the selected UUID and recheck UUID inside the worker.
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(selected['index']) if args.run_single_step else ''
        roots = [PROJECT,Path(args.wall_source).resolve(),Path(args.rlinf_root).resolve()]
        sys.path[:0] = [str(path) for path in roots]+[str(PROJECT/'scripts')]
        os.environ['PYTHONPATH'] = os.pathsep.join([str(path) for path in roots]+[str(PROJECT/'scripts')])
        os.environ['RLINF_EXT_MODULE'] = 'wall_rlt.rlinf_ext'
        os.environ['HF_HUB_OFFLINE'] = '1'
        from train_wall_rlt import validate_wall_source
        from omegaconf import OmegaConf
        from rlinf.config import validate_sft_cfg,validate_fsdp_cfg
        from wall_rlt.rlinf_ext import register
        from wall_rlt.workers.sft import validate_training_config,source_hashes
        from wall_rlt.model.builder import validate_build_config
        register()
        report['wall_source'] = validate_wall_source(roots[1])
        cfg = OmegaConf.merge(OmegaConf.load(args.config),OmegaConf.from_dotlist(args.set))
        cfg = validate_sft_cfg(cfg)
        cfg.actor = validate_fsdp_cfg(cfg.actor)
        rlt,contract = validate_training_config(cfg,world_size=args.world_size)
        report['builder'] = validate_build_config(cfg.actor.model)
        report['contract_hash'] = contract.fingerprint()
        report['source_hashes'] = source_hashes()
        report['config_sha256'] = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
        report['config_overrides'] = args.set
        from wall_rlt.resources import estimate_training_memory
        report['static_memory_estimate'] = estimate_training_memory(
            rlt.checkpoint_path,rlt,world_size=args.world_size,
            sharding_strategy=cfg.actor.fsdp_config.sharding_strategy)
        gate = memory_gate(report['static_memory_estimate'],selected)
        if selections:
            gpu_gates = [dict(memory_gate(report['static_memory_estimate'],gpu),
                              uuid=gpu['uuid'],index=gpu['index']) for gpu in selections]
            gate.update(per_gpu=gpu_gates,
                        admission_passed=all(item['admission_passed'] for item in gpu_gates),
                        available_bytes=min(item['available_bytes'] for item in gpu_gates))
        report['memory_gate'] = gate
        if args.world_size > 1 and selections:
            report['multirank_launch_plan'] = multirank_launch_plan(args,selections,output)
            report['multirank_launch_plan']['memory_admission_passed'] = gate['admission_passed']
            report['multirank_launch_plan']['execution_requires'] = (
                'allocated GPUs, matching UUIDs at launch, sufficient current free memory, '
                'and a new output directory; preflight is not a capacity guarantee')
        if not args.run_single_step:
            report.update(status='PREFLIGHT_PASS',scope='configuration and read-only GPU inventory; no model or Ray',
                          h100_runtime='NOT_RUN',gpu_selection_required_for_execution=not selections,
                          execution_readiness='NO_GPU_SELECTED' if not selections else
                          'INSUFFICIENT_MEMORY' if not gate['admission_passed'] else
                          'ADMISSION_FLOOR_PASSED_ACTIVATION_PEAK_UNVERIFIED')
        else:
            if not gate['admission_passed']:
                raise ValueError('selected H100 has insufficient free memory for the conservative single-GPU admission floor; use an explicitly allocated multi-GPU FSDP candidate instead, without shrinking the model silently')
            cfg,changes = prepare_single_step_config(cfg,selected,output)
            cfg.h100_check.memory_gate = gate
            report['single_step_overrides'] = changes
            report['warmup_override_reason'] = 'the candidate warmup starts with LR=0; this isolated first-update check uses warmup=0 to verify real parameter change'
            OmegaConf.save(cfg,output/'single-step.yaml')
            report['ray_launch_attempted'] = True
            report['ray_started'] = 'UNKNOWN_IF_LAUNCH_FAILS'
            report['weights_constructed'] = 'UNKNOWN_IF_LAUNCH_FAILS'
            report['execution'] = launch_one_update(cfg)
            report['ray_started'] = True
            report['weights_constructed'] = True
            report['status'] = 'SINGLE_STEP_PASS'
    except Exception as exc:
        report.update(status='FAIL',error_type=type(exc).__name__,error=str(exc))
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))
        raise SystemExit(1)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
