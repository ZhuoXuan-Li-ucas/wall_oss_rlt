"""RLinf SFT worker for WALL-RLT, with explicit loss and data recovery semantics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from dataclasses import asdict
import shutil
import sys

import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker
from rlinf.utils.utils import get_rng_state, set_rng_state, seed_everything

from wall_rlt.config import WallRLTConfig
from wall_rlt.data.contract import DataContract, canonical_hash
from wall_rlt.data.dataset import EpisodeWindowDataset
from wall_rlt.data.collator import WallCollator
from wall_rlt.data.npz_source import build_source
from wall_rlt.model.prefix import observation_prefix_mask


PROCESSOR_FILES = ('config.json','processor_config.json','preprocessor_config.json',
                   'tokenizer_config.json','tokenizer.json','special_tokens_map.json',
                   'added_tokens.json','vocab.json','merges.txt','chat_template.json',
                   'chat_template.jinja','video_preprocessor_config.json')


def source_hashes():
    """Bind project Python and directly used upstream training sources, not binaries."""
    project_package = Path(__file__).resolve().parents[1]
    hashes = {'wall_rlt/'+str(p.relative_to(project_package)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(project_package.rglob('*.py'))}
    selected = {
        'rlinf':['models/__init__.py','workers/sft/fsdp_vla_sft_worker.py','workers/sft/fsdp_sft_worker.py',
                 'hybrid_engines/fsdp/fsdp_model_manager.py','hybrid_engines/fsdp/mixed_precision.py',
                 'hybrid_engines/fsdp/strategy/fsdp.py','hybrid_engines/fsdp/strategy/checkpoint.py',
                 'hybrid_engines/fsdp/optim.py','utils/fsdp_config.py','runners/sft_runner.py'],
        'wall_x':['model/qact/qwen2_5/modeling_qwen2_5_vl_act.py',
                  'model/qact/qwen2_5/modeling_qwen2_5_vl.py','model/core/vla_mixin.py',
                  'model/core/attention/mask.py','model/core/action/processor.py',
                  'model/core/action/normalizer.py',
                  'data/backends/lerobot/loader.py','data/backends/lerobot/utils.py']}
    for package,names in selected.items():
        module = sys.modules.get(package)
        if module is None or not getattr(module,'__file__',None):
            continue
        root = Path(module.__file__).resolve().parent
        for name in names:
            path = root/name
            if path.is_file():
                hashes[f'{package}/{name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def plain(value):
    return OmegaConf.to_container(value,resolve=True) if OmegaConf.is_config(value) else dict(value)


def select_eval_dataset(dataset,max_samples):
    """Deterministic uniform window subset; never repeats or pads evaluation rows."""
    full = len(dataset)
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError('eval_max_samples must be null or a positive integer')
    if max_samples is None or max_samples >= full:
        return dataset,{'full_windows':full,'evaluated_windows':full,'subset_strategy':'all_dev_windows'}
    indices = np.linspace(0,full-1,num=max_samples,dtype=np.int64).tolist()
    return Subset(dataset,indices),{'full_windows':full,'evaluated_windows':len(indices),
                                    'subset_strategy':'fixed_uniform_dev_window_indices_smoke_subset'}


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key:to_device(item,device) for key,item in value.items()}
    if isinstance(value, list):
        return [to_device(item,device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item,device) for item in value)
    return value


def validate_training_config(cfg, *, world_size=None):
    """Project-specific constraints, without creating Ray actors or model weights."""
    model = cfg.actor.model
    rlt = WallRLTConfig.from_dict(OmegaConf.to_container(model.rlt,resolve=True))
    contract = DataContract.load(rlt.data_contract_path)
    contract.require_trainable()
    if cfg.data.get('num_workers',0) != 0:
        raise ValueError('strict collator RNG resume currently requires num_workers=0')
    limit = cfg.data.get('eval_max_samples')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('eval_max_samples must be null or a positive integer')
    if rlt.action_horizon != contract.action_horizon or rlt.model_action_dim != contract.model_action_dim:
        raise ValueError('model/data action dimensions disagree')
    if cfg.actor.training_backend != 'fsdp' or cfg.actor.fsdp_config.strategy != 'fsdp':
        raise ValueError('this worker currently supports FSDP1 only')
    initialization = model.get('initialization','new_transfer')
    resume_dir = cfg.runner.get('resume_dir')
    if (initialization == 'resume') != bool(resume_dir):
        raise ValueError('initialization=resume and runner.resume_dir must be set together')
    if resume_dir and not (Path(resume_dir)/'actor'/'wall_rlt_data.json').is_file():
        raise ValueError('resume_dir lacks WALL-RLT checkpoint metadata')
    if cfg.actor.fsdp_config.get('gradient_checkpointing',False):
        raise ValueError('gradient checkpointing requires a separately validated WALL-RLT configuration')
    if world_size is not None:
        if world_size < 1 or cfg.actor.global_batch_size % (cfg.actor.micro_batch_size * world_size):
            raise ValueError('global batch must divide micro batch times world size')
        if cfg.data.get('val_data_paths') is not None and world_size != 1:
            raise ValueError('loss evaluation currently requires world_size=1; disable val_data_paths for multi-rank training')
    for key in ('train_data_paths','val_data_paths'):
        path = cfg.data.get(key)
        if path is not None and Path(str(path)).resolve() != Path(rlt.data_contract_path).resolve():
            raise ValueError('train and validation must use the model contract, with distinct frozen episode splits')
    if not contract.splits.get('train'):
        raise ValueError('empty train split')
    if cfg.data.get('val_data_paths') and not contract.splits.get('dev'):
        raise ValueError('empty validation split')
    return rlt,contract


def loss_weights(batch, rlt_config):
    inputs = batch['wall_inputs']
    dof = inputs['dof_mask'].bool()
    times = inputs['flow_loss_mask'].bool()
    if dof.ndim != 3 or times.shape != dof.shape[:2]:
        raise ValueError('invalid flow mask shapes')
    flow = int((dof & times[...,None]).sum())
    prefix = observation_prefix_mask(inputs['input_ids'],inputs['attention_mask'],rlt_config.action_token_id)
    reconstruction = int(prefix.sum()) * rlt_config.input_dim
    if flow < 1 or reconstruction < 1:
        raise ValueError('loss batch has zero valid elements')
    return flow,reconstruction


class WallRLTSftWorker(FSDPVlaSftWorker):
    def init_worker(self):
        from wall_rlt.placement import verify_worker_allocation
        verify_worker_allocation(self.cfg,self._rank,torch)
        super().init_worker()

    def __init__(self,cfg):
        # Explicit registration also covers worker processes with customized Ray
        # environments; the driver still exports RLINF_EXT_MODULE.
        from wall_rlt.rlinf_ext import register
        register()
        self.rlt_config,self.contract = validate_training_config(cfg)
        super().__init__(cfg)
        validate_training_config(cfg,world_size=self._world_size)

    def build_dataloader(self,data_paths,eval_dataset=False):
        # The base constructor calls this before assigning epoch/offset fields.
        if eval_dataset and self._world_size != 1:
            raise ValueError('validation is restricted to one rank to avoid duplicated episodes or unequal FSDP forwards')
        split = 'dev' if eval_dataset else 'train'
        episodes = self.contract.splits[split]
        source = build_source(self.contract,episode_ids=episodes)
        dataset = EpisodeWindowDataset(source,episodes,self.contract.action_horizon,
            complete_windows_only=bool(self.cfg.data.get('complete_windows_only',False)))
        subset_info = {'full_windows':len(dataset),'evaluated_windows':None,'subset_strategy':'training_sampler'}
        if eval_dataset:
            dataset,subset_info = select_eval_dataset(dataset,self.cfg.data.get('eval_max_samples'))
        collator = WallCollator(self.contract,
            self.cfg.actor.model.get('processor_path',self.rlt_config.checkpoint_path),
            seed=int(self.cfg.actor.seed)+self._rank+(100000 if eval_dataset else 0),
            image_size=int(self.cfg.data.get('image_size',448)),
            max_length=int(self.cfg.data.get('max_length',2048)))
        batch_size = self.eval_batch_size if eval_dataset else self.micro_batch_size
        sampler = None if eval_dataset else DistributedSampler(dataset,
            num_replicas=self._world_size,rank=self._rank,shuffle=True,
            seed=int(self.cfg.actor.seed),drop_last=True)
        generator = torch.Generator().manual_seed(int(self.cfg.actor.seed)+self._rank+(100000 if eval_dataset else 0))
        loader = StatefulDataLoader(dataset,batch_size=batch_size,sampler=sampler,
            shuffle=False,num_workers=0,collate_fn=collator,drop_last=not eval_dataset,
            generator=generator,pin_memory=False)
        if not len(loader):
            raise ValueError(f'{split} split has no full batch on this rank')
        return loader,{'contract_hash':self.contract.fingerprint(),'split':split,
                       'episodes':episodes,'windows':len(dataset),
                       'rank_batches':len(loader),'sampler_drop_last':not eval_dataset,**subset_info}

    def get_train_model_output(self,batch):
        batch = to_device(batch,self.device)
        if batch['contract_hash'] != self.contract.fingerprint():
            raise ValueError('batch data contract mismatch')
        with self.amp_context:
            output = self.model(forward_type='sft',data=batch)
        if any(not torch.is_tensor(output[k]) or output[k].numel()!=1 or not torch.isfinite(output[k])
               for k in ('loss','vla_loss','rlt_loss')):
            raise ValueError('model must return finite scalar SFT losses')
        return output['loss'],{key:float(output[key].detach()) for key in ('loss','vla_loss','rlt_loss')}

    @torch.no_grad()
    def get_eval_model_output(self,batch):
        flow_count,rlt_count = loss_weights(batch,self.rlt_config)
        _,metrics = self.get_train_model_output(batch)
        return {'vla_sum':metrics['vla_loss']*flow_count,'vla_count':flow_count,
                'rlt_sum':metrics['rlt_loss']*rlt_count,'rlt_count':rlt_count,
                'samples':len(batch['metadata'])}

    @torch.no_grad()
    def run_eval(self):
        if self._world_size != 1:
            raise ValueError('loss evaluation currently requires world_size=1')
        if self.eval_data_loader is None:
            raise ValueError('no validation loader')
        # Evaluation must not advance training randomness, including the native
        # WALL collator's private NumPy generator or the loader generator.
        rng = get_rng_state()
        collator_state = self.eval_data_loader.collate_fn.state_dict()
        generator_state = self.eval_data_loader.generator.get_state()
        was_training = self.model.training
        totals = {k:0 for k in ('vla_sum','vla_count','rlt_sum','rlt_count','samples')}
        try:
            seed_everything(int(self.cfg.actor.seed)+100000)
            self.eval_data_loader.collate_fn.native.np_rng = np.random.default_rng(int(self.cfg.actor.seed)+100000)
            self.eval_data_loader.generator.manual_seed(int(self.cfg.actor.seed)+100000)
            self.model.eval()
            with self.worker_timer():
                # Exhaustion resets StatefulDataLoader's next iteration; no
                # capped evaluation that could resume halfway next time.
                for batch in self.eval_data_loader:
                    for key,value in self.get_eval_model_output(batch).items():
                        totals[key] += value
        finally:
            self.model.train(was_training)
            self.eval_data_loader.collate_fn.load_state_dict(collator_state)
            self.eval_data_loader.generator.set_state(generator_state)
            set_rng_state(rng)
        if not totals['vla_count'] or not totals['rlt_count']:
            raise ValueError('validation produced no valid loss elements')
        vla = totals['vla_sum']/totals['vla_count']
        rlt = totals['rlt_sum']/totals['rlt_count']
        full_windows = getattr(self,'eval_data_config',{}).get('full_windows',totals['samples'])
        return {'loss':rlt+self.rlt_config.rlt_alpha*vla,'vla_loss':vla,'rlt_loss':rlt,
                'samples':totals['samples'],'action_elements':totals['vla_count'],
                'prefix_elements':totals['rlt_count'],'full_dev_windows':full_windows,
                'smoke_subset':int(totals['samples']<full_windows)}

    def _data_signature(self):
        model_cfg = self.cfg.actor.get('model',{})
        processor = model_cfg.get('processor_path',self.rlt_config.checkpoint_path)
        lineage = {}
        if self.rlt_config.wall_train_config:
            lineage['wall_train_config'] = hashlib.sha256(Path(self.rlt_config.wall_train_config).read_bytes()).hexdigest()
        if processor:
            for name in PROCESSOR_FILES:
                file = Path(processor)/name
                if file.is_file():
                    lineage[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        return canonical_hash({'contract_hash':self.contract.fingerprint(),
            'world_size':self._world_size,'seed':int(self.cfg.actor.seed),
            'micro_batch_size':self.micro_batch_size,'global_batch_size':self.global_batch_size,
            'eval_batch_size':self.eval_batch_size,
            'eval_max_samples':self.cfg.data.get('eval_max_samples'),
            'image_size':int(self.cfg.data.get('image_size',448)),
            'max_length':int(self.cfg.data.get('max_length',2048)),
            'processor_and_wall_config':lineage,
            'wall_attention':model_cfg.get('wall_attention','sdpa'),
            'vision_attention':model_cfg.get('vision_attention','flash_attention_2'),
            'optimizer':plain(self.cfg.actor.get('optim',{})),
            'fsdp':plain(self.cfg.actor.get('fsdp_config',{})),
            'source_hashes':source_hashes(),
            'complete_windows_only':bool(self.cfg.data.get('complete_windows_only',False)),
            'rlt':{k:v for k,v in vars(self.rlt_config).items() if not k.endswith('_path') and k!='wall_train_config'}})

    def _save_assets(self,root):
        assets = root/'wall_rlt_assets'
        assets.mkdir(exist_ok=True)
        self.rlt_config.save(assets/'rlt_config.json')
        (assets/'training_config.json').write_text(json.dumps(plain(self.cfg),indent=2)+'\n')
        (assets/'source_hashes.json').write_text(json.dumps(source_hashes(),indent=2)+'\n')
        if isinstance(self.contract,DataContract):
            data = asdict(self.contract)
            for key,name in (('manifest_path','episodes.json'),('normalization_path','normalization.json')):
                shutil.copyfile(data[key],assets/name)
                data[key] = name
            (assets/'data_contract.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        model_cfg = self.cfg.actor.get('model',{})
        processor = model_cfg.get('processor_path',self.rlt_config.checkpoint_path)
        if processor:
            dest = assets/'processor'; dest.mkdir(exist_ok=True)
            for name in PROCESSOR_FILES:
                path = Path(processor)/name
                if path.is_file():
                    shutil.copyfile(path,dest/name)
        if self.rlt_config.wall_train_config:
            shutil.copyfile(self.rlt_config.wall_train_config,assets/'wall_train_config.yml')
        baseline = Path(__file__).resolve().parents[2]/'metadata/wall-rlt-design-audit-20261028.json'
        if baseline.is_file():
            evidence = json.loads(baseline.read_text())
            (assets/'source_baseline.json').write_text(json.dumps({
                'declared_source_versions':evidence['source_versions'],
                'evidence_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),
                'base_weight_content_rehashed_here':False,
                'scope':'declared baseline plus separately saved actual source hashes; not a new hash of large base weights'},indent=2)+'\n')
        for module in self.model.modules():
            if 'build_report' in module.__dict__:
                (assets/'build_report.json').write_text(json.dumps(module.build_report,indent=2)+'\n')
                if hasattr(module.config,'to_dict'):
                    (assets/'adapted_wall_config.json').write_text(json.dumps(module.config.to_dict(),indent=2)+'\n')
                break
        return {str(p.relative_to(assets)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(assets.rglob('*')) if p.is_file()}

    def save_checkpoint(self,save_path,step=0):
        super().save_checkpoint(save_path,step)
        state = {'train_collator':self.data_loader.collate_fn.state_dict(),
                 'train_generator':self.data_loader.generator.get_state(),
                 'epoch':self._data_epoch,'offset':self._data_iter_offset,
                 'eval_collator':self.eval_data_loader.collate_fn.state_dict() if self.eval_data_loader else None}
        states = [None]*self._world_size
        torch.distributed.all_gather_object(states,state)
        if self._rank == 0:
            root = Path(save_path)
            torch.save(states,root/'wall_rlt_data.pt.tmp')
            os.replace(root/'wall_rlt_data.pt.tmp',root/'wall_rlt_data.pt')
            meta = {'schema_version':1,'step':step,'signature':self._data_signature(),
                    'sha256':hashlib.sha256((root/'wall_rlt_data.pt').read_bytes()).hexdigest(),
                    'world_size':self._world_size,'contract_hash':self.contract.fingerprint()}
            meta['assets_sha256'] = self._save_assets(root)
            (root/'wall_rlt_data.json.tmp').write_text(json.dumps(meta,indent=2)+'\n')
            os.replace(root/'wall_rlt_data.json.tmp',root/'wall_rlt_data.json')
        torch.distributed.barrier()

    def load_checkpoint(self,load_path):
        root = Path(load_path)
        meta = json.loads((root/'wall_rlt_data.json').read_text())
        if meta['schema_version'] != 1 or meta['signature'] != self._data_signature() or meta['world_size'] != self._world_size:
            raise ValueError('checkpoint data/model configuration differs from this run')
        if hashlib.sha256((root/'wall_rlt_data.pt').read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('checkpoint collator state hash mismatch')
        if not meta.get('assets_sha256'):
            raise ValueError('checkpoint lacks auditable WALL-RLT assets')
        assets = (root/'wall_rlt_assets').resolve()
        for name,expected in meta['assets_sha256'].items():
            path = (assets/name).resolve()
            if not path.is_relative_to(assets) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('checkpoint asset hash mismatch')
        if not (root/'data.pt').is_file() or not (root/'rng.pt').is_file():
            raise ValueError('checkpoint lacks loader or global RNG state')
        # Local trusted training artifact, digest checked above. Never deserialize
        # arbitrary downloaded pickle payloads through this entry point.
        states = torch.load(root/'wall_rlt_data.pt',weights_only=False,map_location='cpu')
        if len(states) != self._world_size:
            raise ValueError('checkpoint rank state count mismatch')
        state = states[self._rank]
        self._data_epoch,self._data_iter_offset = state['epoch'],state['offset']
        self.data_loader.sampler.set_epoch(self._data_epoch)
        # Upstream restores distributed weights, optimizer, scheduler, loader and
        # global RNG and recreates data_iter. Set sampler epoch before that call.
        super().load_checkpoint(load_path)
        self.data_loader.collate_fn.load_state_dict(state['train_collator'])
        self.data_loader.generator.set_state(state['train_generator'])
        if self.eval_data_loader is not None and state['eval_collator'] is not None:
            self.eval_data_loader.collate_fn.load_state_dict(state['eval_collator'])
        self.global_step = meta['step']
        for module in self.model.modules():
            if '_rlt_requires_checkpoint_restore' in module.__dict__:
                module._rlt_requires_checkpoint_restore = False
