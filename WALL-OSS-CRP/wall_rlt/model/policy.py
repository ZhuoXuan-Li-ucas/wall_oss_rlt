"""A hardware-independent Stage 1 wrapper and frozen Stage 2 feature interface."""
from __future__ import annotations

from dataclasses import asdict
import torch
from torch import nn

from wall_rlt.config import WallRLTConfig
from wall_rlt._vendor.rlt_token_transformer import RLTTokenTransformer


class WallRLTPolicy(nn.Module):
    def __init__(self, wall: nn.Module, config: WallRLTConfig):
        super().__init__()
        self.wall = wall
        self.rlt_config = config
        self.config = getattr(wall,'config',None)
        self.rlt_module = RLTTokenTransformer(**config.token_kwargs())
        self._frozen_features = False
        self._no_split_modules = ['Qwen2_5_VLDecoderLayer_with_MoE','RLTTokenTransformer']

    def forward(self, forward_type='sft', data=None, **kwargs):
        mode=getattr(forward_type,'value',forward_type)
        if mode != 'sft':
            raise NotImplementedError(f'WALL-RLT supports SFT forward, got {mode}')
        return self.sft_forward(data,**kwargs)

    def _prefix(self, output):
        h=getattr(output,'rlt_prefix_hidden_states',None)
        mask=getattr(output,'rlt_prefix_mask',None)
        if h is None or mask is None:
            raise RuntimeError('WALL is missing the optional RLT prefix patch; use the isolated patched source')
        if h.ndim != 3 or h.shape[-1] != self.rlt_config.input_dim or mask.shape != h.shape[:2]:
            raise ValueError('WALL prefix dimensions disagree with RLT config')
        if mask.dtype != torch.bool:
            raise ValueError('WALL prefix mask must be boolean')
        if h.shape[1] > self.rlt_config.prefix_seq_len or not mask.any(dim=1).all():
            raise ValueError('empty or oversized RLT prefix')
        if not torch.isfinite(h).all():
            raise ValueError('non-finite WALL prefix')
        return h.detach().float(),mask.bool()

    def sft_forward(self,data,**kwargs):
        if getattr(self,'_rlt_requires_checkpoint_restore',False):
            raise RuntimeError('resume model requires successful checkpoint restore before use')
        if self._frozen_features:
            raise RuntimeError('cannot train the frozen Stage 2 feature model')
        inputs=dict(data['wall_inputs'])
        inputs['labels']=torch.full_like(inputs['input_ids'],-100)
        inputs['return_dict']=True
        output=self.wall(mode='train',return_rlt_prefix=True,flow_loss_only=True,**inputs)
        vla_loss=output.flow_loss
        if vla_loss is None or vla_loss.numel()!=1 or not torch.isfinite(vla_loss):
            raise ValueError('WALL must return a finite scalar flow loss')
        h,mask=self._prefix(output)
        with torch.autocast(device_type=h.device.type,enabled=False):
            rlt_loss,_=self.rlt_module(h,mask)
        if not torch.isfinite(rlt_loss):
            raise ValueError('non-finite RLT reconstruction loss')
        return {'loss':rlt_loss+self.rlt_config.rlt_alpha*vla_loss,
                'vla_loss':vla_loss,'rlt_loss':rlt_loss}

    @torch.no_grad()
    def evaluate_loss(self,data):
        was_training=self.training
        self.eval()
        try:
            return self.sft_forward(data)
        finally:
            self.train(was_training)

    def freeze_features(self):
        self.requires_grad_(False)
        self.eval()
        self._frozen_features=True

    def train(self,mode=True):
        if mode and getattr(self,'_frozen_features',False):
            raise RuntimeError('frozen Stage 2 features must remain in eval mode')
        return super().train(mode)

    @torch.inference_mode()
    def extract_rlt_obs(self, env_obs):
        """Input is an adapter-produced observation batch; no raw robot inference."""
        if getattr(self,'_rlt_requires_checkpoint_restore',False):
            raise RuntimeError('resume model requires successful checkpoint restore before use')
        if not self._frozen_features or self.training:
            raise RuntimeError('call freeze_features() before Stage 2 extraction')
        inputs=dict(env_obs['wall_inputs'])
        if 'action_chunk' in inputs:
            raise ValueError('feature input must not contain demonstration actions')
        result=self.wall.generate_flow_action(
            action_horizon=self.rlt_config.action_horizon,
            action_dim=self.rlt_config.model_action_dim,
            return_rlt_prefix=True,**inputs)
        from types import SimpleNamespace
        h,mask=self._prefix(SimpleNamespace(**result))
        with torch.autocast(device_type=h.device.type,enabled=False):
            z=self.rlt_module.encode_flat(h,mask)
        # Physical action to RL conversion must be supplied explicitly.
        transform=env_obs['action_transform']
        slot_map=env_obs['action_slot_map']
        ref=transform.to_rl(slot_map.decode(result['predict_action']))
        proprio=torch.as_tensor(env_obs['proprio'],device=z.device,dtype=torch.float32)
        if proprio.ndim!=2 or proprio.shape[0]!=z.shape[0] or not torch.isfinite(proprio).all():
            raise ValueError('invalid Stage 2 proprio')
        return {'z_rl':z,'proprio':proprio,'ref_chunk':ref[:,:1].float()}

    def default_forward(self,**kwargs):
        raise NotImplementedError('PPO/logprob forward is outside this RLT adapter')

    @torch.inference_mode()
    def predict_action_batch(self,env_obs,**kwargs):
        result=self.extract_rlt_obs(env_obs)
        return result['ref_chunk'],{'forward_inputs':result}

    def save_pretrained(self,path,**kwargs):
        from pathlib import Path
        from safetensors.torch import save_file
        folder=Path(path); folder.mkdir(parents=True,exist_ok=True)
        self.rlt_config.save(folder/'rlt_config.json')
        state=kwargs.get('state_dict',self.state_dict())
        save_file({k:v.detach().cpu().contiguous().clone() for k,v in state.items()},folder/'model.safetensors')
