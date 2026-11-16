"""WALL's official processor, with explicit masks and reproducible noise time."""
from __future__ import annotations

import copy
import json
import numpy as np
import torch

from wall_rlt.model.prefix import observation_prefix_mask
from .ar5 import EE_REL6D_NEXT, ee_relative_window
from .transforms import SlotMap


class WallCollator:
    def __init__(self, contract, processor_path, *, seed=0, image_size=448, max_length=2048):
        contract.require_trainable(allow_synthetic=True)
        from wall_x.data.backends.lerobot.loader import DataCollator
        from wall_x.data.backends.lerobot.utils import NormStats
        self.contract = contract
        self.action_map = SlotMap(contract.action_slots,contract.model_action_dim)
        self.state_map = SlotMap(contract.state_slots,contract.model_state_dim)
        self.image_size = image_size
        # Official Wall-OSS-0.5 targets are relative to the observed pose, so a
        # window is converted here, after windowing and before normalization.
        self.relative_targets = contract.target_mode == EE_REL6D_NEXT
        raw = json.loads(open(contract.normalization_path).read())['norm_stats']
        stats = {}
        for role,key,mapping in [('action',contract.action_key,self.action_map),
                                 ('state',contract.state_key,self.state_map)]:
            lo,_ = mapping.encode(torch.tensor(raw[key]['q01'],dtype=torch.float32))
            hi,valid = mapping.encode(torch.tensor(raw[key]['q99'],dtype=torch.float32))
            hi = torch.where(valid,hi,torch.ones_like(hi))
            stats[role] = NormStats(min=lo,max=hi,delta=hi-lo)
        self.native = DataCollator(
            {'processor_path':str(processor_path),'model_type':'qwen2_5','use_fast_tokenizer':False,
             'use_state_string_representation':True,'state_bins':512,'padding_side':'left',
             'dof_total_dim':contract.model_action_dim,'agent_pos_total_dim':contract.model_state_dim},
            {'max_length':max_length},stats,{'repo_id':contract.repo_id})
        self.native.np_rng = np.random.default_rng(seed)
        self.action_token_id = self.native.processor.tokenizer.convert_tokens_to_ids('<|action|>')

    def state_dict(self):
        return {'numpy_bit_generator':copy.deepcopy(self.native.np_rng.bit_generator.state)}

    def load_state_dict(self,state):
        self.native.np_rng.bit_generator.state = copy.deepcopy(state['numpy_bit_generator'])

    def _images(self, images):
        from PIL import Image
        from qwen_vl_utils.vision_process import smart_resize, IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS
        output = []
        for key in self.contract.camera_mapping:
            tensor = torch.as_tensor(images[key])
            if tensor.ndim != 3 or tensor.shape[0] != 3 or not torch.isfinite(tensor).all():
                raise ValueError(f'{key} must be finite RGB [3,H,W]')
            if tensor.is_floating_point():
                if tensor.min() < 0 or tensor.max() > 1:
                    raise ValueError('floating RGB input must be in [0,1]')
                tensor = tensor * 255
            elif tensor.dtype != torch.uint8:
                raise ValueError('integer RGB input must be uint8')
            image = Image.fromarray(tensor.to(torch.uint8).permute(1,2,0).cpu().numpy())
            width,height = image.size
            scale = self.image_size/max(width,height)
            width,height = int(width*scale),int(height*scale)
            h,w = smart_resize(height,width,factor=IMAGE_FACTOR,min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS)
            output.append(image.resize((w,h)))
        return output

    def __call__(self, samples, *, convert_targets=True):
        from wall_x.data.backends.lerobot.utils import get_wallx_normal_text
        if not samples:
            raise ValueError('empty batch')
        prepared, times, provenance = [], [], []
        for sample in samples:
            time = torch.as_tensor(sample['time_mask'])
            if time.dtype != torch.bool or time.shape != (self.contract.action_horizon,) or not time.any():
                raise ValueError('invalid or empty time mask')
            # Valid entries form a prefix: no gaps or fabricated interior frames.
            if (time[1:] & ~time[:-1]).any():
                raise ValueError('time mask must be a contiguous valid prefix')
            action = torch.as_tensor(sample['action'],dtype=torch.float32)
            if action.shape != (self.contract.action_horizon,len(self.contract.action_names)):
                raise ValueError('action window shape disagrees with contract')
            raw_state = torch.as_tensor(sample['state'],dtype=torch.float32)
            if raw_state.shape != (len(self.contract.state_names),):
                raise ValueError('state must be a single observation vector')
            if self.relative_targets and convert_targets:
                valid = int(time.sum())
                action = action.clone()
                action[:valid] = torch.from_numpy(ee_relative_window(
                    action[:valid].double().numpy(),raw_state.double().numpy())).float()
            raw_valid = time[:,None].expand_as(action)
            mapped, mask = self.action_map.encode(action,raw_valid)
            state,state_mask = self.state_map.encode(raw_state)
            text,_ = get_wallx_normal_text(
                {'instruction':sample['instruction']},self.contract.action_horizon,0,
                {'instruction':1.0},self.contract.camera_mapping,generate_subtask_ratio=0.0)
            prepared.append({'image_inputs':self._images(sample['images']),'text':text,
                             'action':mapped.masked_fill(~mask,float('nan')),
                             'agent_pos':state.masked_fill(~state_mask,float('nan')),
                             'frame_index':torch.tensor(sample.get('frame_index',0))})
            times.append(time)
            provenance.append({k:sample.get(k) for k in ('episode_id','frame_index')})
        batch = dict(self.native(prepared))
        time_mask = torch.stack(times)
        action_tokens = batch['input_ids'].eq(self.action_token_id)
        if not action_tokens.sum(dim=1).eq(self.contract.action_horizon).all():
            raise ValueError('processor truncated or changed the flow action horizon')
        valid_positions = torch.zeros_like(action_tokens)
        for index in range(len(samples)):
            valid_positions[index,action_tokens[index]] = time_mask[index]
        batch['positional_masks'] = {
            'padding_positions':~batch['attention_mask'].bool(),
            'valid_flow_action_positions':valid_positions,
        }
        batch['flow_loss_mask'] = time_mask.float()
        batch['action_chunk'].masked_fill_(~batch['dof_mask'].bool(),0)
        batch['proprioception'].masked_fill_(~batch['agent_pos_mask'].bool(),0)
        batch['labels'] = torch.full_like(batch['input_ids'],-100)
        observation_prefix_mask(batch['input_ids'],batch['attention_mask'],self.action_token_id)
        batch.pop('frame_index',None)
        return {'wall_inputs':batch,'metadata':provenance,'contract_hash':self.contract.fingerprint()}

    def observation_batch(self,samples):
        """Build the same prefix without reading any demonstration action target."""
        items=[]
        for obs in samples:
            item=dict(obs)
            item['action']=torch.zeros(self.contract.action_horizon,len(self.contract.action_names))
            item['time_mask']=torch.ones(self.contract.action_horizon,dtype=torch.bool)
            items.append(item)
        saved=self.state_dict()
        try:
            result=self(items,convert_targets=False)
        finally:
            self.load_state_dict(saved)
        for key in ('action_chunk','sample_time','flow_loss_mask'):
            result['wall_inputs'].pop(key,None)
        return result
