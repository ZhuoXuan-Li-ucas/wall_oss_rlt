"""Load a trained Stage 1 WALL and encoder without the training decoder.

The returned actions are WALL reference predictions in the data contract's
physical coordinates. The encoder supplies optional features; there is no
trained RL Actor or robot execution in this deployment entry.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import threading
import time
import types

import numpy as np
import torch

from wall_rlt.config import WallRLTConfig
from wall_rlt.data.ar5 import EE_REL6D_NEXT, ee_absolute_window, rot6d_to_matrix
from wall_rlt.data.collator import WallCollator
from wall_rlt.data.contract import DataContract
from wall_rlt.data.transforms import SlotMap


MODEL_SOURCES = {
    'wall_rlt': ('config.py', '_vendor/rlt_token_transformer.py', 'data/contract.py',
                 'data/collator.py', 'data/transforms.py', 'data/ar5.py', 'model/prefix.py',
                 'model/ar5_adapter.py', 'model/builder.py'),
    'wall_x': ('model/qact/qwen2_5/modeling_qwen2_5_vl_act.py',
               'model/qact/qwen2_5/modeling_qwen2_5_vl.py', 'model/core/vla_mixin.py',
               'model/core/attention/mask.py', 'model/core/action/processor.py',
               'model/core/action/normalizer.py', 'data/backends/lerobot/loader.py',
               'data/backends/lerobot/utils.py'),
}


def verify_model_sources(assets):
    """Require the recorded model/preprocessing implementation used in training.

    Training-only workers and newly added serving code are deliberately outside
    this comparison. The delivery source manifest binds the remaining code.
    """
    import importlib
    expected = json.loads((Path(assets)/'source_hashes.json').read_text())
    actual = {}
    for package, names in MODEL_SOURCES.items():
        root = Path(importlib.import_module(package).__file__).resolve().parent
        for name in names:
            key = f'{package}/{name}'
            actual[key] = hashlib.sha256((root/name).read_bytes()).hexdigest()
            if expected.get(key) != actual[key]:
                raise ValueError(f'trained model/preprocessing source differs or is unrecorded: {key}')
    return hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()


# Real-Time Chunking (Black, Galliker, Levine 2025, arXiv 2506.07339). The prefix
# schedule follows LeRobot policies/rtc/modeling_rtc.py (RTCProcessor, LINEAR).
# Two ways to make the new chunk continue the previous one, both installed through
# wall-x's _execute_flow_action strategy hook (no MODEL_SOURCES file changes):
#   inpaint  (default) after every Euler step pull the prefix toward the previous
#            chunk noised to the next time, x <- x + W ((1-t') noise + t' Y - x).
#            No gradients; costs about as much as the vanilla sampler.
#   guidance LeRobot's PiGDM guidance. WALL integrates t from 0 (noise) to 1 (action)
#            with x_t = (1-t) noise + t action and v = action - noise, so the clean
#            estimate is x_t + (1-t) v and the term is added (LeRobot runs time the
#            other way and subtracts). One backward pass per step roughly doubles
#            inference time on the 5070 Ti, because each 32-token step is host-bound
#            (wall-x MoE expert loops, ~2.7k kernel launches per step).
# Offline on bolt dev episode 14 (11-05): prefix mismatch 11.2 mm without RTC,
# 6.4 mm with guidance (473 ms), 0.1 mm with inpainting (271 ms); the free tail
# stays within 1-3 mm of the unguided prediction and no farther from the demo.
RTC_MAX_GUIDANCE = 10.0
RTC_METHODS = ('inpaint', 'guidance')


def rtc_prefix_weights(delay, horizon, total):
    """LeRobot get_prefix_weights (LINEAR): 1 for the first ``delay`` steps the
    robot executes from the previous chunk during inference, then linear decay to
    0 at ``horizon`` (the previous chunk's length), 0 afterwards."""
    start = min(delay, horizon)
    weights = torch.zeros(total)
    weights[:start] = 1.0
    if horizon > start:
        weights[start:horizon] = torch.linspace(1, 0, horizon - start + 2)[1:-1]
    return weights


def rtc_guidance_weight(tau, beta=RTC_MAX_GUIDANCE):
    """min(beta, (1-tau) / (tau r^2)), r^2 = (1-tau)^2 / ((1-tau)^2 + tau^2)."""
    if tau <= 0 or tau >= 1:
        return beta
    r2 = (1 - tau)**2 / ((1 - tau)**2 + tau**2)
    return min(beta, (1 - tau) / (tau * r2))


def ee_relative_targets(targets, state):
    """Absolute ``[K,7]`` flange targets (pos, rotation vector, gripper) -> ``[K,10]``
    rows relative to the ``[10]`` observation state: the inverse of
    ``ee_absolute_window``. The gripper stays the continuous prediction."""
    from scipy.spatial.transform import Rotation
    r_rel = Rotation.from_rotvec(targets[:, 3:6]).as_matrix() @ rot6d_to_matrix(state[None, 3:9])[0].T
    return np.concatenate((targets[:, :3] - state[None, :3], r_rel[:, :2, :].reshape(-1, 6),
                           targets[:, 6:7]), axis=1)


def _rtc_flow_action(model, ctx, guide):
    """wall-x ``_generate_flow_action_vanilla`` with an RTC prefix on every Euler step.

    Inpainting keeps the vanilla step 0 velocity from the prefill. Guidance needs a
    velocity differentiable in the noisy action, so it evaluates every step,
    including 0, through the cached postfix path (bit-identical to the prefill's
    step 0 when the weights are zero) instead of backpropagating through the
    image/text prefix.
    """
    if getattr(model.config, 'use_x_pred', False):
        raise ValueError('RTC is implemented for velocity prediction, not use_x_pred')
    target, weights, method = guide
    prefill = model.model(
        input_ids=None, attention_mask=ctx['attention_mask'], position_ids=ctx['position_ids'],
        past_key_values=None, inputs_embeds=ctx['inputs_embeds'],
        moe_token_types=ctx['moe_token_types'], start_indices=ctx['start_indices'],
        end_indices=ctx['end_indices'], positional_masks=ctx['positional_masks'],
        use_cache=True, output_attentions=False, output_hidden_states=False,
        return_dict=True, adarms_conds=[None, ctx['adarms_cond']])
    if ctx.get('return_rlt_prefix', False):
        from wall_rlt.model.prefix import observation_prefix_mask, pack_prefix
        mask = observation_prefix_mask(ctx['input_ids'], ctx['attention_mask'],
                                       model.action_token_id_set['action_token_id'])
        ctx['rlt_prefix_hidden_states'], ctx['rlt_prefix_mask'] = pack_prefix(
            prefill.last_hidden_state.detach(), mask)
    kv = prefill.past_key_values
    positions, embeds, attention, moe_types, input_ids, starts, ends, _ = \
        model._prepare_flow_postfix(ctx, kv)
    action_mask = input_ids == model.action_token_id_set['action_token_id']
    dof, noise = ctx['dof_mask'], ctx['noise']
    v_padding = (ctx['padding_action'] - noise
                 if not dof.all() and ctx['padding_action'] is not None else None)
    preprocessor = model.action_preprocessor

    def velocity(t, x):
        action_embed, adarms_cond = preprocessor.step(
            timestep=t.reshape(1).repeat(x.shape[0]), noisy_action=x, dof_mask=dof)
        tokens = embeds.clone()
        tokens[action_mask] = action_embed.reshape(-1, embeds.shape[-1]).to(tokens.dtype)
        out = model.model(
            input_ids=None, attention_mask=attention, position_ids=positions, past_key_values=kv,
            inputs_embeds=tokens, moe_token_types=moe_types, start_indices=starts, end_indices=ends,
            use_cache=False, output_attentions=False, output_hidden_states=False,
            return_dict=True, adarms_conds=[None, adarms_cond])
        hidden = out.last_hidden_state[action_mask].to(torch.float32)
        v = preprocessor.action_proj_back(hidden[:, :preprocessor.action_hidden_size])
        return v.reshape(ctx['batch_size'], ctx['action_horizon'], ctx['action_dim'])

    times, x = ctx['times'], noise.clone()
    for i in range(len(times) - 1):
        t, t_next = times[i], times[i + 1]
        if method == 'inpaint':
            if i == 0:
                hidden = prefill.last_hidden_state[ctx['flow_action_mask']].to(torch.float32)
                v = preprocessor.action_proj_back(hidden[:, :preprocessor.action_hidden_size]).reshape(x.shape)
            else:
                v = velocity(t, x)
        else:
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                v = velocity(t, x_in)
                x1 = x_in + (1 - t) * v
                correction, = torch.autograd.grad(x1, x_in, ((target - x1) * weights).detach())
            v = v.detach() + rtc_guidance_weight(float(t)) * correction
        if v_padding is not None:
            v = v_padding * (1 - dof) + v * dof
        x = x + (t_next - t) * v
        if method == 'inpaint':
            x = x + weights * ((1 - t_next) * noise + t_next * target - x)
    return [x], None


def install_rtc_hook(wall):
    """Route wall-x's flow strategy hook through RTC while a guide is set."""
    if not callable(getattr(type(wall), '_execute_flow_action', None)):
        return False
    def execute(self, ctx):
        guide = self._wall_rlt_rtc_guide
        if guide is None:
            return type(self)._execute_flow_action(self, ctx)
        return _rtc_flow_action(self, ctx, guide)
    wall._wall_rlt_rtc_guide = None
    wall._execute_flow_action = types.MethodType(execute, wall)
    return True


def move_inputs(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_inputs(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_inputs(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_inputs(item, device) for item in value)
    return value


class WallRLTInferenceBackend:
    """A serialized model/processor/RNG boundary, also usable without HTTP."""

    def __init__(self, *, wall, encoder, collator, contract, config, manifest,
                 device='cpu', inference_steps=10):
        if type(inference_steps) is not int or not 1 <= inference_steps <= 100:
            raise ValueError('inference_steps must be an integer in [1,100]')
        if config.action_horizon != contract.action_horizon or config.model_action_dim != contract.model_action_dim:
            raise ValueError('model and data action shapes disagree')
        self.wall, self.encoder = wall, encoder
        self.collator, self.contract, self.config = collator, contract, config
        self.manifest = dict(manifest)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.inference_steps = inference_steps
        self.slot_map = SlotMap(contract.action_slots, contract.model_action_dim)
        self.absolute_targets = contract.target_mode == EE_REL6D_NEXT
        self.lock = threading.Lock()
        self.wall.requires_grad_(False).eval()
        self.encoder.requires_grad_(False).eval()
        self.rtc = self.absolute_targets and install_rtc_hook(self.wall)
        self.calls = 0
        self.last_inference_seconds = None
        self.peak_allocated_bytes = None
        self.peak_reserved_bytes = None

    def metadata(self):
        extra = {'absolute_target_dim': 7,
                 'absolute_target_names': ['flange_x', 'flange_y', 'flange_z', 'flange_rotvec_x',
                                           'flange_rotvec_y', 'flange_rotvec_z', 'gripper_command'],
                 'absolute_target_units': ['m'] * 3 + ['rad'] * 3 + ['binary_0_close_1_open'],
                 'absolute_target_frame': self.contract.coordinate_frame,
                 'rtc': self.rtc,
                 } if self.absolute_targets else {}
        return {'backend': 'wall_rlt_stage1', **extra,
                'target_mode': self.contract.target_mode,
                'model_adapter': self.contract.model_adapter,
                'model_origin': self.manifest['model_origin'],
                'artifact_id': self.manifest.get('artifact_id'),
                'training_step': self.manifest.get('training_step'),
                'runtime_model_sources_sha256': self.manifest.get('runtime_model_sources_sha256'),
                'data_contract_hash': self.contract.fingerprint(),
                'action_horizon': self.contract.action_horizon,
                'action_dim': len(self.contract.action_names),
                'action_names': self.contract.action_names,
                'action_units': self.contract.action_units,
                'action_convention': self.contract.action_convention,
                'coordinate_frame': self.contract.coordinate_frame,
                'camera_mapping': self.contract.camera_mapping,
                'state_names': self.contract.state_names,
                'state_dim': len(self.contract.state_names),
                'state_units': self.contract.state_units,
                'action_source': 'WALL_reference_prediction',
                'gripper_output': 'continuous_prediction_of_binary_target',
                'includes_actor': False, 'includes_decoder': False,
                'feature_dim': self.config.embed_dim,
                'robot_execution_supported': False,
                'device': str(self.device),
                'inference_steps': self.inference_steps,
                'image_size': self.collator.image_size,
                'max_length': self.collator.native.dataload_config.get('max_length',768),
                'vision_attention': self.wall.config.vision_config._attn_implementation,
                'wall_attention': self.wall.config._attn_implementation,
                'completed_requests': self.calls,
                'last_inference_seconds': self.last_inference_seconds,
                'peak_allocated_bytes': self.peak_allocated_bytes,
                'peak_reserved_bytes': self.peak_reserved_bytes}

    def _observation(self, observation):
        if set(observation) != {'instruction', 'state', 'images'}:
            raise ValueError('only current instruction, state and images are allowed')
        instruction = observation['instruction']
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('instruction must be a nonempty string')
        state = torch.as_tensor(observation['state'], dtype=torch.float32)
        if state.shape != (len(self.contract.state_names),) or not torch.isfinite(state).all():
            raise ValueError('state dimensions or values disagree with the contract')
        if set(observation['images']) != set(self.contract.camera_mapping):
            raise ValueError('camera keys disagree with the data contract')
        images = {}
        from PIL import Image
        for name, image in observation['images'].items():
            if not isinstance(image, Image.Image) or image.mode != 'RGB':
                raise ValueError('inference images must be decoded RGB PIL images')
            images[name] = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2,0,1)
        return {'instruction': instruction, 'state': state, 'images': images}

    def _rtc_guide(self, rtc, state, inputs):
        """Previous chunk's leftover absolute targets -> normalized model-space prefix
        relative to this observation, plus per-step/per-slot guidance weights."""
        targets = np.asarray(rtc['prefix_targets'], dtype=np.float64)
        horizon = self.contract.action_horizon
        if (targets.ndim != 2 or targets.shape[1] != 7 or not 1 <= len(targets) <= horizon or
                not np.isfinite(targets).all()):
            raise ValueError(f'RTC prefix targets must be finite [1..{horizon},7]')
        delay = rtc['inference_delay']
        if type(delay) is not int or not 0 <= delay <= horizon:
            raise ValueError(f'RTC inference_delay must be an integer in [0,{horizon}]')
        relative = ee_relative_targets(targets, state.double().numpy())
        physical = torch.zeros((1, horizon, relative.shape[1]), dtype=torch.float32)
        physical[0, :len(relative)] = torch.from_numpy(relative).float()
        model, slots = self.slot_map.encode(physical)
        from wall_x.model.core.action.normalizer import normalize_data_with_virtual_tail
        target = normalize_data_with_virtual_tail(
            self.wall.action_preprocessor.normalizer_action, model.to(self.device),
            inputs['dataset_names'], inputs['dof_mask']).float()
        weights = rtc_prefix_weights(delay, len(relative), horizon)[None, :, None] * slots.float()
        method = rtc.get('method', 'inpaint')
        if method not in RTC_METHODS:
            raise ValueError(f'RTC method must be one of {RTC_METHODS}')
        return target, weights.to(self.device), method

    def infer(self, observation, *, return_features=False, seed=None, rtc=None):
        if type(return_features) is not bool or (seed is not None and
                (type(seed) is not int or not 0 <= seed < 2**63)):
            raise ValueError('invalid feature flag or seed')
        if rtc is not None and not self.rtc:
            raise ValueError('this backend does not support RTC guidance')
        sample = self._observation(observation)
        # RTC guidance differentiates through the action postfix; inference tensors
        # cannot be saved for backward, so that path runs under no_grad instead.
        guidance = rtc is not None and rtc.get('method', 'inpaint') == 'guidance'
        with (torch.no_grad() if guidance else torch.inference_mode()):
            return self._infer(sample, return_features, seed, rtc)

    def _infer(self, sample, return_features, seed, rtc):
        with self.lock:
            devices = [self.device.index] if self.device.type == 'cuda' else []
            rng = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
            with rng:
                if seed is not None:
                    torch.random.default_generator.manual_seed(seed)
                    if self.device.type == 'cuda':
                        with torch.cuda.device(self.device):
                            torch.cuda.manual_seed(seed)
                inputs = self.collator.observation_batch([sample])['wall_inputs']
                if any(key in inputs for key in ('action_chunk','sample_time','flow_loss_mask')):
                    raise RuntimeError('observation processor leaked training targets')
                inputs = move_inputs(inputs, self.device)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                if rtc is not None:
                    self.wall._wall_rlt_rtc_guide = self._rtc_guide(rtc, sample['state'], inputs)
                try:
                    result = self.wall.generate_flow_action(
                        action_horizon=self.config.action_horizon,
                        action_dim=self.config.model_action_dim,
                        num_inference_timesteps=self.inference_steps,
                        return_rlt_prefix=return_features, **inputs)
                finally:
                    if self.rtc:
                        self.wall._wall_rlt_rtc_guide = None
                # WALL already unnormalizes its output. Only select the trained
                # physical slots; another unnormalization would corrupt units.
                actions = self.slot_map.decode(result['predict_action']).float()
                expected = (1,self.contract.action_horizon,len(self.contract.action_names))
                if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
                    raise ValueError('model returned invalid physical action predictions')
                response = {'actions': actions[0].cpu().tolist()}
                if self.absolute_targets:
                    # Relative to the observation that produced this chunk, like training.
                    absolute = ee_absolute_window(actions[0].double().cpu().numpy(),
                                                  sample['state'].double().numpy())
                    if not np.isfinite(absolute).all():
                        raise ValueError('model returned invalid absolute EE targets')
                    response['absolute_targets'] = absolute.tolist()
                if return_features:
                    h, mask = result['rlt_prefix_hidden_states'], result['rlt_prefix_mask']
                    if (h.ndim != 3 or h.shape[0] != 1 or h.shape[-1] != self.config.input_dim or
                            mask.shape != h.shape[:2] or mask.dtype != torch.bool or
                            h.shape[1] > self.config.prefix_seq_len or not mask.any(dim=1).all() or
                            not torch.isfinite(h).all()):
                        raise ValueError('model returned an invalid observation prefix')
                    with torch.autocast(device_type=self.device.type, enabled=False):
                        z = self.encoder(h.detach().float(), mask).reshape(1,-1)
                    if z.shape != (1,self.config.embed_dim) or not torch.isfinite(z).all():
                        raise ValueError('encoder returned invalid RL features')
                    response['z_rl'] = z[0].cpu().tolist()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                    self.peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
                    self.peak_reserved_bytes = torch.cuda.max_memory_reserved(self.device)
                self.last_inference_seconds = time.perf_counter()-started
                self.calls += 1
                return response


def load_backend(artifact, *, device='cuda:0', vision_attention='flash_attention_2',
                 inference_steps=10):
    """Production entry: requires an exported, trained AR5 target checkpoint.

    Model parameters are constructed on meta and assigned from inference-only
    safetensors. This avoids allocating a full FP32 training model or decoder.
    It does not reset the learned AR5 projections or load the original base.
    """
    from .artifact import verify_inference_artifact
    root = Path(artifact).resolve()
    manifest = verify_inference_artifact(root)
    if manifest.get('model_origin') != 'stage1_checkpoint' or manifest.get('data_purpose') != 'target':
        raise ValueError('production inference requires a trained target artifact, not a synthetic fixture')
    if vision_attention not in {'eager','flash_attention_2'}:
        raise ValueError('vision attention must be eager or flash_attention_2')
    device = torch.device(device)
    if device.type not in {'cpu','cuda'}:
        raise ValueError('inference supports CPU checks or one CUDA device')
    if device.type == 'cpu' and vision_attention != 'eager':
        raise ValueError('CPU inference requires vision eager attention')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('the selected CUDA device is unavailable')
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('the inference artifact requires BF16 support')
    assets = root/'assets'
    source_digest = verify_model_sources(assets)
    contract = DataContract.load(assets/'data_contract.json')
    contract.require_trainable()
    if contract.fingerprint() != manifest['data_contract_hash']:
        raise ValueError('inference artifact data contract differs from its manifest')
    config = WallRLTConfig.load(assets/'rlt_config.json')
    training_data = json.loads((assets/'training_config.json').read_text()).get('data',{})
    image_size, max_length = training_data.get('image_size',448), training_data.get('max_length',2048)
    if (type(image_size) is not int or not 28 <= image_size <= 448 or
            type(max_length) is not int or not 1 <= max_length <= 2048):
        raise ValueError('trained preprocessing settings exceed the validated inference limits')
    collator = WallCollator(contract, assets/'processor', seed=0, image_size=image_size, max_length=max_length)
    wall, encoder = _load_modules(root, contract, config, collator, device, vision_attention)
    return WallRLTInferenceBackend(wall=wall,encoder=encoder,collator=collator,
        contract=contract,config=config,
        manifest={**manifest,'runtime_model_sources_sha256':source_digest},
        device=device,inference_steps=inference_steps)


def _load_modules(root, contract, config, collator, device, vision_attention):
    """Strict inference construction, separately exercised by tiny-model tests."""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from wall_rlt._vendor.rlt_token_transformer import RLTTokenEncoder
    from wall_rlt.model.builder import build_target_normalizers, _require_prefix_patch
    from wall_rlt.model.ar5_adapter import validate_ar5_model_schema
    from wall_x.model.qact.qwen2_5 import Qwen2_5_VLConfig, Qwen2_5_VLMoEForAction
    assets = root/'assets'
    model_config = Qwen2_5_VLConfig(**json.loads((assets/'adapted_wall_config.json').read_text()))
    if not model_config.mot_opt:
        raise ValueError('selective BF16/FP32 inference requires the validated mot_opt=True path')
    model_config._attn_implementation = 'sdpa'
    model_config.vision_config._attn_implementation = vision_attention
    _require_prefix_patch(Qwen2_5_VLMoEForAction)
    normalizers = build_target_normalizers(contract)
    with init_empty_weights(include_buffers=False):
        wall = Qwen2_5_VLMoEForAction(model_config, collator.native.processor, None)
        encoder = RLTTokenEncoder(**config.token_kwargs())
    wall.set_normalizer(*normalizers)
    wall_state = load_file(str(root/'wall.safetensors'), device='cpu')
    encoder_state = load_file(str(root/'encoder.safetensors'), device='cpu')
    wall.load_state_dict(wall_state, strict=True, assign=True)
    encoder.load_state_dict(encoder_state, strict=True, assign=True)
    if model_config.tie_word_embeddings:
        if not torch.equal(wall.lm_head.weight, wall.model.embed_tokens.weight):
            raise ValueError('tied embedding/head weights disagree in the inference artifact')
        # strict assign restores separate tensors. Recover the trained tying
        # before moving to CUDA, saving a second full vocabulary allocation.
        wall.lm_head.weight = wall.model.embed_tokens.weight
    del wall_state, encoder_state
    if any(p.is_meta for m in (wall,encoder) for p in m.parameters()):
        raise RuntimeError('inference weights did not materialize every model parameter')
    validate_ar5_model_schema(wall, contract.model_adapter)
    if collator.action_token_id != config.action_token_id:
        raise ValueError('processor action token differs from the trained model')
    wall.to(device); encoder.to(device)
    return wall, encoder
