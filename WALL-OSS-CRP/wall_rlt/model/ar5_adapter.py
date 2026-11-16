"""Explicit transfer variants for the AR5 + OmniPicker dataset.

The public WALL checkpoint has EE-position/6D-rotation slots. Keeping 26 tensor
columns is shape compatibility only; it does not make seven joint angles into
EE coordinates. For joint8, construct WALL with the layouts below, load backbone
weights, then reset the projections once for a *new* transfer run (never on
resume). The EE variant keeps the checkpoint's own Wall-OSS-0.5 layout and its
pretrained action projections; AR5 occupies the right-arm slots 10..19.
Target train-only normalizers must replace the checkpoint normalizers in both.
"""
from __future__ import annotations

import math

import torch
from torch import nn


AR5_VARIANT = 'ar5_joint8_reinit_v1'
AR5_ACTION_LAYOUT = {
    'ar5_joint_absolute': 7,
    'omnipicker_command': 1,
    'action_padding': 18,
}
AR5_STATE_LAYOUT = {
    'ar5_joint_measured': 7,
    'omnipicker_opening_measured': 1,
    'action_padding': 18,
}
# Copied verbatim from the released wall-oss-0.5 config.yml (wall-x ce2a1ae).
EE_VARIANT = 'wall_oss05_right_arm_ee_v1'
WALL_OSS05_ACTION_LAYOUT = {
    'follow_left_ee_cartesian_pos_relative': 3,
    'follow_left_ee_rotation_6D_relative': 6,
    'follow_left_gripper': 1,
    'follow_right_ee_cartesian_pos_relative': 3,
    'follow_right_ee_rotation_6D_relative': 6,
    'follow_right_gripper': 1,
    'velocity_decomposed': 3,
    'height': 1,
    'head_actions': 2,
}
WALL_OSS05_STATE_LAYOUT = {
    'follow_left_ee_cartesian_pos': 3,
    'follow_left_ee_rotation_6D': 6,
    'follow_left_gripper': 1,
    'follow_right_ee_cartesian_pos': 3,
    'follow_right_ee_rotation_6D': 6,
    'follow_right_gripper': 1,
    'velocity_decomposed': 3,
    'height': 1,
    'head_actions': 2,
}
VARIANT_LAYOUTS = {
    AR5_VARIANT: (AR5_ACTION_LAYOUT, AR5_STATE_LAYOUT),
    EE_VARIANT: (WALL_OSS05_ACTION_LAYOUT, WALL_OSS05_STATE_LAYOUT),
}
AR5_RESET_PARAMETERS = (
    'action_preprocessor.w1.weight',
    'action_preprocessor.action_proj_back.weight',
)


def variant_layouts(variant):
    if variant not in VARIANT_LAYOUTS:
        raise ValueError(f'unsupported model adaptation variant: {variant}')
    return VARIANT_LAYOUTS[variant]


def validate_ar5_model_schema(wall, variant=AR5_VARIANT):
    """Fail before changing any weight if the declared layout is incompatible."""
    action_layout, state_layout = variant_layouts(variant)
    config = wall.config
    for name, expected in (('dof_config', action_layout),
                           ('agent_pos_config', state_layout)):
        value = getattr(config, name, None)
        if not isinstance(value, dict) or list(value.items()) != list(expected.items()):
            raise ValueError(f'{name} must explicitly declare ordered {expected}')
    if getattr(config, 'use_state_string_representation', None) is not True:
        raise ValueError(f'{variant} requires state-string inputs')
    proc = wall.action_preprocessor
    if hasattr(proc, 'propri_proj'):
        raise ValueError('continuous proprioception projection is unsupported in this variant')
    for name, expected in (('dof_config', action_layout),
                           ('agent_pos_config', state_layout)):
        value = getattr(proc, name, None)
        if not isinstance(value, dict) or list(value.items()) != list(expected.items()):
            raise ValueError(f'action_preprocessor.{name} disagrees with the AR5 schema')
    if getattr(proc, 'action_dim', None) != 26 or getattr(proc, 'propri_dim', None) != 26:
        raise ValueError('AR5 variant requires 26 model action and state columns')
    hidden = getattr(config, 'action_hidden_size', None)
    if type(hidden) is not int or hidden < 1:
        raise ValueError('action_hidden_size must be positive')
    if getattr(proc, 'action_hidden_size', None) != hidden:
        raise ValueError('action hidden size differs between config and processor')
    # The project collator always supplies a dof_mask. WALL's training path
    # concatenates it unconditionally, so this variant must enable proj_with_mask.
    if getattr(config, 'proj_with_mask', None) is not True:
        raise ValueError('AR5 masked training requires proj_with_mask=True')
    modules = (proc.w1, proc.action_proj_back)
    expected_shapes = ((hidden, 52), (26, hidden))
    for module, shape, name in zip(modules, expected_shapes, AR5_RESET_PARAMETERS):
        if not isinstance(module, nn.Linear) or tuple(module.weight.shape) != shape:
            raise ValueError(f'{name} must be an nn.Linear weight with shape {shape}')
        if module.bias is not None:
            raise ValueError(f'{name} must have no bias, as in the pinned WALL source')
        if module.weight.is_meta or not module.weight.is_floating_point():
            raise ValueError(f'{name} must be a materialized floating point parameter')
    if proc.w1.weight is proc.action_proj_back.weight:
        raise ValueError('AR5 action projections cannot share a parameter')
    std = float(getattr(config, 'initializer_range', 0.02))
    if not math.isfinite(std) or std <= 0:
        raise ValueError('initializer_range must be finite and positive')
    return modules, std


@torch.no_grad()
def reset_ar5_action_projections(wall, *, seed=0, variant=AR5_VARIANT):
    """Reset only input/output action projection weights using private RNGs.

    This is initialization of a new adaptation, not a trained AR5 model. The
    caller must save the report with its config and must not call it when loading
    an already adapted checkpoint. Global training RNG state is preserved.
    """
    if variant != AR5_VARIANT:
        raise ValueError(f'unsupported model adaptation variant: {variant}')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    if getattr(wall, '_ar5_projection_reset_report', None) is not None:
        raise ValueError('AR5 projections have already been reset on this model instance')
    modules, std = validate_ar5_model_schema(wall)
    generators = {}
    for module in modules:
        device = module.weight.device
        if device not in generators:
            generators[device] = torch.Generator(device=device).manual_seed(seed)
        nn.init.normal_(module.weight, mean=0.0, std=std, generator=generators[device])
    report = {
        'variant': variant,
        'reset_parameters': list(AR5_RESET_PARAMETERS),
        'initialization': 'normal',
        'mean': 0.0,
        'std': std,
        'seed': seed,
        'action_layout': dict(AR5_ACTION_LAYOUT),
        'state_layout': dict(AR5_STATE_LAYOUT),
        'target_normalizers_required': True,
        'training_required': True,
    }
    wall._ar5_projection_reset_report = report
    return report
