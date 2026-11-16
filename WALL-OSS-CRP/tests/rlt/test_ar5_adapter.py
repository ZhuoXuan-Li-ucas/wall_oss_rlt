from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wall_rlt.model.ar5_adapter import (
    AR5_ACTION_LAYOUT, AR5_STATE_LAYOUT, AR5_RESET_PARAMETERS, AR5_VARIANT,
    reset_ar5_action_projections,
)


def tiny_wall():
    wall = nn.Module()
    wall.config = SimpleNamespace(
        dof_config=dict(AR5_ACTION_LAYOUT), agent_pos_config=dict(AR5_STATE_LAYOUT),
        use_state_string_representation=True, proj_with_mask=True,
        action_hidden_size=4, initializer_range=0.02,
    )
    proc = nn.Module()
    proc.dof_config = dict(AR5_ACTION_LAYOUT)
    proc.agent_pos_config = dict(AR5_STATE_LAYOUT)
    proc.action_dim = proc.propri_dim = 26
    proc.action_hidden_size = 4
    proc.w1 = nn.Linear(52, 4, bias=False)
    proc.action_proj_back = nn.Linear(4, 26, bias=False)
    proc.time_mlp_in = nn.Linear(4, 4)
    wall.action_preprocessor = proc
    wall.backbone = nn.Linear(4, 4)
    wall.register_buffer('normalizer_sentinel', torch.tensor([1., 2.]))
    return wall


def test_only_declared_projections_change_without_touching_rng():
    wall = tiny_wall()
    before = {k:v.clone() for k,v in wall.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    report = reset_ar5_action_projections(wall, seed=29)
    changed = {k for k,v in wall.state_dict().items() if not torch.equal(v,before[k])}
    assert changed == set(AR5_RESET_PARAMETERS)
    assert report['variant'] == AR5_VARIANT
    assert report['reset_parameters'] == list(AR5_RESET_PARAMETERS)
    assert report['training_required'] and report['target_normalizers_required']
    assert torch.equal(rng, torch.random.get_rng_state())


def test_private_initialization_is_reproducible_and_guarded():
    a, b = tiny_wall(), tiny_wall()
    reset_ar5_action_projections(a, seed=29)
    torch.randn(100)
    reset_ar5_action_projections(b, seed=29)
    for name in AR5_RESET_PARAMETERS:
        assert torch.equal(a.state_dict()[name], b.state_dict()[name])
    with pytest.raises(ValueError, match='already been reset'):
        reset_ar5_action_projections(a)


@pytest.mark.parametrize('fault', ['layout', 'order', 'width', 'state', 'mask', 'bias', 'variant'])
def test_incompatible_schema_fails_before_any_mutation(fault):
    wall = tiny_wall()
    kwargs = {}
    if fault == 'layout':
        wall.config.dof_config = {'old_ee_slots':26}
    elif fault == 'order':
        wall.config.dof_config = dict(reversed(list(AR5_ACTION_LAYOUT.items())))
    elif fault == 'width':
        wall.action_preprocessor.action_proj_back = nn.Linear(4,8,bias=False)
    elif fault == 'state':
        wall.config.use_state_string_representation = False
    elif fault == 'mask':
        wall.config.proj_with_mask = False
    elif fault == 'bias':
        wall.action_preprocessor.w1 = nn.Linear(52,4,bias=True)
    else:
        kwargs['variant'] = 'implicit_joint_mapping'
    before = {k:v.clone() for k,v in wall.state_dict().items()}
    with pytest.raises(ValueError):
        reset_ar5_action_projections(wall, **kwargs)
    assert all(torch.equal(v,before[k]) for k,v in wall.state_dict().items())
