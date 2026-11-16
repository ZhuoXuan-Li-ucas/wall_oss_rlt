"""Small fixture tests of wrapper/official RLT semantics; no WALL weights loaded."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wall_rlt.config import WallRLTConfig
from wall_rlt.model.policy import WallRLTPolicy
from wall_rlt.model.prefix import observation_prefix_mask, pack_prefix
from wall_rlt.data.transforms import BoundedActionTransform, SlotMap


class FixtureWall(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16,8)
        self.action_head = nn.Linear(8,2)
        self.config = SimpleNamespace()
        self.last_call = None

    def forward(self, input_ids, attention_mask, action_chunk, **kwargs):
        self.last_call = kwargs
        assert kwargs['mode'] == 'train'
        assert kwargs['flow_loss_only'] and kwargs['return_rlt_prefix'] and kwargs['return_dict']
        assert kwargs['labels'].eq(-100).all()
        h,mask = pack_prefix(self.embedding(input_ids),observation_prefix_mask(input_ids,attention_mask,15))
        loss = (self.action_head(h[:,0]) - action_chunk[:,0]).square().mean()
        return SimpleNamespace(flow_loss=loss,rlt_prefix_hidden_states=h,rlt_prefix_mask=mask)

    def generate_flow_action(self,input_ids,attention_mask,**kwargs):
        h,mask = pack_prefix(self.embedding(input_ids),observation_prefix_mask(input_ids,attention_mask,15))
        return {'rlt_prefix_hidden_states':h,'rlt_prefix_mask':mask,
                'predict_action':torch.zeros(len(input_ids),2,2)}


@pytest.fixture
def policy():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg=WallRLTConfig(input_dim=8,embed_dim=8,num_heads=2,num_layers=1,
                      prefix_seq_len=8,mlp_ratio=2,action_horizon=2,model_action_dim=2,
                      action_token_id=15,rlt_alpha=2.)
    return WallRLTPolicy(FixtureWall(),cfg)


def batch():
    return {'wall_inputs':{'input_ids':torch.tensor([[1,2,3,15,15],[0,4,5,15,15]]),
            'attention_mask':torch.tensor([[1,1,1,1,1],[0,1,1,1,1]]),
            'action_chunk':torch.ones(2,2,2)}}


def has_grad(module):
    return any(p.grad is not None and bool(p.grad.abs().sum()) for p in module.parameters())


def test_reconstruction_gradient_does_not_train_wall(policy):
    output=policy(data=batch())
    output['rlt_loss'].backward()
    assert not has_grad(policy.wall)
    assert has_grad(policy.rlt_module.encoder) and has_grad(policy.rlt_module.decoder)


def test_flow_gradient_trains_wall_only(policy):
    policy(data=batch())['vla_loss'].backward()
    assert has_grad(policy.wall.embedding) and has_grad(policy.wall.action_head)
    assert not has_grad(policy.rlt_module)


def test_joint_optimizer_updates_both_branches_and_total_loss(policy):
    before={k:v.detach().clone() for k,v in policy.named_parameters()}
    optimizer=torch.optim.AdamW(policy.parameters(),lr=.001)
    output=policy(data=batch())
    torch.testing.assert_close(output['loss'],output['rlt_loss']+2*output['vla_loss'])
    output['loss'].backward(); optimizer.step()
    for prefix in ('wall.','rlt_module.encoder.','rlt_module.decoder.'):
        assert any(not torch.equal(before[n],p) for n,p in policy.named_parameters() if n.startswith(prefix))


def test_evaluate_loss_restores_training_and_never_keeps_graph(policy):
    output=policy.evaluate_loss(batch())
    assert policy.training
    assert all(not value.requires_grad for value in output.values())


def test_decoder_teacher_forcing_is_causal_at_fixed_latent(policy):
    module=policy.rlt_module
    h=torch.randn(2,6,8); mask=torch.ones(2,6,dtype=torch.bool)
    z=torch.randn(2,1,8)
    changed=h.clone(); changed[:,3:]+=100
    a,b=module.decode(z,h,mask),module.decode(z,changed,mask)
    torch.testing.assert_close(a[:,:4],b[:,:4],rtol=0,atol=0)
    assert not torch.equal(a[:,4:],b[:,4:])


def test_masked_tokens_do_not_change_encoder_or_valid_reconstruction(policy):
    h=torch.randn(2,6,8)
    mask=torch.tensor([[1,1,1,0,0,0],[1,1,1,1,1,0]],dtype=torch.bool)
    changed=h.clone(); changed[~mask]=1000
    module=policy.rlt_module
    a,z=module.reconstruct(h,mask); b,z2=module.reconstruct(changed,mask)
    torch.testing.assert_close(z,z2,rtol=0,atol=0)
    torch.testing.assert_close(a[mask],b[mask],rtol=0,atol=0)
    torch.testing.assert_close(module(h,mask)[0],module(changed,mask)[0],rtol=0,atol=0)


def test_frozen_feature_interface_and_no_action_targets(policy):
    obs=batch(); obs['wall_inputs'].pop('action_chunk')
    obs.update(action_transform=BoundedActionTransform([-1,-1],[1,1]),
               action_slot_map=SlotMap([0,1],2),proprio=torch.zeros(2,8))
    with pytest.raises(RuntimeError,match='freeze_features'):
        policy.extract_rlt_obs(obs)
    policy.freeze_features()
    before={k:v.clone() for k,v in policy.state_dict().items()}
    features=policy.extract_rlt_obs(obs)
    assert features['z_rl'].shape == (2,8)
    assert features['ref_chunk'].shape == (2,1,2)
    assert all(not p.requires_grad for p in policy.parameters())
    assert all(torch.equal(v,policy.state_dict()[k]) for k,v in before.items())
    with pytest.raises(RuntimeError,match='frozen Stage 2'):
        policy.train()
    with pytest.raises(ValueError,match='demonstration actions'):
        policy.extract_rlt_obs({**obs,'wall_inputs':batch()['wall_inputs']})


@pytest.mark.parametrize('mode',['default','sac','ppo'])
def test_unsupported_training_modes_fail(policy,mode):
    with pytest.raises(NotImplementedError):
        policy(forward_type=mode,data=batch())


def test_resume_instance_cannot_run_before_weights_are_restored(policy):
    policy._rlt_requires_checkpoint_restore=True
    with pytest.raises(RuntimeError,match='checkpoint restore'):
        policy(data=batch())
    policy.freeze_features()
    with pytest.raises(RuntimeError,match='checkpoint restore'):
        policy.extract_rlt_obs({})


def test_invalid_prefix_mask_dtype_is_rejected(policy):
    with pytest.raises(ValueError,match='boolean'):
        policy._prefix(SimpleNamespace(rlt_prefix_hidden_states=torch.zeros(1,2,8),
                                       rlt_prefix_mask=torch.ones(1,2)))
