"""Single-step replay contract and a synthetic CPU check of upstream RLT AC.

The top_error demonstrations do not supply rewards or termination labels.
This module never fabricates those fields for a demonstration recording.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import json
import math

import torch


def _features(obs, *, batch_size, action_dim):
    result = {}
    for key in ('z_rl','proprio','ref_chunk'):
        value = obs[key]
        if not isinstance(value,torch.Tensor) or not value.is_floating_point() or value.requires_grad:
            raise ValueError('replay features must be detached floating tensors')
        if value.ndim < 1 or value.shape[0] != batch_size or not torch.isfinite(value).all():
            raise ValueError('invalid feature batch size or finite values')
        if key == 'ref_chunk':
            if value.shape != (batch_size,1,action_dim) or (value.abs()>1+1e-6).any():
                raise ValueError('reference must be a single action in RL [-1,1] space')
        elif value.ndim != 2 or value.shape[1] < 1:
            raise ValueError('z_rl and proprio must be nonempty [B,D]')
        result[key] = value.detach().float().clone()
    return result


def single_step_transition(*, curr_obs, next_obs, actions, rewards, terminated, truncated,
                           origin, delta_t, next_obs_kind, final_obs=None, provenance=None,
                           fps=30.0, intervene_flags=None):
    """Build one real step per row, with explicit terminal/reset observation kind.

    Standard bootstrap uses termination, not time-limit truncation. For an
    autoreset transition, final_obs must contain the pre-reset terminal features.
    Do not pass a demonstration target as an executed action.
    """
    if origin not in {'synthetic','recorded_transition'}:
        raise ValueError('origin must explicitly identify synthetic or recorded transitions')
    if origin == 'recorded_transition' and (not provenance or any(
            not provenance.get(k) for k in ('executed_actions','rewards','termination_flags'))):
        raise ValueError('recorded transitions require execution/reward/termination provenance')
    if not isinstance(actions,torch.Tensor) or actions.ndim != 3 or actions.shape[1] != 1:
        raise ValueError('Stage 2 baseline requires actions [B,1,A], without chunk padding')
    b,_,a = actions.shape
    if b<1 or a<1 or not actions.is_floating_point() or not torch.isfinite(actions).all() or (actions.abs()>1+1e-6).any():
        raise ValueError('actions must be finite in RL [-1,1] space')
    for flag in (terminated,truncated):
        if not isinstance(flag,torch.Tensor) or flag.dtype != torch.bool or flag.shape != (b,1):
            raise ValueError('termination/truncation flags must be bool [B,1]')
    if not isinstance(rewards,torch.Tensor) or rewards.shape != (b,1) or not rewards.is_floating_point() or not torch.isfinite(rewards).all():
        raise ValueError('explicit finite single-step rewards [B,1] are required')
    dt = torch.as_tensor(delta_t,dtype=torch.float32)
    if not math.isfinite(fps) or fps <= 0 or dt.shape != (b,1) or not torch.isfinite(dt).all() or (torch.abs(dt-1/fps) > .001).any():
        raise ValueError('one transition must correspond to one declared observation interval')
    curr = _features(curr_obs,batch_size=b,action_dim=a)
    nxt = _features(next_obs,batch_size=b,action_dim=a)
    if any(curr[k].shape != nxt[k].shape for k in curr):
        raise ValueError('current and next feature dimensions must match')
    ended = terminated | truncated
    if next_obs_kind == 'auto_reset':
        if ended.any():
            if final_obs is None:
                raise ValueError('autoreset transitions require actual final observations')
            final = _features(final_obs,batch_size=b,action_dim=a)
            for key in nxt:
                if final[key].shape != nxt[key].shape:
                    raise ValueError('final feature dimensions must match next observations')
                mask = ended.reshape((b,)+(1,)*(nxt[key].ndim-1))
                nxt[key] = torch.where(mask,final[key],nxt[key])
    elif next_obs_kind != 'pre_reset':
        raise ValueError('next_obs_kind must explicitly be pre_reset or auto_reset')
    batch = {'curr_obs':curr,'next_obs':nxt,'actions':actions.detach().float().clone(),
             'rewards':rewards.detach().float().clone(),'terminations':terminated.clone(),
             'truncations':truncated.clone(),'dones':ended.clone(),
             'origin':origin,'provenance':dict(provenance or {}),'delta_t':dt.clone()}
    if intervene_flags is not None:
        if intervene_flags.dtype != torch.bool or intervene_flags.shape != actions.shape:
            raise ValueError('intervention flags must be bool [B,1,A]')
        batch['intervene_flags'] = intervene_flags.clone()
    return batch


def synthetic_ac_check(seed=0):
    """One critic and actor update using the *actual* pinned RLinf loss mixin."""
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    class Harness(RLTACLossMixin):
        def worker_timer(self,*args,**kwargs):
            return nullcontext()
    harness = Harness()
    harness.cfg = OmegaConf.create({'actor':{'model':{'num_action_chunks':1,'action_dim':8}},
                                    'algorithm':{'gamma':.99,'bc_weight':1.,'q_weight':1.,
                                                 'q_head_type':'default','bootstrap_type':'standard'},
                                    'env':{'train':None}})
    harness.torch_dtype = torch.float32
    harness.model = RLTMLPPolicy(z_dim=8,proprio_dim=8,action_dim=8,num_action_chunks=1)
    harness.target_model = copy.deepcopy(harness.model).eval().requires_grad_(False)
    # fixed_std is part of the pinned RLT policy. Inherited learned std head is
    # unused and must not be mistaken for an entropy/alpha update.
    harness.model.actor_logstd.requires_grad_(False)
    actor_params = list(harness.model.backbone.parameters())+list(harness.model.actor_mean.parameters())
    critic_params = list(harness.model.q_head.parameters())
    actor_opt = torch.optim.Adam(actor_params,lr=1e-3)
    critic_opt = torch.optim.Adam(critic_params,lr=1e-3)
    def obs():
        return {'z_rl':torch.randn(3,8),'proprio':torch.randn(3,8),'ref_chunk':torch.zeros(3,1,8)}
    batch = single_step_transition(curr_obs=obs(),next_obs=obs(),actions=torch.zeros(3,1,8),
                rewards=torch.tensor([[0.],[1.],[-.1]]),terminated=torch.tensor([[False],[True],[False]]),
                truncated=torch.tensor([[False],[False],[True]]),origin='synthetic',
                delta_t=torch.full((3,1),1/30),next_obs_kind='pre_reset')
    before_actor = [p.detach().clone() for p in actor_params]
    before_critic = [p.detach().clone() for p in critic_params]
    before_target = {k:v.clone() for k,v in harness.target_model.state_dict().items()}
    features = {side:{k:v.clone() for k,v in batch[side].items()} for side in ('curr_obs','next_obs')}
    # Bypass only the Worker.timer decorator (requires a launched accelerator
    # worker); execute the exact upstream objective functions on the CPU model.
    critic_loss,_ = RLTACLossMixin.forward_critic.__wrapped__(harness,batch)
    critic_loss.backward(); critic_opt.step(); critic_opt.zero_grad(set_to_none=True)
    for p in critic_params:
        p.requires_grad_(False)
    actor_loss,_,metrics = RLTACLossMixin.forward_actor.__wrapped__(harness,batch)
    actor_loss.backward(); actor_opt.step(); actor_opt.zero_grad(set_to_none=True)
    for p in critic_params:
        p.requires_grad_(True)
    if not torch.isfinite(critic_loss) or not torch.isfinite(actor_loss):
        raise AssertionError('non-finite AC loss')
    actor_changed = any(not torch.equal(a,b) for a,b in zip(before_actor,actor_params))
    critic_changed = any(not torch.equal(a,b) for a,b in zip(before_critic,critic_params))
    if not actor_changed or not critic_changed:
        raise AssertionError('actor and critic must both update')
    for key,value in before_target.items():
        torch.testing.assert_close(harness.target_model.state_dict()[key],value,rtol=0,atol=0)
    for side in features:
        for key,value in features[side].items():
            torch.testing.assert_close(batch[side][key],value,rtol=0,atol=0)
    return {'status':'PASS','data_origin':'synthetic','implementation':'pinned RLinf RLTMLPPolicy + RLTACLossMixin',
            'actor_updates':1,'critic_updates':1,'actor_loss':actor_loss.item(),
            'critic_loss':critic_loss.item(),'bc_loss':metrics['bc_loss'],
            'feature_tensors_unchanged':True,'target_network_unchanged':True,
            'entropy_alpha_updates':0,'trained_WALL_encoder_used':False,
            'real_replay_or_online_learning':'NOT_RUN'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic',action='store_true',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    from wall_rlt.data.audit import write_json
    report = synthetic_ac_check()
    write_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
