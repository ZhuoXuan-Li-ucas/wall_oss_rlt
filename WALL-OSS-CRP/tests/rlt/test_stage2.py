from contextlib import nullcontext
import copy

import pytest
import torch

from wall_rlt.stage2 import single_step_transition,synthetic_ac_check
from wall_rlt.data.transforms import BoundedActionTransform


def arguments():
    obs = {'z_rl':torch.ones(3,4),'proprio':torch.zeros(3,8),'ref_chunk':torch.zeros(3,1,2)}
    return dict(curr_obs=obs,next_obs=copy.deepcopy(obs),actions=torch.zeros(3,1,2),
                rewards=torch.tensor([[1.],[2.],[3.]]),
                terminated=torch.tensor([[False],[True],[False]]),
                truncated=torch.tensor([[False],[False],[True]]),
                origin='synthetic',delta_t=torch.full((3,1),1/30),next_obs_kind='pre_reset')


def test_autoreset_uses_final_observation_and_preserves_timeout_bootstrap():
    args=arguments(); args['next_obs_kind']='auto_reset'
    with pytest.raises(ValueError,match='actual final'):
        single_step_transition(**args)
    args['final_obs']={k:torch.full_like(v,.5) for k,v in args['next_obs'].items()}
    batch=single_step_transition(**args)
    assert batch['terminations'].flatten().tolist()==[False,True,False]
    assert batch['dones'].flatten().tolist()==[False,True,True]
    assert batch['next_obs']['z_rl'][0].eq(1).all()
    assert batch['next_obs']['z_rl'][1:].eq(.5).all()


@pytest.mark.parametrize('key,value,error',[
    ('actions',torch.zeros(3,2,2),'without chunk padding'),
    ('actions',torch.full((3,1,2),2.),'RL'),
    ('rewards',torch.zeros(3,2),'single-step rewards'),
    ('delta_t',torch.full((3,1),.1),'one declared observation interval'),
    ('terminated',torch.zeros(3,1),'bool'),
    ('origin','demonstration','origin'),
    ('origin','recorded_transition','provenance'),
])
def test_invalid_or_fabricated_transition_is_rejected(key,value,error):
    args=arguments(); args[key]=value
    with pytest.raises(ValueError,match=error):
        single_step_transition(**args)


def test_upstream_critic_matches_hand_computed_one_step_discount():
    from omegaconf import OmegaConf
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin
    from rlinf.models.embodiment.base_policy import ForwardType
    class Model:
        def __call__(self,forward_type,**kwargs):
            if forward_type==ForwardType.SAC:
                return torch.zeros(3,2),torch.zeros(3,2),None
            return torch.tensor([[2.,3.],[4.,5.],[6.,7.]])
    class Harness(RLTACLossMixin):
        def worker_timer(self,*a,**kw): return nullcontext()
    h=Harness()
    h.cfg=OmegaConf.create({'algorithm':{'gamma':.5},'env':{'train':None}})
    h.torch_dtype=torch.float32
    h.model=Model()
    h.target_model=lambda **kwargs:torch.tensor([[10.,20.],[10.,20.],[10.,20.]])
    loss,_=RLTACLossMixin.forward_critic.__wrapped__(h,single_step_transition(**arguments()))
    # Termination: reward only. Truncation: bootstrap from pre-reset final obs.
    expected=torch.tensor([[6.,6.],[2.,2.],[8.,8.]])
    torch.testing.assert_close(loss,((h.model(ForwardType.SAC_Q)-expected)**2).mean())


def test_upstream_bc_routes_only_actual_interventions():
    from omegaconf import OmegaConf
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin
    h=RLTACLossMixin()
    h.cfg=OmegaConf.create({'actor':{'model':{'num_action_chunks':1,'action_dim':2}}})
    loss,_=h._bc_metrics(torch.zeros(2,2),torch.full((2,1,2),.75),torch.full((2,1,2),.25),
                         torch.tensor([[[False,False]],[[True,True]]]))
    torch.testing.assert_close(loss,torch.tensor((.25**2+.75**2)/2))


def test_actual_rlinf_actor_and_critic_each_update_on_synthetic_data():
    result=synthetic_ac_check(seed=5)
    assert result['status']=='PASS'
    assert result['actor_updates']==result['critic_updates']==1
    assert result['trained_WALL_encoder_used'] is False


def test_bounded_actions_round_trip_without_silent_clipping():
    t=BoundedActionTransform([-2,0],[2,1])
    action=torch.tensor([[-2.,0.],[0.,.5],[2.,1.]])
    torch.testing.assert_close(t.from_rl(t.to_rl(action)),action)
    with pytest.raises(ValueError,match='no implicit clipping'):
        t.to_rl(torch.tensor([[3.,1.]]))
