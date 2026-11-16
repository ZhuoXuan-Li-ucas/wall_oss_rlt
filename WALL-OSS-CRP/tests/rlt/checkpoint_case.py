"""Small real RLT optimizer fixture used in a fresh Python process."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader

from wall_rlt.config import WallRLTConfig
from wall_rlt.model.policy import WallRLTPolicy
from wall_rlt.checkpoint import save_training_checkpoint, load_training_checkpoint

METADATA = {'contract_hash':'synthetic-checkpoint-fixture-v1',
            'config':{'input_dim':8,'embed_dim':8,'dropout':.15,'alpha':1}}


class WallFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3,8)
        self.head = nn.Linear(8,2)

    def forward(self, *, input_ids, obs, action_chunk, sample_time, **kwargs):
        h = self.encoder(obs)
        noise = torch.rand_like(action_chunk)*.01 + random.random()*.01 + float(np.random.rand())*.01
        loss = ((self.head(h[:,0]) - action_chunk - sample_time[:,None]*noise)**2).mean()
        return SimpleNamespace(flow_loss=loss,rlt_prefix_hidden_states=h,
                               rlt_prefix_mask=torch.ones(h.shape[:2],dtype=torch.bool))


class Collator:
    def __init__(self):
        self.rng = np.random.default_rng(91)

    def state_dict(self):
        return copy.deepcopy(self.rng.bit_generator.state)

    def load_state_dict(self,state):
        self.rng.bit_generator.state = copy.deepcopy(state)

    def __call__(self,rows):
        values = torch.stack(rows)
        return {'wall_inputs':{'input_ids':torch.ones(len(rows),4,dtype=torch.long),
                              'obs':values[:,:12].reshape(-1,4,3),
                              'action_chunk':values[:,12:14],
                              'sample_time':torch.tensor(self.rng.beta(1.5,1.,len(rows)),dtype=torch.float32)}}


def components(seed=17):
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg = WallRLTConfig(input_dim=8,embed_dim=8,prefix_seq_len=8,num_layers=1,num_heads=2,
                        mlp_ratio=2.,dropout_rate=.15,action_horizon=1,model_action_dim=2)
    model = WallRLTPolicy(WallFixture(),cfg)
    opt = torch.optim.AdamW(model.parameters(),lr=.002)
    scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=2,gamma=.8)
    collator = Collator()
    rows = torch.arange(20*14,dtype=torch.float32).reshape(20,14)/280
    loader = StatefulDataLoader(rows,batch_size=2,shuffle=True,num_workers=0,collate_fn=collator,
                                generator=torch.Generator().manual_seed(42))
    return model,opt,scheduler,loader,collator


def update(model,opt,scheduler,batch):
    output = model(data=batch)
    output['loss'].backward()
    opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)
    return {k:v.detach().clone() for k,v in output.items()}


def continuation(path, *, resume):
    model,opt,scheduler,loader,collator = components(991 if resume else 17)
    if resume:
        model._rlt_requires_checkpoint_restore = True
        restored = load_training_checkpoint(path,model,opt,scheduler,expected_metadata=METADATA,
                                            loader=loader,collator=collator)
        assert restored['step'] == 1
        iterator = restored['iterator']
    else:
        iterator = iter(loader)
        update(model,opt,scheduler,next(iterator))
        save_training_checkpoint(path,model,opt,scheduler,step=1,metadata=METADATA,
                                 loader=loader,collator=collator)
    batch = next(iterator)
    losses = update(model,opt,scheduler,batch)
    return {'batch':batch['wall_inputs'],'losses':losses,'weights':model.state_dict(),
            'lr':opt.param_groups[0]['lr']}


if __name__ == '__main__':
    import sys
    torch.save(continuation(sys.argv[1],resume=True),sys.argv[2])
