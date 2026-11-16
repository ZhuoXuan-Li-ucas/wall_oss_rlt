"""Actor/critic container and configs for Evo-RLT's ``OnlineRLTrainer``.

``OnlineRLTrainer`` (vendored unchanged) needs a policy object with
``actor``, ``critic``, ``target_critic``, a ``_critic_step`` buffer,
``forward(batch) -> (loss, info)``, ``train/eval/parameters`` and
``save_pretrained``. It also mutates ``policy_cfg.tau`` and
``policy_cfg.actor_update_interval`` around each update, so ``forward`` must
read both from the *same* config object on every call. Upstream's
``ChunkACPolicy`` provides this but drags in LeRobot and pi0.5; this module
is the minimal replacement. ``forward`` and the actor-loss helper follow
``evo_rlt/adapters/lerobot/policies/modeling_rlt_ac.py`` lines 191-275 of the
2026-08-03 archive.

Every field that changes the experiment is required: the values are user
decisions (D1-D13 in ``WALL_RLT_Stage2_Research_Plan.md``), not code defaults.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from rlt_online.evo import ChunkActor, TwinCritic, critic_loss, soft_update

RL_TOKEN_DIM = 2048        # z_rl returned by the inference service
ACTION_DIM = 10            # AR5 EE contract row
PROPRIO_DIM = 10           # AR5 EE contract state
WALL_HORIZON = 32          # steps per WALL prediction window


@dataclass(kw_only=True)
class ChunkACConfig:
    """Network and update-rule settings. Names match upstream ``ChunkACPolicyConfig``."""

    chunk_length: int                       # D1
    gamma: float                            # per 30 Hz step
    beta: float                             # BC weight, summed over C*A dims (scale with C)
    tau: float
    actor_update_interval: int              # critic updates per actor update
    utd_ratio: int
    target_q_clip: float | None
    actor_hidden_dim: int
    actor_num_layers: int
    actor_activation: str
    actor_layer_norm: bool
    actor_residual: bool
    actor_residual_to_ref: bool
    actor_fixed_std: float                  # 0 under D6: exploration is rlt_online.exploration
    actor_ref_dropout_p: float
    actor_action_clip_delta: float | None   # rollout-only safety bound around the reference
    actor_clip_exempt_dims: list[int]       # row dims the clip ignores; D14: [9] (gripper)
    actor_bc_dim_weights: list[float]       # per row dim, repeated over C; all 1 == upstream
    critic_hidden_dim: int
    critic_num_layers: int
    critic_activation: str
    critic_layer_norm: bool
    critic_residual: bool
    rankq_alpha_success: float = 0.0
    rankq_alpha_failure: float = 0.0
    rankq_noise_scale: float = 0.15
    actor_rl_dims: list[bool] | None = None  # per-step mask over the 10 dims; None = all RL
    action_dim: int = ACTION_DIM
    proprio_dim: int = PROPRIO_DIM
    rl_token_dim: int = RL_TOKEN_DIM
    # Read by OnlineRLTrainer's offline-cache identity checks only.
    actor_rl_arm: str = 'both'
    vla_pretrained_path: str | None = None
    rl_token_pretrained_path: str | None = None

    def __post_init__(self):
        if self.chunk_length < 1 or self.actor_update_interval < 1 or self.utd_ratio < 1:
            raise ValueError('chunk_length, actor_update_interval and utd_ratio must be >= 1')
        if not 0 <= self.gamma < 1 or not 0 < self.tau <= 1:
            raise ValueError('gamma must be in [0, 1) and tau in (0, 1]')
        if not 0 <= self.actor_ref_dropout_p <= 1 or self.actor_fixed_std < 0:
            raise ValueError('ref dropout must be in [0, 1] and fixed_std >= 0')
        if self.actor_rl_dims is not None and len(self.actor_rl_dims) != self.action_dim:
            raise ValueError(f'actor_rl_dims needs {self.action_dim} entries')
        if len(self.actor_bc_dim_weights) != self.action_dim or min(self.actor_bc_dim_weights) < 0:
            raise ValueError(f'actor_bc_dim_weights needs {self.action_dim} non-negative entries')
        if any(not 0 <= d < self.action_dim for d in self.actor_clip_exempt_dims):
            raise ValueError(f'actor_clip_exempt_dims must index the {self.action_dim} row dims')

    @property
    def state_dim(self) -> int:
        return self.rl_token_dim + self.proprio_dim

    @property
    def chunk_dim(self) -> int:
        return self.chunk_length * self.action_dim

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(kw_only=True)
class OnlineConfig:
    """Fields ``OnlineRLTrainer`` reads (upstream ``backend.OnlineRLConfig``)."""

    warmup_episodes: int
    critic_only_episodes: int
    min_warmup_transitions: int
    min_warmup_successes: int
    min_warmup_failures: int
    replay_capacity: int
    batch_size: int
    lr_actor: float
    lr_critic: float
    max_updates_per_episode: int
    use_stratified_sampling: bool
    terminal_reward: float
    milestone_reward: float                 # 0 = paper (no shaping)
    time_decay: float                       # 1 = paper (no speed shaping)
    save_dir: str
    save_every_episodes: int
    offline_cache_path: str | None = None
    offline_batch_fraction: float = 0.0
    wandb: bool = False
    wandb_project: str = 'wall-rlt-stage2'
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None
    wandb_resume: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_configs(path) -> tuple[ChunkACConfig, OnlineConfig, dict]:
    """Load ``{status, policy: {...}, online: {...}}`` from YAML or JSON.

    A ``null`` value for a required field is an undecided parameter and is
    rejected, so a proposal file cannot be run until it is filled in.
    """
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if path.suffix in ('.yaml', '.yml'):
        import yaml
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    result = []
    for key, cls in (('policy', ChunkACConfig), ('online', OnlineConfig)):
        section = dict(payload[key])
        required = [f.name for f in dataclasses.fields(cls)
                    if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
        undecided = [name for name in required if section.get(name) is None]
        missing_ok = {'target_q_clip', 'actor_action_clip_delta'}   # None is a real choice here
        undecided = [name for name in undecided if not (name in missing_ok and name in section)]
        if undecided:
            raise ValueError(f'{path}: {key} has undecided fields {undecided}')
        result.append(cls(**section))
    meta = {k: v for k, v in payload.items() if k not in ('policy', 'online')}
    offset = meta.get('execution_offset')
    if offset is None or not 0 <= offset or offset + result[0].chunk_length > WALL_HORIZON:
        raise ValueError(f'{path}: execution_offset + chunk_length must fit in the '
                         f'{WALL_HORIZON}-step WALL window (got {offset} + {result[0].chunk_length})')
    return result[0], result[1], meta


def weighted_actor_loss(actor, critic, batch: dict[str, Tensor], beta: float,
                        bc_weights_flat: Tensor) -> Tensor:
    """Upstream ``evo_rlt.core.losses.actor_loss`` with per-dimension BC weights.

    ``-min Q(x, mu) + beta * sum_d w_d (mu_d - ref_d)^2`` averaged over the
    batch, ``mu`` the deterministic mean with reference dropout. With all
    weights 1 this is exactly upstream's loss. D14: the gripper needs its own
    small weight, otherwise switching closed -> open (2 normalized units per
    step) costs far more than the success value.
    """
    x, ref = batch['state_vec'], batch['ref_chunk_flat']
    mu, _ = actor.forward(x, ref, training=True)
    q = critic.min_q(x, mu)
    bc_reg = (((mu - ref) ** 2) * bc_weights_flat).sum(dim=-1).mean()
    return -q.mean() + beta * bc_reg


class WallChunkACPolicy(nn.Module):
    """Chunk actor + twin critic + target critic, trainable by ``OnlineRLTrainer``."""

    def __init__(self, config: ChunkACConfig):
        super().__init__()
        self.config = config
        mask = None
        if config.actor_rl_dims is not None:
            mask = torch.tensor([float(bool(v)) for v in config.actor_rl_dims]).repeat(config.chunk_length)
        self.actor = ChunkActor(
            state_dim=config.state_dim, chunk_dim=config.chunk_dim,
            hidden_dim=config.actor_hidden_dim, num_layers=config.actor_num_layers,
            fixed_std=config.actor_fixed_std, ref_dropout_p=config.actor_ref_dropout_p,
            activation=config.actor_activation, layer_norm=config.actor_layer_norm,
            residual=config.actor_residual, residual_to_ref=config.actor_residual_to_ref,
            action_mask=mask)
        self.critic = TwinCritic(
            state_dim=config.state_dim, chunk_dim=config.chunk_dim,
            hidden_dim=config.critic_hidden_dim, num_layers=config.critic_num_layers,
            activation=config.critic_activation, layer_norm=config.critic_layer_norm,
            residual=config.critic_residual)
        self.target_critic = copy.deepcopy(self.critic)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False
        self.target_critic.eval()
        self.register_buffer('_critic_step', torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer('_bc_weights_flat', torch.tensor(config.actor_bc_dim_weights,
                             dtype=torch.float32).repeat(config.chunk_length), persistent=False)
        clip_mask = torch.ones(config.action_dim)
        clip_mask[list(config.actor_clip_exempt_dims)] = 0.0
        self.register_buffer('_clip_mask', clip_mask, persistent=False)

    # ---- training (upstream ChunkACPolicy._coerce_batch / forward) ----
    @staticmethod
    def _coerce_batch(batch: dict[str, Any]) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for key in ('state_vec', 'exec_chunk', 'ref_chunk', 'reward_seq', 'next_state_vec',
                    'next_ref_chunk', 'done', 'actual_steps'):
            if key not in batch:
                raise KeyError(f'forward missing batch key: {key!r}')
            value = batch[key]
            out[key] = value if isinstance(value, Tensor) else torch.as_tensor(value)
        out['exec_chunk_flat'] = out['exec_chunk'].flatten(start_dim=-2)
        out['ref_chunk_flat'] = out['ref_chunk'].flatten(start_dim=-2)
        out['next_ref_flat'] = out['next_ref_chunk'].flatten(start_dim=-2)
        if 'outcome' in batch:
            value = batch['outcome']
            out['outcome'] = value if isinstance(value, Tensor) else torch.as_tensor(value)
        return out

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        cfg = self.config
        tx = self._coerce_batch(batch)
        c_loss = critic_loss(self.critic, self.target_critic, self.actor, tx, gamma=cfg.gamma,
                             C=cfg.chunk_length, target_q_clip=cfg.target_q_clip,
                             rankq_noise_scale=cfg.rankq_noise_scale,
                             rankq_alpha_success=cfg.rankq_alpha_success,
                             rankq_alpha_failure=cfg.rankq_alpha_failure)
        soft_update(self.target_critic, self.critic, cfg.tau)     # tau is 0 inside OnlineRLTrainer
        self._critic_step += 1
        if int(self._critic_step.item()) % cfg.actor_update_interval == 0:
            a_loss = self._actor_loss_without_critic_grads(tx)
            total = c_loss + a_loss
            return total, {'loss': total.detach(), 'loss_critic': c_loss.detach(),
                           'loss_actor': a_loss.detach(),
                           'critic_step': self._critic_step.detach().clone()}
        return c_loss, {'loss': c_loss.detach(), 'loss_critic': c_loss.detach(),
                        'critic_step': self._critic_step.detach().clone()}

    def _actor_loss_without_critic_grads(self, tx: dict[str, Tensor]) -> Tensor:
        critic_params = [p for p in self.critic.parameters() if p.requires_grad]
        for parameter in critic_params:
            parameter.requires_grad_(False)
        try:
            return weighted_actor_loss(self.actor, self.critic, tx, self.config.beta,
                                       self._bc_weights_flat)
        finally:
            for parameter in critic_params:
                parameter.requires_grad_(True)

    # ---- rollout ----
    @torch.no_grad()
    def act(self, state_vec: Tensor, ref_chunk: Tensor, *, explore: bool) -> Tensor:
        """``[B, D]``, ``[B, C, A]`` normalized -> ``[B, C, A]`` in [-1, 1].

        ``explore=True`` samples with ``actor_fixed_std`` (0 under D6, where
        exploration is added later by ``rlt_online.exploration``); ``False``
        uses the mean. Then clamp to [-1, 1] and apply ``actor_action_clip_delta``
        around the reference, as upstream ``RLTActionModifier.compute_chunk``,
        except on ``actor_clip_exempt_dims`` (D14: the gripper).
        """
        cfg = self.config
        ref_flat = ref_chunk.flatten(start_dim=-2)
        if explore:
            sample, _ = self.actor.sample(state_vec, ref_flat, training=False)
        else:
            sample, _ = self.actor.forward(state_vec, ref_flat, training=False)
        chunk = sample.unflatten(-1, (cfg.chunk_length, cfg.action_dim)).clamp(-1.0, 1.0)
        if cfg.actor_action_clip_delta is not None:
            delta = (chunk - ref_chunk).clamp(-cfg.actor_action_clip_delta, cfg.actor_action_clip_delta)
            clipped = (ref_chunk + delta).clamp(-1.0, 1.0)
            chunk = torch.where(self._clip_mask.bool(), clipped, chunk)
        return chunk

    # ---- persistence ----
    def save_pretrained(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tmp = path / 'actor_critic.pt.tmp'
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                    'target_critic': self.target_critic.state_dict(),
                    'critic_step': self._critic_step.detach().cpu()}, tmp)
        tmp.replace(path / 'actor_critic.pt')
        (path / 'config.json').write_text(json.dumps(self.config.to_dict(), indent=1) + '\n',
                                          encoding='utf-8')

    @classmethod
    def from_pretrained(cls, path, map_location='cpu') -> 'WallChunkACPolicy':
        path = Path(path)
        config = ChunkACConfig(**json.loads((path / 'config.json').read_text(encoding='utf-8')))
        policy = cls(config)
        state = torch.load(path / 'actor_critic.pt', map_location=map_location, weights_only=True)
        policy.actor.load_state_dict(state['actor'])
        policy.critic.load_state_dict(state['critic'])
        policy.target_critic.load_state_dict(state['target_critic'])
        policy._critic_step.copy_(state['critic_step'])
        return policy
