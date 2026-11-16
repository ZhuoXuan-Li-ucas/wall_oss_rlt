"""WallChunkACPolicy with the vendored OnlineRLTrainer, end to end on synthetic episodes."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rlt_online import evo
from rlt_online.policy import ChunkACConfig, OnlineConfig, WallChunkACPolicy, load_configs

ROOT = Path(__file__).resolve().parents[2]
C, A, Z = 4, 10, 16


def _policy_config(**overrides):
    values = dict(chunk_length=C, gamma=0.99, beta=1.0, tau=0.005, actor_update_interval=2,
                  utd_ratio=5, target_q_clip=100.0, actor_hidden_dim=32, actor_num_layers=2,
                  actor_activation='relu', actor_layer_norm=False, actor_residual=False,
                  actor_residual_to_ref=True, actor_fixed_std=0.05, actor_ref_dropout_p=0.5,
                  actor_action_clip_delta=0.1, critic_hidden_dim=32, critic_num_layers=2,
                  critic_activation='relu', critic_layer_norm=True, critic_residual=False,
                  actor_clip_exempt_dims=[], actor_bc_dim_weights=[1.0] * A, rl_token_dim=Z)
    values.update(overrides)
    return ChunkACConfig(**values)


def _online_config(tmp_path, **overrides):
    values = dict(warmup_episodes=2, critic_only_episodes=2, min_warmup_transitions=10,
                  min_warmup_successes=1, min_warmup_failures=1, replay_capacity=10_000,
                  batch_size=8, lr_actor=1e-3, lr_critic=1e-3, max_updates_per_episode=50,
                  use_stratified_sampling=True, terminal_reward=1.0, milestone_reward=0.0,
                  time_decay=1.0, save_dir=str(tmp_path / 'run'), save_every_episodes=3)
    values.update(overrides)
    return OnlineConfig(**values)


def _run_episode(trainer, policy, episode, success, chunks=4, seed=0):
    generator = torch.Generator().manual_seed(seed + episode)
    base = trainer.start_episode(episode)
    for _ in range(chunks):
        state = torch.randn(1, Z + 10, generator=generator)
        ref = torch.rand(1, C, A, generator=generator) * 2 - 1
        chunk = policy.act(state, ref, explore=True)
        for i in range(C):
            trainer.collector.on_frame(action=chunk[0, i], state_vec=state if i == 0 else None,
                                       ref_chunk=ref if i == 0 else None,
                                       source_type=evo.SOURCE_RL, is_critical=1.0)
    trainer.collector.flush_episode(success)
    return trainer.maybe_update(episode, base)


def _params(module):
    return [p.detach().clone() for p in module.parameters()]


def _changed(before, module):
    return any(not torch.equal(a, b) for a, b in zip(before, module.parameters()))


def test_actor_loss_does_not_add_critic_grads():
    policy = WallChunkACPolicy(_policy_config(actor_update_interval=1))
    batch = {'state_vec': torch.randn(8, Z + 10), 'exec_chunk': torch.rand(8, C, A),
             'ref_chunk': torch.rand(8, C, A), 'reward_seq': torch.zeros(8, C),
             'next_state_vec': torch.randn(8, Z + 10), 'next_ref_chunk': torch.rand(8, C, A),
             'done': torch.zeros(8), 'actual_steps': torch.full((8,), C)}
    tx = policy._coerce_batch(batch)
    loss = policy._actor_loss_without_critic_grads(tx)
    loss.backward()
    assert all(p.grad is None for p in policy.critic.parameters())
    assert any(p.grad is not None for p in policy.actor.parameters())
    assert all(p.requires_grad for p in policy.critic.parameters())


def test_zero_init_actor_reproduces_reference_and_exploration_is_bounded():
    policy = WallChunkACPolicy(_policy_config())
    state = torch.randn(3, Z + 10)
    ref = torch.rand(3, C, A) * 2 - 1
    torch.testing.assert_close(policy.act(state, ref, explore=False), ref)
    explored = policy.act(state, ref, explore=True)
    assert not torch.equal(explored, ref)
    assert (explored - ref).abs().max() <= 0.1 + 1e-6


def _batch(n=8):
    return {'state_vec': torch.randn(n, Z + 10), 'ref_chunk_flat': torch.rand(n, C * A) * 2 - 1}


def test_unit_bc_weights_reproduce_upstream_actor_loss():
    from rlt_online.policy import weighted_actor_loss
    policy = WallChunkACPolicy(_policy_config(actor_residual_to_ref=False))
    batch = _batch()
    torch.manual_seed(3)
    ours = weighted_actor_loss(policy.actor, policy.critic, batch, 1.0, policy._bc_weights_flat)
    torch.manual_seed(3)                                   # same reference-dropout draw
    upstream = evo.actor_loss(policy.actor, policy.critic, batch, beta=1.0)
    torch.testing.assert_close(ours, upstream)


def test_gripper_bc_weight_scales_only_the_gripper_term():
    from rlt_online.policy import weighted_actor_loss
    weights = [1.0] * 9 + [0.0]
    policy = WallChunkACPolicy(_policy_config(actor_bc_dim_weights=weights, actor_ref_dropout_p=0.0))
    batch = _batch()
    with torch.no_grad():                                  # residual actor: mu = ref + bias
        bias = torch.zeros(C, A)
        bias[:, 9] = 2.0                                   # "open" against a closed reference
        policy.actor.net[-1].bias.copy_(bias.flatten())
    q = policy.critic.min_q(batch['state_vec'], batch['ref_chunk_flat'] + bias.flatten()).mean()
    loss = weighted_actor_loss(policy.actor, policy.critic, batch, 1.0, policy._bc_weights_flat)
    torch.testing.assert_close(loss, -q)                   # gripper deviation costs nothing here


def test_clip_exempt_gripper_can_leave_the_reference():
    policy = WallChunkACPolicy(_policy_config(actor_clip_exempt_dims=[9]))
    with torch.no_grad():
        bias = torch.full((C, A), 0.5)
        policy.actor.net[-1].bias.copy_(bias.flatten())
    state, ref = torch.randn(2, Z + 10), torch.full((2, C, A), -0.8)
    chunk = policy.act(state, ref, explore=False)
    torch.testing.assert_close(chunk[..., :9], torch.full((2, C, 9), -0.7))   # clipped at 0.1
    torch.testing.assert_close(chunk[..., 9], torch.full((2, C), -0.3))       # exempt


def test_rl_dims_mask_keeps_masked_dims_on_the_reference():
    mask = [True] * 9 + [False]                         # gripper stays WALL's
    policy = WallChunkACPolicy(_policy_config(actor_rl_dims=mask))
    state, ref = torch.randn(2, Z + 10), torch.rand(2, C, A) * 2 - 1
    explored = policy.act(state, ref, explore=True)
    torch.testing.assert_close(explored[..., 9], ref[..., 9])


def test_online_trainer_warmup_critic_only_then_actor(tmp_path):
    torch.manual_seed(0)
    policy_cfg = _policy_config()
    policy = WallChunkACPolicy(policy_cfg)
    trainer = evo.OnlineRLTrainer(policy, _online_config(tmp_path), policy_cfg)
    # Warmup: episode 0 has only a success, so no update yet.
    assert _run_episode(trainer, policy, 0, True) is None
    assert _run_episode(trainer, policy, 1, False) is None          # 0-based index < warmup_episodes
    actor_before, critic_before = _params(policy.actor), _params(policy.critic)
    stats = _run_episode(trainer, policy, 2, True)
    assert stats is not None and stats['actual_updates'] == 4 * 5
    assert trainer.warmup_completed_at_episode == 2
    assert _changed(critic_before, policy.critic)
    assert not _changed(actor_before, policy.actor)                 # critic-only window
    assert policy_cfg.tau == 0.005 and policy_cfg.actor_update_interval == 2   # restored
    _run_episode(trainer, policy, 3, False)
    assert not _changed(actor_before, policy.actor)
    _run_episode(trainer, policy, 4, True)                          # window over
    assert _changed(actor_before, policy.actor)
    assert (Path(trainer.cfg.save_dir) / 'step_000003' / 'actor_critic.pt').is_file()
    trainer.close()


def test_latest_state_resume_round_trip(tmp_path):
    torch.manual_seed(1)
    policy_cfg = _policy_config()
    policy = WallChunkACPolicy(policy_cfg)
    trainer = evo.OnlineRLTrainer(policy, _online_config(tmp_path), policy_cfg)
    for episode, success in enumerate([True, False, True, False]):
        _run_episode(trainer, policy, episode, success)
    trainer.save_latest_state(4)
    fresh_cfg = _policy_config()
    fresh = WallChunkACPolicy(fresh_cfg)
    resumed = evo.OnlineRLTrainer(fresh, _online_config(tmp_path), fresh_cfg)
    assert resumed.load_latest_state(Path(trainer.cfg.save_dir) / 'latest_online_state.pt') == 4
    for a, b in zip(policy.state_dict().values(), fresh.state_dict().values()):
        torch.testing.assert_close(a, b)
    assert len(resumed.replay_buffer) == len(trainer.replay_buffer)
    assert resumed.warmup_completed_at_episode == trainer.warmup_completed_at_episode


def test_save_and_load_pretrained(tmp_path):
    policy = WallChunkACPolicy(_policy_config())
    policy._critic_step += 7
    policy.save_pretrained(tmp_path / 'ckpt')
    loaded = WallChunkACPolicy.from_pretrained(tmp_path / 'ckpt')
    assert loaded.config == policy.config
    for a, b in zip(policy.state_dict().values(), loaded.state_dict().values()):
        torch.testing.assert_close(a, b)


def test_run_config_needs_only_the_per_run_save_dir():
    with pytest.raises(ValueError, match=r"undecided fields \['save_dir'\]"):
        load_configs(ROOT / 'configs/rlt_online/arm1.bolt.yaml')        # the tool fills save_dir per run


def test_run_config_carries_the_decisions(tmp_path):
    import yaml
    payload = yaml.safe_load((ROOT / 'configs/rlt_online/arm1.bolt.yaml').read_text())
    assert payload['policy']['chunk_length'] == 12 and payload['execution_offset'] == 11   # D1
    assert payload['policy']['actor_fixed_std'] == 0.0                                   # D6
    assert payload['exploration']['form'] == 'per_chunk_linear_ramp'
    assert payload['policy']['actor_clip_exempt_dims'] == [9]                            # D14
    assert payload['policy']['actor_bc_dim_weights'] == [1.0] * 9 + [0.002]             # D14 weight
    payload['online']['save_dir'] = str(tmp_path)
    path = tmp_path / 'filled.yaml'
    path.write_text(yaml.safe_dump(payload))
    policy_cfg, online_cfg, meta = load_configs(path)
    assert policy_cfg.chunk_dim == 120 and policy_cfg.state_dim == 2058
    assert online_cfg.milestone_reward == 0.0 and meta['status'].startswith('run config')
    payload['execution_offset'] = 21                     # 21 + 12 > 32
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match='execution_offset'):
        load_configs(path)
