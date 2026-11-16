"""Vendored Evo-RLT integrity, and the chunk TD target under early termination."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from rlt_online import evo


def test_vendored_files_match_manifest():
    assert evo.verify_vendor() == []
    assert Path(evo.evo_rlt.__file__).resolve().parent == evo.VENDOR / 'evo_rlt'


def _collector(C, A=3):
    replay = evo.ReplayBuffer(capacity=100)
    collector = evo.RLTOnlineCollector(replay, chunk_length=C, action_dim=A,
                                       milestone_reward=0.0, terminal_reward=1.0, time_decay=1.0)
    return replay, collector


def _feed_chunk(collector, state, ref, actions):
    """Feed one actor chunk frame by frame, with the chunk's own state/ref (aligned)."""
    out = None
    for i, action in enumerate(actions):
        out = collector.on_frame(action=action, state_vec=state if i == 0 else None,
                                 ref_chunk=ref if i == 0 else None,
                                 source_type=evo.SOURCE_RL, is_critical=1.0)
    return out


def test_success_mid_chunk_closes_short_terminal_transition():
    C, A = 4, 3
    replay, collector = _collector(C, A)
    collector.start_episode(0)
    s0, s1 = torch.zeros(5), torch.ones(5)
    ref = torch.zeros(C, A)
    _feed_chunk(collector, s0, ref, torch.full((C, A), 0.1))
    _feed_chunk(collector, s1, ref, torch.full((3, A), 0.2))      # success after 3 of 4 steps
    collector.flush_episode(True)
    assert len(replay) == 2
    first, last = replay.buffer
    assert first.done.item() == 0 and first.actual_steps.item() == C
    torch.testing.assert_close(first.next_state_vec, s1)
    assert last.done.item() == 1 and last.actual_steps.item() == 3
    torch.testing.assert_close(last.reward_seq, torch.tensor([0., 0., 1., 0.]))
    torch.testing.assert_close(last.exec_chunk[3], torch.zeros(A))      # padded, not executed
    assert first.outcome.item() == 1.0 and last.outcome.item() == 1.0


def _constant_critic(state_dim, chunk_dim, value):
    critic = evo.TwinCritic(state_dim, chunk_dim, hidden_dim=4, num_layers=1)
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        for q in (critic.q1, critic.q2):
            q.net[-1].bias.fill_(value)
    return critic


def _target_from_loss(batch, gamma, C, q_next):
    state_dim, chunk_dim = batch['state_vec'].shape[1], batch['exec_chunk_flat'].shape[1]
    critic = _constant_critic(state_dim, chunk_dim, 0.0)
    target = _constant_critic(state_dim, chunk_dim, q_next)
    actor = evo.ChunkActor(state_dim, chunk_dim, hidden_dim=4, num_layers=1)
    loss = evo.critic_loss(critic, target, actor, batch, gamma=gamma, C=C, target_q_clip=None)
    return math.sqrt(loss.item() / 2.0)      # q1 = q2 = 0, single sample


def _batch(reward, done, steps, C=4, A=3):
    return {'state_vec': torch.zeros(1, 5), 'exec_chunk_flat': torch.zeros(1, C * A),
            'ref_chunk_flat': torch.zeros(1, C * A), 'next_state_vec': torch.zeros(1, 5),
            'next_ref_flat': torch.zeros(1, C * A), 'reward_seq': torch.tensor([reward]),
            'done': torch.tensor([float(done)]), 'actual_steps': torch.tensor([steps])}


def test_td_target_full_chunk_bootstraps_with_gamma_power_c():
    got = _target_from_loss(_batch([0., 0., 0., 0.], 0, 4), 0.9, 4, 0.5)
    assert math.isclose(got, 0.9 ** 4 * 0.5, rel_tol=1e-5)


def test_td_target_terminal_short_chunk_has_no_bootstrap():
    got = _target_from_loss(_batch([0., 0., 1., 0.], 1, 3), 0.9, 4, 0.5)
    assert math.isclose(got, 0.9 ** 2, rel_tol=1e-5)


def test_td_target_truncated_short_chunk_bootstraps_with_actual_steps():
    got = _target_from_loss(_batch([0., 0., 0., 0.], 0, 3), 0.9, 4, 0.5)
    assert math.isclose(got, 0.9 ** 3 * 0.5, rel_tol=1e-5)
