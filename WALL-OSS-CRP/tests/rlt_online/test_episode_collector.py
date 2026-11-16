"""Episode protocol (D15/D16) and the chunk-aligned collector with truncation."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from rlt_online import evo
from rlt_online.collector import ChunkCollector, attach_chunk_collector
from rlt_online.episode import EpisodeProtocol, ProtocolConfig

ROOT = Path(__file__).resolve().parents[2]
META = yaml.safe_load((ROOT / 'configs/rlt_online/arm1.bolt.yaml').read_text())
NEAR, FAR = (0.20, -0.31, 0.38), (0.30, -0.16, 0.27)
HOLDING, EMPTY = 0.20, 0.01


@pytest.fixture
def protocol():
    return EpisodeProtocol(ProtocolConfig.from_meta(META), t_start=0.0)


def test_config_matches_the_decisions():
    cfg = ProtocolConfig.from_meta(META)
    assert cfg.radius_m == 0.03 and cfg.holding_opening == (0.12, 0.30)
    assert cfg.rl_phase_s == 20 and cfg.wall_phase_s == 30
    assert cfg.keys['release'] == 'o' and 'space' in cfg.keys['ignored']


def test_handover_needs_position_and_a_held_bolt(protocol):
    assert not protocol.on_observation(5.0, FAR, HOLDING)
    assert not protocol.on_observation(9.0, NEAR, EMPTY)          # closed on nothing
    assert protocol.on_observation(11.0, NEAR, HOLDING)
    assert protocol.phase == 'rl' and protocol.handover_t == 11.0
    assert not protocol.on_observation(11.4, NEAR, HOLDING)       # only once, never back to WALL


def test_wall_phase_timeout_is_not_stored(protocol):
    protocol.on_tick(29.9)
    assert protocol.phase == 'wall'
    protocol.on_tick(30.1)
    assert protocol.outcome == 'wall_failure'
    assert not protocol.stores_transitions and not protocol.counts_toward_budget


def test_rl_phase_timeout_truncates_after_20_s(protocol):
    protocol.on_observation(10.0, NEAR, HOLDING)
    protocol.on_tick(29.9)
    assert protocol.phase == 'rl'
    protocol.on_tick(30.0)
    assert protocol.outcome == 'truncated' and protocol.stores_transitions and not protocol.task_success


def test_human_release_then_success(protocol):
    assert protocol.on_key('o', 1.0) is None                      # not in RL yet
    protocol.on_observation(10.0, NEAR, HOLDING)
    assert protocol.on_key('s', 11.0) is None                     # success needs a release first
    assert protocol.on_key('o', 12.0) == 'command_open'
    assert protocol.on_key('o', 12.1) is None                     # once
    protocol.on_release(12.3, 'human')
    assert protocol.phase == 'released' and protocol.release_source == 'human'
    assert protocol.on_key('s', 14.0) == 'label'
    assert protocol.outcome == 'success' and protocol.task_success
    assert protocol.summary()['rl_phase_s'] == pytest.approx(2.3)


def test_failure_discard_and_ignored_keys(protocol):
    protocol.on_observation(10.0, NEAR, HOLDING)
    assert protocol.on_key('space', 10.5) is None and protocol.phase == 'rl'
    assert protocol.on_key('f', 11.0) == 'label' and protocol.outcome == 'failure'
    assert protocol.on_key('r', 12.0) is None                     # already done
    other = EpisodeProtocol(ProtocolConfig.from_meta(META), t_start=0.0)
    assert other.on_key('r', 3.0) == 'discard'
    assert other.outcome == 'discarded' and not other.stores_transitions


C, A, D = 12, 10, 5


def _collector():
    replay = evo.ReplayBuffer(capacity=100)
    return replay, ChunkCollector(replay, chunk_length=C, action_dim=A, milestone_reward=0.0,
                                  terminal_reward=1.0, time_decay=1.0)


def test_release_mid_chunk_closes_a_short_terminal_transition():
    replay, collector = _collector()
    collector.start_episode(0)
    for k in range(2):
        collector.add_chunk(torch.full((D,), float(k)), torch.zeros(C, A), torch.full((C, A), 0.1), evo.SOURCE_RL)
    collector.add_chunk(torch.full((D,), 2.0), torch.zeros(C, A), torch.full((5, A), 0.2), evo.SOURCE_RL,
                        pending=True)
    with pytest.raises(RuntimeError):
        collector.add_chunk(torch.zeros(D), torch.zeros(C, A), torch.zeros(C, A), evo.SOURCE_RL)
    collector.flush_episode(True)
    t0, t1, t2 = replay.buffer
    torch.testing.assert_close(t0.next_state_vec, torch.full((D,), 1.0))
    torch.testing.assert_close(t1.next_state_vec, torch.full((D,), 2.0))
    assert [t.done.item() for t in (t0, t1, t2)] == [0, 0, 1]
    assert t2.actual_steps.item() == 5 and t2.reward_seq[4].item() == 1.0 and t2.reward_seq.sum().item() == 1.0


def test_truncation_keeps_bootstrap_and_links_the_last_observation():
    replay, collector = _collector()
    collector.start_episode(0)
    for k in range(3):
        collector.add_chunk(torch.full((D,), float(k)), torch.zeros(C, A), torch.full((C, A), 0.1), evo.SOURCE_RL)
    collector.flush_truncated(torch.full((D,), 9.0), torch.full((C, A), 0.5))
    assert len(replay) == 3 and all(t.done.item() == 0 for t in replay.buffer)
    torch.testing.assert_close(replay.buffer[-1].next_state_vec, torch.full((D,), 9.0))
    torch.testing.assert_close(replay.buffer[-1].next_ref_chunk, torch.full((C, A), 0.5))
    assert replay.count_outcomes() == (0, 0)                       # neither success nor failure
    collector.flush_truncated(torch.zeros(D), torch.zeros(C, A))   # idempotent
    assert len(replay) == 3


def test_human_chunk_becomes_the_bc_target():
    replay, collector = _collector()
    collector.start_episode(0)
    executed = torch.full((7, A), 0.3)
    collector.add_chunk(torch.zeros(D), torch.zeros(C, A), executed, evo.SOURCE_HUMAN, pending=True)
    collector.flush_episode(True)
    (t,) = replay.buffer
    assert t.intervention.item() == 1.0
    torch.testing.assert_close(t.ref_chunk[:7], executed)


def test_attach_replaces_the_trainer_collector(tmp_path):
    from test_policy_trainer import _online_config, _policy_config      # same directory (rootdir import mode)
    from rlt_online.policy import WallChunkACPolicy
    policy_cfg = _policy_config()
    trainer = evo.OnlineRLTrainer(WallChunkACPolicy(policy_cfg), _online_config(tmp_path), policy_cfg)
    collector = attach_chunk_collector(trainer)
    assert isinstance(trainer.collector, ChunkCollector) and collector._buffer is trainer.replay_buffer
