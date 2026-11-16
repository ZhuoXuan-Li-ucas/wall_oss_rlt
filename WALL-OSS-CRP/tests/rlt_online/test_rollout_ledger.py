"""ChunkLedger: executed-chunk bookkeeping for releases, failures and truncation."""
from __future__ import annotations

import numpy as np

from rlt_online import evo
from rlt_online.collector import ChunkCollector
from rlt_online.rollout import ChunkLedger, RLChunk

C, A, Dz, FPS = 12, 10, 4, 30.0
STEP = 1 / FPS


def _setup():
    replay = evo.ReplayBuffer(capacity=100)
    collector = ChunkCollector(replay, chunk_length=C, action_dim=A, milestone_reward=0.0,
                               terminal_reward=1.0, time_decay=1.0)
    collector.start_episode(0)
    return replay, collector, ChunkLedger(collector, C, FPS)


def _chunk(cid, t_start, random_release=False):
    return RLChunk(cid, t_start, np.full(Dz, float(cid), np.float32), np.zeros((C, A), np.float32),
                   np.full((C, A), -1.0, np.float32), evo.SOURCE_RL, random_release)


def test_chunks_enter_only_after_their_window():
    replay, collector, ledger = _setup()
    ledger.register(_chunk(0, 0.0))
    ledger.register(_chunk(1, C * STEP))
    assert ledger.advance(C * STEP - 1e-6) == 0
    assert ledger.advance(C * STEP + 1e-6) == 1 and ledger.added == 1
    assert ledger.advance(2 * C * STEP) == 1


def test_release_mid_chunk_cuts_and_drops_unstarted():
    replay, collector, ledger = _setup()
    for k in range(3):
        ledger.register(_chunk(k, k * C * STEP))
    t_release = 2 * C * STEP - 3 * STEP + 1e-4   # 9 rows (and a bit) into chunk 1? no: chunk 1 runs [C, 2C) steps
    k = ledger.close_at(t_release)
    assert k == 10 and ledger.dropped == 1
    collector.flush_episode(True)
    t0, t1 = replay.buffer
    assert t0.actual_steps.item() == C and t1.actual_steps.item() == 10 and t1.done.item() == 1
    assert t1.reward_seq[9].item() == 1.0


def test_human_open_patches_rows_and_source():
    replay, collector, ledger = _setup()
    ledger.register(_chunk(0, 0.0))
    ledger.mark_human_open(4 * STEP + 1e-4)
    ledger.close_at(6 * STEP + 1e-4)
    collector.flush_episode(True)
    (t,) = replay.buffer
    assert t.intervention.item() == 1.0 and t.actual_steps.item() == 7
    np.testing.assert_array_equal(t.exec_chunk[:4, 9].numpy(), -1.0)
    np.testing.assert_array_equal(t.exec_chunk[4:7, 9].numpy(), 1.0)
    np.testing.assert_array_equal(t.ref_chunk[:7].numpy(), t.exec_chunk[:7].numpy())   # BC target = human


def test_truncation_waits_for_the_chunk_end_and_the_next_observation():
    replay, collector, ledger = _setup()
    ledger.register(_chunk(0, 0.0))
    ledger.register(_chunk(1, C * STEP))
    assert not ledger.truncate(C * STEP + 5 * STEP)          # chunk 1 still running
    assert not ledger.truncate(2 * C * STEP + 1e-6)          # no chunk 2 fetched yet
    ledger.register(_chunk(2, 2 * C * STEP + 0.5 * STEP), executes=False)
    assert ledger.truncate(2 * C * STEP + 2e-6)
    assert len(replay) == 2 and all(t.done.item() == 0 for t in replay.buffer)
    np.testing.assert_array_equal(replay.buffer[-1].next_state_vec.numpy(), np.full(Dz, 2.0))


def test_shorter_spacing_gives_short_non_terminal_transitions():
    replay, collector, ledger = _setup()
    ledger.register(_chunk(0, 0.0))
    ledger.register(_chunk(1, 11 * STEP))                    # next observation came 11 steps later
    ledger.register(_chunk(2, 24 * STEP))                    # then 13 steps: capped at C, last row held
    ledger.advance(24 * STEP + 1e-6)
    t0, t1 = replay.buffer if len(replay) else collector._episode_staging
    assert (t0.actual_steps.item(), t0.done.item()) == (11, 0)
    assert t1.actual_steps.item() == C
    np.testing.assert_array_equal(t0.next_state_vec.numpy(), np.full(Dz, 1.0))
    assert ledger.short_chunks == 1


def test_release_before_any_rl_chunk_ran_stores_nothing():
    replay, collector, ledger = _setup()
    ledger.register(_chunk(0, 1.0))
    assert ledger.close_at(0.5) is None and ledger.dropped == 1
    assert len(collector._episode_staging) == 0
