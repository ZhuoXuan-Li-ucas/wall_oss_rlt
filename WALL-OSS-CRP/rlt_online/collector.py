"""Chunk-aligned feeding and truncation on top of the vendored ``RLTOnlineCollector``.

Upstream feeds the collector frame by frame and counts C frames from
``start_episode``, so its transitions can straddle two actor chunks (audit
item 2 in ``_vendor/NOTICE.md``). Here every actor chunk is added in one call
with its own state and reference, so boundaries always coincide. Rows must be
in the actor's normalized [-1, 1] space (audit item 1).

Upstream ``flush_episode`` always marks the last transition terminal. A time
limit is a truncation, not a terminal state: ``flush_truncated`` keeps
``done = 0`` and links the last transition to the observation taken at the
limit, so its TD target still bootstraps.
"""
from __future__ import annotations

import torch

from rlt_online.evo import SOURCE_HUMAN, SOURCE_RL, SOURCE_VLA, RLTOnlineCollector

SOURCES = (SOURCE_VLA, SOURCE_RL, SOURCE_HUMAN)


class ChunkCollector(RLTOnlineCollector):
    def add_chunk(self, state_vec, ref_chunk, executed, source_type: float, *, pending: bool = False) -> None:
        """One actor chunk: ``state_vec [D]``, ``ref_chunk [C, A]``, ``executed [k, A]`` (k <= C).

        ``k < C`` happens in two cases. ``pending=True``: the chunk in which the
        gripper released; it stays open until ``flush_episode`` closes it as the
        terminal transition. ``pending=False``: the next chunk took over after k
        rows (observation spacing shorter than C); it is closed right away as a
        short non-terminal transition, whose TD target bootstraps with gamma^k.
        """
        if source_type not in SOURCES:
            raise ValueError(f'source_type must be one of {SOURCES}')
        if self._frame_actions:
            raise RuntimeError('previous chunk was partial; close the episode before adding more')
        executed = torch.as_tensor(executed, dtype=torch.float32)
        if executed.ndim != 2 or not 1 <= len(executed) <= self._C:
            raise ValueError(f'executed must be [1..{self._C}, A], got {tuple(executed.shape)}')
        state = torch.as_tensor(state_vec, dtype=torch.float32)
        ref = torch.as_tensor(ref_chunk, dtype=torch.float32)
        for i, action in enumerate(executed):
            self.on_frame(action=action, state_vec=state if i == 0 else None,
                          ref_chunk=ref if i == 0 else None, source_type=source_type, is_critical=1.0)
        if self._frame_actions and not pending:
            self._emit_transition(done=False)

    def flush_truncated(self, next_state_vec, next_ref_chunk) -> None:
        """Commit the episode at the RL time limit without a terminal flag."""
        if self._flushed:
            return
        if self._frame_actions:
            self._emit_transition(done=False)
        if self._prev_transition is not None:
            self._prev_transition.next_state_vec = torch.as_tensor(next_state_vec, dtype=torch.float32).clone()
            self._prev_transition.next_ref_chunk = torch.as_tensor(next_ref_chunk, dtype=torch.float32).clone()
        for transition in self._episode_staging:
            transition.outcome = torch.tensor(-1.0)      # unresolved: not a success or failure sample
            self._buffer.add(transition)
        self._episode_staging = []
        self._flushed = True


def attach_chunk_collector(trainer) -> ChunkCollector:
    """Replace the plain collector that ``OnlineRLTrainer.__init__`` built, same settings."""
    old = trainer.collector
    collector = ChunkCollector(trainer.replay_buffer, chunk_length=old._C, action_dim=old._action_dim,
                               milestone_reward=old._milestone_reward, terminal_reward=old._terminal_reward,
                               time_decay=old._time_decay)
    trainer.collector = collector
    return collector
