"""Which actor chunks were actually executed, and how far (D1 timing, D14/D16 endings).

The executor fetches RL chunks asynchronously: chunk k is computed from
observation k, its C rows occupy window steps D..D+C-1 and it becomes active at
``t_start = t0 + D / fps``. Chunk k runs until chunk k+1 becomes active. The
observation spacing is set by camera frame timing, so it is usually but not
always exactly C steps; chunk k therefore executes
``rows = clamp(round((t_start[k+1] - t_start[k]) * fps), 1, C)`` rows (if the
next chunk comes later than C steps, the executor holds the last row; the
extra hold is not an action row). The ledger hands chunks to the
``ChunkCollector`` only once they have really been executed:

- ``advance(now)``: every chunk whose execution is over goes in, with its
  executed rows (a short non-terminal transition when rows < C).
- ``close_at(t)``: release (or a failure label during RL) at time t. Finished
  chunks go in, the active one is cut to the rows executed up to t and left
  pending for ``flush_episode``, chunks that never started are dropped.
- ``truncate(...)``: RL time limit, applied when the chunk running at the
  limit has finished; a later fetched (unexecuted) chunk supplies the bootstrap
  observation.
- ``mark_human_open(t)``: the operator pressed the release key. Rows from that
  step on command "open" and the chunk becomes a human intervention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rlt_online.action_space import GRIPPER
from rlt_online.evo import SOURCE_HUMAN

OPEN_NORMALIZED = 1.0          # gripper bound [0, 1] maps 1 (open) to +1


@dataclass
class RLChunk:
    cid: int
    t_start: float
    state_vec: np.ndarray          # [D]
    ref_norm: np.ndarray           # [C, A]
    exec_norm: np.ndarray          # [C, A], what was commanded
    source: float
    random_release: bool = False
    info: dict = field(default_factory=dict)
    rows: int | None = None        # executed rows, known once the next chunk is registered


class ChunkLedger:
    def __init__(self, collector, chunk_length: int, fps: float):
        self.collector, self.C, self.fps = collector, chunk_length, fps
        self.chunks: list[RLChunk] = []
        self.added = 0                 # chunks [0, added) are in the collector
        self.closed = False
        self.rows_added = 0
        self.short_chunks = 0
        self.dropped = 0
        self._trunc_chunk = None
        self._trunc_started = False

    def register(self, chunk: RLChunk, *, executes: bool = True) -> None:
        """``executes=False``: fetched only as the bootstrap observation at truncation."""
        if self.closed:
            return
        if self.chunks and chunk.t_start <= self.chunks[-1].t_start:
            raise ValueError('chunks must be registered in execution order')
        if executes and self.chunks and self.chunks[-1].rows is None:
            prev = self.chunks[-1]
            prev.rows = int(min(self.C, max(1, round((chunk.t_start - prev.t_start) * self.fps))))
        self.chunks.append(chunk)

    def active(self, t: float) -> RLChunk | None:
        started = [c for c in self.chunks if c.t_start <= t]
        return started[-1] if started else None

    def _end(self, chunk: RLChunk) -> float:
        return chunk.t_start + (chunk.rows if chunk.rows is not None else self.C) / self.fps

    def _add(self, chunk: RLChunk, rows: int, *, pending: bool = False) -> None:
        self.collector.add_chunk(chunk.state_vec, chunk.ref_norm, chunk.exec_norm[:rows], chunk.source,
                                 pending=pending)
        self.rows_added += rows
        self.short_chunks += rows < self.C and not pending

    def advance(self, now: float) -> int:
        """Add every chunk whose execution is over. Returns how many were added."""
        n = 0
        while not self.closed and self.added < len(self.chunks):
            chunk = self.chunks[self.added]
            if now < self._end(chunk) or (chunk is self._trunc_chunk):
                break
            self._add(chunk, chunk.rows if chunk.rows is not None else self.C)
            self.added += 1
            n += 1
        return n

    def rows_executed(self, chunk: RLChunk, t: float) -> int:
        limit = chunk.rows if chunk.rows is not None else self.C
        return int(min(limit, max(1, math.ceil((t - chunk.t_start) * self.fps))))

    def mark_human_open(self, t: float) -> None:
        """Rows of the active and later chunks from step ``t`` on command open; chunk becomes human."""
        for chunk in self.chunks[self.added:]:
            first = 0 if chunk.t_start >= t else int(math.floor((t - chunk.t_start) * self.fps))
            if first >= self.C:
                continue
            chunk.exec_norm = chunk.exec_norm.copy()
            chunk.exec_norm[first:, GRIPPER] = OPEN_NORMALIZED
            chunk.source = SOURCE_HUMAN
            chunk.info['human_open_from_row'] = first

    def close_at(self, t: float) -> int | None:
        """Release / failure at t. Returns the executed rows of the cut chunk (None if none ran)."""
        if self.closed:
            return None
        self.advance(t)
        active = self.active(t)
        k = None
        if active is not None and self.chunks.index(active) == self.added:
            k = self.rows_executed(active, t)
            self._add(active, k, pending=True)
            self.added += 1
        self.dropped = len(self.chunks) - self.added
        self.chunks = self.chunks[:self.added]
        self.closed = True
        return k

    def truncate(self, now: float) -> bool:
        """At the RL time limit: once the chunk running at the limit has finished, add it and
        flush as truncated, bootstrapping from the next fetched chunk.

        Returns False while it still has to wait. The chunk running at the
        limit is executed to completion (C rows); later chunks are not executed.
        """
        if self.closed:
            return True
        if not self._trunc_started:
            self._trunc_started = True
            self._trunc_chunk = self.active(now)
            if self._trunc_chunk is not None:
                self._trunc_chunk.rows = self.C
        active = self._trunc_chunk
        if active is None:                  # nothing ever ran: nothing to store
            self.closed = True
            return True
        index = self.chunks.index(active)
        if now < active.t_start + self.C / self.fps or index + 1 >= len(self.chunks):
            return False
        self.advance(now)                   # everything before the truncation chunk
        self._add(active, self.C)
        self.added += 1
        nxt = self.chunks[index + 1]
        self.collector.flush_truncated(nxt.state_vec, nxt.ref_norm)
        self.chunks = self.chunks[:self.added]
        self.closed = True
        return True
