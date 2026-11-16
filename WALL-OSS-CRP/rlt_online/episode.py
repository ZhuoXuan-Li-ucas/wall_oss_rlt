"""One Stage 2 episode as a state machine (decisions D5, D7, D14, D15, D16; 2026-11-10).

Phases: ``wall`` (WALL picks up and carries the bolt) -> ``rl`` (actor in
control after the automatic handover) -> ``released`` (gripper opened, waiting
for the operator's label) -> ``done``. No hardware here: the execution loop
calls ``on_observation`` at every WALL request, ``on_tick`` every control
tick, ``on_release`` when the executor's gripper command actually turns open,
and ``on_key`` for each key press.

Outcomes:
  success / failure   labeled by the operator (``s`` after a release; ``f`` any
                      time in rl/released), stored as terminal transitions
  truncated           no release within ``rl_phase_s`` after handover; stored,
                      bootstrap kept, counted as a task failure in evaluation
  wall_failure        no handover within ``wall_phase_s``; nothing stored, not
                      counted in the training budget, counted end to end
  discarded           ``r`` (setup error / hardware problem); nothing stored
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STORED = ('success', 'failure', 'truncated')
RELEASE_SOURCES = ('actor', 'random', 'human', 'wall')   # wall: before the first actor chunk ran, or vla-only


@dataclass(kw_only=True)
class ProtocolConfig:
    reference_point_m: tuple[float, float, float]
    radius_m: float
    holding_opening: tuple[float, float]
    rl_phase_s: float
    wall_phase_s: float
    keys: dict

    @classmethod
    def from_meta(cls, meta: dict) -> 'ProtocolConfig':
        handover, limits, keys = meta['handover'], meta['limits'], meta['keys']
        if handover.get('trigger') != 'auto_position':
            raise ValueError(f"unsupported handover trigger {handover.get('trigger')!r}")
        return cls(reference_point_m=tuple(handover['reference_point_m']), radius_m=handover['radius_m'],
                   holding_opening=tuple(handover['holding_opening']), rl_phase_s=limits['rl_phase_s'],
                   wall_phase_s=limits['wall_phase_s'], keys=dict(keys))


@dataclass
class EpisodeProtocol:
    config: ProtocolConfig
    t_start: float
    phase: str = 'wall'
    outcome: str | None = None
    handover_t: float | None = None
    release_t: float | None = None
    release_source: str | None = None
    human_release_requested: bool = False
    events: list = field(default_factory=list)

    # ---- inputs from the execution loop ----
    def on_observation(self, t: float, flange_xyz, opening: float) -> bool:
        """Call at every WALL request. Returns True exactly once, at the handover."""
        if self.phase != 'wall':
            return False
        if t - self.t_start > self.config.wall_phase_s:
            self._finish('wall_failure', t)
            return False
        distance = float(np.linalg.norm(np.asarray(flange_xyz, float) - np.asarray(self.config.reference_point_m)))
        low, high = self.config.holding_opening
        if distance < self.config.radius_m and low <= opening <= high:
            self.phase, self.handover_t = 'rl', t
            self.events.append(('handover', t, round(distance, 4), round(float(opening), 3)))
            return True
        return False

    def on_tick(self, t: float) -> None:
        if self.phase == 'wall' and t - self.t_start > self.config.wall_phase_s:
            self._finish('wall_failure', t)
        elif self.phase == 'rl' and t - self.handover_t >= self.config.rl_phase_s:
            self._finish('truncated', t)

    def on_release(self, t: float, source: str) -> None:
        """The executor's gripper command actually turned open (after its confirmation)."""
        if source not in RELEASE_SOURCES:
            raise ValueError(f'release source must be one of {RELEASE_SOURCES}')
        if self.phase != 'rl':
            return
        self.phase, self.release_t, self.release_source = 'released', t, source
        self.events.append(('release', t, source))

    def on_key(self, key: str, t: float) -> str | None:
        """Returns the action the loop must perform, or None if the key does nothing now."""
        keys = self.config.keys
        if key in keys.get('ignored', ()) or self.phase == 'done':
            return None
        if key == keys['discard']:
            self._finish('discarded', t)
            return 'discard'
        if key == keys['release'] and self.phase == 'rl' and not self.human_release_requested:
            self.human_release_requested = True
            self.events.append(('human_release_key', t))
            return 'command_open'            # the loop opens the gripper, then calls on_release(t, 'human')
        if key == keys['success'] and self.phase == 'released':
            self._finish('success', t)
            return 'label'
        if key == keys['failure'] and self.phase in ('rl', 'released'):
            self._finish('failure', t)
            return 'label'
        return None

    # ---- results ----
    def _finish(self, outcome: str, t: float) -> None:
        self.phase, self.outcome = 'done', outcome
        self.events.append(('done', t, outcome))

    @property
    def stores_transitions(self) -> bool:
        return self.outcome in STORED

    @property
    def counts_toward_budget(self) -> bool:
        return self.outcome in STORED

    @property
    def task_success(self) -> bool:
        return self.outcome == 'success'

    def summary(self) -> dict:
        rl_s = None
        if self.handover_t is not None:
            end = self.release_t if self.release_t is not None else (self.events[-1][1] if self.events else None)
            rl_s = None if end is None else round(end - self.handover_t, 3)
        return {'outcome': self.outcome, 'handover_s': None if self.handover_t is None
                else round(self.handover_t - self.t_start, 3), 'rl_phase_s': rl_s,
                'release_source': self.release_source, 'events': self.events}
