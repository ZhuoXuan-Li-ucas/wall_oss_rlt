# Vendored Evo-RLT sources

`evo_rlt/` and `evo_rlt_tests/` are **byte-identical** copies of a subset of the
Evo-RLT project (`My Passport: wy/EVO/Evo-RLT.zip`, the 2026-08-03 archive,
project version 0.1.0). Every file's sha256 and the zip central directory's
CRC32/size are recorded in `MANIFEST.json`; `rlt_online.evo.verify_vendor()`
re-checks them. Do not edit these files. Project-specific behaviour lives in
`rlt_online/` outside `_vendor/`.

License: the archive's `pyproject.toml` declares Apache-2.0. The archive does
not contain a LICENSE file. Some `adapters/lerobot/record/*` files carry the
HuggingFace Apache-2.0 header inherited from LeRobot.

## What was taken

- `evo_rlt/__init__.py`, all of `evo_rlt/core/` (the package `__init__`
  imports every core module, so the whole directory is needed).
- `evo_rlt/adapters/{__init__,lerobot/__init__,lerobot/registry}.py` — import
  scaffolding only. `registry.py` imports LeRobot lazily inside functions and
  is never called here.
- `evo_rlt/adapters/lerobot/online_collector.py` — chunk transition builder with
  early chunk close and `actual_steps` on termination.
- `evo_rlt/adapters/lerobot/record/{__init__,annotations,online_trainer,prune_state_cli}.py`
  — the online trainer (warmup thresholds, critic-only window, UTD budget,
  offline/online mixing, stratified sampling, atomic resume state) and the
  episode prune tool. None of them imports LeRobot.
- `evo_rlt_tests/tests/rlt/*` — the upstream tests that do not need LeRobot.
  Run them with `rlt_online/_vendor` and `rlt_online/_vendor/evo_rlt_tests`
  on `PYTHONPATH` (they import `tests.rlt.helpers`, which would clash with
  this project's own `tests/rlt`). Result on 2026-11-10, Python 3.10 /
  torch 2.10: 143 passed, 1 failed. The failure is
  `test_training_step.py::test_lerobot_ac_actor_loss_does_not_add_critic_grads`,
  which imports the LeRobot `ChunkACPolicy` that was deliberately not
  vendored. The equivalent property is tested on this project's stand-in in
  `tests/rlt_online/test_policy_trainer.py`.

## What was not taken

LeRobot recording loop, robot/teleop/HIL code, pi0.5 adapter, RL-token
training, LeRobot policy wrappers (`policies/*`), RTC code, offline cache
builders. The WALL inference service supplies `z_rl` and the reference chunk
instead.

## Known upstream issues that the adapter must handle (audit 2026-11-05)

1. Units: upstream's recording loop feeds the collector robot-unit actions
   while `ref_chunk`/`state_vec` are normalized. Feed normalized chunks only.
2. Chunk alignment: upstream's collector counts C frames from
   `start_episode`, not from the actor's chunk start. Feed it per chunk.
3. Upstream online runs use `fixed_std=0` and `ref_dropout=0` (mean rollout,
   no exploration). The paper uses both.
4. Quantile clamping makes the zero-init residual actor differ from the VLA
   where the VLA exceeds q01/q99. Bounds here come from train min/max plus a
   margin (`rlt_online.action_space`).
5. With `residual_to_ref=True`, reference dropout zeroes only the network
   input; the output still adds the full reference chunk.
6. `online_trainer.py` pickles the whole replay buffer every episode and
   aborts on an unlabeled episode in upstream's loop; the replacement loop
   decides when to call `save_latest_state()`.
