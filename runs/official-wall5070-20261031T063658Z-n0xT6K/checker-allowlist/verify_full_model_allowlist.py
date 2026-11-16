#!/usr/bin/env python3
"""H100-only full-checkpoint smoke check; synthetic inputs, no robot execution.

Allowlist variant of /opt/walloss/docker/verify_full_model.py (sha256 cae9f11c...a8c9d6),
approved by the user on 2026-10-31. Only the key-incompatibility check differs: a missing tied
lm_head.weight and unexpected normalizer statistics are accepted after verification.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import logging
from pathlib import Path
import re
import time
import traceback

OFFICIAL_CHECKER_SHA256 = "cae9f11c4bac1fed32945f3ad3cd99a99b148e1a864aaee69f28728232a8c9d6"
ALLOWED_MISSING = {"lm_head.weight"}
ALLOWED_UNEXPECTED = re.compile(r"^action_preprocessor\.normalizer_(action|propri)\.(min|delta)\.([^.]+)$")


def review_key_incompatibilities(adapter, loads, checkpoint_path, report):
    """Accept only allowlisted keys, each verified; raise on anything else."""
    import torch
    from safetensors import safe_open
    from wall_x._vendor.harrix.utils.normalizer import Normalizer

    review = report["key_review"] = {}
    missing = {key for item in loads for key in item["missing_keys"]}
    unexpected = {key for item in loads for key in item["unexpected_keys"]}
    other_missing = sorted(missing - ALLOWED_MISSING)
    other_unexpected = sorted(key for key in unexpected if not ALLOWED_UNEXPECTED.match(key))
    if other_missing or other_unexpected:
        review.update(other_missing=other_missing, other_unexpected=other_unexpected)
        raise RuntimeError("Checkpoint key incompatibilities outside the allowlist; see key_review")

    if "lm_head.weight" in missing:
        model = adapter._model
        lm_head, embed = model.lm_head.weight, model.model.embed_tokens.weight
        tied = bool(getattr(model.config, "tie_word_embeddings", False))
        shared = lm_head.data_ptr() == embed.data_ptr() and lm_head.shape == embed.shape
        review["lm_head"] = {"tie_word_embeddings": tied, "shares_embed_tokens_storage": shared}
        if not (tied and shared):
            raise RuntimeError("lm_head.weight is missing and not tied to embed_tokens")

    if unexpected:
        # build_normalizers prefers norm_stats.json; the .pth files are the source only without it.
        if (checkpoint_path / "norm_stats.json").exists():
            raise RuntimeError("norm_stats.json present; .pth normalizers are not the runtime source")
        sources = {
            kind: Normalizer.from_ckpt(str(checkpoint_path / f"normalizer_{kind}.pth"))
            for kind in ("action", "propri")
        }
        mismatched = []
        with safe_open(str(checkpoint_path / "model.safetensors"), "pt") as tensors:
            for key in sorted(unexpected):
                kind, stat, dataset = ALLOWED_UNEXPECTED.match(key).groups()
                table = getattr(sources[kind], stat)
                embedded = tensors.get_tensor(key).float()
                if dataset not in table or not torch.equal(embedded, table[dataset].float().cpu()):
                    mismatched.append(key)
        review["normalizer_keys_compared"] = len(unexpected)
        review["normalizer_keys_mismatched"] = mismatched
        if mismatched:
            raise RuntimeError("Embedded normalizer statistics differ from normalizer_*.pth; see key_review")
    review["status"] = "PASS"


def main(args, report):
    import numpy as np
    import torch
    import wall_x._vendor.harrix.adapters  # noqa: F401
    from wall_x._vendor.harrix.adapters.registry import build_adapter
    from wall_x._vendor.harrix.eval_config import (
        EvalConfig, LiberoEnvParams, autofill_from_checkpoint,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this check")
    torch.manual_seed(args.seed)
    report.update(gpu=torch.cuda.get_device_name(0), torch_cuda=torch.version.cuda)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "walloss_fake_inference", root / "wall-x/scripts/fake_inference.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    cfg = EvalConfig()
    cfg.model.checkpoint_path = str(args.checkpoint_path)
    cfg.model.norm_key = args.norm_key
    cfg.model.cam_names = ["face_view", "right_wrist_view"]
    cfg.model.architecture = "qwen2_5"
    cfg.model.action_mode = "flow"
    cfg.env.libero = LiberoEnvParams(num_trials_per_task=1, task_indices=[0])
    cfg = autofill_from_checkpoint(cfg)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    # The upstream adapter uses strict=False; record incompatibilities explicitly.
    loads = report["checkpoint_loads"] = []
    original_load = torch.nn.Module.load_state_dict

    def audited_load(module, *positional, **keyword):
        result = original_load(module, *positional, **keyword)
        loads.append({
            "module": type(module).__qualname__,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        })
        return result

    torch.nn.Module.load_state_dict = audited_load
    try:
        adapter = build_adapter(cfg)
    finally:
        torch.nn.Module.load_state_dict = original_load
    torch.cuda.synchronize()
    report["model_load_seconds"] = time.perf_counter() - started
    if not loads:
        raise RuntimeError("Checkpoint load was not observed; inspect the adapter loading path")
    if any(item["missing_keys"] or item["unexpected_keys"] for item in loads):
        review_key_incompatibilities(adapter, loads, args.checkpoint_path, report)
    gripper_dim = helper._right_gripper_dim_from_config(adapter._train_config)
    payload = {
        "observation": helper._build_fake_observation(args.seed, 128, gripper_dim),
        "instruction": "pick up the object", "noise": None,
    }
    started = time.perf_counter()
    with torch.inference_mode():
        action = helper._predict_raw_action(adapter, payload)
    torch.cuda.synchronize()
    report["first_inference_seconds"] = time.perf_counter() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["action_shape"] = list(action.shape)
    report["action_dtype"] = str(action.dtype)
    expected = (adapter._action_horizon, adapter._action_dim)
    if expected != (32, 26) or action.shape != expected:
        raise ValueError(f"Expected checkpoint action (32, 26); configured={expected}, actual={action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Action output contains NaN or infinity")
    report["action_min"] = float(action.min())
    report["action_max"] = float(action.max())
    report["status"] = "PASS"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("/opt/walloss/models/wall-oss-0.5"))
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "full checkpoint with synthetic observations; not AR5, training or service validation",
        "checker": "allowlist variant (user-approved 2026-10-31) of official verify_full_model.py",
        "official_checker_sha256": OFFICIAL_CHECKER_SHA256,
        "status": "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        main(args, report)
    except Exception as error:
        report["error"] = repr(error)
        traceback.print_exc()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "PASS" else 1)
