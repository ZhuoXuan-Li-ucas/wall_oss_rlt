#!/usr/bin/env python3
"""H100-only full-checkpoint smoke check; synthetic inputs, no robot execution."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import logging
from pathlib import Path
import time
import traceback


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
        raise RuntimeError("Checkpoint key incompatibilities require review; see checkpoint_loads")
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
