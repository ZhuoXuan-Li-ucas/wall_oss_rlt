#!/usr/bin/env python3
"""Verify installation, native kernels and local assets without full-model inference."""
from collections import Counter
import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image
from safetensors import safe_open
import torch
from transformers import AutoProcessor
import yaml

ROOT = Path(__file__).resolve().parents[1]
result = {
    "scope": "setup only; no full model load or action inference",
    "checked_at_utc": datetime.now(timezone.utc).isoformat(), "checks": {},
}


def main():
    assert Path(sys.prefix).resolve() == (ROOT / "env").resolve(), sys.prefix
    assert not any("/opt/ros/" in path for path in sys.path), sys.path
    result["python"] = sys.version
    result["python_search_path"] = list(sys.path)
    result["versions"] = {
        name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "torchaudio", "transformers", "flash-attn",
            "accelerate", "peft", "lerobot", "wall_x", "dmuon", "huggingface-hub",
        )
    }
    assert torch.cuda.is_available()
    result["gpu"] = torch.cuda.get_device_name(0)
    result["torch_cuda"] = torch.version.cuda
    result["bf16_supported"] = torch.cuda.is_bf16_supported()
    assert result["bf16_supported"]
    tensor = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
    product = tensor @ tensor
    assert torch.isfinite(product).all() and (product == 64).all()
    result["checks"]["cuda_bf16_matmul"] = "PASS"

    from flash_attn import flash_attn_func
    torch.manual_seed(0)
    qkv = [torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    attention = flash_attn_func(*qkv, causal=False)
    reference = torch.nn.functional.scaled_dot_product_attention(
        *[t.transpose(1, 2) for t in qkv]
    ).transpose(1, 2)
    assert attention.shape == (1, 32, 2, 64) and torch.isfinite(attention).all()
    assert torch.allclose(attention, reference, atol=0.025, rtol=0.025)
    torch.cuda.synchronize()
    result["checks"]["flash_attention_bf16"] = "PASS"

    for name in (
        "wall_x.model.core.ops._cuda_ext_bin",
        "wall_x.model.qact.qwen2_5",
        "wall_x._vendor.harrix.adapters",
        "lerobot.datasets.lerobot_dataset",
    ):
        module = importlib.import_module(name)
        result["checks"][name] = {"status": "PASS", "path": str(module.__file__)}

    checkpoint = ROOT / "models/wall-oss-0.5"
    cfg = yaml.safe_load((checkpoint / "config.yml").read_text())
    result["action_dim"] = sum(cfg["dof_config"].values())
    assert result["action_dim"] == 26
    result["action_horizon"] = cfg["data"]["action_horizon_flow"]
    assert result["action_horizon"] == 32
    for dirname in ("wall-oss-0.5", "Qwen2.5-VL-3B-Instruct"):
        processor = AutoProcessor.from_pretrained(ROOT / "models" / dirname, local_files_only=True)
        image = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))
        batch = processor(
            text=["<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>pick up the object"],
            images=[image, image], return_tensors="pt",
        )
        result["checks"][f"processor:{dirname}"] = {
            "status": "PASS", "class": type(processor).__name__,
            "tokenizer_size": len(processor.tokenizer),
            "image_grid_thw": batch["image_grid_thw"].tolist(),
            "pixel_values_shape": list(batch["pixel_values"].shape),
        }

    from wall_x._vendor.harrix.utils.normalizer import build_normalizers
    action, propri, key = build_normalizers(str(checkpoint), cfg, "x2_normal")
    assert key == "x2_normal" and key in action.delta and key in propri.delta
    for normalizer in (action, propri):
        assert normalizer.delta[key].numel() == 26
        assert torch.isfinite(normalizer.delta[key]).all()
        assert torch.isfinite(normalizer.min[key]).all()
    result["checks"]["official_normalizers"] = {"status": "PASS", "norm_key": key, "dimension": 26}
    from wall_x._vendor.harrix.utils.train_config import load_train_config_with_ckpt_overlay
    from wall_x.trainer.trainer_utils import load_wallx_processors
    runtime_cfg = load_train_config_with_ckpt_overlay(
        str(checkpoint / "config.yml"), str(checkpoint)
    )
    runtime_processor = load_wallx_processors(runtime_cfg, normalizer=action, device="cpu")["processor"]
    token_ids = {
        token: runtime_processor.tokenizer.convert_tokens_to_ids(token)
        for token in ("<|propri|>", "<|action|>")
    }
    assert all(value is not None and value != runtime_processor.tokenizer.unk_token_id
               for value in token_ids.values())
    result["checks"]["official_flow_processor"] = {
        "status": "PASS", "path": runtime_cfg["processor_path"],
        "tokenizer_size": len(runtime_processor.tokenizer), "special_token_ids": token_ids,
    }
    count = 0
    dtypes = Counter()
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        for name in weights.keys():
            view = weights.get_slice(name)
            elements = math.prod(view.get_shape())
            count += elements
            dtypes[view.get_dtype()] += elements
        result["checkpoint_tensor_count"] = len(weights.keys())
    result["checkpoint_stored_elements"] = count
    result["checkpoint_dtypes_by_elements"] = dict(dtypes)
    result["checkpoint_tensor_storage_bytes"] = dtypes["BF16"] * 2 + dtypes["F32"] * 4
    result["checks"]["safetensors_header"] = "PASS"
    result["status"] = "PASS"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "setup-verification.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception as error:
        result["status"] = "FAIL"
        result["error"] = repr(error)
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    sys.exit(0 if result.get("status") == "PASS" else 1)
