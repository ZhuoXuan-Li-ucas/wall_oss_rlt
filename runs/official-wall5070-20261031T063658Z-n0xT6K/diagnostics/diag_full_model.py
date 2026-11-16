"""Diagnostic only: mirrors /opt/walloss/docker/verify_full_model.py setup, but records
lm_head tying and normalizer state instead of failing on key incompatibilities.
Not an acceptance check."""
import importlib.util
import json
import sys
import time

import numpy as np
import torch
import wall_x._vendor.harrix.adapters  # noqa: F401
from wall_x._vendor.harrix.adapters.registry import build_adapter
from wall_x._vendor.harrix.eval_config import EvalConfig, LiberoEnvParams, autofill_from_checkpoint

out_path = sys.argv[1]
report = {"scope": "diagnostic only; not acceptance", "gpu": torch.cuda.get_device_name(0)}
torch.manual_seed(0)
spec = importlib.util.spec_from_file_location("fi", "/opt/walloss/wall-x/scripts/fake_inference.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
cfg = EvalConfig()
cfg.model.checkpoint_path = "/opt/walloss/models/wall-oss-0.5"
cfg.model.norm_key = "x2_normal"
cfg.model.cam_names = ["face_view", "right_wrist_view"]
cfg.model.architecture = "qwen2_5"
cfg.model.action_mode = "flow"
cfg.env.libero = LiberoEnvParams(num_trials_per_task=1, task_indices=[0])
cfg = autofill_from_checkpoint(cfg)

adapter = build_adapter(cfg)
model = adapter._model
lm, emb = model.lm_head.weight, model.model.embed_tokens.weight
report["lm_head_is_embed_tokens"] = lm is emb
report["lm_head_shares_storage"] = lm.data_ptr() == emb.data_ptr()
report["lm_head_equals_embed_tokens"] = bool(lm.shape == emb.shape and torch.equal(lm, emb))
report["tie_word_embeddings"] = getattr(model.config, "tie_word_embeddings", None)
report["resolved_norm_key"] = adapter._norm_key

gripper_dim = helper._right_gripper_dim_from_config(adapter._train_config)
payload = {"observation": helper._build_fake_observation(0, 128, gripper_dim),
           "instruction": "pick up the object", "noise": None}
torch.cuda.synchronize()
t = time.perf_counter()
with torch.inference_mode():
    action = helper._predict_raw_action(adapter, payload)
torch.cuda.synchronize()
a = np.asarray(action.float().cpu() if torch.is_tensor(action) else action, dtype=np.float32)
report.update(first_inference_seconds=time.perf_counter() - t,
              action_shape=list(a.shape[-2:]) if a.ndim >= 2 else list(a.shape),
              action_all_finite=bool(np.isfinite(a).all()),
              action_min=float(a.min()), action_max=float(a.max()),
              peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
