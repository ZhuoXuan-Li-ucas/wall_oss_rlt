"""Serializable, strict configuration for the model-side RLT integration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WallRLTConfig:
    input_dim: int = 2048
    embed_dim: int = 2048
    prefix_seq_len: int = 1024
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    rlt_alpha: float = 1.0
    action_horizon: int = 32
    model_action_dim: int = 26
    action_token_id: int = 151666
    checkpoint_path: str = ""
    data_contract_path: str = ""
    wall_train_config: str = ""
    schema_version: int = 1

    def __post_init__(self):
        for name in ("input_dim", "embed_dim", "prefix_seq_len", "num_layers", "num_heads",
                     "action_horizon", "model_action_dim"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not math.isfinite(self.rlt_alpha) or self.rlt_alpha <= 0:
            raise ValueError("rlt_alpha must be finite and positive")
        if not math.isfinite(self.mlp_ratio) or self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be finite and positive")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0,1)")
        if self.schema_version != 1:
            raise ValueError(f"unsupported config schema {self.schema_version}")

    @classmethod
    def from_dict(cls, value: dict[str, Any]):
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown WALL-RLT config fields: {sorted(unknown)}")
        return cls(**value)

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n")

    def token_kwargs(self):
        return {key: getattr(self, key) for key in (
            "input_dim", "embed_dim", "prefix_seq_len", "num_layers", "num_heads",
            "mlp_ratio", "dropout_rate")}
