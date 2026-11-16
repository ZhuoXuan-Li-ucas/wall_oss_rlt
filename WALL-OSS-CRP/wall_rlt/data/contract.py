"""Validate data semantics independently from model and deployment settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DataContract:
    root: str
    repo_id: str
    fps: float
    action_names: list[str]
    state_names: list[str]
    camera_mapping: dict[str, str]
    splits: dict[str, list[int]]
    action_key: str = "action"
    state_key: str = "observation.state"
    action_horizon: int = 32
    model_action_dim: int = 26
    model_state_dim: int = 26
    action_slots: list[int] = field(default_factory=list)
    state_slots: list[int] = field(default_factory=list)
    action_units: list[str] = field(default_factory=list)
    state_units: list[str] = field(default_factory=list)
    action_convention: str = "unconfirmed"
    coordinate_frame: str = "unconfirmed"
    robot_type: str = "unconfirmed"
    semantics_confirmed: bool = False
    purpose: str = "target"
    normalization_path: str = ""
    normalization_sha256: str = ""
    source_format: str = "lerobot"
    manifest_path: str = ""
    manifest_sha256: str = ""
    field_conventions: list[str] = field(default_factory=list)
    alignment: str = "unconfirmed"
    target_mode: str = "recorded"
    model_adapter: str = "unconfirmed"
    provenance: dict = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.purpose not in {"target", "synthetic", "inspection"}:
            raise ValueError("unsupported contract schema or purpose")
        if self.source_format not in {"lerobot", "npz_video", "memory"}:
            raise ValueError("unsupported source format")
        if self.target_mode not in {"recorded", "ar5_joint_next_v1", "ar5_ee_delta_next_v1",
                                    "ar5_ee_rel6d_next_v1"}:
            raise ValueError("unsupported action target mode")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if type(self.action_horizon) is not int or self.action_horizon < 1:
            raise ValueError("action_horizon must be a positive integer")
        for names in (self.action_names, self.state_names):
            if not names or len(names) != len(set(names)) or not all(isinstance(n,str) and n for n in names):
                raise ValueError("action/state names must be nonempty and unique")
        if not self.camera_mapping or len(set(self.camera_mapping.values())) != len(self.camera_mapping):
            raise ValueError("camera mapping must be nonempty and one-to-one")
        used = set()
        if set(self.splits) - {"train", "dev", "test"}:
            raise ValueError("split names must be train/dev/test")
        for name, episodes in self.splits.items():
            if any(type(e) is not int or e < 0 for e in episodes):
                raise ValueError(f"invalid episode IDs in {name}")
            if len(episodes) != len(set(episodes)) or used.intersection(episodes):
                raise ValueError("duplicate episode or leakage between splits")
            used.update(episodes)
        from .transforms import SlotMap
        for indices, names, width in ((self.action_slots, self.action_names, self.model_action_dim),
                                      (self.state_slots, self.state_names, self.model_state_dim)):
            if indices:
                if len(indices) != len(names):
                    raise ValueError("slot map must have one index per source field")
                SlotMap(indices, width)

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        value = json.loads(path.read_text())
        for key in ("root", "normalization_path", "manifest_path"):
            if value.get(key) and not Path(value[key]).is_absolute():
                value[key] = str((path.parent / value[key]).resolve())
        return cls(**value)

    def fingerprint(self):
        # Storage location may differ across training and inference hosts.
        value = asdict(self)
        value.pop("root")
        value.pop("normalization_path")
        value.pop("manifest_path")
        return canonical_hash(value)

    def require_trainable(self, *, allow_synthetic=False):
        if self.purpose != "target" and not (allow_synthetic and self.purpose == "synthetic"):
            raise ValueError("inspection/synthetic data is not a target training dataset")
        if not self.semantics_confirmed:
            raise ValueError("action/state semantics have not been confirmed")
        for names, units, slots in ((self.action_names,self.action_units,self.action_slots),
                                    (self.state_names,self.state_units,self.state_slots)):
            if len(names) != len(units) or len(names) != len(slots) or not all(units):
                raise ValueError("every field needs a unit and an explicit model slot")
        if self.action_convention not in {"absolute", "delta", "mixed"} or self.coordinate_frame in {"", "unconfirmed"}:
            raise ValueError("action convention and coordinate frame must be confirmed")
        if self.action_convention == "mixed" and (
                len(self.field_conventions) != len(self.action_names) or
                any(x not in {"absolute", "delta"} for x in self.field_conventions)):
            raise ValueError("mixed actions require a convention for every field")
        if self.alignment in {"", "unconfirmed"}:
            raise ValueError("observation/action temporal alignment must be confirmed")
        if self.target_mode == "ar5_joint_next_v1":
            if (self.model_adapter != "ar5_joint8_reinit_v1" or
                    self.action_slots != list(range(8)) or self.state_slots != list(range(8)) or
                    self.model_action_dim != 26 or self.model_state_dim != 26 or
                    self.alignment != "observation_t_target_t_plus_1_drop_last"):
                raise ValueError("AR5 joint targets require the explicit joint8 adaptation contract")
        if self.target_mode == "ar5_ee_rel6d_next_v1":
            # Pretrained Wall-OSS-0.5 right-arm slots: pos 10..12, 6D 13..18, gripper 19.
            if (self.model_adapter != "wall_oss05_right_arm_ee_v1" or
                    self.action_slots != list(range(10, 20)) or self.state_slots != list(range(10, 20)) or
                    self.model_action_dim != 26 or self.model_state_dim != 26 or
                    self.action_convention != "mixed" or
                    self.field_conventions != ["delta"] * 9 + ["absolute"] or
                    self.alignment != "observation_t_target_t_plus_1_drop_last"):
                raise ValueError("AR5 EE targets require the explicit Wall-OSS-0.5 right-arm EE contract")
        if self.source_format == "npz_video":
            self.read_manifest()
        if not self.splits.get("train"):
            raise ValueError("train split must be nonempty")
        if not self.normalization_path or not self.normalization_sha256:
            raise ValueError("training requires fingerprinted train-only normalization")
        raw = Path(self.normalization_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.normalization_sha256:
            raise ValueError("normalization SHA256 mismatch")
        payload = json.loads(raw)
        if payload.get("train_episode_ids") != self.splits["train"]:
            raise ValueError("normalization was not computed for this exact train split")
        if payload.get("target_mode", "recorded") != self.target_mode:
            raise ValueError("normalization target representation mismatch")
        if self.source_format == "npz_video" and payload.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError("normalization source manifest mismatch")
        for key, names in ((self.action_key,self.action_names),(self.state_key,self.state_names)):
            stats = payload["norm_stats"][key]
            for quantile in ("q01", "q99"):
                if len(stats[quantile]) != len(names) or not all(math.isfinite(v) for v in stats[quantile]):
                    raise ValueError("invalid normalization dimension or value")
            if any(hi < lo for lo,hi in zip(stats['q01'],stats['q99'])):
                raise ValueError("inverted normalization quantiles")

    def read_manifest(self):
        if not self.manifest_path or not self.manifest_sha256:
            raise ValueError("NPZ source requires a fingerprinted episode manifest")
        raw = Path(self.manifest_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.manifest_sha256:
            raise ValueError("episode manifest SHA256 mismatch")
        manifest = json.loads(raw)
        ids = [e['episode_id'] for e in manifest['episodes']]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate manifest episode ID")
        selected = {i for split in self.splits.values() for i in split}
        if not selected.issubset(ids):
            raise ValueError("split references an unknown manifest episode")
        if manifest.get('splits') != self.splits:
            raise ValueError("contract split differs from the frozen manifest")
        excluded = set(manifest.get('excluded_groups', []))
        if any(e['episode_id'] in selected and (e.get('task_group') in excluded or
               e.get('structural_errors') or e.get('semantic_errors')) for e in manifest['episodes']):
            raise ValueError("split contains an excluded or failed episode")
        return manifest
