"""Explicit slot and bounded-action transforms; no inferred robot geometry."""
from __future__ import annotations

import torch


class SlotMap:
    def __init__(self, indices: list[int], model_dim: int):
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("slot indices must be nonempty and unique")
        if any(type(i) is not int or not 0 <= i < model_dim for i in indices):
            raise ValueError("slot index outside model dimension")
        self.indices = tuple(indices)
        self.model_dim = model_dim

    def encode(self, value: torch.Tensor, valid: torch.Tensor | None = None):
        if value.shape[-1] != len(self.indices):
            raise ValueError("input width does not match explicit slot map")
        if valid is None:
            valid = torch.ones_like(value, dtype=torch.bool)
        if valid.shape != value.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean with the input shape")
        if not torch.isfinite(value[valid]).all():
            raise ValueError("non-finite value in a valid input slot")
        result = value.new_zeros((*value.shape[:-1], self.model_dim))
        mask = torch.zeros_like(result, dtype=torch.bool)
        result[..., list(self.indices)] = value.masked_fill(~valid, 0)
        mask[..., list(self.indices)] = valid
        return result, mask

    def decode(self, value: torch.Tensor):
        if value.shape[-1] != self.model_dim:
            raise ValueError("model output width does not match slot map")
        return value[..., list(self.indices)]


class BoundedActionTransform:
    """Affine conversion for an already agreed physical action representation."""
    def __init__(self, lower, upper):
        self.lower = torch.as_tensor(lower, dtype=torch.float32)
        self.upper = torch.as_tensor(upper, dtype=torch.float32)
        if self.lower.ndim != 1 or self.lower.numel() == 0 or self.lower.shape != self.upper.shape:
            raise ValueError("action bounds must be matching nonempty vectors")
        if not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all():
            raise ValueError("action bounds must be finite")
        if not (self.upper > self.lower).all():
            raise ValueError("each action upper bound must exceed its lower bound")

    def _bounds(self, value):
        if value.shape[-1] != self.lower.numel() or not torch.isfinite(value).all():
            raise ValueError("action has incorrect width or non-finite values")
        return self.lower.to(value), self.upper.to(value)

    def to_rl(self, value: torch.Tensor):
        lo, hi = self._bounds(value)
        if ((value < lo - 1e-6) | (value > hi + 1e-6)).any():
            raise ValueError("physical action outside declared bounds; no implicit clipping")
        return 2 * (value - lo) / (hi - lo) - 1

    def from_rl(self, value: torch.Tensor):
        lo, hi = self._bounds(value)
        if (value.abs() > 1 + 1e-6).any():
            raise ValueError("RL action outside [-1,1]")
        return lo + (value + 1) * (hi - lo) / 2
