"""Select only observable, non-padding tokens before the flow suffix."""
from __future__ import annotations

import torch


def observation_prefix_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                            action_token_id: int) -> torch.Tensor:
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must both have shape [B,L]")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("prefix selection requires a binary 2-D padding mask")
    valid = attention_mask.bool()
    actions = input_ids.eq(action_token_id) & valid
    if not actions.any(dim=1).all():
        raise ValueError("each prompt must contain at least one valid flow action token")
    first = actions.long().argmax(dim=1)
    indices = torch.arange(input_ids.shape[1], device=input_ids.device)
    mask = valid & (indices[None] < first[:, None])
    if not mask.any(dim=1).all():
        raise ValueError("each sample must have a nonempty observation prefix")
    return mask


def pack_prefix(hidden_states: torch.Tensor, mask: torch.Tensor,
                max_length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 3 or mask.shape != hidden_states.shape[:2]:
        raise ValueError("hidden_states [B,L,D] and mask [B,L] must align")
    if mask.dtype != torch.bool:
        raise ValueError("prefix mask must be boolean")
    lengths = mask.sum(dim=1)
    if not (lengths > 0).all():
        raise ValueError("empty observation prefix")
    width = int(lengths.max())
    if max_length is not None and width > max_length:
        raise ValueError(f"prefix length {width} exceeds configured maximum {max_length}")
    # Gather in original sequence order; padding values cannot enter the encoder.
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    gather = positions.masked_fill(~mask, mask.shape[1]).sort(dim=1).values[:, :width]
    packed_mask = torch.arange(width, device=mask.device)[None] < lengths[:, None]
    gather = gather.clamp_max(mask.shape[1] - 1)
    packed = hidden_states.gather(1, gather[..., None].expand(-1, -1, hidden_states.shape[2]))
    packed = packed.masked_fill(~packed_mask[..., None], 0)
    if not torch.isfinite(packed).all():
        raise ValueError("non-finite observation prefix")
    return packed, packed_mask
