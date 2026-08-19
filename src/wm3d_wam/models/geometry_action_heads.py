"""Leakage-safe grouped auxiliary action heads for online VGGT-GAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from wm3d_wam.data.action_events import GroupedActionBatch

from .grouped_action_flow import GroupedActionCodec, GroupedActionCodecConfig


@dataclass(frozen=True)
class GroupedAuxiliaryActionOutput:
    direct: torch.Tensor
    refined: torch.Tensor


class _AnchorConditionedActionBranch(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_ratio: float) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("auxiliary action width must be divisible by num_heads")
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * ffn_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * ffn_ratio), hidden_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.cross_attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_mask,
            need_weights=False,
        )[0]
        hidden = query + attended
        return hidden + self.ffn(self.ffn_norm(hidden))


class GroupedGeometryActionHeads(nn.Module):
    """Decode direct predictor seeds and deep VGGT action tokens to events.

    The event template contributes only timestamps, masks, embodiment, and
    semantic/group metadata.  Its clean scalar values are replaced by zeros
    before query encoding, making target-to-policy leakage impossible here.
    """

    def __init__(
        self,
        *,
        codec_config: GroupedActionCodecConfig | Mapping[str, object],
        future_anchor_count: int = 4,
        num_heads: int = 16,
        ffn_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.codec = GroupedActionCodec(codec_config)
        self.future_anchor_count = int(future_anchor_count)
        if self.future_anchor_count <= 0:
            raise ValueError("future_anchor_count must be positive")
        hidden = self.codec.hidden_dim
        self.anchor_position = nn.Parameter(
            torch.zeros(self.future_anchor_count, hidden)
        )
        nn.init.normal_(self.anchor_position, std=0.02)
        self.direct_branch = _AnchorConditionedActionBranch(
            hidden, int(num_heads), float(ffn_ratio)
        )
        self.refined_branch = _AnchorConditionedActionBranch(
            hidden, int(num_heads), float(ffn_ratio)
        )

    @staticmethod
    def _pool_views(tokens: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        weights = view_mask.to(device=tokens.device, dtype=tokens.dtype)
        denominator = weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        return (tokens * weights.unsqueeze(-1)).sum(dim=2) / denominator

    def forward(
        self,
        *,
        predicted_action_seed_tokens: torch.Tensor,
        refined_action_tokens: torch.Tensor,
        total_steps: int,
        future_view_valid_mask: torch.Tensor,
        action_template: GroupedActionBatch,
    ) -> GroupedAuxiliaryActionOutput:
        if predicted_action_seed_tokens.ndim != 4:
            raise ValueError("direct action seeds must be [B,A,V,D]")
        batch_size, anchors, views, hidden = predicted_action_seed_tokens.shape
        if anchors != self.future_anchor_count or hidden != self.codec.hidden_dim:
            raise ValueError("direct action seed layout does not match the head")
        if refined_action_tokens.ndim != 2 or refined_action_tokens.shape != (
            batch_size * int(total_steps) * views,
            hidden,
        ):
            raise ValueError("refined VGGT action token layout is invalid")
        view_mask = future_view_valid_mask.to(
            device=predicted_action_seed_tokens.device, dtype=torch.bool
        )
        if view_mask.shape != (batch_size, anchors, views):
            raise ValueError("future view mask does not match auxiliary action seeds")
        memory_mask = view_mask.any(dim=2)
        if not bool(memory_mask.all()):
            raise ValueError("every future anchor needs at least one real view")

        direct_memory = self._pool_views(
            predicted_action_seed_tokens, view_mask
        )
        refined = refined_action_tokens.reshape(
            batch_size, int(total_steps), views, hidden
        )[:, -anchors:]
        refined_memory = self._pool_views(refined, view_mask)
        position = self.anchor_position.to(
            device=direct_memory.device, dtype=direct_memory.dtype
        ).unsqueeze(0)
        direct_memory = direct_memory + position
        refined_memory = refined_memory + position

        blank_template = action_template.with_values(
            torch.zeros_like(action_template.values)
        )
        query = self.codec.encode(blank_template)
        direct_tokens = self.direct_branch(query, direct_memory, memory_mask)
        refined_tokens = self.refined_branch(query, refined_memory, memory_mask)
        return GroupedAuxiliaryActionOutput(
            direct=self.codec.decode(direct_tokens, blank_template),
            refined=self.codec.decode(refined_tokens, blank_template),
        )
