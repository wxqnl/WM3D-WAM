"""Sparse per-layer geometry K/V adapters for Wan-Action mixed attention."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class _GeometryKVProjection(nn.Module):
    def __init__(self, geometry_dim: int, attention_width: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(geometry_dim, eps=eps)
        self.projection = nn.Linear(geometry_dim, 2 * attention_width, bias=False)
        self.gate = nn.Parameter(torch.full((attention_width,), 0.01))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = self.projection(self.norm(tokens)).chunk(2, dim=-1)
        gate = self.gate.to(device=tokens.device, dtype=tokens.dtype)
        return key * gate, value * gate


class SparseGeometryKVAdapters(nn.Module):
    """Project predicted VGGT tokens into selected MoT attention layers.

    Geometry supplies K/V only. It never becomes a query stream and clean
    target geometry is deliberately absent from this interface.
    """

    def __init__(
        self,
        *,
        geometry_dim: int = 1024,
        num_heads: int = 24,
        attention_head_dim: int = 128,
        fusion_layers: Sequence[int] = (5, 11, 17, 23, 29),
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        layers = tuple(int(layer) for layer in fusion_layers)
        if not layers or len(layers) != len(set(layers)) or min(layers) < 0:
            raise ValueError("fusion_layers must contain unique non-negative indices")
        if geometry_dim <= 0 or num_heads <= 0 or attention_head_dim <= 0:
            raise ValueError("geometry and attention dimensions must be positive")
        self.geometry_dim = int(geometry_dim)
        self.num_heads = int(num_heads)
        self.attention_head_dim = int(attention_head_dim)
        self.attention_width = self.num_heads * self.attention_head_dim
        self.fusion_layers = layers
        self.projections = nn.ModuleDict(
            {
                str(layer): _GeometryKVProjection(
                    self.geometry_dim, self.attention_width, eps
                )
                for layer in layers
            }
        )

    def forward(
        self,
        *,
        geometry_tokens: torch.Tensor,
        geometry_token_mask: torch.Tensor,
        query_token_mask: torch.Tensor,
    ) -> dict[int, dict[str, torch.Tensor]]:
        if geometry_tokens.ndim != 3:
            raise ValueError("geometry_tokens must be [B,Sg,Dg]")
        batch_size, geometry_length, geometry_dim = geometry_tokens.shape
        if geometry_dim != self.geometry_dim:
            raise ValueError(
                f"geometry token width must be {self.geometry_dim}, got {geometry_dim}"
            )
        geometry_mask = geometry_token_mask.to(
            device=geometry_tokens.device, dtype=torch.bool
        )
        query_mask = query_token_mask.to(
            device=geometry_tokens.device, dtype=torch.bool
        )
        if geometry_mask.shape != (batch_size, geometry_length):
            raise ValueError("geometry_token_mask shape mismatch")
        if query_mask.ndim != 2 or query_mask.shape[0] != batch_size:
            raise ValueError("query_token_mask must be [B,Sq]")
        if not bool(geometry_mask.any(dim=1).all()):
            raise ValueError("every sample must provide at least one predicted geometry token")
        attention_mask = (
            query_mask.unsqueeze(2) & geometry_mask.unsqueeze(1)
        ).unsqueeze(1)
        outputs: dict[int, dict[str, torch.Tensor]] = {}
        for layer in self.fusion_layers:
            key, value = self.projections[str(layer)](geometry_tokens)
            outputs[layer] = {
                "k": key,
                "v": value,
                "attention_mask": attention_mask,
            }
        return outputs
