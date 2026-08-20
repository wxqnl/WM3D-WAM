"""Original-WM3D state prior and factual dynamics for online VGGT tokens.

This module deliberately contains no policy query, action decoder, or RGB
decoder.  It adapts the world-state half of the original WM3D core to the
online VGGT shallow-token ABI; Wan owns RGB generation and ActionDiT owns
action generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)


def _round_multiple(value: float, multiple: int = 256) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


@dataclass(frozen=True)
class WM3DStateDynamicsConfig:
    """Architecture contract for the state-only WM3D core."""

    observed_steps: int = 4
    future_steps: int = 16
    token_count: int = 261
    token_dim: int = 1024
    max_views: int = 3
    history_dim: int = 1024
    language_dim: int = 4096

    state_hidden: int = 1600
    state_layers: int = 18
    state_heads: int = 16
    state_ff_mult: float = 2.5
    dynamics_layers: int = 1

    view_hidden: int = 1024
    view_heads: int = 8
    view_ff_mult: float = 2.5

    time_fourier_dim: int = 128
    time_min_period_s: float = 0.01
    time_max_period_s: float = 120.0
    dropout: float = 0.0
    activation_checkpointing: bool = True

    def validate(self) -> None:
        for name in (
            "observed_steps",
            "future_steps",
            "token_count",
            "token_dim",
            "max_views",
            "history_dim",
            "language_dim",
            "state_hidden",
            "state_layers",
            "state_heads",
            "dynamics_layers",
            "view_hidden",
            "view_heads",
            "time_fourier_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.state_hidden % self.state_heads:
            raise ValueError("state_hidden must be divisible by state_heads")
        if self.view_hidden % self.view_heads:
            raise ValueError("view_hidden must be divisible by view_heads")
        if self.time_fourier_dim % 2:
            raise ValueError("time_fourier_dim must be even")
        if not 0 < self.time_min_period_s < self.time_max_period_s:
            raise ValueError("time periods must satisfy 0 < min < max")
        if self.state_ff_mult <= 0 or self.view_ff_mult <= 0:
            raise ValueError("feed-forward multipliers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value_fp32 = value.float()
        normalized = value_fp32 * torch.rsqrt(
            value_fp32.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=value.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float, dropout: float) -> None:
        super().__init__()
        inner = _round_multiple(dim * mult)
        self.gate_up = nn.Linear(dim, inner * 2, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(value).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * up))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = float(dropout)

    def forward(
        self,
        value: torch.Tensor,
        *,
        is_causal: bool = False,
        allowed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, length, _ = value.shape
        query, key, item = self.qkv(value).chunk(3, dim=-1)
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        item = item.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        result = F.scaled_dot_product_attention(
            query,
            key,
            item,
            attn_mask=allowed_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and allowed_mask is None,
        )
        return self.out(result.transpose(1, 2).reshape(batch, length, self.dim))


class CrossAttention(nn.Module):
    def __init__(
        self, query_dim: int, context_dim: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.heads = int(heads)
        self.head_dim = self.query_dim // self.heads
        self.query = nn.Linear(query_dim, query_dim, bias=False)
        self.key_value = nn.Linear(context_dim, query_dim * 2, bias=False)
        self.out = nn.Linear(query_dim, query_dim, bias=False)
        self.dropout = float(dropout)

    def forward(
        self,
        query_value: torch.Tensor,
        context: torch.Tensor,
        *,
        allowed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, query_length, _ = query_value.shape
        context_length = int(context.shape[1])
        query = self.query(query_value).view(
            batch, query_length, self.heads, self.head_dim
        ).transpose(1, 2)
        key, value = self.key_value(context).chunk(2, dim=-1)
        key = key.view(batch, context_length, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, context_length, self.heads, self.head_dim).transpose(1, 2)
        result = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(
            result.transpose(1, 2).reshape(batch, query_length, self.query_dim)
        )


class ContinuousTimeEmbedding(nn.Module):
    """Fourier time encoding parameterized in physical seconds."""

    def __init__(self, output_dim: int, config: WM3DStateDynamicsConfig) -> None:
        super().__init__()
        half = config.time_fourier_dim // 2
        frequencies = torch.exp(
            torch.linspace(
                log(1.0 / config.time_max_period_s),
                log(1.0 / config.time_min_period_s),
                half,
            )
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.proj = nn.Sequential(
            nn.Linear(config.time_fourier_dim + 1, output_dim, bias=False),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim, bias=False),
        )

    def forward(self, seconds: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(seconds):
            seconds = seconds.float()
        angles = seconds[..., None] * self.frequencies.to(dtype=seconds.dtype)
        features = torch.cat(
            (seconds[..., None], torch.sin(angles), torch.cos(angles)), dim=-1
        )
        return self.proj(features)


class MultiViewTokenFuser(nn.Module):
    """Fuse only the real views at each time/token coordinate."""

    def __init__(self, config: WM3DStateDynamicsConfig) -> None:
        super().__init__()
        self.max_views = int(config.max_views)
        self.in_proj = nn.Linear(config.token_dim, config.view_hidden, bias=False)
        self.view_embed = nn.Parameter(
            torch.empty(1, 1, self.max_views, 1, config.view_hidden)
        )
        nn.init.normal_(self.view_embed, std=0.02)
        self.attn_norm = RMSNorm(config.view_hidden)
        self.attn = SelfAttention(
            config.view_hidden, config.view_heads, config.dropout
        )
        self.ff_norm = RMSNorm(config.view_hidden)
        self.ff = SwiGLU(
            config.view_hidden, config.view_ff_mult, config.dropout
        )
        self.gate = nn.Linear(config.view_hidden, 1, bias=False)
        self.out_proj = nn.Linear(
            config.view_hidden, config.state_hidden, bias=False
        )

    def forward(
        self, tokens: torch.Tensor, view_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, frames, views, token_count, _ = tokens.shape
        if not 1 <= views <= self.max_views:
            raise ValueError(
                f"real view count {views} is outside [1,{self.max_views}]"
            )
        if tuple(view_mask.shape) != (batch, frames, views):
            raise ValueError("view_mask must align with [B,T,V]")
        if not bool(view_mask.any(dim=-1).all()):
            raise ValueError("every observed state needs at least one real view")
        value = self.in_proj(tokens) + self.view_embed[:, :, :views]
        value = value.permute(0, 1, 3, 2, 4).reshape(
            batch * frames * token_count, views, -1
        )
        valid = view_mask[:, :, None, :].expand(
            batch, frames, token_count, views
        ).reshape(batch * frames * token_count, views)
        value = value + self.attn(
            self.attn_norm(value), allowed_mask=valid[:, None, None, :]
        )
        value = value + self.ff(self.ff_norm(value))
        logits = self.gate(value).squeeze(-1).masked_fill(~valid, -torch.inf)
        fused = (value * logits.softmax(dim=-1)[..., None]).sum(dim=1)
        return self.out_proj(fused.view(batch, frames, token_count, -1))


class FactorizedStateBlock(nn.Module):
    """Spatial attention per time and causal temporal attention per token."""

    def __init__(self, config: WM3DStateDynamicsConfig) -> None:
        super().__init__()
        dim = config.state_hidden
        self.spatial_norm = RMSNorm(dim)
        self.spatial = SelfAttention(dim, config.state_heads, config.dropout)
        self.temporal_norm = RMSNorm(dim)
        self.temporal = SelfAttention(dim, config.state_heads, config.dropout)
        self.ff_norm = RMSNorm(dim)
        self.ff = SwiGLU(dim, config.state_ff_mult, config.dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, token_count, dim = value.shape
        spatial = value.reshape(batch * frames, token_count, dim)
        spatial = spatial + self.spatial(self.spatial_norm(spatial))
        value = spatial.view(batch, frames, token_count, dim)
        temporal = value.transpose(1, 2).reshape(
            batch * token_count, frames, dim
        )
        temporal = temporal + self.temporal(
            self.temporal_norm(temporal), is_causal=True
        )
        value = temporal.view(batch, token_count, frames, dim).transpose(1, 2)
        return value + self.ff(self.ff_norm(value))


class DynamicsConditionBlock(nn.Module):
    """Refine a completed action-free prior with factual future actions."""

    def __init__(self, config: WM3DStateDynamicsConfig) -> None:
        super().__init__()
        dim = config.state_hidden
        self.null_action = nn.Parameter(torch.empty(1, 1, 1, dim))
        nn.init.normal_(self.null_action, std=0.02)
        self.state_norm = RMSNorm(dim)
        self.action_norm = RMSNorm(dim)
        self.cross = CrossAttention(dim, dim, config.state_heads, config.dropout)
        self.factorized = FactorizedStateBlock(config)

    def forward(
        self,
        future_state: torch.Tensor,
        factual_action: torch.Tensor,
        factual_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, token_count, dim = future_state.shape
        action_slots = int(factual_action.shape[2])
        if tuple(factual_action.shape[:2]) != (batch, horizon):
            raise ValueError("factual actions must align with the future horizon")
        if factual_mask.shape != factual_action.shape[:-1]:
            raise ValueError("factual action mask must align with factual actions")
        null = self.null_action.expand(batch, horizon, -1, -1)
        context = torch.cat((null, factual_action), dim=2)
        valid = torch.cat(
            (
                torch.ones(
                    batch,
                    horizon,
                    1,
                    dtype=torch.bool,
                    device=factual_mask.device,
                ),
                factual_mask.bool(),
            ),
            dim=2,
        )
        query = future_state.reshape(batch * horizon, token_count, dim)
        context = context.reshape(batch * horizon, action_slots + 1, dim)
        valid = valid.reshape(batch * horizon, action_slots + 1)
        update = self.cross(
            self.state_norm(query),
            self.action_norm(context),
            allowed_mask=valid[:, None, None, :],
        )
        state = (query + update).view(batch, horizon, token_count, dim)
        return self.factorized(state)


class ViewTokenDecoder(nn.Module):
    """Decode one fused WM3D state into the original per-view VGGT ABI."""

    def __init__(self, config: WM3DStateDynamicsConfig) -> None:
        super().__init__()
        self.max_views = int(config.max_views)
        self.view_embed = nn.Parameter(
            torch.empty(1, 1, self.max_views, 1, config.state_hidden)
        )
        nn.init.normal_(self.view_embed, std=0.02)
        self.norm = RMSNorm(config.state_hidden)
        self.output = nn.Linear(config.state_hidden, config.token_dim, bias=False)

    def forward(self, state: torch.Tensor, *, views: int) -> torch.Tensor:
        if not 1 <= int(views) <= self.max_views:
            raise ValueError("decoder view count is outside configured capacity")
        value = state[:, :, None] + self.view_embed[:, :, : int(views)]
        return self.output(self.norm(value))


@dataclass(frozen=True)
class WM3DStateDynamicsOutput:
    action_free_native_state: torch.Tensor
    native_state: torch.Tensor
    action_free_tokens: torch.Tensor
    factual_tokens: torch.Tensor


class WM3DStateDynamicsCore(nn.Module):
    """Action-free WM3D state prior followed by factual dynamics refinement."""

    def __init__(self, config: WM3DStateDynamicsConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.view_fuser = MultiViewTokenFuser(config)
        self.state_time = ContinuousTimeEmbedding(config.state_hidden, config)
        self.state_space = nn.Parameter(
            torch.empty(1, 1, config.token_count, config.state_hidden)
        )
        self.future_queries = nn.Parameter(
            torch.empty(
                1,
                config.future_steps,
                config.token_count,
                config.state_hidden,
            )
        )
        nn.init.normal_(self.state_space, std=0.02)
        nn.init.normal_(self.future_queries, std=0.02)
        self.history_state = nn.Linear(
            config.history_dim, config.state_hidden, bias=False
        )
        self.history_action = nn.Linear(
            config.history_dim, config.state_hidden, bias=False
        )
        self.factual_action = nn.Linear(
            config.history_dim, config.state_hidden, bias=False
        )
        self.task = nn.Linear(config.language_dim, config.state_hidden, bias=False)
        self.state_input_norm = RMSNorm(config.state_hidden)
        self.state_blocks = self._checkpoint_module_list(
            (FactorizedStateBlock(config) for _ in range(config.state_layers)),
            enabled=config.activation_checkpointing,
        )
        self.dynamics_blocks = self._checkpoint_module_list(
            (DynamicsConditionBlock(config) for _ in range(config.dynamics_layers)),
            enabled=config.activation_checkpointing,
        )
        self.state_norm = RMSNorm(config.state_hidden)
        self.token_decoder = ViewTokenDecoder(config)

    @staticmethod
    def _checkpoint_module_list(
        modules: Iterable[nn.Module], *, enabled: bool
    ) -> nn.ModuleList:
        values = tuple(modules)
        if enabled:
            values = tuple(checkpoint_wrapper(module) for module in values)
        return nn.ModuleList(values)

    @staticmethod
    def _pool_language(
        context: torch.Tensor, context_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError("language context must be [B,L,D]")
        if context_mask is None:
            return context.mean(dim=1)
        valid = context_mask.to(device=context.device, dtype=torch.bool)
        if valid.shape != context.shape[:2]:
            raise ValueError("language mask must align with context tokens")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every sample needs at least one language token")
        weight = valid.unsqueeze(-1).to(dtype=context.dtype)
        return (context * weight).sum(dim=1) / weight.sum(dim=1)

    def forward(
        self,
        *,
        observed_tokens: torch.Tensor,
        observed_view_mask: torch.Tensor,
        world_times_s: torch.Tensor,
        history_state_tokens: torch.Tensor,
        history_action_tokens: torch.Tensor,
        language_context: torch.Tensor,
        language_mask: Optional[torch.Tensor] = None,
        factual_action_tokens: Optional[torch.Tensor] = None,
        factual_action_mask: Optional[torch.Tensor] = None,
    ) -> WM3DStateDynamicsOutput:
        cfg = self.config
        if observed_tokens.ndim != 5:
            raise ValueError("observed VGGT tokens must be [B,T,V,N,D]")
        batch, observed_steps, views, token_count, token_dim = observed_tokens.shape
        expected = (
            cfg.observed_steps,
            cfg.token_count,
            cfg.token_dim,
        )
        if (observed_steps, token_count, token_dim) != expected:
            raise ValueError(
                "observed token layout does not match the WM3D state contract"
            )
        if tuple(world_times_s.shape) != (
            batch,
            cfg.observed_steps + cfg.future_steps,
        ):
            raise ValueError("world_times_s must cover observed + K future steps")
        if not bool(torch.isfinite(world_times_s).all()) or not bool(
            torch.diff(world_times_s, dim=1).gt(0).all()
        ):
            raise ValueError("world times must be finite and strictly increasing")
        history_shape = (batch, cfg.observed_steps, cfg.history_dim)
        if history_state_tokens.shape != history_shape:
            raise ValueError("history state tokens do not align with visual keyframes")
        if history_action_tokens.shape != history_shape:
            raise ValueError("history action tokens do not align with visual keyframes")
        if language_context.shape[0] != batch or language_context.shape[-1] != cfg.language_dim:
            raise ValueError("language context does not match the state core")

        observed = self.view_fuser(observed_tokens, observed_view_mask)
        observed = observed + self.history_state(history_state_tokens)[:, :, None]
        observed = observed + self.history_action(history_action_tokens)[:, :, None]
        future = self.future_queries.expand(batch, -1, -1, -1)
        state = torch.cat((observed, future), dim=1)
        relative_time = world_times_s - world_times_s[:, cfg.observed_steps - 1 : cfg.observed_steps]
        state = self.state_input_norm(state)
        state = state + self.state_space + self.state_time(relative_time)[:, :, None]
        task = self.task(self._pool_language(language_context, language_mask))
        state = state + task[:, None, None]
        for block in self.state_blocks:
            state = block(state)

        prior = self.state_norm(state)[:, cfg.observed_steps :]
        factual = prior
        if factual_action_tokens is not None:
            if factual_action_mask is None:
                raise ValueError("factual action tokens require an explicit mask")
            expected_action = (batch, cfg.future_steps, cfg.history_dim)
            if factual_action_tokens.shape != expected_action:
                raise ValueError("factual actions must contain exactly K time bins")
            if factual_action_mask.shape != expected_action[:2]:
                raise ValueError("factual action mask must be [B,K]")
            action = self.factual_action(factual_action_tokens)[:, :, None]
            action_mask = factual_action_mask[:, :, None].bool()
            for block in self.dynamics_blocks:
                factual = block(factual, action, action_mask)
            factual = self.state_norm(factual)
        elif factual_action_mask is not None:
            raise ValueError("factual action mask was supplied without action tokens")

        return WM3DStateDynamicsOutput(
            action_free_native_state=prior,
            native_state=factual,
            action_free_tokens=self.token_decoder(prior, views=views),
            factual_tokens=self.token_decoder(factual, views=views),
        )

    def parameter_counts(self) -> dict[str, int]:
        groups = {
            "multiview_fuser": self.view_fuser,
            "state_trunk": self.state_blocks,
            "factual_dynamics": self.dynamics_blocks,
            "token_decoder": self.token_decoder,
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in groups.items()
        }
        counts["total"] = sum(parameter.numel() for parameter in self.parameters())
        counts["other"] = counts["total"] - sum(counts.values())
        return counts

