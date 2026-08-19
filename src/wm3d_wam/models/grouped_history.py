"""Grouped state/action history connector for GAMFuturePredictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from wm3d_wam.data.grouped_history import (
    GroupedActionTimelineBatch,
    GroupedStateHistoryBatch,
)

from .grouped_action_flow import GroupedActionCodec, GroupedActionCodecConfig


@dataclass(frozen=True)
class GroupedHistoryConnectorConfig:
    d_model: int = 1024
    max_groups: int = 8
    max_state_dim: int = 32
    state_semantic_vocab_size: int = 128
    group_vocab_size: int = 256
    embodiment_vocab_size: int = 4096
    time_fourier_dim: int = 128
    time_max_frequency_hz: float = 32.0
    transformer_depth: int = 4
    num_heads: int = 16
    ffn_ratio: float = 4.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "max_groups",
            "max_state_dim",
            "state_semantic_vocab_size",
            "group_vocab_size",
            "embodiment_vocab_size",
            "time_fourier_dim",
            "transformer_depth",
            "num_heads",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.time_fourier_dim % 2:
            raise ValueError("time_fourier_dim must be even")


def _coerce_config(
    value: GroupedHistoryConnectorConfig | Mapping[str, object],
) -> GroupedHistoryConnectorConfig:
    if isinstance(value, GroupedHistoryConnectorConfig):
        return value
    return GroupedHistoryConnectorConfig(**dict(value))


class GroupedStateHistoryCodec(nn.Module):
    """Encode one timestamped grouped state into one semantic token."""

    def __init__(self, config: GroupedHistoryConnectorConfig):
        super().__init__()
        self.config = config
        hidden = config.d_model
        self.value_encoder = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.semantic_embedding = nn.Embedding(
            config.state_semantic_vocab_size, hidden, padding_idx=0
        )
        self.group_embedding = nn.Embedding(
            config.group_vocab_size, hidden, padding_idx=0
        )
        self.embodiment_embedding = nn.Embedding(
            config.embodiment_vocab_size, hidden, padding_idx=0
        )
        self.group_slot_embedding = nn.Embedding(config.max_groups, hidden)
        self.dimension_embedding = nn.Embedding(config.max_state_dim, hidden)
        frequency_count = config.time_fourier_dim // 2
        frequencies = torch.logspace(
            0.0,
            torch.log10(torch.tensor(config.time_max_frequency_hz)).item(),
            frequency_count,
            dtype=torch.float32,
        )
        self.register_buffer("time_frequencies_hz", frequencies, persistent=True)
        self.time_encoder = nn.Sequential(
            nn.Linear(config.time_fourier_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, batch: GroupedStateHistoryBatch) -> torch.Tensor:
        values = batch.values
        if values.ndim != 4:
            raise ValueError("grouped state values must be [B,H,G,D]")
        b, h, g, d = values.shape
        cfg = self.config
        expected = {
            "value_mask": (b, h, g, d),
            "step_mask": (b, h),
            "times_s": (b, h),
            "group_ids": (b, g),
            "group_mask": (b, g),
            "state_semantic_ids": (b, g, d),
            "embodiment_ids": (b,),
        }
        for name, shape in expected.items():
            if tuple(getattr(batch, name).shape) != shape:
                raise ValueError(f"{name} must be {shape}")
        if (g, d) != (cfg.max_groups, cfg.max_state_dim):
            raise ValueError("grouped state capacities do not match the codec")
        if not bool(torch.isfinite(values).all()) or not bool(
            torch.isfinite(batch.times_s).all()
        ):
            raise ValueError("grouped state contains non-finite values")
        valid = (
            batch.value_mask.bool()
            & batch.step_mask[:, :, None, None].bool()
            & batch.group_mask[:, None, :, None].bool()
        )
        if bool((batch.value_mask.bool() & ~valid).any()):
            raise ValueError("state value_mask marks a padded step/group as valid")

        group_slots = torch.arange(g, device=values.device)
        dimension_slots = torch.arange(d, device=values.device)
        metadata = (
            self.semantic_embedding(batch.state_semantic_ids)
            + self.group_embedding(batch.group_ids).unsqueeze(2)
            + self.group_slot_embedding(group_slots)[None, :, None]
            + self.dimension_embedding(dimension_slots)[None, None, :]
        )
        scalar = self.value_encoder(values.unsqueeze(-1)) + metadata[:, None]
        weights = valid.unsqueeze(-1).to(dtype=scalar.dtype)
        denominator = weights.sum(dim=(2, 3)).clamp_min(1.0)
        token = (scalar * weights).sum(dim=(2, 3)) / denominator

        frequency = self.time_frequencies_hz.to(
            device=values.device, dtype=batch.times_s.dtype
        )
        phase = batch.times_s.unsqueeze(-1) * frequency * (2.0 * torch.pi)
        time_token = self.time_encoder(torch.cat([phase.sin(), phase.cos()], dim=-1))
        token = self.norm(
            token
            + time_token
            + self.embodiment_embedding(batch.embodiment_ids).unsqueeze(1)
        )
        return token * batch.step_mask.unsqueeze(-1).to(dtype=token.dtype)


class GroupedHistoryConnector(nn.Module):
    """Condition four visual keyframes on all 16 grouped robot history steps.

    State and source-native action events are encoded independently, then an
    interleaved block-causal transformer lets every selected keyframe summary
    carry the complete earlier robot history.  The output can be passed to the
    GAM predictor through its pre-embedded grouped-history API.
    """

    def __init__(
        self,
        config: GroupedHistoryConnectorConfig | Mapping[str, object],
        *,
        action_codec_config: GroupedActionCodecConfig | Mapping[str, object],
    ) -> None:
        super().__init__()
        self.config = _coerce_config(config)
        self.state_codec = GroupedStateHistoryCodec(self.config)
        self.action_codec = GroupedActionCodec(action_codec_config)
        if self.action_codec.hidden_dim == self.config.d_model:
            self.action_projection = nn.Identity()
        else:
            self.action_projection = nn.Linear(
                self.action_codec.hidden_dim, self.config.d_model
            )
        self.action_score = nn.Linear(self.config.d_model, 1, bias=False)
        self.empty_action_token = nn.Parameter(torch.zeros(self.config.d_model))
        self.state_type = nn.Parameter(torch.zeros(self.config.d_model))
        self.action_type = nn.Parameter(torch.zeros(self.config.d_model))
        nn.init.normal_(self.empty_action_token, std=0.02)
        nn.init.normal_(self.state_type, std=0.02)
        nn.init.normal_(self.action_type, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=int(self.config.d_model * self.config.ffn_ratio),
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer,
            num_layers=self.config.transformer_depth,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.config.d_model)

    def encode_action_steps(
        self, timeline: GroupedActionTimelineBatch
    ) -> torch.Tensor:
        event_tokens = self.action_projection(self.action_codec.encode(timeline.events))
        b, _, d = event_tokens.shape
        h = timeline.num_steps
        if timeline.step_indices.shape != timeline.events.event_mask.shape:
            raise ValueError("action step indices must align with event slots")
        output = self.empty_action_token.view(1, 1, d).expand(b, h, d).clone()
        for step in range(h):
            valid = timeline.events.event_mask.bool() & (
                timeline.step_indices == step
            )
            if not bool(valid.any()):
                continue
            score = self.action_score(event_tokens).squeeze(-1)
            score = score.masked_fill(~valid, -torch.inf)
            weights = torch.softmax(score, dim=1)
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            pooled = torch.einsum("be,bed->bd", weights, event_tokens)
            present = valid.any(dim=1)
            output[present, step] = pooled[present]
        return output * timeline.step_mask.unsqueeze(-1).to(dtype=output.dtype)

    @staticmethod
    def _block_causal_mask(steps: int, device: torch.device) -> torch.Tensor:
        token_steps = torch.arange(steps, device=device).repeat_interleave(2)
        # Transformer bool mask uses True for disallowed attention.
        return token_steps[:, None] < token_steps[None, :]

    def forward(
        self,
        *,
        state_history: GroupedStateHistoryBatch,
        action_history: GroupedActionTimelineBatch,
        keyframe_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state_codec(state_history)
        action = self.encode_action_steps(action_history)
        if state.shape != action.shape:
            raise ValueError("state and action history timelines must align")
        b, h, d = state.shape
        if keyframe_indices.ndim != 1 or keyframe_indices.numel() < 1:
            raise ValueError("keyframe_indices must be a non-empty rank-1 tensor")
        keyframe_indices = keyframe_indices.to(device=state.device, dtype=torch.long)
        if int(keyframe_indices.min()) < 0 or int(keyframe_indices.max()) >= h:
            raise ValueError("keyframe index is outside the history timeline")
        if not bool((torch.diff(keyframe_indices) > 0).all()):
            raise ValueError("keyframe indices must be strictly increasing")

        interleaved = torch.stack(
            [state + self.state_type, action + self.action_type], dim=2
        ).reshape(b, 2 * h, d)
        valid = torch.stack(
            [state_history.step_mask, action_history.step_mask], dim=2
        ).reshape(b, 2 * h)
        hidden = self.temporal(
            interleaved,
            mask=self._block_causal_mask(h, state.device),
            src_key_padding_mask=~valid.bool(),
        ).reshape(b, h, 2, d)
        hidden = self.output_norm(hidden)
        return hidden[:, keyframe_indices, 0], hidden[:, keyframe_indices, 1]
