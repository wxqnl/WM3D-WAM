"""Source-native grouped action codec on top of FastWAM's ActionDiT blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from wm3d_wam.data.action_events import GroupedActionBatch
from wm3d_wam.vendor.fastwam.wan22.action_dit import ActionDiT
from wm3d_wam.vendor.fastwam.wan22.helpers.gradient import (
    gradient_checkpoint_forward,
)
from wm3d_wam.vendor.fastwam.wan22.wan_video_dit import sinusoidal_embedding_1d


@dataclass(frozen=True)
class GroupedActionCodecConfig:
    """Static capacities for the padded grouped robot ABI.

    Capacities do not define an embodiment. Valid groups, dimensions, and
    events are always selected by the masks carried in ``GroupedActionBatch``.
    """

    hidden_dim: int = 1024
    max_groups: int = 8
    max_action_dim: int = 16
    semantic_vocab_size: int = 128
    group_vocab_size: int = 256
    composition_vocab_size: int = 32
    embodiment_vocab_size: int = 4096
    time_fourier_dim: int = 128
    time_max_frequency_hz: float = 64.0
    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        integer_fields = (
            "hidden_dim",
            "max_groups",
            "max_action_dim",
            "semantic_vocab_size",
            "group_vocab_size",
            "composition_vocab_size",
            "embodiment_vocab_size",
            "time_fourier_dim",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.time_fourier_dim % 4:
            raise ValueError("time_fourier_dim must be divisible by four")
        if self.time_max_frequency_hz <= 0:
            raise ValueError("time_max_frequency_hz must be positive")


def _coerce_codec_config(
    value: GroupedActionCodecConfig | Mapping[str, Any],
) -> GroupedActionCodecConfig:
    if isinstance(value, GroupedActionCodecConfig):
        return value
    return GroupedActionCodecConfig(**dict(value))


class GroupedActionCodec(nn.Module):
    """Encode/decode scalar robot fields through one token per real event.

    The scalar fields are never flattened into a source-specific action vector.
    Semantic, physical group, composition rule, embodiment, and real timestamp
    remain explicit so differently shaped embodiments can share the expert.
    """

    def __init__(self, config: GroupedActionCodecConfig | Mapping[str, Any]):
        super().__init__()
        self.config = _coerce_codec_config(config)
        hidden = self.config.hidden_dim

        self.value_encoder = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        binding_rank = min(32, hidden)
        self.field_binding_down = nn.Linear(hidden, binding_rank, bias=False)
        self.field_binding_up = nn.Linear(binding_rank, hidden, bias=False)
        self.semantic_embedding = nn.Embedding(
            self.config.semantic_vocab_size, hidden, padding_idx=0
        )
        self.group_embedding = nn.Embedding(
            self.config.group_vocab_size, hidden, padding_idx=0
        )
        self.composition_embedding = nn.Embedding(
            self.config.composition_vocab_size, hidden, padding_idx=0
        )
        self.embodiment_embedding = nn.Embedding(
            self.config.embodiment_vocab_size, hidden, padding_idx=0
        )
        self.group_slot_embedding = nn.Embedding(self.config.max_groups, hidden)
        self.dimension_embedding = nn.Embedding(self.config.max_action_dim, hidden)

        frequency_count = self.config.time_fourier_dim // 4
        frequencies = torch.logspace(
            0.0,
            torch.log10(torch.tensor(self.config.time_max_frequency_hz)).item(),
            frequency_count,
            dtype=torch.float32,
        )
        self.register_buffer("time_frequencies_hz", frequencies, persistent=True)
        self.time_encoder = nn.Sequential(
            nn.Linear(self.config.time_fourier_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.event_norm = nn.LayerNorm(hidden, eps=self.config.eps)
        self.scalar_decoder = nn.Sequential(
            nn.LayerNorm(hidden, eps=self.config.eps),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    @property
    def hidden_dim(self) -> int:
        return self.config.hidden_dim

    def _validate_batch(self, batch: GroupedActionBatch) -> tuple[int, int, int, int]:
        if batch.values.ndim != 4:
            raise ValueError(
                "GroupedActionBatch.values must be [B,E,G,D], got "
                f"{tuple(batch.values.shape)}"
            )
        batch_size, events, groups, dimensions = batch.values.shape
        if groups != self.config.max_groups or dimensions != self.config.max_action_dim:
            raise ValueError(
                "Grouped action padding capacities do not match the codec: "
                f"batch G/D={groups}/{dimensions}, codec "
                f"G/D={self.config.max_groups}/{self.config.max_action_dim}"
            )
        expected_shapes = {
            "value_mask": (batch_size, events, groups, dimensions),
            "event_mask": (batch_size, events),
            "times_s": (batch_size, events),
            "event_dt_s": (batch_size, events),
            "group_ids": (batch_size, groups),
            "group_mask": (batch_size, groups),
            "action_semantic_ids": (batch_size, groups, dimensions),
            "composition_operator_ids": (batch_size, groups, dimensions),
            "embodiment_ids": (batch_size,),
        }
        for name, expected in expected_shapes.items():
            actual = tuple(getattr(batch, name).shape)
            if actual != expected:
                raise ValueError(f"{name} must be {expected}, got {actual}")
        if not torch.is_floating_point(batch.values):
            raise ValueError("GroupedActionBatch.values must be floating point")
        if not bool(torch.isfinite(batch.values).all()):
            raise ValueError("GroupedActionBatch.values contains non-finite values")
        if not bool(torch.isfinite(batch.times_s).all()) or not bool(
            torch.isfinite(batch.event_dt_s).all()
        ):
            raise ValueError("action event times contain non-finite values")
        self._validate_ids(
            batch.action_semantic_ids,
            upper=self.config.semantic_vocab_size,
            name="action_semantic_ids",
        )
        self._validate_ids(
            batch.group_ids,
            upper=self.config.group_vocab_size,
            name="group_ids",
        )
        self._validate_ids(
            batch.composition_operator_ids,
            upper=self.config.composition_vocab_size,
            name="composition_operator_ids",
        )
        self._validate_ids(
            batch.embodiment_ids,
            upper=self.config.embodiment_vocab_size,
            name="embodiment_ids",
        )
        event_mask = batch.event_mask.to(dtype=torch.bool)
        group_mask = batch.group_mask.to(dtype=torch.bool)
        valid_scalar = (
            event_mask[:, :, None, None]
            & group_mask[:, None, :, None]
            & batch.value_mask.to(dtype=torch.bool)
        )
        if bool((batch.value_mask.to(dtype=torch.bool) & ~valid_scalar).any()):
            raise ValueError("value_mask marks a padded event or group as valid")
        if bool((event_mask & ~valid_scalar.flatten(2).any(dim=2)).any()):
            raise ValueError("every valid action event must supervise at least one scalar")
        return batch_size, events, groups, dimensions

    @staticmethod
    def _validate_ids(value: torch.Tensor, *, upper: int, name: str) -> None:
        if value.dtype == torch.bool or torch.is_floating_point(value):
            raise ValueError(f"{name} must use an integer dtype")
        if value.numel() and (
            int(value.min().item()) < 0 or int(value.max().item()) >= upper
        ):
            raise ValueError(f"{name} must be in [0, {upper})")

    def _time_features(self, times_s: torch.Tensor, event_dt_s: torch.Tensor) -> torch.Tensor:
        frequencies = self.time_frequencies_hz.to(
            device=times_s.device, dtype=times_s.dtype
        )
        angular = frequencies * (2.0 * torch.pi)
        absolute_phase = times_s.unsqueeze(-1) * angular
        delta_phase = event_dt_s.unsqueeze(-1) * angular
        return torch.cat(
            (
                absolute_phase.sin(),
                absolute_phase.cos(),
                delta_phase.sin(),
                delta_phase.cos(),
            ),
            dim=-1,
        )

    def _field_metadata(self, batch: GroupedActionBatch) -> torch.Tensor:
        _, _, groups, dimensions = batch.values.shape
        device = batch.values.device
        group_slots = torch.arange(groups, device=device)
        dimension_slots = torch.arange(dimensions, device=device)
        return (
            self.semantic_embedding(batch.action_semantic_ids)
            + self.group_embedding(batch.group_ids).unsqueeze(2)
            + self.composition_embedding(batch.composition_operator_ids)
            + self.group_slot_embedding(group_slots)[None, :, None, :]
            + self.dimension_embedding(dimension_slots)[None, None, :, :]
        )

    def encode(self, batch: GroupedActionBatch) -> torch.Tensor:
        """Return one hidden token for each padded event slot, [B,E,H]."""

        self._validate_batch(batch)
        scalar_mask = (
            batch.value_mask.to(dtype=torch.bool)
            & batch.event_mask[:, :, None, None].to(dtype=torch.bool)
            & batch.group_mask[:, None, :, None].to(dtype=torch.bool)
        )
        field_metadata = self._field_metadata(batch)
        value_features = self.value_encoder(batch.values.unsqueeze(-1))
        scalar_tokens = value_features + field_metadata.unsqueeze(1)
        scalar_weights = scalar_mask.unsqueeze(-1).to(dtype=scalar_tokens.dtype)
        denominator = scalar_weights.sum(dim=(2, 3)).clamp_min(1.0)
        event_tokens = (scalar_tokens * scalar_weights).sum(dim=(2, 3)) / denominator

        # The additive pooled token above is invariant to swapping values
        # between fields.  Bind value and field in rank 32, pool that compact
        # interaction, then lift it once per event.  This keeps the semantic
        # association without a hidden-size interaction at every scalar.
        field_basis = torch.tanh(self.field_binding_down(field_metadata)).unsqueeze(1)
        compact_binding = field_basis * torch.tanh(batch.values.unsqueeze(-1))
        compact_binding = (
            (compact_binding * scalar_weights).sum(dim=(2, 3)) / denominator
        )
        event_tokens = event_tokens + self.field_binding_up(compact_binding)

        time_tokens = self.time_encoder(
            self._time_features(batch.times_s, batch.event_dt_s)
        )
        embodiment_tokens = self.embodiment_embedding(batch.embodiment_ids).unsqueeze(1)
        event_tokens = self.event_norm(event_tokens + time_tokens + embodiment_tokens)
        return event_tokens * batch.event_mask.unsqueeze(-1).to(event_tokens.dtype)

    def decode(self, event_tokens: torch.Tensor, batch: GroupedActionBatch) -> torch.Tensor:
        """Decode event tokens into the original grouped scalar tensor."""

        batch_size, events, _, _ = self._validate_batch(batch)
        if event_tokens.shape != (batch_size, events, self.hidden_dim):
            raise ValueError(
                "event_tokens must be "
                f"{(batch_size, events, self.hidden_dim)}, got {tuple(event_tokens.shape)}"
            )
        field_metadata = self._field_metadata(batch).unsqueeze(1)
        time_tokens = self.time_encoder(
            self._time_features(batch.times_s, batch.event_dt_s)
        )
        embodiment_tokens = self.embodiment_embedding(batch.embodiment_ids).unsqueeze(1)
        query = (
            event_tokens[:, :, None, None, :]
            + field_metadata
            + time_tokens[:, :, None, None, :]
            + embodiment_tokens[:, :, None, None, :]
        )
        decoded = self.scalar_decoder(query).squeeze(-1)
        scalar_mask = (
            batch.value_mask.to(dtype=torch.bool)
            & batch.event_mask[:, :, None, None].to(dtype=torch.bool)
            & batch.group_mask[:, None, :, None].to(dtype=torch.bool)
        )
        return decoded * scalar_mask.to(decoded.dtype)

    def extra_repr(self) -> str:
        values = asdict(self.config)
        return ", ".join(f"{key}={value}" for key, value in values.items())


class GroupedActionFlowExpert(ActionDiT):
    """FastWAM ActionDiT backbone with a grouped, timestamp-aware I/O codec.

    Its transformer blocks are unchanged from FastWAM. In joint execution the
    expert is passed to ``MoT`` together with the Wan Video Expert, so action
    tokens participate in the same layer-wise mixed attention while retaining
    their own hidden size, output projection, and FFN.
    """

    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.", "codec.")

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        codec_config: GroupedActionCodecConfig | Mapping[str, Any],
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            action_dim=1,
            ffn_dim=ffn_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            num_layers=num_layers,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        codec = _coerce_codec_config(codec_config)
        if codec.hidden_dim != hidden_dim:
            raise ValueError(
                f"codec hidden_dim={codec.hidden_dim} must match expert hidden_dim={hidden_dim}"
            )
        self.codec = GroupedActionCodec(codec)
        # Remove the source-specific flat-vector I/O from upstream ActionDiT.
        self.action_encoder = nn.Identity()
        self.head = nn.Identity()
        self.action_dim = None

    def pre_dit(
        self,
        action_batch: GroupedActionBatch,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if timestep.ndim != 1:
            raise ValueError(
                f"timestep must be [B] or [1], got {tuple(timestep.shape)}"
            )
        if context.ndim != 3:
            raise ValueError(f"context must be [B,L,D], got {tuple(context.shape)}")

        tokens = self.codec.encode(action_batch)
        batch_size, seq_len, _ = tokens.shape
        if context.shape[0] != batch_size:
            raise ValueError("action and text context batch sizes do not match")
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError("timestep length must be one or the action batch size")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("training requires one action timestep per sample")
            timestep = timestep.expand(batch_size)
        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        elif context_mask.shape != (batch_size, context.shape[1]):
            raise ValueError(
                "context_mask must match the first two context dimensions, got "
                f"{tuple(context_mask.shape)}"
            )
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"action event length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}"
            )

        time_embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep)
        )
        time_modulation = self.time_projection(time_embedding).unflatten(
            1, (6, self.hidden_dim)
        )
        context_embedding = self.text_embedding(context)
        context_attention_mask = context_mask.to(dtype=torch.bool).unsqueeze(1).expand(
            -1, seq_len, -1
        )
        frequencies = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)
        return {
            "tokens": tokens,
            "freqs": frequencies,
            "t": time_embedding,
            "t_mod": time_modulation,
            "context": context_embedding,
            "context_mask": context_attention_mask,
            "action_batch": action_batch,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.codec.decode(tokens, pre_state["action_batch"])

    @staticmethod
    def _standalone_attention_mask(event_mask: torch.Tensor) -> torch.Tensor:
        valid = event_mask.to(dtype=torch.bool)
        mask = valid.unsqueeze(1) & valid.unsqueeze(2)
        invalid = ~valid
        if bool(invalid.any()):
            batch_indices, token_indices = torch.nonzero(invalid, as_tuple=True)
            mask[batch_indices, token_indices, token_indices] = True
        return mask.unsqueeze(1)

    def forward(
        self,
        action_batch: GroupedActionBatch,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_batch=action_batch,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        tokens = pre_state["tokens"]
        self_attention_mask = self._standalone_attention_mask(
            action_batch.event_mask
        )
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    tokens,
                    pre_state["context"],
                    pre_state["t_mod"],
                    pre_state["freqs"],
                    context_mask=pre_state["context_mask"],
                    self_attn_mask=self_attention_mask,
                )
            else:
                tokens = block(
                    tokens,
                    pre_state["context"],
                    pre_state["t_mod"],
                    pre_state["freqs"],
                    context_mask=pre_state["context_mask"],
                    self_attn_mask=self_attention_mask,
                )
        return self.post_dit(tokens, pre_state)


def load_grouped_action_backbone(
    expert: GroupedActionFlowExpert,
    path: str | Path,
) -> None:
    """Load the locally preprocessed Wan2.2 backbone into a grouped expert.

    Grouped codec parameters remain independently initialized.  Every shared
    transformer/text/time key must be present with the exact target shape.
    """

    path = Path(path).expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"ActionDiT backbone must be a regular local file: {path}")
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("ActionDiT backbone payload must be a dict")
    backbone = payload.get("backbone_state_dict")
    meta = payload.get("meta")
    if not isinstance(backbone, dict) or not isinstance(meta, dict):
        raise ValueError("ActionDiT backbone payload is missing state/meta")
    expected_meta = {
        "hidden_dim": int(expert.hidden_dim),
        "ffn_dim": int(expert.ffn_dim),
        "num_layers": int(len(expert.blocks)),
        "num_heads": int(expert.num_heads),
        "attn_head_dim": int(expert.attn_head_dim),
        "text_dim": int(expert.text_dim),
        "freq_dim": int(expert.freq_dim),
    }
    for name, expected in expected_meta.items():
        if name not in meta or int(meta[name]) != expected:
            raise ValueError(
                f"ActionDiT backbone meta.{name}={meta.get(name)!r}, expected {expected}"
            )
    state = expert.state_dict()
    expected_keys = expert.backbone_key_set(state.keys())
    provided_keys = set(backbone)
    if provided_keys != expected_keys:
        raise ValueError(
            "ActionDiT backbone key mismatch: "
            f"missing={sorted(expected_keys-provided_keys)[:8]}, "
            f"unexpected={sorted(provided_keys-expected_keys)[:8]}"
        )
    merged = dict(state)
    for key in expected_keys:
        value = backbone[key]
        if not isinstance(value, torch.Tensor) or value.shape != state[key].shape:
            raise ValueError(f"ActionDiT backbone tensor {key!r} has the wrong shape")
        merged[key] = value.to(device=state[key].device, dtype=state[key].dtype)
    expert.load_state_dict(merged, strict=True)


def grouped_action_flow_loss(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    batch: GroupedActionBatch,
) -> torch.Tensor:
    """Per-sample flow loss normalized by each sample's valid scalar count."""

    if prediction.shape != batch.values.shape or target_velocity.shape != batch.values.shape:
        raise ValueError("prediction, target_velocity, and grouped values must share shape")
    valid = (
        batch.value_mask.to(dtype=torch.bool)
        & batch.event_mask[:, :, None, None].to(dtype=torch.bool)
        & batch.group_mask[:, None, :, None].to(dtype=torch.bool)
    )
    counts = valid.flatten(1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("every sample needs at least one valid action scalar")
    squared_error = (prediction.float() - target_velocity.float()).square()
    per_sample = (squared_error * valid).flatten(1).sum(dim=1) / counts
    return per_sample.mean()
