"""Wan2.2 and grouped ActionDiT execution through layer-wise MoT attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast

import torch
import torch.nn as nn

from wm3d_wam.data.action_events import GroupedActionBatch
from wm3d_wam.vendor.fastwam.wan22.mot import MoT
from wm3d_wam.vendor.fastwam.wan22.wan_video_dit import WanVideoDiT

from .grouped_action_flow import GroupedActionFlowExpert
from .geometry_adapters import SparseGeometryKVAdapters
from .interaction_masks import InteractionProgram, build_mot_attention_mask


@dataclass(frozen=True)
class ObservedVideoKVCache:
    """One-call, per-layer Wan K/V used during iterative action denoising."""

    layers: list[dict[str, torch.Tensor]]
    token_mask: torch.Tensor
    tokens_per_frame: int
    geometry_kv: Optional[dict[int, dict[str, torch.Tensor]]] = None
    geometry_token_mask: Optional[torch.Tensor] = None

    @property
    def sequence_length(self) -> int:
        return int(self.token_mask.shape[1])


@dataclass(frozen=True)
class WanActionOutput:
    video_velocity: Optional[torch.Tensor]
    action_velocity: Optional[torch.Tensor]
    video_pre_state: dict[str, Any]
    action_pre_state: dict[str, Any]


class WanActionMoT(nn.Module):
    """Composition boundary between FastWAM's Wan and grouped action expert."""

    def __init__(
        self,
        *,
        video_expert: WanVideoDiT,
        action_expert: GroupedActionFlowExpert,
        geometry_adapters: Optional[SparseGeometryKVAdapters] = None,
        mot_checkpoint_mixed_attn: bool = True,
    ) -> None:
        super().__init__()
        self.geometry_adapters = geometry_adapters
        self.mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        self._validate_expert_layout()

    @property
    def video_expert(self) -> WanVideoDiT:
        """Return the sole registered video-expert instance.

        ``MoT.mixtures`` owns both experts. Registering aliases on this module
        creates duplicate state-dict paths for the same parameters, which is
        unsafe for FSDP/DCP collective state-dict construction.
        """

        return cast(WanVideoDiT, self.mot.mixtures["video"])

    @property
    def action_expert(self) -> GroupedActionFlowExpert:
        """Return the sole registered grouped-action expert instance."""

        return cast(GroupedActionFlowExpert, self.mot.mixtures["action"])

    def _validate_expert_layout(self) -> None:
        if len(self.video_expert.blocks) != len(self.action_expert.blocks):
            raise ValueError("video and action experts must have the same block count")
        if self.video_expert.num_heads != self.action_expert.num_heads:
            raise ValueError("video and action experts must have the same head count")
        if self.video_expert.attn_head_dim != self.action_expert.attn_head_dim:
            raise ValueError("video and action experts must have the same head dimension")
        if self.geometry_adapters is not None:
            if self.geometry_adapters.num_heads != self.video_expert.num_heads:
                raise ValueError("geometry adapter head count must match MoT experts")
            if (
                self.geometry_adapters.attention_head_dim
                != self.video_expert.attn_head_dim
            ):
                raise ValueError("geometry adapter head dimension must match MoT experts")
            if max(self.geometry_adapters.fusion_layers) >= len(self.video_expert.blocks):
                raise ValueError("geometry fusion layer exceeds the MoT block count")

    @staticmethod
    def _video_token_mask(
        tokens: torch.Tensor, value: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = tokens.shape
        if value is None:
            return torch.ones(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=tokens.device,
            )
        result = value.to(device=tokens.device, dtype=torch.bool)
        if result.shape != (batch_size, sequence_length):
            raise ValueError(
                "video_token_mask must be "
                f"{(batch_size, sequence_length)}, got {tuple(result.shape)}"
            )
        return result

    def _prepare_states(
        self,
        *,
        video_latents: torch.Tensor,
        video_timestep: torch.Tensor,
        action_batch: GroupedActionBatch,
        action_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        video_state = self.video_expert.pre_dit(
            x=video_latents,
            timestep=video_timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        action_state = self.action_expert.pre_dit(
            action_batch=action_batch,
            timestep=action_timestep,
            context=context,
            context_mask=context_mask,
        )
        return video_state, action_state

    def _prepare_geometry_kv(
        self,
        *,
        geometry_tokens: Optional[torch.Tensor],
        geometry_token_mask: Optional[torch.Tensor],
        query_token_mask: torch.Tensor,
    ) -> Optional[dict[int, dict[str, torch.Tensor]]]:
        values = (geometry_tokens, geometry_token_mask)
        if self.geometry_adapters is None:
            if any(value is not None for value in values):
                raise ValueError("geometry inputs require configured geometry_adapters")
            return None
        if any(value is None for value in values):
            raise ValueError("geometry_tokens and geometry_token_mask are both required")
        return self.geometry_adapters(
            geometry_tokens=geometry_tokens,
            geometry_token_mask=geometry_token_mask,
            query_token_mask=query_token_mask,
        )

    @staticmethod
    def _retarget_geometry_kv(
        geometry_kv: Optional[dict[int, dict[str, torch.Tensor]]],
        *,
        geometry_token_mask: Optional[torch.Tensor],
        query_token_mask: torch.Tensor,
    ) -> Optional[dict[int, dict[str, torch.Tensor]]]:
        if geometry_kv is None:
            return None
        if geometry_token_mask is None:
            raise ValueError("cached geometry K/V is missing its token mask")
        attention_mask = (
            query_token_mask.to(dtype=torch.bool).unsqueeze(2)
            & geometry_token_mask.to(
                device=query_token_mask.device, dtype=torch.bool
            ).unsqueeze(1)
        ).unsqueeze(1)
        return {
            layer: {
                "k": payload["k"],
                "v": payload["v"],
                "attention_mask": attention_mask,
            }
            for layer, payload in geometry_kv.items()
        }

    def _prefill_prepared_video(
        self,
        *,
        state: dict[str, Any],
        token_mask: torch.Tensor,
        geometry_tokens: Optional[torch.Tensor],
        geometry_token_mask: Optional[torch.Tensor],
    ) -> ObservedVideoKVCache:
        shared_self_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(state["tokens"].shape[1]),
            video_tokens_per_frame=int(state["meta"]["tokens_per_frame"]),
            device=state["tokens"].device,
        )
        batch_self_mask = (
            shared_self_mask.unsqueeze(0)
            & token_mask.unsqueeze(1)
            & token_mask.unsqueeze(2)
        ).unsqueeze(1)
        geometry_kv = self._prepare_geometry_kv(
            geometry_tokens=geometry_tokens,
            geometry_token_mask=geometry_token_mask,
            query_token_mask=token_mask,
        )
        layers = self.mot.prefill_video_cache(
            video_tokens=state["tokens"],
            video_freqs=state["freqs"],
            video_t_mod=state["t_mod"],
            video_context_payload={
                "context": state["context"],
                "mask": state["context_mask"],
            },
            video_attention_mask=batch_self_mask,
            extra_kv_all=geometry_kv,
        )
        return ObservedVideoKVCache(
            layers=layers,
            token_mask=token_mask,
            tokens_per_frame=int(state["meta"]["tokens_per_frame"]),
            geometry_kv=geometry_kv,
            geometry_token_mask=geometry_token_mask,
        )

    def _action_velocity_from_prepared_cache(
        self,
        *,
        state: dict[str, Any],
        action_batch: GroupedActionBatch,
        cache: ObservedVideoKVCache,
    ) -> torch.Tensor:
        attention_mask = build_mot_attention_mask(
            program=InteractionProgram.ACTION_ONLY,
            video_token_mask=cache.token_mask,
            action_event_mask=action_batch.event_mask,
            video_tokens_per_frame=cache.tokens_per_frame,
        )
        geometry_kv = self._retarget_geometry_kv(
            cache.geometry_kv,
            geometry_token_mask=cache.geometry_token_mask,
            query_token_mask=action_batch.event_mask,
        )
        tokens = self.mot.forward_action_with_video_cache(
            action_tokens=state["tokens"],
            action_freqs=state["freqs"],
            action_t_mod=state["t_mod"],
            action_context_payload={
                "context": state["context"],
                "mask": state["context_mask"],
            },
            video_kv_cache=cache.layers,
            attention_mask=attention_mask,
            video_seq_len=cache.sequence_length,
            extra_kv_all=geometry_kv,
        )
        return self.action_expert.post_dit(tokens, state)

    def forward(
        self,
        *,
        program: InteractionProgram | str,
        video_latents: torch.Tensor,
        video_timestep: torch.Tensor,
        action_batch: GroupedActionBatch,
        action_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        video_token_mask: Optional[torch.Tensor] = None,
        video_to_action_visibility: Optional[torch.Tensor] = None,
        geometry_tokens: Optional[torch.Tensor] = None,
        geometry_token_mask: Optional[torch.Tensor] = None,
    ) -> WanActionOutput:
        """Run a deployment-related interaction program through the real MoT.

        ``action_batch.values`` carries noisy future actions for action/joint
        flow, or the clean candidate action for forward-world flow. Clean
        targets are intentionally absent from this API.
        """

        mode = InteractionProgram(program)
        video_state, action_state = self._prepare_states(
            video_latents=video_latents,
            video_timestep=video_timestep,
            action_batch=action_batch,
            action_timestep=action_timestep,
            context=context,
            context_mask=context_mask,
        )
        if mode is InteractionProgram.ACTION_ONLY and int(
            video_state["meta"]["grid_size"][0]
        ) != 1:
            raise ValueError(
                "action_only accepts exactly one observed Wan latent frame; "
                "future video must not enter the policy path"
            )
        if mode is InteractionProgram.ACTION_ONLY and bool(
            (video_timestep != 0).any()
        ):
            raise ValueError("action_only observed-video timestep must be zero")

        video_mask = self._video_token_mask(
            video_state["tokens"], video_token_mask
        )
        if mode is InteractionProgram.ACTION_ONLY:
            cache = self._prefill_prepared_video(
                state=video_state,
                token_mask=video_mask,
                geometry_tokens=geometry_tokens,
                geometry_token_mask=geometry_token_mask,
            )
            action_velocity = self._action_velocity_from_prepared_cache(
                state=action_state,
                action_batch=action_batch,
                cache=cache,
            )
            return WanActionOutput(
                video_velocity=None,
                action_velocity=action_velocity,
                video_pre_state=video_state,
                action_pre_state=action_state,
            )
        video_self = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(video_state["tokens"].shape[1]),
            video_tokens_per_frame=int(video_state["meta"]["tokens_per_frame"]),
            device=video_state["tokens"].device,
        )
        attention_mask = build_mot_attention_mask(
            program=mode,
            video_token_mask=video_mask,
            action_event_mask=action_batch.event_mask,
            video_tokens_per_frame=int(video_state["meta"]["tokens_per_frame"]),
            video_self_mask=video_self,
            video_to_action_visibility=video_to_action_visibility,
        )
        joint_query_mask = torch.cat(
            (
                video_mask,
                action_batch.event_mask.to(
                    device=video_mask.device, dtype=torch.bool
                ),
            ),
            dim=1,
        )
        geometry_kv = self._prepare_geometry_kv(
            geometry_tokens=geometry_tokens,
            geometry_token_mask=geometry_token_mask,
            query_token_mask=joint_query_mask,
        )
        outputs = self.mot(
            embeds_all={
                "video": video_state["tokens"],
                "action": action_state["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_state["freqs"],
                "action": action_state["freqs"],
            },
            context_all={
                "video": {
                    "context": video_state["context"],
                    "mask": video_state["context_mask"],
                },
                "action": {
                    "context": action_state["context"],
                    "mask": action_state["context_mask"],
                },
            },
            t_mod_all={
                "video": video_state["t_mod"],
                "action": action_state["t_mod"],
            },
            extra_kv_all=geometry_kv,
        )
        video_velocity = None
        action_velocity = None
        if mode is not InteractionProgram.ACTION_ONLY:
            video_velocity = self.video_expert.post_dit(outputs["video"], video_state)
        if mode is not InteractionProgram.FORWARD_WORLD:
            action_velocity = self.action_expert.post_dit(
                outputs["action"], action_state
            )
        return WanActionOutput(
            video_velocity=video_velocity,
            action_velocity=action_velocity,
            video_pre_state=video_state,
            action_pre_state=action_state,
        )

    def prefill_observed_video(
        self,
        *,
        observed_video_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        video_token_mask: Optional[torch.Tensor] = None,
        geometry_tokens: Optional[torch.Tensor] = None,
        geometry_token_mask: Optional[torch.Tensor] = None,
    ) -> ObservedVideoKVCache:
        """Prefill one observed latent frame once for iterative action flow."""

        if observed_video_latents.ndim != 5 or observed_video_latents.shape[2] != 1:
            raise ValueError("observed_video_latents must be [B,C,1,H,W]")
        timestep = torch.zeros(
            (observed_video_latents.shape[0],),
            device=observed_video_latents.device,
            dtype=observed_video_latents.dtype,
        )
        state = self.video_expert.pre_dit(
            x=observed_video_latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        token_mask = self._video_token_mask(state["tokens"], video_token_mask)
        return self._prefill_prepared_video(
            state=state,
            token_mask=token_mask,
            geometry_tokens=geometry_tokens,
            geometry_token_mask=geometry_token_mask,
        )

    def action_velocity_from_cache(
        self,
        *,
        action_batch: GroupedActionBatch,
        action_timestep: torch.Tensor,
        context: torch.Tensor,
        cache: ObservedVideoKVCache,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one action denoising step against an observed-video K/V cache."""

        state = self.action_expert.pre_dit(
            action_batch=action_batch,
            timestep=action_timestep,
            context=context,
            context_mask=context_mask,
        )
        return self._action_velocity_from_prepared_cache(
            state=state,
            action_batch=action_batch,
            cache=cache,
        )
