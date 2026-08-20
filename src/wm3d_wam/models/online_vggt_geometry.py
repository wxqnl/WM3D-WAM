"""Online VGGT split-and-resume around the WM3D state-dynamics core."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm3d_wam.data.grouped_history import (
    GroupedActionTimelineBatch,
    GroupedStateHistoryBatch,
)
from wm3d_wam.vendor.vggt_gam.vggt_encoder import VGGTEncoder

from .grouped_history import GroupedHistoryConnector
from .wm3d_state_dynamics import WM3DStateDynamicsCore


class GeometryConditionMode(str, Enum):
    ACTION_FREE = "action_free"
    FACTUAL = "factual"


@dataclass(frozen=True)
class OnlineGeometryOutput:
    mode: GeometryConditionMode
    observed_shallow_tokens: torch.Tensor
    predicted_future_shallow_tokens: torch.Tensor
    action_free_future_shallow_tokens: torch.Tensor
    action_free_native_state: torch.Tensor
    native_state: torch.Tensor
    deep_visual_tokens: torch.Tensor
    geometry_tokens: torch.Tensor
    geometry_token_mask: torch.Tensor
    geometry_anchor_indices: tuple[int, ...]
    geometry_anchor_valid_mask: torch.Tensor
    student_geometry: dict[str, torch.Tensor]
    target_future_shallow_tokens: Optional[torch.Tensor]
    target_geometry: Optional[dict[str, torch.Tensor]]


class GeometryTokenReducer(nn.Module):
    """Keep VGGT special tokens and spatially pool its dense patch grid."""

    def __init__(
        self,
        *,
        token_dim: int = 1024,
        num_register_tokens: int = 4,
        source_patch_grid: int = 16,
        output_patch_grid: int = 4,
    ) -> None:
        super().__init__()
        if output_patch_grid <= 0 or output_patch_grid > source_patch_grid:
            raise ValueError("invalid geometry output patch grid")
        self.token_dim = int(token_dim)
        self.num_special = 1 + int(num_register_tokens)
        self.source_patch_grid = int(source_patch_grid)
        self.output_patch_grid = int(output_patch_grid)
        self.norm = nn.LayerNorm(token_dim)

    @property
    def tokens_per_view(self) -> int:
        return self.num_special + self.output_patch_grid**2

    def forward(
        self, tokens: torch.Tensor, view_valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 5:
            raise ValueError("deep VGGT tokens must be [B,T,V,N,D]")
        batch, steps, views, token_count, token_dim = tokens.shape
        expected = self.num_special + self.source_patch_grid**2
        if token_count != expected or token_dim != self.token_dim:
            raise ValueError(
                f"unexpected deep token layout N/D={token_count}/{token_dim}, "
                f"expected {expected}/{self.token_dim}"
            )
        valid = view_valid_mask.to(device=tokens.device, dtype=torch.bool)
        if valid.shape != (batch, steps, views):
            raise ValueError("view_valid_mask does not align with deep tokens")
        special = tokens[:, :, :, : self.num_special]
        patches = tokens[:, :, :, self.num_special :].reshape(
            batch * steps * views,
            self.source_patch_grid,
            self.source_patch_grid,
            token_dim,
        )
        patches = patches.permute(0, 3, 1, 2)
        patches = F.adaptive_avg_pool2d(
            patches.float(), (self.output_patch_grid, self.output_patch_grid)
        ).to(dtype=tokens.dtype)
        patches = patches.permute(0, 2, 3, 1).reshape(
            batch, steps, views, self.output_patch_grid**2, token_dim
        )
        reduced = self.norm(torch.cat((special, patches), dim=3))
        reduced = reduced * valid[:, :, :, None, None].to(dtype=reduced.dtype)
        mask = valid[:, :, :, None].expand(
            batch, steps, views, self.tokens_per_view
        )
        return reduced.reshape(batch, -1, token_dim), mask.reshape(batch, -1)


class OnlineVGGTGeometryCore(nn.Module):
    """Encode observations, roll out K=16 WM3D states, then resume VGGT.

    The student path never reads clean future RGB. Frozen shallow VGGT creates
    online feature targets only when the active training program asks for
    them. Deep VGGT runs at sparse geometry anchors while the WM3D state prior
    itself remains dense at 10 Hz.
    """

    def __init__(
        self,
        *,
        encoder: VGGTEncoder,
        state_dynamics: WM3DStateDynamicsCore,
        history_connector: GroupedHistoryConnector,
        observed_keyframe_indices: tuple[int, ...] = (0, 5, 10, 15),
        future_steps: int = 16,
        geometry_anchor_indices: tuple[int, ...] = (3, 7, 11, 15),
        geometry_output_patch_grid: int = 4,
        shallow_scene_chunk_size: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.state_dynamics = state_dynamics
        self.history_connector = history_connector
        self.observed_keyframe_indices = tuple(
            int(value) for value in observed_keyframe_indices
        )
        self.future_steps = int(future_steps)
        self.geometry_anchor_indices = tuple(
            int(value) for value in geometry_anchor_indices
        )
        self.shallow_scene_chunk_size = int(shallow_scene_chunk_size)
        if len(self.observed_keyframe_indices) < 1 or any(
            right <= left
            for left, right in zip(
                self.observed_keyframe_indices,
                self.observed_keyframe_indices[1:],
            )
        ):
            raise ValueError("observed keyframe indices must be strictly increasing")
        if self.future_steps <= 0:
            raise ValueError("future_steps must be positive")
        if self.shallow_scene_chunk_size <= 0:
            raise ValueError("shallow_scene_chunk_size must be positive")
        if not self.geometry_anchor_indices or any(
            index < 0 or index >= self.future_steps
            for index in self.geometry_anchor_indices
        ):
            raise ValueError("geometry anchors must refer to the K-step future")
        if tuple(sorted(set(self.geometry_anchor_indices))) != self.geometry_anchor_indices:
            raise ValueError("geometry anchor indices must be unique and increasing")
        state_config = state_dynamics.config
        expected_token_count = 1 + encoder.num_register_tokens + encoder.num_patches
        if state_config.observed_steps != len(self.observed_keyframe_indices):
            raise ValueError("WM3D observed-step count does not match keyframes")
        if state_config.future_steps != self.future_steps:
            raise ValueError("WM3D K does not match the online geometry horizon")
        if state_config.token_count != expected_token_count:
            raise ValueError("WM3D and VGGT token counts do not match")
        if state_config.token_dim != encoder.embed_dim:
            raise ValueError("WM3D and VGGT token dimensions do not match")
        if state_config.history_dim != history_connector.config.d_model:
            raise ValueError("WM3D and grouped-history widths do not match")
        self.geometry_reducer = GeometryTokenReducer(
            token_dim=encoder.embed_dim,
            num_register_tokens=encoder.num_register_tokens,
            source_patch_grid=encoder.h_patches,
            output_patch_grid=geometry_output_patch_grid,
        )

    @property
    def geometry_anchor_count(self) -> int:
        return len(self.geometry_anchor_indices)

    @staticmethod
    def _validate_images(name: str, images: torch.Tensor) -> tuple[int, int, int]:
        if images.ndim != 6 or images.shape[3] != 3:
            raise ValueError(f"{name} must be [B,T,V,3,H,W]")
        if not torch.is_floating_point(images) or not bool(
            torch.isfinite(images).all()
        ):
            raise ValueError(f"{name} must contain finite floating-point RGB")
        return int(images.shape[0]), int(images.shape[1]), int(images.shape[2])

    def _encode_shallow(self, images: torch.Tensor) -> torch.Tensor:
        batch, steps, views = images.shape[:3]
        scenes = images.reshape(batch * steps, views, *images.shape[3:])
        encoded: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, batch * steps, self.shallow_scene_chunk_size):
                chunk = scenes[start : start + self.shallow_scene_chunk_size]
                result = self.encoder.encode_shallow_visual_slots(
                    chunk, T=1, V=views
                )
                encoded.append(result["visual_tokens"][:, 0])
        visual = torch.cat(encoded, dim=0)
        return visual.reshape(batch, steps, views, *visual.shape[-2:]).detach()

    @staticmethod
    def _action_step_valid(timeline: GroupedActionTimelineBatch) -> torch.Tensor:
        batch = timeline.events.event_mask.shape[0]
        output = torch.zeros(
            (batch, timeline.num_steps),
            dtype=torch.bool,
            device=timeline.step_indices.device,
        )
        valid_event = timeline.events.event_mask.to(
            device=timeline.step_indices.device, dtype=torch.bool
        )
        for step in range(timeline.num_steps):
            output[:, step] = (
                valid_event & timeline.step_indices.eq(step)
            ).any(dim=1)
        return output & timeline.step_mask.bool()

    def forward(
        self,
        *,
        observed_images: torch.Tensor,
        state_history: GroupedStateHistoryBatch,
        action_history: GroupedActionTimelineBatch,
        future_world_times_s: torch.Tensor,
        mode: GeometryConditionMode | str,
        future_action_history: Optional[GroupedActionTimelineBatch] = None,
        future_target_images: Optional[torch.Tensor] = None,
        language_features: Optional[torch.Tensor] = None,
        language_padding_mask: Optional[torch.Tensor] = None,
        observed_view_valid_mask: Optional[torch.Tensor] = None,
        future_view_valid_mask: Optional[torch.Tensor] = None,
        future_world_valid_mask: Optional[torch.Tensor] = None,
        decode_geometry_heads: bool = True,
        compute_target_geometry: bool = False,
        gradient_checkpointing: bool = False,
    ) -> OnlineGeometryOutput:
        mode = GeometryConditionMode(mode)
        if mode is GeometryConditionMode.ACTION_FREE and future_action_history is not None:
            raise ValueError("action-free geometry must not receive future actions")
        if mode is GeometryConditionMode.FACTUAL and future_action_history is None:
            raise ValueError("factual geometry requires future action events")
        if language_features is None:
            raise ValueError("WM3D state dynamics requires language features")

        batch, observed_steps, views = self._validate_images(
            "observed_images", observed_images
        )
        if observed_steps != len(self.observed_keyframe_indices):
            raise ValueError("observed image count must match configured keyframes")
        if observed_images.shape[-2:] != (
            self.encoder.encoder_input_size,
            self.encoder.encoder_input_size,
        ):
            raise ValueError("observed images are not at the VGGT input size")
        if observed_view_valid_mask is None:
            observed_view_valid_mask = torch.ones(
                (batch, observed_steps, views),
                device=observed_images.device,
                dtype=torch.bool,
            )
        else:
            observed_view_valid_mask = observed_view_valid_mask.to(
                device=observed_images.device, dtype=torch.bool
            )
        if observed_view_valid_mask.shape != (batch, observed_steps, views):
            raise ValueError("observed_view_valid_mask shape mismatch")
        if not bool(observed_view_valid_mask.any(dim=2).all()):
            raise ValueError("every observed timestep needs a real camera view")

        if future_view_valid_mask is None:
            future_view_valid_mask = observed_view_valid_mask[:, -1:].expand(
                batch, self.future_steps, views
            )
        else:
            future_view_valid_mask = future_view_valid_mask.to(
                device=observed_images.device, dtype=torch.bool
            )
        if future_view_valid_mask.shape != (batch, self.future_steps, views):
            raise ValueError("future_view_valid_mask must be [B,K,V]")
        if not bool(future_view_valid_mask.any(dim=2).all()):
            raise ValueError("every predicted future step needs a real view slot")
        if future_world_valid_mask is None:
            future_world_valid_mask = future_view_valid_mask
        else:
            future_world_valid_mask = future_world_valid_mask.to(
                device=observed_images.device, dtype=torch.bool
            )
        if future_world_valid_mask.shape != (batch, self.future_steps, views):
            raise ValueError("future_world_valid_mask must be [B,K,V]")
        if bool((future_world_valid_mask & ~future_view_valid_mask).any()):
            raise ValueError("RGB supervision cannot mark a padded view as valid")

        keyframes = torch.tensor(
            self.observed_keyframe_indices,
            device=state_history.values.device,
            dtype=torch.long,
        )
        state_tokens, history_action_tokens = self.history_connector(
            state_history=state_history,
            action_history=action_history,
            keyframe_indices=keyframes,
        )
        factual_tokens = None
        factual_mask = None
        if future_action_history is not None:
            if future_action_history.num_steps != self.future_steps:
                raise ValueError("future action timeline must contain K bins")
            factual_tokens = self.history_connector.encode_action_steps(
                future_action_history
            )
            factual_mask = self._action_step_valid(future_action_history)

        future_world_times_s = future_world_times_s.to(
            device=state_history.times_s.device,
            dtype=state_history.times_s.dtype,
        )
        if future_world_times_s.ndim == 1:
            future_world_times_s = future_world_times_s.unsqueeze(0)
        if future_world_times_s.shape != (batch, self.future_steps):
            raise ValueError("future_world_times_s must be [B,K]")
        observed_times = state_history.times_s.index_select(1, keyframes)
        world_times = torch.cat((observed_times, future_world_times_s), dim=1)

        observed_shallow = self._encode_shallow(observed_images)
        state_output = self.state_dynamics(
            observed_tokens=observed_shallow,
            observed_view_mask=observed_view_valid_mask,
            world_times_s=world_times,
            history_state_tokens=state_tokens,
            history_action_tokens=history_action_tokens,
            language_context=language_features,
            language_mask=language_padding_mask,
            factual_action_tokens=factual_tokens,
            factual_action_mask=factual_mask,
        )
        predicted_shallow = (
            state_output.factual_tokens
            if mode is GeometryConditionMode.FACTUAL
            else state_output.action_free_tokens
        )

        anchor_index = torch.tensor(
            self.geometry_anchor_indices,
            device=predicted_shallow.device,
            dtype=torch.long,
        )
        predicted_anchors = predicted_shallow.index_select(1, anchor_index)
        anchor_view_mask = future_view_valid_mask.index_select(1, anchor_index)
        all_shallow = torch.cat((observed_shallow, predicted_anchors), dim=1)
        all_view_mask = torch.cat(
            (observed_view_valid_mask, anchor_view_mask), dim=1
        )
        student_geometry = self.encoder.propagate_shallow_without_actions_grad(
            all_shallow,
            decode_visuals=decode_geometry_heads,
            dpt_chunk_size=1,
            gradient_checkpointing=gradient_checkpointing,
            return_multi_level=False,
            step_valid_mask=all_view_mask.any(dim=2),
            deep_temporal_causal_mask=True,
        )
        deep_visual = student_geometry["deep_visual_tokens"]
        future_deep = deep_visual[:, -self.geometry_anchor_count :]
        geometry_tokens, geometry_mask = self.geometry_reducer(
            future_deep, anchor_view_mask
        )

        target_shallow = None
        target_geometry = None
        if future_target_images is not None:
            target_batch, target_steps, target_views = self._validate_images(
                "future_target_images", future_target_images
            )
            if (target_batch, target_steps, target_views) != (
                batch,
                self.future_steps,
                views,
            ):
                raise ValueError("future RGB targets must align with [B,K,V]")
            target_shallow = self._encode_shallow(future_target_images).detach()
            if compute_target_geometry:
                target_anchor_mask = future_world_valid_mask.index_select(
                    1, anchor_index
                )
                if not bool(target_anchor_mask.all()):
                    raise ValueError(
                        "every sparse geometry anchor needs a recorded RGB target"
                    )
                target_anchors = target_shallow.index_select(1, anchor_index)
                with torch.no_grad():
                    target_geometry = self.encoder.propagate_shallow_without_actions(
                        target_anchors,
                        decode_visuals=decode_geometry_heads,
                        dpt_chunk_size=1,
                    )
                    target_geometry = {
                        key: value.detach() if isinstance(value, torch.Tensor) else value
                        for key, value in target_geometry.items()
                    }
        elif compute_target_geometry:
            raise ValueError("target geometry requires future RGB targets")

        return OnlineGeometryOutput(
            mode=mode,
            observed_shallow_tokens=observed_shallow,
            predicted_future_shallow_tokens=predicted_shallow,
            action_free_future_shallow_tokens=state_output.action_free_tokens,
            action_free_native_state=state_output.action_free_native_state,
            native_state=state_output.native_state,
            deep_visual_tokens=deep_visual,
            geometry_tokens=geometry_tokens,
            geometry_token_mask=geometry_mask,
            geometry_anchor_indices=self.geometry_anchor_indices,
            geometry_anchor_valid_mask=anchor_view_mask,
            student_geometry=student_geometry,
            target_future_shallow_tokens=target_shallow,
            target_geometry=target_geometry,
        )
