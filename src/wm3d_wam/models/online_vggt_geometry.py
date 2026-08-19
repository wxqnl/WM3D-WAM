"""Online VGGT split-and-resume geometry core for WM3D-WAM."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm3d_wam.data.action_events import GroupedActionBatch
from wm3d_wam.data.grouped_history import (
    GroupedActionTimelineBatch,
    GroupedStateHistoryBatch,
)
from wm3d_wam.vendor.vggt_gam.future_predictor import GAMFuturePredictor
from wm3d_wam.vendor.vggt_gam.vggt_encoder import VGGTEncoder

from .grouped_history import GroupedHistoryConnector
from .geometry_action_heads import (
    GroupedAuxiliaryActionOutput,
    GroupedGeometryActionHeads,
)


class GeometryConditionMode(str, Enum):
    POLICY = "policy"
    FACTUAL = "factual"


@dataclass(frozen=True)
class OnlineGeometryOutput:
    mode: GeometryConditionMode
    observed_shallow_tokens: torch.Tensor
    predicted_future_shallow_tokens: torch.Tensor
    predicted_action_seed_tokens: torch.Tensor
    predicted_future_proprio_tokens: torch.Tensor
    deep_visual_tokens: torch.Tensor
    geometry_tokens: torch.Tensor
    geometry_token_mask: torch.Tensor
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
        b, t, v, n, d = tokens.shape
        expected = self.num_special + self.source_patch_grid**2
        if n != expected or d != self.token_dim:
            raise ValueError(
                f"unexpected deep token layout N/D={n}/{d}, expected {expected}/{self.token_dim}"
            )
        valid = view_valid_mask.to(device=tokens.device, dtype=torch.bool)
        if valid.shape != (b, t, v):
            raise ValueError("view_valid_mask does not align with deep tokens")
        special = tokens[:, :, :, : self.num_special]
        patches = tokens[:, :, :, self.num_special :].reshape(
            b * t * v,
            self.source_patch_grid,
            self.source_patch_grid,
            d,
        )
        patches = patches.permute(0, 3, 1, 2)
        patches = F.adaptive_avg_pool2d(
            patches.float(), (self.output_patch_grid, self.output_patch_grid)
        ).to(dtype=tokens.dtype)
        patches = patches.permute(0, 2, 3, 1).reshape(
            b, t, v, self.output_patch_grid**2, d
        )
        reduced = self.norm(torch.cat([special, patches], dim=3))
        reduced = reduced * valid[:, :, :, None, None].to(dtype=reduced.dtype)
        mask = valid[:, :, :, None].expand(
            b, t, v, self.tokens_per_view
        )
        return reduced.reshape(b, -1, d), mask.reshape(b, -1)


class OnlineVGGTGeometryCore(nn.Module):
    """Run shallow VGGT, causal geometry rollout, and deep VGGT online.

    Clean future RGB is accepted only as an optional target branch.  It is
    encoded under ``no_grad`` and never enters the student predictor/deep path.
    """

    def __init__(
        self,
        *,
        encoder: VGGTEncoder,
        future_predictor: GAMFuturePredictor,
        history_connector: GroupedHistoryConnector,
        auxiliary_action_heads: GroupedGeometryActionHeads | None = None,
        observed_keyframe_indices: tuple[int, ...] = (0, 5, 10, 15),
        future_anchor_count: int = 4,
        geometry_output_patch_grid: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.future_predictor = future_predictor
        self.history_connector = history_connector
        self.auxiliary_action_heads = auxiliary_action_heads
        self.observed_keyframe_indices = tuple(
            int(value) for value in observed_keyframe_indices
        )
        self.future_anchor_count = int(future_anchor_count)
        if self.future_anchor_count <= 0:
            raise ValueError("future_anchor_count must be positive")
        if len(self.observed_keyframe_indices) < 1 or any(
            right <= left
            for left, right in zip(
                self.observed_keyframe_indices,
                self.observed_keyframe_indices[1:],
            )
        ):
            raise ValueError("observed keyframe indices must be strictly increasing")
        if future_predictor.d_da3 != encoder.embed_dim:
            raise ValueError("GAM predictor and VGGT token dimensions do not match")
        if future_predictor.d_model != history_connector.config.d_model:
            raise ValueError("history connector and GAM predictor widths do not match")
        if future_predictor.num_patches_per_view != encoder.num_patches:
            raise ValueError("GAM predictor and VGGT patch counts do not match")
        if future_predictor.num_register_tokens != encoder.num_register_tokens:
            raise ValueError("GAM predictor and VGGT register counts do not match")
        self.mode_embedding = nn.Embedding(2, future_predictor.d_model)
        self.terminal_action_seed = nn.Parameter(torch.zeros(encoder.embed_dim))
        nn.init.normal_(self.mode_embedding.weight, std=0.02)
        nn.init.normal_(self.terminal_action_seed, std=0.02)
        self.geometry_reducer = GeometryTokenReducer(
            token_dim=encoder.embed_dim,
            num_register_tokens=encoder.num_register_tokens,
            source_patch_grid=encoder.h_patches,
            output_patch_grid=geometry_output_patch_grid,
        )

    def predict_auxiliary_actions(
        self,
        output: OnlineGeometryOutput,
        *,
        action_template: GroupedActionBatch,
        future_view_valid_mask: torch.Tensor,
    ) -> GroupedAuxiliaryActionOutput:
        if self.auxiliary_action_heads is None:
            raise RuntimeError("grouped geometry auxiliary action heads are not configured")
        refined = output.student_geometry.get("action_tokens")
        if refined is None:
            raise RuntimeError("deep VGGT output is missing refined action tokens")
        return self.auxiliary_action_heads(
            predicted_action_seed_tokens=output.predicted_action_seed_tokens,
            refined_action_tokens=refined,
            total_steps=int(output.deep_visual_tokens.shape[1]),
            future_view_valid_mask=future_view_valid_mask,
            action_template=action_template,
        )

    @staticmethod
    def _validate_images(name: str, images: torch.Tensor) -> tuple[int, int, int]:
        if images.ndim != 6 or images.shape[3] != 3:
            raise ValueError(f"{name} must be [B,T,V,3,H,W]")
        if not torch.is_floating_point(images) or not bool(torch.isfinite(images).all()):
            raise ValueError(f"{name} must contain finite floating-point RGB")
        return int(images.shape[0]), int(images.shape[1]), int(images.shape[2])

    def _encode_shallow(self, images: torch.Tensor) -> torch.Tensor:
        b, t, v = images.shape[:3]
        flattened = images.reshape(b, t * v, *images.shape[3:])
        # The encoder method is itself no-grad; keep this explicit at the
        # integration boundary so shallow activations can never leak into the
        # trainable graph.
        with torch.no_grad():
            result = self.encoder.encode_shallow_visual_slots(flattened, T=t, V=v)
        return result["visual_tokens"].detach()

    def _predict_future(
        self,
        *,
        observed_shallow: torch.Tensor,
        proprio_tokens: torch.Tensor,
        action_history_tokens: torch.Tensor,
        mode: GeometryConditionMode,
        future_action_tokens: Optional[torch.Tensor],
        language_features: Optional[torch.Tensor],
        language_padding_mask: Optional[torch.Tensor],
        view_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        mode_index = 0 if mode is GeometryConditionMode.POLICY else 1
        mode_token = self.mode_embedding.weight[mode_index].to(
            device=observed_shallow.device, dtype=observed_shallow.dtype
        )
        proprio_stream = proprio_tokens.to(dtype=observed_shallow.dtype) + mode_token
        action_stream = action_history_tokens.to(dtype=observed_shallow.dtype) + mode_token
        visual_stream = observed_shallow
        view_stream = view_valid_mask
        predicted_visual: list[torch.Tensor] = []
        predicted_seed: list[torch.Tensor] = []
        predicted_proprio: list[torch.Tensor] = []
        last_output: dict[str, torch.Tensor] | None = None

        for anchor_index in range(self.future_anchor_count):
            conditioned_action = action_stream
            if future_action_tokens is not None:
                conditioned_action = action_stream.clone()
                conditioned_action[:, -1] = (
                    conditioned_action[:, -1]
                    + future_action_tokens[:, anchor_index].to(
                        dtype=conditioned_action.dtype
                    )
                )
            last_output = self.future_predictor(
                past_visual_tokens=visual_stream,
                proprio_token_embeddings=proprio_stream,
                past_action_token_embeddings=conditioned_action,
                lang_feats=language_features,
                lang_padding_mask=language_padding_mask,
                view_valid_mask=view_stream,
            )
            next_visual = last_output["predicted_next_visual_tokens"][:, -1:]
            next_seed = last_output["predicted_action_tokens"][:, -1:]
            next_proprio = last_output["predicted_next_proprio_tokens"][:, -1:]
            next_action_history = last_output[
                "predicted_next_action_history_tokens"
            ][:, -1:]
            if future_action_tokens is not None:
                next_action_history = future_action_tokens[:, anchor_index : anchor_index + 1]
            predicted_visual.append(next_visual)
            predicted_seed.append(next_seed)
            predicted_proprio.append(next_proprio)
            visual_stream = torch.cat([visual_stream, next_visual], dim=1)
            proprio_stream = torch.cat([proprio_stream, next_proprio], dim=1)
            action_stream = torch.cat([action_stream, next_action_history], dim=1)
            view_stream = torch.cat([view_stream, view_stream[:, -1:]], dim=1)

        if last_output is None:
            raise RuntimeError("future rollout produced no anchors")
        return (
            torch.cat(predicted_visual, dim=1),
            torch.cat(predicted_seed, dim=1),
            torch.cat(predicted_proprio, dim=1),
            last_output,
        )

    def forward(
        self,
        *,
        observed_images: torch.Tensor,
        state_history: GroupedStateHistoryBatch,
        action_history: GroupedActionTimelineBatch,
        mode: GeometryConditionMode | str,
        future_action_history: Optional[GroupedActionTimelineBatch] = None,
        future_target_images: Optional[torch.Tensor] = None,
        language_features: Optional[torch.Tensor] = None,
        language_padding_mask: Optional[torch.Tensor] = None,
        observed_view_valid_mask: Optional[torch.Tensor] = None,
        future_view_valid_mask: Optional[torch.Tensor] = None,
        decode_geometry_heads: bool = True,
        compute_target_geometry: bool = False,
        gradient_checkpointing: bool = False,
    ) -> OnlineGeometryOutput:
        mode = GeometryConditionMode(mode)
        if mode is GeometryConditionMode.POLICY and future_action_history is not None:
            raise ValueError("policy geometry must not receive future factual actions")
        if mode is GeometryConditionMode.FACTUAL and future_action_history is None:
            raise ValueError("factual geometry requires a candidate future action timeline")

        b, observed_steps, views = self._validate_images(
            "observed_images", observed_images
        )
        if observed_steps != len(self.observed_keyframe_indices):
            raise ValueError("observed image count must match configured keyframes")
        if not 1 <= views <= 3:
            raise ValueError("WM3D-WAM supports one to three real views per sample")
        if observed_images.shape[-2:] != (
            self.encoder.encoder_input_size,
            self.encoder.encoder_input_size,
        ):
            raise ValueError("observed images are not at the VGGT input size")
        if observed_view_valid_mask is None:
            observed_view_valid_mask = torch.ones(
                (b, observed_steps, views),
                device=observed_images.device,
                dtype=torch.bool,
            )
        else:
            observed_view_valid_mask = observed_view_valid_mask.to(
                device=observed_images.device, dtype=torch.bool
            )
            if observed_view_valid_mask.shape != (b, observed_steps, views):
                raise ValueError("observed_view_valid_mask shape mismatch")
        if not bool(observed_view_valid_mask.any(dim=2).all()):
            raise ValueError("every observed timestep needs a real camera view")
        if not bool(observed_view_valid_mask.all()):
            raise ValueError(
                "padded camera views are not allowed in the VGGT deep path; "
                "bucket windows by their exact real view count"
            )

        keyframes = torch.tensor(
            self.observed_keyframe_indices,
            device=state_history.values.device,
            dtype=torch.long,
        )
        proprio_tokens, action_tokens = self.history_connector(
            state_history=state_history,
            action_history=action_history,
            keyframe_indices=keyframes,
        )
        future_action_tokens = None
        if future_action_history is not None:
            if future_action_history.num_steps != self.future_anchor_count:
                raise ValueError("future action bins must match future anchors")
            future_action_tokens = self.history_connector.encode_action_steps(
                future_action_history
            )

        observed_shallow = self._encode_shallow(observed_images)
        predicted_shallow, predicted_seed, predicted_proprio, final_predictor = (
            self._predict_future(
                observed_shallow=observed_shallow,
                proprio_tokens=proprio_tokens,
                action_history_tokens=action_tokens,
                mode=mode,
                future_action_tokens=future_action_tokens,
                language_features=language_features,
                language_padding_mask=language_padding_mask,
                view_valid_mask=observed_view_valid_mask,
            )
        )
        all_shallow = torch.cat([observed_shallow, predicted_shallow], dim=1)
        # Predictor action seeds for the final input sequence cover all but the
        # terminal predicted state.  That state receives an explicit learned
        # null because no action beyond the requested horizon is available.
        input_seed = final_predictor["predicted_action_tokens"]
        terminal = self.terminal_action_seed.view(1, 1, 1, -1).expand(
            b, 1, views, -1
        ).to(device=input_seed.device, dtype=input_seed.dtype)
        deep_action_seed = torch.cat([input_seed, terminal], dim=1)
        if deep_action_seed.shape[:3] != all_shallow.shape[:3]:
            raise RuntimeError("VGGT visual/action rollout lengths do not align")

        if future_view_valid_mask is None:
            future_view_valid_mask = observed_view_valid_mask[:, -1:].expand(
                b, self.future_anchor_count, views
            )
        else:
            future_view_valid_mask = future_view_valid_mask.to(
                device=observed_images.device, dtype=torch.bool
            )
            if future_view_valid_mask.shape != (
                b,
                self.future_anchor_count,
                views,
            ):
                raise ValueError("future_view_valid_mask shape mismatch")
        if not bool(future_view_valid_mask.all()):
            raise ValueError(
                "padded future views are not allowed in the VGGT deep path; "
                "bucket windows by their exact real view count"
            )
        all_view_mask = torch.cat(
            [observed_view_valid_mask, future_view_valid_mask], dim=1
        )
        step_valid = all_view_mask.any(dim=2)
        student_geometry = self.encoder.propagate_shallow_with_actions_grad(
            all_shallow,
            deep_action_seed,
            decode_visuals=decode_geometry_heads,
            dpt_chunk_size=1,
            gradient_checkpointing=gradient_checkpointing,
            return_multi_level=False,
            step_valid_mask=step_valid,
            deep_temporal_causal_mask=True,
        )
        deep_visual = student_geometry["deep_visual_tokens"]
        future_deep = deep_visual[:, -self.future_anchor_count :]
        geometry_tokens, geometry_mask = self.geometry_reducer(
            future_deep, future_view_valid_mask
        )

        target_shallow = None
        target_geometry = None
        if future_target_images is not None:
            tb, target_steps, target_views = self._validate_images(
                "future_target_images", future_target_images
            )
            if (tb, target_steps, target_views) != (
                b,
                self.future_anchor_count,
                views,
            ):
                raise ValueError("future target RGB does not align with anchors/views")
            target_shallow = self._encode_shallow(future_target_images).detach()
            if compute_target_geometry:
                with torch.no_grad():
                    target_geometry = self.encoder.propagate_shallow_without_actions(
                        target_shallow,
                        decode_visuals=decode_geometry_heads,
                        dpt_chunk_size=1,
                    )
                    target_geometry = {
                        key: value.detach() if isinstance(value, torch.Tensor) else value
                        for key, value in target_geometry.items()
                    }
        elif compute_target_geometry:
            raise ValueError("target geometry requires future_target_images")

        return OnlineGeometryOutput(
            mode=mode,
            observed_shallow_tokens=observed_shallow,
            predicted_future_shallow_tokens=predicted_shallow,
            predicted_action_seed_tokens=predicted_seed,
            predicted_future_proprio_tokens=predicted_proprio,
            deep_visual_tokens=deep_visual,
            geometry_tokens=geometry_tokens,
            geometry_token_mask=geometry_mask,
            student_geometry=student_geometry,
            target_future_shallow_tokens=target_shallow,
            target_geometry=target_geometry,
        )
