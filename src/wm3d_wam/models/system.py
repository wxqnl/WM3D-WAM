"""End-to-end online geometry + Wan video + grouped action composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from wm3d_wam.data.action_events import GroupedActionBatch
from wm3d_wam.data.online_episode import OnlineRobotWindow
from wm3d_wam.vendor.fastwam.wan22.wan_video_vae import WanVideoVAE38

from .interaction_masks import InteractionProgram
from .online_vggt_geometry import (
    GeometryConditionMode,
    OnlineGeometryOutput,
    OnlineVGGTGeometryCore,
)
from .wan_action_mot import WanActionMoT, WanActionOutput


@dataclass(frozen=True)
class WM3DWAMProgramOutput:
    program: InteractionProgram
    geometry: OnlineGeometryOutput
    mot: WanActionOutput


class WM3DWAMSystem(nn.Module):
    """The deployable model boundary shared by all interaction programs.

    The VAE is attached for online RGB encoding but permanently frozen.  The
    UMT5 encoder is intentionally outside this class: its no-grad output is a
    common immutable input to both geometry and Wan/Action branches.
    """

    def __init__(
        self,
        *,
        geometry_core: OnlineVGGTGeometryCore,
        wan_action: WanActionMoT,
        video_vae: WanVideoVAE38,
    ) -> None:
        super().__init__()
        self.geometry_core = geometry_core
        self.wan_action = wan_action
        self.video_vae = video_vae
        self.video_vae.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> "WM3DWAMSystem":
        super().train(mode)
        # VAE statistics and causal caches are never part of optimization.
        self.video_vae.eval()
        return self

    @torch.no_grad()
    def encode_wan_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode [0,1] RGB to the exact Wan2.2 48-channel latent layout."""

        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("Wan RGB must be [B,3,T,H,W] or [3,T,H,W]")
        if video.shape[2] not in {1, 17}:
            raise ValueError("Wan windows must contain an anchor or 17-frame clip")
        if video.shape[-2] % 16 or video.shape[-1] % 16:
            raise ValueError("Wan RGB spatial dimensions must be divisible by 16")
        if not torch.is_floating_point(video) or not bool(torch.isfinite(video).all()):
            raise ValueError("Wan RGB must be finite floating point")
        if bool((video < 0).any()) or bool((video > 1).any()):
            raise ValueError("Wan RGB must be normalized to [0,1]")
        model_device = next(self.video_vae.parameters()).device
        model_dtype = next(self.video_vae.parameters()).dtype
        normalized = video.to(device=model_device, dtype=model_dtype) * 2.0 - 1.0
        latents = self.video_vae.encode(normalized, device=model_device)
        expected_latent_steps = 1 if video.shape[2] == 1 else 5
        if (
            latents.ndim != 5
            or latents.shape[1] != 48
            or latents.shape[2] != expected_latent_steps
        ):
            raise RuntimeError(
                "Wan2.2 VAE produced an unexpected latent shape: "
                f"{tuple(latents.shape)}"
            )
        return latents.detach()

    @staticmethod
    def _batched_images(value: torch.Tensor, *, name: str) -> torch.Tensor:
        if value.ndim == 5:
            value = value.unsqueeze(0)
        if value.ndim != 6:
            raise ValueError(f"{name} must be [T,V,3,H,W] or [B,T,V,3,H,W]")
        return value

    @staticmethod
    def _batched_view_mask(value: torch.Tensor, *, name: str) -> torch.Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3:
            raise ValueError(f"{name} must be [T,V] or [B,T,V]")
        return value

    def forward_program(
        self,
        *,
        program: InteractionProgram | str,
        window: OnlineRobotWindow,
        video_latents: torch.Tensor,
        video_timestep: torch.Tensor,
        action_batch: GroupedActionBatch,
        action_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        include_future_targets: bool = False,
        decode_geometry_heads: bool = False,
        compute_target_geometry: bool = False,
        geometry_gradient_checkpointing: bool = False,
    ) -> WM3DWAMProgramOutput:
        """Run one strict interaction program with no target-to-policy edge."""

        mode = InteractionProgram(program)
        if window.batch_size != context.shape[0]:
            raise ValueError("window and language context batch sizes do not match")
        if video_latents.shape[0] != context.shape[0]:
            raise ValueError("video latent and language context batches do not match")
        if action_batch.values.shape[0] != context.shape[0]:
            raise ValueError("action and language context batches do not match")
        geometry_mode = (
            GeometryConditionMode.FACTUAL
            if mode is InteractionProgram.FORWARD_WORLD
            else GeometryConditionMode.ACTION_FREE
        )
        future_action_history = (
            window.future_action_history
            if geometry_mode is GeometryConditionMode.FACTUAL
            else None
        )
        future_target_images = (
            self._batched_images(
                window.future_world_images, name="future_world_images"
            )
            if include_future_targets
            else None
        )
        geometry = self.geometry_core(
            observed_images=self._batched_images(
                window.observed_images, name="observed_images"
            ),
            state_history=window.state_history,
            action_history=window.action_history,
            future_world_times_s=window.future_world_times_s,
            mode=geometry_mode,
            future_action_history=future_action_history,
            future_target_images=future_target_images,
            language_features=context,
            language_padding_mask=context_mask,
            observed_view_valid_mask=self._batched_view_mask(
                window.observed_view_valid_mask,
                name="observed_view_valid_mask",
            ),
            future_view_valid_mask=self._batched_view_mask(
                window.future_view_valid_mask,
                name="future_view_valid_mask",
            ),
            future_world_valid_mask=self._batched_view_mask(
                window.future_world_valid_mask,
                name="future_world_valid_mask",
            ),
            decode_geometry_heads=decode_geometry_heads,
            compute_target_geometry=compute_target_geometry,
            gradient_checkpointing=geometry_gradient_checkpointing,
        )
        mot = self.wan_action(
            program=mode,
            video_latents=video_latents,
            video_timestep=video_timestep,
            action_batch=action_batch,
            action_timestep=action_timestep,
            context=context,
            context_mask=context_mask,
            geometry_tokens=geometry.geometry_tokens,
            geometry_token_mask=geometry.geometry_token_mask,
        )
        return WM3DWAMProgramOutput(
            program=mode,
            geometry=geometry,
            mot=mot,
        )
