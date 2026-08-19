"""Flow-matching samples and masked objectives for video and grouped actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wm3d_wam.data.action_events import GroupedActionBatch
from wm3d_wam.vendor.fastwam.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


@dataclass(frozen=True)
class GroupedActionFlowSample:
    clean: GroupedActionBatch
    noisy: GroupedActionBatch
    noise: torch.Tensor
    target_velocity: torch.Tensor
    timestep: torch.Tensor
    sample_weight: torch.Tensor


@dataclass(frozen=True)
class VideoFlowSample:
    clean_latents: torch.Tensor
    noisy_latents: torch.Tensor
    noise: torch.Tensor
    target_velocity: torch.Tensor
    timestep: torch.Tensor
    sample_weight: torch.Tensor
    prediction_mask: torch.Tensor  # [B,T], false for the observed latent


def _batch_weights(
    value: torch.Tensor, *, batch_size: int, device: torch.device
) -> torch.Tensor:
    result = value.to(device=device, dtype=torch.float32)
    if result.ndim == 0:
        result = result.expand(batch_size)
    if result.shape != (batch_size,):
        raise ValueError("flow sample weights must be scalar or [B]")
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise ValueError("flow sample weights must be finite and non-negative")
    return result


def _grouped_valid_mask(batch: GroupedActionBatch) -> torch.Tensor:
    return (
        batch.value_mask.to(dtype=torch.bool)
        & batch.event_mask[:, :, None, None].to(dtype=torch.bool)
        & batch.group_mask[:, None, :, None].to(dtype=torch.bool)
    )


def sample_grouped_action_flow(
    clean: GroupedActionBatch,
    scheduler: WanContinuousFlowMatchScheduler,
    *,
    timestep: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> GroupedActionFlowSample:
    """Noise only valid source-native scalars; padded ABI fields stay zero."""

    if clean.values.ndim != 4 or not torch.is_floating_point(clean.values):
        raise ValueError("grouped action values must be floating [B,E,G,D]")
    batch_size = int(clean.values.shape[0])
    device = clean.values.device
    dtype = clean.values.dtype
    valid = _grouped_valid_mask(clean)
    if not bool(valid.flatten(1).any(dim=1).all()):
        raise ValueError("every sample needs a valid future action scalar")
    if timestep is None:
        timestep = scheduler.sample_training_t(batch_size, device, dtype)
    else:
        timestep = timestep.to(device=device, dtype=dtype)
    if timestep.shape != (batch_size,):
        raise ValueError("action flow timestep must be [B]")
    if noise is None:
        noise = torch.randn_like(clean.values)
    else:
        noise = noise.to(device=device, dtype=dtype)
        if noise.shape != clean.values.shape:
            raise ValueError("action noise shape must match grouped values")
    noise = noise * valid.to(dtype=dtype)
    noisy_values = scheduler.add_noise(clean.values, noise, timestep)
    noisy_values = noisy_values * valid.to(dtype=dtype)
    target = scheduler.training_target(clean.values, noise, timestep)
    target = target * valid.to(dtype=dtype)
    weights = _batch_weights(
        scheduler.training_weight(timestep),
        batch_size=batch_size,
        device=device,
    )
    return GroupedActionFlowSample(
        clean=clean,
        noisy=clean.with_values(noisy_values),
        noise=noise,
        target_velocity=target,
        timestep=timestep,
        sample_weight=weights,
    )


def weighted_grouped_action_flow_loss(
    prediction: torch.Tensor,
    sample: GroupedActionFlowSample,
) -> torch.Tensor:
    if prediction.shape != sample.clean.values.shape:
        raise ValueError("action velocity shape does not match the grouped batch")
    valid = _grouped_valid_mask(sample.clean)
    counts = valid.flatten(1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("every action sample needs a supervised scalar")
    squared = (prediction.float() - sample.target_velocity.float()).square()
    per_sample = (squared * valid).flatten(1).sum(dim=1) / counts
    return (per_sample * sample.sample_weight).mean()


def sample_video_flow(
    clean_latents: torch.Tensor,
    scheduler: WanContinuousFlowMatchScheduler,
    *,
    timestep: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> VideoFlowSample:
    """Noise future Wan latents while preserving latent frame zero exactly."""

    if clean_latents.ndim != 5 or clean_latents.shape[2] < 2:
        raise ValueError("Wan latents must be [B,C,T,H,W] with T >= 2")
    if not torch.is_floating_point(clean_latents) or not bool(
        torch.isfinite(clean_latents).all()
    ):
        raise ValueError("Wan latents must be finite floating point")
    batch_size, _, frames = clean_latents.shape[:3]
    device = clean_latents.device
    dtype = clean_latents.dtype
    if timestep is None:
        timestep = scheduler.sample_training_t(batch_size, device, dtype)
    else:
        timestep = timestep.to(device=device, dtype=dtype)
    if timestep.shape != (batch_size,):
        raise ValueError("video flow timestep must be [B]")
    if noise is None:
        noise = torch.randn_like(clean_latents)
    else:
        noise = noise.to(device=device, dtype=dtype)
        if noise.shape != clean_latents.shape:
            raise ValueError("video noise shape must match clean latents")
    noise = noise.clone()
    noise[:, :, 0] = 0
    noisy = scheduler.add_noise(clean_latents, noise, timestep)
    noisy[:, :, 0] = clean_latents[:, :, 0]
    target = scheduler.training_target(clean_latents, noise, timestep)
    target[:, :, 0] = 0
    prediction_mask = torch.ones(
        (batch_size, frames), dtype=torch.bool, device=device
    )
    prediction_mask[:, 0] = False
    weights = _batch_weights(
        scheduler.training_weight(timestep),
        batch_size=batch_size,
        device=device,
    )
    return VideoFlowSample(
        clean_latents=clean_latents,
        noisy_latents=noisy,
        noise=noise,
        target_velocity=target,
        timestep=timestep,
        sample_weight=weights,
        prediction_mask=prediction_mask,
    )


def weighted_video_flow_loss(
    prediction: torch.Tensor,
    sample: VideoFlowSample,
) -> torch.Tensor:
    if prediction.shape != sample.clean_latents.shape:
        raise ValueError("video velocity shape does not match Wan latents")
    mask = sample.prediction_mask[:, None, :, None, None]
    squared = (prediction.float() - sample.target_velocity.float()).square()
    weighted = squared * mask
    counts = mask.expand_as(squared).flatten(1).sum(dim=1)
    per_sample = weighted.flatten(1).sum(dim=1) / counts.clamp_min(1)
    return (per_sample * sample.sample_weight).mean()
