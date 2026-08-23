"""Program-specific visibility masks for Wan video and action token streams."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch


class InteractionProgram(str, Enum):
    ACTION_ONLY = "action_only"
    FORWARD_WORLD = "forward_world"
    JOINT_WORLD_ACTION = "joint_world_action"


def _as_batch_mask(
    value: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    length: int,
) -> torch.Tensor:
    result = value.to(dtype=torch.bool)
    if result.shape != (batch_size, length):
        raise ValueError(
            f"{name} must have shape {(batch_size, length)}, got {tuple(result.shape)}"
        )
    return result


def build_group_diagonal_video_to_action_visibility(
    *,
    action_step_indices: torch.Tensor,
    action_event_mask: torch.Tensor,
    video_length: int,
    video_tokens_per_frame: int,
    num_action_steps: int,
) -> torch.Tensor:
    """Map physical action bins to their exact Wan latent-frame group.

    Wan's 17 RGB frames produce five latent frames: one observed anchor and
    four future temporal groups.  The grouped robot ABI already assigns every
    source-native event to one of 16 exact 10 Hz future bins.  Each future
    latent frame therefore reads exactly its corresponding four-step group;
    the observed anchor reads none.  This matches the production FastWAM
    configuration
    ``action_group_causal_mask_mode=group_diagonal`` while avoiding FastWAM's
    equal-event-count assumption and preserving asynchronous source clocks.
    """

    if action_step_indices.ndim != 2 or action_event_mask.ndim != 2:
        raise ValueError("action step indices and event mask must be rank-2")
    if action_step_indices.shape != action_event_mask.shape:
        raise ValueError("action step indices must align with action events")
    if video_length <= 0 or video_tokens_per_frame <= 0:
        raise ValueError("video lengths must be positive")
    if video_length % video_tokens_per_frame:
        raise ValueError("video token length must contain complete latent frames")
    if num_action_steps <= 0:
        raise ValueError("num_action_steps must be positive")

    latent_frames = video_length // video_tokens_per_frame
    if latent_frames < 2:
        raise ValueError("action-conditioned video requires a future latent frame")
    future_latent_groups = latent_frames - 1
    if num_action_steps % future_latent_groups:
        raise ValueError(
            "physical action steps must divide evenly across future latent groups"
        )
    steps_per_group = num_action_steps // future_latent_groups
    event_mask = action_event_mask.to(dtype=torch.bool)
    step_indices = action_step_indices.to(
        device=event_mask.device, dtype=torch.long
    )
    invalid_valid_event = event_mask & (
        (step_indices < 0) | (step_indices >= num_action_steps)
    )
    if bool(invalid_valid_event.any()):
        raise ValueError("a valid action event has an invalid physical step index")
    if bool((~event_mask & (step_indices != -1)).any()):
        raise ValueError("padded action events must use step index -1")

    frame_indices = torch.arange(
        latent_frames, device=event_mask.device, dtype=torch.long
    ).repeat_interleave(video_tokens_per_frame)
    event_groups = torch.div(
        step_indices.clamp_min(0), steps_per_group, rounding_mode="floor"
    )
    video_groups = frame_indices - 1
    visibility = event_groups[:, None, :] == video_groups[None, :, None]
    visibility &= event_mask[:, None, :]
    return visibility


def build_mot_attention_mask(
    *,
    program: InteractionProgram | str,
    video_token_mask: torch.Tensor,
    action_event_mask: torch.Tensor,
    video_tokens_per_frame: int,
    video_self_mask: Optional[torch.Tensor] = None,
    video_to_action_visibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a batch-aware MoT mask with no clean-future target channel.

    The returned bool tensor has shape [B, 1, Sv+Sa, Sv+Sa], directly
    broadcastable by PyTorch scaled-dot-product attention.
    """

    mode = InteractionProgram(program)
    if video_token_mask.ndim != 2 or action_event_mask.ndim != 2:
        raise ValueError("video_token_mask and action_event_mask must be rank-2")
    batch_size, video_length = video_token_mask.shape
    action_batch, action_length = action_event_mask.shape
    if action_batch != batch_size:
        raise ValueError("video and action masks must have the same batch size")
    if video_tokens_per_frame <= 0 or video_tokens_per_frame > video_length:
        raise ValueError(
            "video_tokens_per_frame must be positive and no larger than video length"
        )
    device = video_token_mask.device
    video_valid = _as_batch_mask(
        video_token_mask,
        name="video_token_mask",
        batch_size=batch_size,
        length=video_length,
    )
    action_valid = _as_batch_mask(
        action_event_mask.to(device=device),
        name="action_event_mask",
        batch_size=batch_size,
        length=action_length,
    )
    total = video_length + action_length
    mask = torch.zeros((batch_size, total, total), dtype=torch.bool, device=device)

    if video_self_mask is None:
        base_video_self = torch.ones(
            (video_length, video_length), dtype=torch.bool, device=device
        )
    else:
        base_video_self = video_self_mask.to(device=device, dtype=torch.bool)
        if base_video_self.shape != (video_length, video_length):
            raise ValueError(
                f"video_self_mask must have shape {(video_length, video_length)}, "
                f"got {tuple(base_video_self.shape)}"
            )
    mask[:, :video_length, :video_length] = (
        base_video_self.unsqueeze(0)
        & video_valid.unsqueeze(1)
        & video_valid.unsqueeze(2)
    )

    action_pair_valid = action_valid.unsqueeze(1) & action_valid.unsqueeze(2)
    mask[:, video_length:, video_length:] = action_pair_valid

    if mode is InteractionProgram.ACTION_ONLY:
        observed = video_valid[:, :video_tokens_per_frame]
        mask[:, video_length:, :video_tokens_per_frame] = (
            action_valid.unsqueeze(2) & observed.unsqueeze(1)
        )
    elif mode is InteractionProgram.FORWARD_WORLD:
        if video_to_action_visibility is None:
            raise ValueError("forward_world requires temporal action visibility")
        else:
            visibility = video_to_action_visibility.to(device=device, dtype=torch.bool)
            if visibility.shape == (video_length, action_length):
                visibility = visibility.unsqueeze(0).expand(batch_size, -1, -1)
            if visibility.shape != (batch_size, video_length, action_length):
                raise ValueError(
                    "video_to_action_visibility must be [Sv,Sa] or [B,Sv,Sa]"
                )
        mask[:, :video_length, video_length:] = (
            visibility
            & video_valid.unsqueeze(2)
            & action_valid.unsqueeze(1)
        )
    else:  # Mature FastWAM joint direction: video conditions action, not inverse.
        mask[:, video_length:, :video_length] = (
            action_valid.unsqueeze(2) & video_valid.unsqueeze(1)
        )
        if video_to_action_visibility is not None:
            raise ValueError("joint_world_action video must not read noisy actions")

    all_valid = torch.cat((video_valid, action_valid), dim=1)
    invalid_queries = ~all_valid
    if bool(invalid_queries.any()):
        batch_indices, token_indices = torch.nonzero(
            invalid_queries, as_tuple=True
        )
        mask[batch_indices, token_indices, token_indices] = True
    return mask.unsqueeze(1)
