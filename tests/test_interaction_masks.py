from __future__ import annotations

import torch

from wm3d_wam.models.interaction_masks import (
    InteractionProgram,
    build_mot_attention_mask,
)


def _inputs():
    video = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    action = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    return video, action


def test_action_only_sees_first_frame_and_never_future_video() -> None:
    video, action = _inputs()
    mask = build_mot_attention_mask(
        program=InteractionProgram.ACTION_ONLY,
        video_token_mask=video,
        action_event_mask=action,
        video_tokens_per_frame=2,
    )[:, 0]

    assert mask[0, 4:, :2].all()
    assert not mask[0, 4:, 2:4].any()
    assert not mask[0, :4, 4:].any()
    assert mask[1, 5, 5]
    assert not mask[1, 5, :5].any()


def test_forward_world_video_reads_candidate_action_one_way() -> None:
    video, action = _inputs()
    mask = build_mot_attention_mask(
        program=InteractionProgram.FORWARD_WORLD,
        video_token_mask=video,
        action_event_mask=action,
        video_tokens_per_frame=2,
    )[:, 0]

    assert mask[0, :4, 4:].all()
    assert not mask[0, 4:, :4].any()
    assert mask[1, :2, 4].all()
    assert not mask[1, :2, 5:].any()


def test_joint_program_is_bidirectional_only_for_valid_tokens() -> None:
    video, action = _inputs()
    mask = build_mot_attention_mask(
        program=InteractionProgram.JOINT_WORLD_ACTION,
        video_token_mask=video,
        action_event_mask=action,
        video_tokens_per_frame=2,
    )[:, 0]

    assert mask[0, :4, 4:].all()
    assert mask[0, 4:, :4].all()
    assert not mask[1, 4, 2:4].any()
    assert mask[1, 5, 5]


def test_temporal_visibility_can_restrict_video_queries() -> None:
    video, action = _inputs()
    visibility = torch.zeros((4, 3), dtype=torch.bool)
    visibility[2:, :2] = True
    mask = build_mot_attention_mask(
        program=InteractionProgram.FORWARD_WORLD,
        video_token_mask=video,
        action_event_mask=action,
        video_tokens_per_frame=2,
        video_to_action_visibility=visibility,
    )[0, 0]

    assert not mask[:2, 4:].any()
    assert mask[2:, 4:6].all()
    assert not mask[2:4, 6].any()
