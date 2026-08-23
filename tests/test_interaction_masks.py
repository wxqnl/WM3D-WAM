from __future__ import annotations

import torch

from wm3d_wam.models.interaction_masks import (
    InteractionProgram,
    build_group_diagonal_video_to_action_visibility,
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
        video_to_action_visibility=torch.ones((4, 3), dtype=torch.bool),
    )[:, 0]

    assert mask[0, :4, 4:].all()
    assert not mask[0, 4:, :4].any()
    assert mask[1, :2, 4].all()
    assert not mask[1, :2, 5:].any()


def test_joint_program_keeps_noisy_action_out_of_video_queries() -> None:
    video, action = _inputs()
    mask = build_mot_attention_mask(
        program=InteractionProgram.JOINT_WORLD_ACTION,
        video_token_mask=video,
        action_event_mask=action,
        video_tokens_per_frame=2,
    )[:, 0]

    assert not mask[0, :4, 4:].any()
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


def test_physical_action_steps_map_to_four_group_diagonal_wan_groups() -> None:
    event_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    step_indices = torch.tensor([[0, 3, 4, 8, 15, -1]], dtype=torch.long)
    visibility = build_group_diagonal_video_to_action_visibility(
        action_step_indices=step_indices,
        action_event_mask=event_mask,
        video_length=10,
        video_tokens_per_frame=2,
        num_action_steps=16,
    )[0]

    assert not visibility[:2].any()
    assert visibility[2:4, :2].all()
    assert not visibility[2:4, 2:].any()
    assert visibility[4:6, 2].all()
    assert not visibility[4:6, :2].any()
    assert not visibility[4:6, 3:].any()
    assert visibility[6:8, 3].all()
    assert not visibility[6:8, :3].any()
    assert not visibility[6:8, 4:].any()
    assert visibility[8:10, 4].all()
    assert not visibility[8:10, :4].any()
    assert not visibility[:, 5].any()


def test_forward_world_rejects_missing_temporal_visibility() -> None:
    video, action = _inputs()
    try:
        build_mot_attention_mask(
            program=InteractionProgram.FORWARD_WORLD,
            video_token_mask=video,
            action_event_mask=action,
            video_tokens_per_frame=2,
        )
    except ValueError as error:
        assert "temporal action visibility" in str(error)
    else:
        raise AssertionError("forward_world accepted all-to-all action visibility")
