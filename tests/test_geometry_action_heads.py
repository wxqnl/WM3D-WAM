import torch

from wm3d_wam.data.action_events import GroupedActionBatch
from wm3d_wam.models.geometry_action_heads import GroupedGeometryActionHeads
from wm3d_wam.training.objectives import grouped_auxiliary_action_loss


def _template(values: torch.Tensor) -> GroupedActionBatch:
    return GroupedActionBatch(
        values=values,
        value_mask=torch.tensor([[[[True, True]], [[True, False]]]]),
        event_mask=torch.tensor([[True, True]]),
        times_s=torch.tensor([[0.1, 0.7]]),
        event_dt_s=torch.tensor([[0.1, 0.6]]),
        group_ids=torch.tensor([[1]]),
        group_mask=torch.tensor([[True]]),
        action_semantic_ids=torch.tensor([[[1, 2]]]),
        composition_operator_ids=torch.tensor([[[1, 1]]]),
        embodiment_ids=torch.tensor([1]),
    )


def test_geometry_action_heads_cannot_read_clean_target_values():
    torch.manual_seed(7)
    head = GroupedGeometryActionHeads(
        codec_config={
            "hidden_dim": 8,
            "max_groups": 1,
            "max_action_dim": 2,
            "semantic_vocab_size": 8,
            "group_vocab_size": 8,
            "composition_vocab_size": 8,
            "embodiment_vocab_size": 8,
            "time_fourier_dim": 4,
        },
        future_anchor_count=2,
        num_heads=2,
    ).eval()
    seeds = torch.randn(1, 2, 1, 8)
    refined = torch.randn(3, 8)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    first = head(
        predicted_action_seed_tokens=seeds,
        refined_action_tokens=refined,
        total_steps=3,
        future_view_valid_mask=mask,
        action_template=_template(torch.tensor([[[[1.0, 2.0]], [[3.0, 0.0]]]])),
    )
    second = head(
        predicted_action_seed_tokens=seeds,
        refined_action_tokens=refined,
        total_steps=3,
        future_view_valid_mask=mask,
        action_template=_template(torch.tensor([[[[9.0, -4.0]], [[8.0, 0.0]]]])),
    )
    torch.testing.assert_close(first.direct, second.direct)
    torch.testing.assert_close(first.refined, second.refined)
    assert first.direct[0, 1, 0, 1].item() == 0.0
    loss = grouped_auxiliary_action_loss(
        first,
        _template(torch.tensor([[[[1.0, 2.0]], [[3.0, 0.0]]]])),
    )
    assert torch.isfinite(loss.total)
