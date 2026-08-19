import torch

from wm3d_wam.data.action_events import GroupedActionBatch
from wm3d_wam.data.grouped_history import (
    GroupedActionTimelineBatch,
    GroupedStateHistoryBatch,
    assign_action_events_to_steps,
)
from wm3d_wam.models.grouped_action_flow import GroupedActionCodecConfig
from wm3d_wam.models.grouped_history import (
    GroupedHistoryConnector,
    GroupedHistoryConnectorConfig,
)


def _events(values: torch.Tensor) -> GroupedActionBatch:
    batch, events, groups, dimensions = values.shape
    return GroupedActionBatch(
        values=values,
        value_mask=torch.ones_like(values, dtype=torch.bool),
        event_mask=torch.ones((batch, events), dtype=torch.bool),
        times_s=torch.arange(events, dtype=torch.float32).unsqueeze(0) * 0.5,
        event_dt_s=torch.full((batch, events), 0.5),
        group_ids=torch.tensor([[1, 2]], dtype=torch.long),
        group_mask=torch.ones((batch, groups), dtype=torch.bool),
        action_semantic_ids=torch.ones(
            (batch, groups, dimensions), dtype=torch.long
        ),
        composition_operator_ids=torch.ones(
            (batch, groups, dimensions), dtype=torch.long
        ),
        embodiment_ids=torch.ones((batch,), dtype=torch.long),
    )


def test_exact_boundary_events_belong_to_the_following_step():
    events = _events(torch.randn(1, 4, 2, 2))
    timeline = assign_action_events_to_steps(
        events,
        step_boundaries_s=torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0]]),
    )
    assert timeline.step_indices.tolist() == [[0, 1, 2, 3]]


def test_connector_uses_early_grouped_state_and_action_history():
    state_values = torch.randn(1, 4, 2, 3, requires_grad=True)
    action_values = torch.randn(1, 4, 2, 2, requires_grad=True)
    state = GroupedStateHistoryBatch(
        values=state_values,
        value_mask=torch.ones_like(state_values, dtype=torch.bool),
        step_mask=torch.ones((1, 4), dtype=torch.bool),
        times_s=torch.tensor([[-1.5, -1.0, -0.5, 0.0]]),
        group_ids=torch.tensor([[1, 2]], dtype=torch.long),
        group_mask=torch.ones((1, 2), dtype=torch.bool),
        state_semantic_ids=torch.ones((1, 2, 3), dtype=torch.long),
        embodiment_ids=torch.ones((1,), dtype=torch.long),
    )
    timeline = GroupedActionTimelineBatch(
        events=_events(action_values),
        step_indices=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        step_mask=torch.ones((1, 4), dtype=torch.bool),
    )
    connector = GroupedHistoryConnector(
        GroupedHistoryConnectorConfig(
            d_model=32,
            max_groups=2,
            max_state_dim=3,
            time_fourier_dim=16,
            transformer_depth=2,
            num_heads=4,
        ),
        action_codec_config=GroupedActionCodecConfig(
            hidden_dim=32,
            max_groups=2,
            max_action_dim=2,
            time_fourier_dim=16,
        ),
    )
    state_tokens, action_tokens = connector(
        state_history=state,
        action_history=timeline,
        keyframe_indices=torch.tensor([0, 3]),
    )
    assert state_tokens.shape == (1, 2, 32)
    assert action_tokens.shape == (1, 2, 32)
    (state_tokens[:, -1].square().mean() + action_tokens[:, -1].square().mean()).backward()
    assert state_values.grad is not None
    assert action_values.grad is not None
    assert torch.count_nonzero(state_values.grad[:, 0]) > 0
    assert torch.count_nonzero(action_values.grad[:, 0]) > 0
