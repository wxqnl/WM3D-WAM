from __future__ import annotations

import numpy as np
import torch

from wm3d_wam.data.action_events import (
    collate_grouped_action_events,
    extract_grouped_action_events,
)
from wm3d_wam.data.grouped_robot import (
    GroupedRobotLimits,
    RawActionSeries,
    RawStateSnapshot,
    pack_grouped_robot_window,
    panda_single_arm_spec,
)
from wm3d_wam.models.grouped_action_flow import (
    GroupedActionCodecConfig,
    GroupedActionFlowExpert,
    grouped_action_flow_loss,
)


def _batch(rates=(5, 10), max_events=16):
    samples = []
    for rate_hz in rates:
        count = int(rate_hz * 0.4)
        times = np.arange(count, dtype=np.float64) / rate_hz
        values = np.arange(count * 7, dtype=np.float32).reshape(count, 7) / 10
        window = pack_grouped_robot_window(
            embodiment=panda_single_arm_spec(),
            limits=GroupedRobotLimits(max_substeps=count),
            world_boundaries_s=[0.0, 0.4],
            action_series=[
                RawActionSeries("arm", "fine_command", values, times)
            ],
            current_state=[
                RawStateSnapshot("arm", 0.0, np.zeros(10, dtype=np.float32))
            ],
            policy_chunk_start_s=0.0,
        )
        samples.append(extract_grouped_action_events(window, max_events=max_events))
    return collate_grouped_action_events(samples, max_events=max_events)


def _expert(hidden_dim=16, num_layers=2):
    return GroupedActionFlowExpert(
        hidden_dim=hidden_dim,
        ffn_dim=hidden_dim * 2,
        text_dim=12,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=2,
        attn_head_dim=4,
        num_layers=num_layers,
        codec_config=GroupedActionCodecConfig(
            hidden_dim=hidden_dim,
            max_groups=8,
            max_action_dim=16,
            time_fourier_dim=16,
        ),
    )


def test_grouped_action_expert_runs_real_action_dit_blocks_and_backpropagates() -> None:
    torch.manual_seed(3)
    batch = _batch()
    expert = _expert()
    context = torch.randn(2, 5, 12)
    output = expert(
        action_batch=batch,
        timestep=torch.tensor([100.0, 700.0]),
        context=context,
        context_mask=torch.ones(2, 5, dtype=torch.bool),
    )

    assert output.shape == batch.values.shape
    assert not output[0, 2:].any()
    loss = grouped_action_flow_loss(output, torch.zeros_like(output), batch)
    loss.backward()
    assert expert.blocks[0].self_attn.q.weight.grad is not None
    assert expert.codec.value_encoder[0].weight.grad is not None


def test_flow_loss_normalizes_each_sample_by_its_valid_scalar_count() -> None:
    batch = _batch()
    prediction = torch.ones_like(batch.values)
    target = torch.zeros_like(batch.values)
    loss = grouped_action_flow_loss(prediction, target, batch)
    torch.testing.assert_close(loss, torch.tensor(1.0))
