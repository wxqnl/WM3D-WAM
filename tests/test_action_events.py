from __future__ import annotations

import numpy as np
import pytest

from wm3d_wam.data.action_events import (
    GroupedActionEventError,
    collate_grouped_action_events,
    extract_grouped_action_events,
)
from wm3d_wam.data.grouped_robot import (
    GroupedRobotLimits,
    RawActionSeries,
    RawStateSnapshot,
    bimanual_arm_spec,
    pack_grouped_robot_window,
    panda_single_arm_spec,
)


def _single_arm_window(rate_hz: int, horizon_s: float = 1.6):
    count = int(round(rate_hz * horizon_s))
    timestamps = np.arange(count, dtype=np.float64) / rate_hz
    values = np.arange(count * 7, dtype=np.float32).reshape(count, 7)
    boundaries = np.arange(9, dtype=np.float64) * 0.2
    if horizon_s != 1.6:
        boundaries = np.asarray([0.0, horizon_s], dtype=np.float64)
    return pack_grouped_robot_window(
        embodiment=panda_single_arm_spec(),
        limits=GroupedRobotLimits(max_substeps=max(rate_hz // 5, count)),
        world_boundaries_s=boundaries,
        action_series=[
            RawActionSeries("arm", "fine_command", values, timestamps)
        ],
        current_state=[
            RawStateSnapshot("arm", 0.0, np.zeros(10, dtype=np.float32))
        ],
        policy_chunk_start_s=0.0,
    ), values, timestamps


@pytest.mark.parametrize("rate_hz,event_count", [(5, 8), (10, 16), (15, 24), (20, 32)])
def test_source_native_rates_remain_lossless(rate_hz: int, event_count: int) -> None:
    window, values, timestamps = _single_arm_window(rate_hz)
    events = extract_grouped_action_events(window)

    assert events.event_count == event_count
    np.testing.assert_array_equal(events.values[:, 0, :7], values)
    np.testing.assert_allclose(events.times_s, timestamps, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        events.event_dt_s[1:], np.diff(timestamps), rtol=0, atol=1e-7
    )


def test_synchronous_bimanual_commands_share_events_without_merging_groups() -> None:
    embodiment = bimanual_arm_spec()
    timestamps = np.arange(8, dtype=np.float64) / 5.0
    left = np.full((8, 7), 1.0, dtype=np.float32)
    right = np.full((8, 7), 2.0, dtype=np.float32)
    window = pack_grouped_robot_window(
        embodiment=embodiment,
        limits=GroupedRobotLimits(max_substeps=2),
        world_boundaries_s=np.arange(9, dtype=np.float64) * 0.2,
        action_series=[
            RawActionSeries("left_arm", "fine_command", left, timestamps),
            RawActionSeries("right_arm", "fine_command", right, timestamps),
        ],
        current_state=[
            RawStateSnapshot("left_arm", 0.0, np.zeros(10, dtype=np.float32)),
            RawStateSnapshot("right_arm", 0.0, np.zeros(10, dtype=np.float32)),
        ],
        policy_chunk_start_s=0.0,
    )
    events = extract_grouped_action_events(window)

    assert events.event_count == 8
    assert events.value_mask[:, :2, :7].all()
    np.testing.assert_array_equal(events.values[:, 0, :7], left)
    np.testing.assert_array_equal(events.values[:, 1, :7], right)


def test_capacity_failure_never_drops_the_33rd_command() -> None:
    window, _, _ = _single_arm_window(20, horizon_s=1.65)
    with pytest.raises(GroupedActionEventError, match="33 source-native events"):
        extract_grouped_action_events(window, max_events=32)


def test_collation_pads_events_but_preserves_each_sample_count() -> None:
    events_5 = extract_grouped_action_events(_single_arm_window(5)[0])
    events_20 = extract_grouped_action_events(_single_arm_window(20)[0])
    batch = collate_grouped_action_events([events_5, events_20], max_events=32)

    assert batch.values.shape == (2, 32, 8, 16)
    assert batch.event_mask.sum(dim=1).tolist() == [8, 32]
    assert not batch.value_mask[0, 8:].any()
    assert batch.value_mask[1, :, 0, :7].all()
