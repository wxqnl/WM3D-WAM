"""Convert lossless grouped robot windows into source-native action events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from .grouped_robot import GROUPED_ROBOT_SCHEMA, GroupedRobotWindow


class GroupedActionEventError(ValueError):
    """Raised when grouped fine commands cannot form a lossless event stream."""


@dataclass(frozen=True)
class GroupedActionEvents:
    """One unpadded source-native action stream.

    Values retain group and semantic dimensions. Commands from different robot
    groups that share a timestamp occupy one event; asynchronous commands stay
    as distinct events.
    """

    values: np.ndarray
    value_mask: np.ndarray
    times_s: np.ndarray
    event_dt_s: np.ndarray
    group_ids: np.ndarray
    group_mask: np.ndarray
    action_semantic_ids: np.ndarray
    composition_operator_ids: np.ndarray
    embodiment_id: np.int64

    @property
    def event_count(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class GroupedActionBatch:
    """Padded torch batch consumed by the grouped Action Flow Expert."""

    values: torch.Tensor
    value_mask: torch.Tensor
    event_mask: torch.Tensor
    times_s: torch.Tensor
    event_dt_s: torch.Tensor
    group_ids: torch.Tensor
    group_mask: torch.Tensor
    action_semantic_ids: torch.Tensor
    composition_operator_ids: torch.Tensor
    embodiment_ids: torch.Tensor

    def to(
        self,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "GroupedActionBatch":
        float_dtype = dtype or self.values.dtype
        return GroupedActionBatch(
            values=self.values.to(device=device, dtype=float_dtype),
            value_mask=self.value_mask.to(device=device),
            event_mask=self.event_mask.to(device=device),
            times_s=self.times_s.to(device=device, dtype=float_dtype),
            event_dt_s=self.event_dt_s.to(device=device, dtype=float_dtype),
            group_ids=self.group_ids.to(device=device),
            group_mask=self.group_mask.to(device=device),
            action_semantic_ids=self.action_semantic_ids.to(device=device),
            composition_operator_ids=self.composition_operator_ids.to(device=device),
            embodiment_ids=self.embodiment_ids.to(device=device),
        )

    def with_values(self, values: torch.Tensor) -> "GroupedActionBatch":
        if values.shape != self.values.shape:
            raise GroupedActionEventError(
                f"replacement values shape {tuple(values.shape)} does not match "
                f"{tuple(self.values.shape)}"
            )
        return GroupedActionBatch(
            values=values,
            value_mask=self.value_mask,
            event_mask=self.event_mask,
            times_s=self.times_s,
            event_dt_s=self.event_dt_s,
            group_ids=self.group_ids,
            group_mask=self.group_mask,
            action_semantic_ids=self.action_semantic_ids,
            composition_operator_ids=self.composition_operator_ids,
            embodiment_ids=self.embodiment_ids,
        )


def _validate_window_shapes(window: GroupedRobotWindow) -> tuple[int, int, int, int]:
    if window.schema != GROUPED_ROBOT_SCHEMA:
        raise GroupedActionEventError(
            f"unsupported grouped robot schema {window.schema!r}; expected {GROUPED_ROBOT_SCHEMA!r}"
        )
    values = np.asarray(window.fine_action_values)
    value_mask = np.asarray(window.fine_action_mask)
    sample_mask = np.asarray(window.fine_sample_mask)
    offsets = np.asarray(window.fine_action_dt)
    if values.ndim != 4:
        raise GroupedActionEventError(
            "fine_action_values must have shape [interval, group, substep, dimension]"
        )
    intervals, groups, substeps, dimensions = values.shape
    if value_mask.shape != values.shape:
        raise GroupedActionEventError("fine_action_mask shape mismatch")
    if sample_mask.shape != (intervals, groups, substeps):
        raise GroupedActionEventError("fine_sample_mask shape mismatch")
    if offsets.shape != sample_mask.shape:
        raise GroupedActionEventError("fine_action_dt shape mismatch")
    boundaries = np.asarray(window.world_boundaries_s, dtype=np.float64)
    if boundaries.shape != (intervals + 1,):
        raise GroupedActionEventError("world_boundaries_s does not match interval count")
    if np.asarray(window.group_ids).shape != (groups,):
        raise GroupedActionEventError("group_ids shape mismatch")
    if np.asarray(window.action_semantic_ids).shape != (groups, dimensions):
        raise GroupedActionEventError("action_semantic_ids shape mismatch")
    return intervals, groups, substeps, dimensions


def extract_grouped_action_events(
    window: GroupedRobotWindow,
    *,
    start_s: Optional[float] = None,
    stop_s: Optional[float] = None,
    max_events: Optional[int] = 32,
    timestamp_tolerance_s: float = 1.0e-6,
) -> GroupedActionEvents:
    """Flatten fine commands without downsampling, interpolation, or repetition."""

    intervals, groups, substeps, dimensions = _validate_window_shapes(window)
    boundaries = np.asarray(window.world_boundaries_s, dtype=np.float64)
    start = float(boundaries[0] if start_s is None else start_s)
    stop = float(boundaries[-1] if stop_s is None else stop_s)
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        raise GroupedActionEventError("action event bounds must be finite and increasing")
    if timestamp_tolerance_s < 0 or not np.isfinite(timestamp_tolerance_s):
        raise GroupedActionEventError("timestamp_tolerance_s must be finite and non-negative")

    values = np.asarray(window.fine_action_values, dtype=np.float32)
    value_mask = np.asarray(window.fine_action_mask, dtype=np.bool_)
    sample_mask = np.asarray(window.fine_sample_mask, dtype=np.bool_)
    offsets = np.asarray(window.fine_action_dt, dtype=np.float64)

    records: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for interval in range(intervals):
        interval_start = float(boundaries[interval])
        interval_stop = float(boundaries[interval + 1])
        for group in range(groups):
            for substep in range(substeps):
                if not bool(sample_mask[interval, group, substep]):
                    continue
                offset = float(offsets[interval, group, substep])
                timestamp = interval_start + offset
                if (
                    not np.isfinite(timestamp)
                    or timestamp < interval_start - timestamp_tolerance_s
                    or timestamp >= interval_stop
                ):
                    raise GroupedActionEventError(
                        f"fine command at interval={interval}, group={group}, "
                        f"substep={substep} has invalid timestamp {timestamp:.9f}"
                    )
                if timestamp < start - timestamp_tolerance_s or timestamp >= stop:
                    continue
                mask_row = value_mask[interval, group, substep].copy()
                if not bool(mask_row.any()):
                    raise GroupedActionEventError(
                        f"fine command at interval={interval}, group={group}, "
                        f"substep={substep} has no supervised dimensions"
                    )
                records.append(
                    (
                        max(timestamp, start),
                        group,
                        values[interval, group, substep].copy(),
                        mask_row,
                    )
                )

    if not records:
        raise GroupedActionEventError(
            f"window contains no fine commands in half-open range [{start}, {stop})"
        )
    records.sort(key=lambda row: (row[0], row[1]))

    grouped: list[tuple[float, dict[int, tuple[np.ndarray, np.ndarray]]]] = []
    for timestamp, group, row_values, row_mask in records:
        if not grouped or timestamp - grouped[-1][0] > timestamp_tolerance_s:
            grouped.append((timestamp, {}))
        event_time, event_groups = grouped[-1]
        if group in event_groups:
            raise GroupedActionEventError(
                f"group slot {group} has multiple commands at timestamp {event_time:.9f}"
            )
        event_groups[group] = (row_values, row_mask)

    event_count = len(grouped)
    if max_events is not None and event_count > int(max_events):
        raise GroupedActionEventError(
            f"window has {event_count} source-native events, exceeding max_events={max_events}; "
            "increase capacity instead of dropping commands"
        )

    event_values = np.zeros((event_count, groups, dimensions), dtype=np.float32)
    event_value_mask = np.zeros_like(event_values, dtype=np.bool_)
    absolute_times = np.empty((event_count,), dtype=np.float64)
    for event_index, (timestamp, event_groups) in enumerate(grouped):
        absolute_times[event_index] = timestamp
        for group, (row_values, row_mask) in event_groups.items():
            event_values[event_index, group] = row_values
            event_value_mask[event_index, group] = row_mask

    if event_count > 1 and np.any(np.diff(absolute_times) <= 0):
        raise GroupedActionEventError("grouped action event times are not strictly increasing")
    relative_times = (absolute_times - start).astype(np.float32)
    event_dt = np.diff(
        np.concatenate((np.asarray([start], dtype=np.float64), absolute_times))
    ).astype(np.float32)

    return GroupedActionEvents(
        values=event_values,
        value_mask=event_value_mask,
        times_s=relative_times,
        event_dt_s=event_dt,
        group_ids=np.asarray(window.group_ids, dtype=np.int64).copy(),
        group_mask=np.asarray(window.group_mask, dtype=np.bool_).copy(),
        action_semantic_ids=np.asarray(
            window.action_semantic_ids, dtype=np.int64
        ).copy(),
        composition_operator_ids=np.asarray(
            window.composition_operator_ids, dtype=np.int64
        ).copy(),
        embodiment_id=np.int64(window.embodiment_id),
    )


def collate_grouped_action_events(
    samples: Sequence[GroupedActionEvents],
    *,
    max_events: int = 32,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> GroupedActionBatch:
    """Pad event sequences while preserving every valid command and mask."""

    if not samples:
        raise GroupedActionEventError("cannot collate an empty action batch")
    if max_events <= 0:
        raise GroupedActionEventError("max_events must be positive")
    groups = int(samples[0].values.shape[1])
    dimensions = int(samples[0].values.shape[2])
    batch_size = len(samples)

    values = np.zeros((batch_size, max_events, groups, dimensions), dtype=np.float32)
    value_mask = np.zeros_like(values, dtype=np.bool_)
    event_mask = np.zeros((batch_size, max_events), dtype=np.bool_)
    times_s = np.zeros((batch_size, max_events), dtype=np.float32)
    event_dt_s = np.zeros_like(times_s)
    group_ids = np.zeros((batch_size, groups), dtype=np.int64)
    group_mask = np.zeros((batch_size, groups), dtype=np.bool_)
    semantics = np.zeros((batch_size, groups, dimensions), dtype=np.int64)
    composition = np.zeros_like(semantics)
    embodiment_ids = np.zeros((batch_size,), dtype=np.int64)

    for batch_index, sample in enumerate(samples):
        if sample.values.shape[1:] != (groups, dimensions):
            raise GroupedActionEventError(
                "all samples must share grouped padding capacities before collation"
            )
        count = sample.event_count
        if count > max_events:
            raise GroupedActionEventError(
                f"sample {batch_index} has {count} events, exceeding max_events={max_events}"
            )
        values[batch_index, :count] = sample.values
        value_mask[batch_index, :count] = sample.value_mask
        event_mask[batch_index, :count] = True
        times_s[batch_index, :count] = sample.times_s
        event_dt_s[batch_index, :count] = sample.event_dt_s
        group_ids[batch_index] = sample.group_ids
        group_mask[batch_index] = sample.group_mask
        semantics[batch_index] = sample.action_semantic_ids
        composition[batch_index] = sample.composition_operator_ids
        embodiment_ids[batch_index] = sample.embodiment_id

    return GroupedActionBatch(
        values=torch.as_tensor(values, device=device, dtype=dtype),
        value_mask=torch.as_tensor(value_mask, device=device),
        event_mask=torch.as_tensor(event_mask, device=device),
        times_s=torch.as_tensor(times_s, device=device, dtype=dtype),
        event_dt_s=torch.as_tensor(event_dt_s, device=device, dtype=dtype),
        group_ids=torch.as_tensor(group_ids, device=device),
        group_mask=torch.as_tensor(group_mask, device=device),
        action_semantic_ids=torch.as_tensor(semantics, device=device),
        composition_operator_ids=torch.as_tensor(composition, device=device),
        embodiment_ids=torch.as_tensor(embodiment_ids, device=device),
    )
