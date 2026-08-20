"""Batched grouped robot history for the online WM3D/VGGT path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .action_events import GroupedActionBatch


@dataclass(frozen=True)
class GroupedStateHistoryBatch:
    """Measured state at the renderer clock without source-specific flattening.

    ``values`` is padded only in group/dimension capacity.  ``value_mask`` and
    the semantic metadata retain the physical meaning of every scalar.
    ``times_s`` contains the real timestamp relative to the policy anchor.
    """

    values: torch.Tensor                 # [B,H,G,D]
    value_mask: torch.Tensor             # [B,H,G,D]
    step_mask: torch.Tensor              # [B,H]
    times_s: torch.Tensor                # [B,H]
    group_ids: torch.Tensor              # [B,G]
    group_mask: torch.Tensor             # [B,G]
    state_semantic_ids: torch.Tensor      # [B,G,D]
    embodiment_ids: torch.Tensor          # [B]

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def history_steps(self) -> int:
        return int(self.values.shape[1])

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GroupedStateHistoryBatch":
        float_dtype = dtype or self.values.dtype
        return GroupedStateHistoryBatch(
            values=self.values.to(device=device, dtype=float_dtype),
            value_mask=self.value_mask.to(device=device),
            step_mask=self.step_mask.to(device=device),
            times_s=self.times_s.to(device=device, dtype=float_dtype),
            group_ids=self.group_ids.to(device=device),
            group_mask=self.group_mask.to(device=device),
            state_semantic_ids=self.state_semantic_ids.to(device=device),
            embodiment_ids=self.embodiment_ids.to(device=device),
        )


@dataclass(frozen=True)
class GroupedActionTimelineBatch:
    """Source-native action events assigned to physical history/anchor bins."""

    events: GroupedActionBatch
    step_indices: torch.Tensor            # [B,E], -1 for padded events
    step_mask: torch.Tensor               # [B,H]

    @property
    def num_steps(self) -> int:
        return int(self.step_mask.shape[1])

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GroupedActionTimelineBatch":
        return GroupedActionTimelineBatch(
            events=self.events.to(device=device, dtype=dtype),
            step_indices=self.step_indices.to(device=device),
            step_mask=self.step_mask.to(device=device),
        )


def assign_action_events_to_steps(
    events: GroupedActionBatch,
    *,
    step_boundaries_s: torch.Tensor,
    step_mask: torch.Tensor | None = None,
) -> GroupedActionTimelineBatch:
    """Assign every valid event to one exact half-open time bin.

    Boundaries are per-sample and must be strictly increasing.  No command is
    interpolated, repeated, truncated, or moved to a nearby bin.
    """

    if step_boundaries_s.ndim != 2:
        raise ValueError("step_boundaries_s must be [B,H+1]")
    batch_size, boundary_count = step_boundaries_s.shape
    if boundary_count < 2 or batch_size != events.values.shape[0]:
        raise ValueError("action batch and step boundaries do not align")
    if not bool(torch.isfinite(step_boundaries_s).all()):
        raise ValueError("step boundaries contain non-finite values")
    if not bool((torch.diff(step_boundaries_s, dim=1) > 0).all()):
        raise ValueError("step boundaries must be strictly increasing")
    history_steps = boundary_count - 1
    if step_mask is None:
        step_mask = torch.ones(
            (batch_size, history_steps),
            dtype=torch.bool,
            device=step_boundaries_s.device,
        )
    else:
        step_mask = step_mask.to(device=step_boundaries_s.device, dtype=torch.bool)
        if step_mask.shape != (batch_size, history_steps):
            raise ValueError("step_mask shape does not match boundaries")

    event_mask = events.event_mask.to(
        device=step_boundaries_s.device, dtype=torch.bool
    )
    event_times = events.times_s.to(
        device=step_boundaries_s.device, dtype=step_boundaries_s.dtype
    )
    step_indices = torch.full(
        event_times.shape, -1, dtype=torch.long, device=event_times.device
    )
    for batch_index in range(batch_size):
        valid = event_mask[batch_index]
        if not bool(valid.any()):
            continue
        times = event_times[batch_index, valid]
        lower = step_boundaries_s[batch_index, 0]
        upper = step_boundaries_s[batch_index, -1]
        if bool(((times < lower) | (times >= upper)).any()):
            raise ValueError(
                "valid action event lies outside the half-open timeline"
            )
        # right=True makes an event on an internal boundary belong to the
        # following bin, matching the grouped robot ABI.
        indices = torch.bucketize(
            times, step_boundaries_s[batch_index], right=True
        ) - 1
        if bool((indices < 0).any()) or bool((indices >= history_steps).any()):
            raise RuntimeError("action event binning produced an invalid index")
        if bool((~step_mask[batch_index, indices]).any()):
            raise ValueError("an action event was assigned to a padded step")
        step_indices[batch_index, valid] = indices

    return GroupedActionTimelineBatch(
        events=events,
        step_indices=step_indices,
        step_mask=step_mask,
    )
