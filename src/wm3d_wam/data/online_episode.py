"""Online RGB/robot loading from the existing LeRobot OXE/RoboCasa layout.

This module reads only raw Parquet and MP4 assets plus their manifest/index
metadata.  It deliberately does not consume VGGT, depth, point, pose, or Wan
latent caches.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .action_events import (
    GroupedActionBatch,
    GroupedActionEvents,
    collate_grouped_action_events,
)
from .grouped_history import (
    GroupedActionTimelineBatch,
    GroupedStateHistoryBatch,
    assign_action_events_to_steps,
)
from .grouped_robot import ACTION_SEMANTIC_IDS, STATE_SEMANTIC_IDS
from .window_selection import WindowSelectionError, select_observed_world_window


class OnlineEpisodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class OnlineRobotWindow:
    source: str
    episode_id: str
    task_text: str
    observed_images: torch.Tensor             # [4,V,3,224,224]
    future_anchor_images: torch.Tensor         # [4,V,3,224,224]
    wan_video: torch.Tensor                    # [3,9,256,256]
    observed_view_valid_mask: torch.Tensor     # [4,V]
    future_view_valid_mask: torch.Tensor       # [4,V]
    state_history: GroupedStateHistoryBatch
    action_history: GroupedActionTimelineBatch
    future_action_history: GroupedActionTimelineBatch
    future_actions: GroupedActionBatch
    observation_times_s: torch.Tensor          # [16], relative to anchor
    future_video_times_s: torch.Tensor         # [9], relative to anchor
    action_history_times_s: torch.Tensor
    future_action_times_s: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.state_history.batch_size)

    @property
    def view_count(self) -> int:
        return int(self.observed_images.shape[1])

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "OnlineRobotWindow":
        """Move one decoded window without changing its physical clocks."""

        float_dtype = dtype or self.observed_images.dtype
        return OnlineRobotWindow(
            source=self.source,
            episode_id=self.episode_id,
            task_text=self.task_text,
            observed_images=self.observed_images.to(
                device=device, dtype=float_dtype
            ),
            future_anchor_images=self.future_anchor_images.to(
                device=device, dtype=float_dtype
            ),
            wan_video=self.wan_video.to(device=device, dtype=float_dtype),
            observed_view_valid_mask=self.observed_view_valid_mask.to(
                device=device
            ),
            future_view_valid_mask=self.future_view_valid_mask.to(device=device),
            state_history=self.state_history.to(device=device, dtype=float_dtype),
            action_history=self.action_history.to(device=device, dtype=float_dtype),
            future_action_history=self.future_action_history.to(
                device=device, dtype=float_dtype
            ),
            future_actions=self.future_actions.to(
                device=device, dtype=float_dtype
            ),
            observation_times_s=self.observation_times_s.to(
                device=device, dtype=float_dtype
            ),
            future_video_times_s=self.future_video_times_s.to(
                device=device, dtype=float_dtype
            ),
            action_history_times_s=self.action_history_times_s.to(
                device=device, dtype=float_dtype
            ),
            future_action_times_s=self.future_action_times_s.to(
                device=device, dtype=float_dtype
            ),
        )


class ParquetEpisodeAccessor:
    """Read only row groups intersecting one manifest episode slice."""

    def __init__(self, path: Path, *, row_start: int, row_stop: int):
        import pyarrow.parquet as pq

        self.path = Path(path).resolve(strict=True)
        self.parquet = pq.ParquetFile(self.path)
        self.start = int(row_start)
        self.stop = int(row_stop)
        if (
            self.start < 0
            or self.stop <= self.start
            or self.stop > self.parquet.metadata.num_rows
        ):
            raise OnlineEpisodeError(
                f"invalid Parquet episode slice [{self.start},{self.stop})"
            )
        self._cache: dict[str, np.ndarray] = {}

    def array(self, key: str) -> np.ndarray:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if key not in self.parquet.schema_arrow.names:
            raise OnlineEpisodeError(f"Parquet payload misses field {key!r}")
        values: list[object] = []
        row_group_start = 0
        for row_group in range(self.parquet.metadata.num_row_groups):
            count = self.parquet.metadata.row_group(row_group).num_rows
            row_group_stop = row_group_start + count
            left = max(self.start, row_group_start)
            right = min(self.stop, row_group_stop)
            if left < right:
                column = self.parquet.read_row_group(
                    row_group, columns=[key]
                ).column(0)
                values.extend(
                    column.slice(
                        left - row_group_start, right - left
                    ).to_pylist()
                )
            row_group_start = row_group_stop
            if row_group_start >= self.stop:
                break
        result = np.asarray(values)
        if result.shape[0] != self.stop - self.start:
            raise OnlineEpisodeError(f"Parquet field {key!r} slice is incomplete")
        self._cache[key] = result
        return result


def iter_episode_manifest(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise OnlineEpisodeError(
                    f"manifest line {line_number} is not an object"
                )
            yield value


def first_eligible_episode(path: Path, *, minimum_rows: int) -> dict[str, Any]:
    for value in iter_episode_manifest(path):
        if int(value.get("observation_samples", 0)) >= int(minimum_rows):
            return value
    raise OnlineEpisodeError(
        f"manifest contains no episode with at least {minimum_rows} rows"
    )


def _load_adapter(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("raw_format") != "lerobot_parquet_video":
        raise OnlineEpisodeError("online loader requires a LeRobot Parquet/video adapter")
    groups = value.get("groups")
    views = value.get("views")
    if not isinstance(groups, list) or len(groups) != 1:
        raise OnlineEpisodeError("v1 online loader requires one audited controller group")
    if not isinstance(views, list) or not views:
        raise OnlineEpisodeError("adapter contains no real RGB views")
    return value


def _mapped(accessor: ParquetEpisodeAccessor, terms: list[dict[str, Any]]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for term in terms:
        source = np.asarray(accessor.array(str(term["key"])))
        columns = np.asarray(term["columns"], dtype=np.int64)
        if source.ndim != 2 or columns.size < 1 or int(columns.max()) >= source.shape[1]:
            raise OnlineEpisodeError("adapter mapping exceeds a Parquet field")
        selected = source[:, columns].astype(np.float32, copy=False)
        selected = selected * np.asarray(term["scale"], dtype=np.float32)
        selected = selected + np.asarray(term["offset"], dtype=np.float32)
        pieces.append(selected)
    result = np.concatenate(pieces, axis=1)
    if not np.isfinite(result).all():
        raise OnlineEpisodeError("mapped robot values contain NaN/Inf")
    return result


def _decode_video_segment(
    path: Path, *, start_s: float, stop_s: float, expected_rows: int
) -> np.ndarray:
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path.resolve(strict=True)), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise OnlineEpisodeError("expected exactly one video stream")
        stream = streams[0]
        container.seek(
            int(start_s / float(stream.time_base)),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise OnlineEpisodeError("video frame has no recorded PTS")
            pts = float(frame.pts * frame.time_base)
            if pts + 1.0e-12 < start_s:
                continue
            if pts >= stop_s - 1.0e-12:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) == expected_rows + 1:
        # Encoder flush frames have no corresponding observation row.  They
        # may be discarded; a missing real row is never repeated.
        frames = frames[:expected_rows]
    if len(frames) != expected_rows:
        raise OnlineEpisodeError(
            f"video/observation ordinal mismatch: {len(frames)} != {expected_rows}"
        )
    output = np.stack(frames)
    if output.dtype != np.uint8 or output.ndim != 4 or output.shape[-1] != 3:
        raise OnlineEpisodeError("decoded video is not uint8 RGB")
    return output


def _resize_center_crop(frames: np.ndarray, size: int) -> torch.Tensor:
    tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    height, width = tensor.shape[-2:]
    scale = float(size) / float(min(height, width))
    resized_h = max(size, int(round(height * scale)))
    resized_w = max(size, int(round(width * scale)))
    tensor = F.interpolate(
        tensor,
        size=(resized_h, resized_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)
    top = (resized_h - size) // 2
    left = (resized_w - size) // 2
    return tensor[:, :, top : top + size, left : left + size].contiguous()


def _action_events(
    *,
    values: np.ndarray,
    timestamps_s: np.ndarray,
    start_s: float,
    stop_s: float,
    time_origin_s: float | None = None,
    embodiment_id: int,
    max_groups: int,
    max_action_dim: int,
    max_events: int,
    timestamp_tolerance_s: float = 1.0e-6,
) -> GroupedActionBatch:
    tolerance = np.float64(timestamp_tolerance_s)
    keep = (timestamps_s >= np.float64(start_s) - tolerance) & (
        timestamps_s < np.float64(stop_s) - tolerance
    )
    selected_values = values[keep]
    selected_times = timestamps_s[keep]
    if len(selected_values) < 1 or len(selected_values) > max_events:
        raise OnlineEpisodeError(
            f"action event count {len(selected_values)} is outside [1,{max_events}]"
        )
    action_dim = int(values.shape[1])
    if action_dim > max_action_dim:
        raise OnlineEpisodeError("source action dimension exceeds grouped capacity")
    padded = np.zeros(
        (len(selected_values), max_groups, max_action_dim), dtype=np.float32
    )
    padded[:, 0, :action_dim] = selected_values
    value_mask = np.zeros_like(padded, dtype=np.bool_)
    value_mask[:, 0, :action_dim] = True
    group_ids = np.zeros((max_groups,), dtype=np.int64)
    group_ids[0] = 30
    group_mask = np.zeros((max_groups,), dtype=np.bool_)
    group_mask[0] = True
    semantics = np.zeros((max_groups, max_action_dim), dtype=np.int64)
    semantics[0, :action_dim] = ACTION_SEMANTIC_IDS["controller_command"]
    composition = np.zeros_like(semantics)
    # Existing audited v4 adapters declare source-native controller channels
    # with last-value composition until per-source semantics are upgraded.
    composition[0, :action_dim] = 3
    origin = start_s if time_origin_s is None else float(time_origin_s)
    relative = (selected_times - np.float64(origin)).astype(np.float32)
    event_dt = np.diff(
        np.concatenate(
            [np.asarray([start_s], dtype=np.float64), selected_times]
        )
    ).astype(np.float32)
    events = GroupedActionEvents(
        values=padded,
        value_mask=value_mask,
        times_s=relative,
        event_dt_s=event_dt,
        group_ids=group_ids,
        group_mask=group_mask,
        action_semantic_ids=semantics,
        composition_operator_ids=composition,
        embodiment_id=np.int64(embodiment_id),
    )
    return collate_grouped_action_events(
        [events], max_events=max_events, dtype=torch.float32
    )


def _state_history(
    *,
    values: np.ndarray,
    timestamps_s: np.ndarray,
    rows: np.ndarray,
    anchor_s: float,
    embodiment_id: int,
    max_groups: int,
    max_state_dim: int,
) -> GroupedStateHistoryBatch:
    state_dim = int(values.shape[1])
    if state_dim > max_state_dim:
        raise OnlineEpisodeError("source state dimension exceeds grouped capacity")
    history = np.zeros(
        (1, len(rows), max_groups, max_state_dim), dtype=np.float32
    )
    history[0, :, 0, :state_dim] = values[rows]
    mask = np.zeros_like(history, dtype=np.bool_)
    mask[0, :, 0, :state_dim] = True
    group_ids = np.zeros((1, max_groups), dtype=np.int64)
    group_ids[0, 0] = 30
    group_mask = np.zeros((1, max_groups), dtype=np.bool_)
    group_mask[0, 0] = True
    semantics = np.zeros(
        (1, max_groups, max_state_dim), dtype=np.int64
    )
    semantics[0, 0, :state_dim] = STATE_SEMANTIC_IDS["controller_state"]
    return GroupedStateHistoryBatch(
        values=torch.from_numpy(history),
        value_mask=torch.from_numpy(mask),
        step_mask=torch.ones((1, len(rows)), dtype=torch.bool),
        times_s=torch.from_numpy(
            (timestamps_s[rows] - np.float64(anchor_s)).astype(np.float32)
        ).unsqueeze(0),
        group_ids=torch.from_numpy(group_ids),
        group_mask=torch.from_numpy(group_mask),
        state_semantic_ids=torch.from_numpy(semantics),
        embodiment_ids=torch.tensor([embodiment_id], dtype=torch.long),
    )


def load_online_robot_window(
    *,
    source_root: Path,
    adapter_path: Path,
    episode: dict[str, Any],
    embodiment_id: int,
    source_hz: int,
    max_views: int = 3,
    max_groups: int = 8,
    max_action_dim: int = 16,
    max_state_dim: int = 32,
    vggt_size: int = 224,
    wan_size: int = 256,
) -> OnlineRobotWindow:
    """Load the first valid 3.2 s history + 1.6 s future window online."""

    source_root = Path(source_root).resolve(strict=True)
    adapter = _load_adapter(adapter_path)
    stride = int(source_hz) // 5
    if int(source_hz) not in {5, 10, 15, 20} or stride * 5 != int(source_hz):
        raise OnlineEpisodeError("source_hz must be one of 5/10/15/20")
    row_count = int(episode["observation_samples"])
    payload = source_root / str(episode["payload"])
    accessor = ParquetEpisodeAccessor(
        payload,
        row_start=int(episode["payload_row_start"]),
        row_stop=int(episode["payload_row_stop"]),
    )
    observation_times = np.asarray(
        accessor.array(str(adapter["observation_time_key"])), dtype=np.float64
    ).reshape(-1)
    if (
        observation_times.shape != (row_count,)
        or not np.isfinite(observation_times).all()
        or np.any(np.diff(observation_times) <= 0)
    ):
        raise OnlineEpisodeError("observation timestamps are invalid")
    group = adapter["groups"][0]
    actions = _mapped(accessor, group["action"])
    states = _mapped(accessor, group["state"])
    action_times = np.asarray(
        accessor.array(str(group["action_time_key"])), dtype=np.float64
    ).reshape(-1)
    state_times = np.asarray(
        accessor.array(str(group["state_time_key"])), dtype=np.float64
    ).reshape(-1)
    if action_times.shape != (row_count,) or state_times.shape != (row_count,):
        raise OnlineEpisodeError("robot clocks do not align with episode rows")

    selected_window = None
    # Use recorded timestamps, not a fixed ordinal stride.  This retains the
    # audited 90%-coverage episodes while still selecting only real rows.
    for anchor_row in range(1, row_count):
        if observation_times[anchor_row] - observation_times[0] < 2.88:
            continue
        try:
            selected_window = select_observed_world_window(
                observation_times,
                anchor_index=anchor_row,
                context_samples=16,
                future_samples=8,
                context_horizon_s=3.2,
                future_horizon_s=1.6,
                minimum_horizon_coverage=0.9,
                future_offsets_s=np.arange(1, 9, dtype=np.float64) * 0.2,
            )
        except WindowSelectionError:
            continue
        boundary_span = (
            observation_times[anchor_row]
            - observation_times[selected_window.leading_boundary_index]
        )
        if boundary_span < 3.2 - 1.0e-3:
            continue
        break
    if selected_window is None:
        raise OnlineEpisodeError(
            "episode has no real-timestamp window meeting 90% horizon coverage"
        )
    state_rows = selected_window.context_indices
    future_rows = selected_window.future_indices
    anchor_row = int(state_rows[-1])
    anchor_s = float(observation_times[anchor_row])
    history_start_s = float(
        observation_times[selected_window.leading_boundary_index]
    )
    history_stop_s = anchor_s
    future_stop_s = anchor_s + 1.6
    keyframe_rows = state_rows[np.asarray([0, 5, 10, 15])]
    future_video_rows = np.concatenate(
        [np.asarray([anchor_row], dtype=np.int64), future_rows]
    )
    future_anchor_rows = future_rows[np.asarray([1, 3, 5, 7])]

    asset_by_role = {
        str(item["role"]): source_root / str(item["path"])
        for item in episode["assets"]
    }
    view_specs = list(episode["views"])[: int(max_views)]
    if not view_specs:
        raise OnlineEpisodeError("episode has no manifest-bound RGB views")
    selected_by_view: list[torch.Tensor] = []
    selected_wan: torch.Tensor | None = None
    for view_index, view in enumerate(view_specs):
        role = str(view["asset_role"])
        if role not in asset_by_role:
            raise OnlineEpisodeError(f"view asset role {role!r} is missing")
        frames = _decode_video_segment(
            asset_by_role[role],
            start_s=float(view["start_s"]),
            stop_s=float(view["stop_s"]),
            expected_rows=row_count,
        )
        adapter_view = next(
            (item for item in adapter["views"] if item["name"] == view["name"]),
            None,
        )
        if adapter_view is not None and adapter_view.get("color_order", "rgb") == "bgr":
            frames = frames[..., ::-1].copy()
        selected_rows = np.concatenate([keyframe_rows, future_anchor_rows])
        selected_by_view.append(_resize_center_crop(frames[selected_rows], vggt_size))
        if view_index == 0:
            selected_wan = _resize_center_crop(frames[future_video_rows], wan_size)
    stacked = torch.stack(selected_by_view, dim=1)  # [8,V,3,H,W]
    if selected_wan is None:
        raise RuntimeError("primary Wan view was not decoded")

    history_actions = _action_events(
        values=actions,
        timestamps_s=action_times,
        start_s=history_start_s,
        stop_s=history_stop_s,
        time_origin_s=anchor_s,
        embodiment_id=embodiment_id,
        max_groups=max_groups,
        max_action_dim=max_action_dim,
        max_events=64,
    )
    future_actions = _action_events(
        values=actions,
        timestamps_s=action_times,
        start_s=anchor_s,
        stop_s=future_stop_s,
        embodiment_id=embodiment_id,
        max_groups=max_groups,
        max_action_dim=max_action_dim,
        max_events=32,
    )
    history_boundary_rows = np.concatenate(
        [
            np.asarray(
                [selected_window.leading_boundary_index], dtype=np.int64
            ),
            state_rows,
        ]
    )
    history_boundaries = torch.from_numpy(
        (observation_times[history_boundary_rows] - anchor_s).astype(np.float32)
    ).unsqueeze(0)
    future_boundaries = torch.from_numpy(
        np.arange(5, dtype=np.float32) * np.float32(0.4)
    ).unsqueeze(0)
    history_timeline = assign_action_events_to_steps(
        history_actions, step_boundaries_s=history_boundaries
    )
    future_timeline = assign_action_events_to_steps(
        future_actions, step_boundaries_s=future_boundaries
    )
    state_history = _state_history(
        values=states,
        timestamps_s=state_times,
        rows=state_rows,
        anchor_s=anchor_s,
        embodiment_id=embodiment_id,
        max_groups=max_groups,
        max_state_dim=max_state_dim,
    )
    view_count = len(view_specs)
    return OnlineRobotWindow(
        source=str(episode["source"]),
        episode_id=str(episode["episode_id"]),
        task_text=str(episode.get("task_text", "")),
        observed_images=stacked[:4],
        future_anchor_images=stacked[4:],
        wan_video=selected_wan.permute(1, 0, 2, 3).contiguous(),
        observed_view_valid_mask=torch.ones((4, view_count), dtype=torch.bool),
        future_view_valid_mask=torch.ones((4, view_count), dtype=torch.bool),
        state_history=state_history,
        action_history=history_timeline,
        future_action_history=future_timeline,
        future_actions=future_actions,
        observation_times_s=state_history.times_s[0],
        future_video_times_s=torch.from_numpy(
            (observation_times[future_video_rows] - anchor_s).astype(np.float32)
        ),
        action_history_times_s=history_actions.times_s[0, history_actions.event_mask[0]],
        future_action_times_s=future_actions.times_s[0, future_actions.event_mask[0]],
    )
