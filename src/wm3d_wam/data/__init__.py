"""Lossless, timestamp-aware robot data contracts."""

from .action_events import (
    GroupedActionBatch,
    GroupedActionEventError,
    GroupedActionEvents,
    collate_grouped_action_events,
    extract_grouped_action_events,
)
from .grouped_robot import GROUPED_ROBOT_SCHEMA, GroupedRobotWindow
from .episode_splits import (
    EpisodeSplit,
    is_v1_eligible_manifest_record,
    materialize_source_split,
    requested_holdout_count,
)
from .grouped_history import (
    GroupedActionTimelineBatch,
    GroupedStateHistoryBatch,
    assign_action_events_to_steps,
)
from .online_episode import (
    OnlineEpisodeError,
    OnlineRobotWindow,
    first_eligible_episode,
    load_online_robot_window,
)

__all__ = [
    "GROUPED_ROBOT_SCHEMA",
    "EpisodeSplit",
    "GroupedActionBatch",
    "GroupedActionEventError",
    "GroupedActionEvents",
    "GroupedActionTimelineBatch",
    "GroupedRobotWindow",
    "GroupedStateHistoryBatch",
    "OnlineEpisodeError",
    "OnlineRobotWindow",
    "assign_action_events_to_steps",
    "collate_grouped_action_events",
    "extract_grouped_action_events",
    "first_eligible_episode",
    "load_online_robot_window",
    "is_v1_eligible_manifest_record",
    "materialize_source_split",
    "requested_holdout_count",
]
