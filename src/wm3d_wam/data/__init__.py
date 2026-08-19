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
from .hierarchical_sampler import (
    RecoverableHierarchicalSampler,
    WindowRequest,
    source_sampling_weights,
)
from .online_dataset import (
    OnlineRobotDataset,
    OnlineTrainingSample,
    build_online_dataloader,
)
from .source_contracts import (
    NormalizationRegistry,
    SourceContract,
    SourceContractError,
    SourceContractRegistry,
    denormalize_action_values,
    pack_robot_arrays,
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
    "OnlineRobotDataset",
    "OnlineTrainingSample",
    "NormalizationRegistry",
    "RecoverableHierarchicalSampler",
    "SourceContract",
    "SourceContractError",
    "SourceContractRegistry",
    "WindowRequest",
    "assign_action_events_to_steps",
    "collate_grouped_action_events",
    "extract_grouped_action_events",
    "first_eligible_episode",
    "load_online_robot_window",
    "build_online_dataloader",
    "denormalize_action_values",
    "is_v1_eligible_manifest_record",
    "materialize_source_split",
    "requested_holdout_count",
    "pack_robot_arrays",
    "source_sampling_weights",
]
