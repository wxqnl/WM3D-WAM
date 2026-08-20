"""Formal online Dataset/DataLoader over fixed episode splits."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from .action_events import GroupedActionBatch
from .episode_splits import is_v1_eligible_manifest_record
from .grouped_history import GroupedActionTimelineBatch, GroupedStateHistoryBatch
from .hierarchical_sampler import RecoverableHierarchicalSampler, WindowRequest
from .online_episode import (
    OnlineEpisodeError,
    OnlineRobotWindow,
    iter_episode_manifest,
    load_online_robot_window,
)
from .source_contracts import (
    NormalizationRegistry,
    SourceContract,
    SourceContractError,
    SourceContractRegistry,
    validate_profile_coverage,
)


@dataclass(frozen=True)
class SourceEpisodeCatalog:
    contract: SourceContract
    source_root: Path
    adapter_path: Path
    manifest_path: Path
    profile_weight: float
    episodes: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class OnlineTrainingSample:
    requests: tuple[WindowRequest, ...]
    window: OnlineRobotWindow
    task_texts: tuple[str, ...]
    episode_ids: tuple[str, ...]
    quality_weights: tuple[float, ...]
    decode_retry_count: int

    def __post_init__(self) -> None:
        size = len(self.requests)
        if size < 1 or any(
            len(values) != size
            for values in (self.task_texts, self.episode_ids, self.quality_weights)
        ):
            raise SourceContractError("online micro-batch metadata is misaligned")
        if self.window.batch_size != size:
            raise SourceContractError("online tensor and request batch sizes differ")
        if not np.isfinite(self.quality_weights).all() or any(
            weight <= 0.0 for weight in self.quality_weights
        ):
            raise SourceContractError(
                "online micro-batch has an invalid quality weight"
            )
        routes = {
            (request.program, request.family, request.source)
            for request in self.requests
        }
        if len(routes) != 1:
            raise SourceContractError(
                "one micro-batch must share a routed model schema"
            )

    @property
    def request(self) -> WindowRequest:
        """Representative route; every request in the batch shares it."""

        return self.requests[0]

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    @property
    def quality_weight(self) -> float:
        first = float(self.quality_weights[0])
        if any(
            abs(float(value) - first) > 1.0e-12 for value in self.quality_weights[1:]
        ):
            raise SourceContractError(
                "one micro-batch must share a source quality weight"
            )
        return first


def _load_split_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise SourceContractError(f"episode split file is missing: {path}")
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not values:
        raise SourceContractError(f"episode split file is empty: {path}")
    return values


class OnlineRobotDataset(Dataset[OnlineTrainingSample]):
    """Decode deterministic raw Parquet/MP4 windows requested by a sampler."""

    def __init__(
        self,
        *,
        data_profile_path: str | Path,
        source_contract_path: str | Path,
        normalization_path: str | Path,
        split_root: str | Path,
        split: str,
        max_decode_retries: int = 8,
        max_views: int = 3,
        max_groups: int = 8,
        max_action_dim: int = 16,
        max_state_dim: int = 32,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise SourceContractError("dataset split must be train, val, or test")
        if max_decode_retries <= 0:
            raise SourceContractError("max_decode_retries must be positive")
        profile_path = Path(data_profile_path).expanduser().resolve(strict=True)
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        source_rows = profile.get("sources") if isinstance(profile, Mapping) else None
        if not isinstance(source_rows, list) or not source_rows:
            raise SourceContractError("data profile contains no source mappings")
        registry = SourceContractRegistry.load(source_contract_path)
        validate_profile_coverage(source_rows, registry)
        normalization = NormalizationRegistry.load(normalization_path)
        split_root = Path(split_root).expanduser().resolve(strict=True)

        catalogs: dict[str, SourceEpisodeCatalog] = {}
        for row in source_rows:
            if not isinstance(row, Mapping):
                raise SourceContractError("data profile source entry is not a mapping")
            source = str(row["name"])
            contract = registry.sources[source]
            if not contract.trainable:
                continue
            ids = _load_split_ids(split_root / split / f"{source}.txt")
            manifest_path = Path(str(row["manifest"])).expanduser().resolve(strict=True)
            episodes = tuple(
                record
                for record in iter_episode_manifest(manifest_path)
                if str(record.get("episode_id", "")) in ids
                and is_v1_eligible_manifest_record(record)
            )
            found = {str(record["episode_id"]) for record in episodes}
            if found != ids:
                missing = sorted(ids - found)
                raise SourceContractError(
                    f"split/catalog mismatch for {source}: missing {missing[:8]}"
                )
            catalogs[source] = SourceEpisodeCatalog(
                contract=contract,
                source_root=Path(str(row["raw_root"]))
                .expanduser()
                .resolve(strict=True),
                adapter_path=Path(str(row["adapter_config"]))
                .expanduser()
                .resolve(strict=True),
                manifest_path=manifest_path,
                profile_weight=float(row["weight"]),
                episodes=episodes,
            )
        if not catalogs:
            raise SourceContractError(
                "no verified source has episodes in the requested split"
            )
        self.profile_path = profile_path
        self.registry = registry
        self.normalization = normalization
        self.split_root = split_root
        self.split = split
        self.catalogs = catalogs
        self.max_decode_retries = int(max_decode_retries)
        self.max_views = int(max_views)
        self.max_groups = int(max_groups)
        self.max_action_dim = int(max_action_dim)
        self.max_state_dim = int(max_state_dim)

    @property
    def episode_counts(self) -> dict[str, int]:
        return {name: len(catalog.episodes) for name, catalog in self.catalogs.items()}

    @property
    def profile_weights(self) -> dict[str, float]:
        return {name: catalog.profile_weight for name, catalog in self.catalogs.items()}

    @property
    def contracts(self) -> tuple[SourceContract, ...]:
        return tuple(catalog.contract for catalog in self.catalogs.values())

    def __len__(self) -> int:
        return sum(self.episode_counts.values())

    def __getitem__(self, request: WindowRequest) -> OnlineTrainingSample:
        if not isinstance(request, WindowRequest):
            raise TypeError("OnlineRobotDataset indices must be WindowRequest objects")
        try:
            catalog = self.catalogs[request.source]
        except KeyError as exc:
            raise SourceContractError(
                f"sampler requested source {request.source!r} outside this dataset"
            ) from exc
        if request.program not in catalog.contract.programs:
            raise SourceContractError(
                f"source {request.source!r} is not approved for {request.program!r}"
            )
        errors: list[str] = []
        episode_count = len(catalog.episodes)
        for retry in range(self.max_decode_retries):
            episode_index = (
                request.episode_index + retry * request.retry_stride
            ) % episode_count
            episode = catalog.episodes[episode_index]
            anchor_fraction = (
                request.anchor_fraction + retry * 0.6180339887498949
            ) % 1.0
            # The target view is part of the routed micro-batch schema. A
            # decode retry may move to another episode/window, but it must not
            # silently route one sample to another camera slot while the other
            # samples in the same micro-batch retain the original slot.
            view_fraction = request.target_view_fraction
            try:
                window = load_online_robot_window(
                    source_root=catalog.source_root,
                    adapter_path=catalog.adapter_path,
                    episode=episode,
                    source_contract=catalog.contract,
                    normalization=self.normalization,
                    program=request.program,
                    anchor_fraction=anchor_fraction,
                    target_view_fraction=view_fraction,
                    sample_index=request.global_sample_index,
                    decode_retry_count=retry,
                    max_views=self.max_views,
                    max_groups=self.max_groups,
                    max_action_dim=self.max_action_dim,
                    max_state_dim=self.max_state_dim,
                )
                return OnlineTrainingSample(
                    requests=(request,),
                    window=window,
                    task_texts=(window.task_text,),
                    episode_ids=(window.episode_id,),
                    quality_weights=(window.quality_weight,),
                    decode_retry_count=window.decode_retry_count,
                )
            except (OnlineEpisodeError, OSError) as exc:
                errors.append(
                    f"{episode.get('episode_id', '<unknown>')}: {type(exc).__name__}: {exc}"
                )
        raise OnlineEpisodeError(
            f"failed {self.max_decode_retries} deterministic windows for "
            f"source={request.source}, sample={request.global_sample_index}; "
            f"last_errors={errors[-3:]}"
        )


def _cat_action_batches(values: list[GroupedActionBatch]) -> GroupedActionBatch:
    return GroupedActionBatch(
        values=torch.cat([value.values for value in values], dim=0),
        value_mask=torch.cat([value.value_mask for value in values], dim=0),
        event_mask=torch.cat([value.event_mask for value in values], dim=0),
        times_s=torch.cat([value.times_s for value in values], dim=0),
        event_dt_s=torch.cat([value.event_dt_s for value in values], dim=0),
        group_ids=torch.cat([value.group_ids for value in values], dim=0),
        group_mask=torch.cat([value.group_mask for value in values], dim=0),
        action_semantic_ids=torch.cat(
            [value.action_semantic_ids for value in values], dim=0
        ),
        composition_operator_ids=torch.cat(
            [value.composition_operator_ids for value in values], dim=0
        ),
        embodiment_ids=torch.cat([value.embodiment_ids for value in values], dim=0),
    )


def _cat_state_histories(
    values: list[GroupedStateHistoryBatch],
) -> GroupedStateHistoryBatch:
    return GroupedStateHistoryBatch(
        values=torch.cat([value.values for value in values], dim=0),
        value_mask=torch.cat([value.value_mask for value in values], dim=0),
        step_mask=torch.cat([value.step_mask for value in values], dim=0),
        times_s=torch.cat([value.times_s for value in values], dim=0),
        group_ids=torch.cat([value.group_ids for value in values], dim=0),
        group_mask=torch.cat([value.group_mask for value in values], dim=0),
        state_semantic_ids=torch.cat(
            [value.state_semantic_ids for value in values], dim=0
        ),
        embodiment_ids=torch.cat([value.embodiment_ids for value in values], dim=0),
    )


def _cat_action_timelines(
    values: list[GroupedActionTimelineBatch],
) -> GroupedActionTimelineBatch:
    return GroupedActionTimelineBatch(
        events=_cat_action_batches([value.events for value in values]),
        step_indices=torch.cat([value.step_indices for value in values], dim=0),
        step_mask=torch.cat([value.step_mask for value in values], dim=0),
    )


def collate_online_training_samples(
    values: list[OnlineTrainingSample],
) -> OnlineTrainingSample:
    """Stack a real micro-batch without changing clocks, views, or events."""

    if not values or any(value.batch_size != 1 for value in values):
        raise SourceContractError(
            "DataLoader collate expects non-empty singleton samples"
        )
    windows = [value.window for value in values]
    first = windows[0]
    for window in windows[1:]:
        expected_schema = (
            first.program,
            first.source,
            first.source_id,
            first.target_view_index,
            first.valid_view_count,
        )
        actual_schema = (
            window.program,
            window.source,
            window.source_id,
            window.target_view_index,
            window.valid_view_count,
        )
        if actual_schema != expected_schema:
            raise SourceContractError(
                "micro-batch windows do not share source/view schema: "
                f"expected={expected_schema!r}, actual={actual_schema!r}, "
                f"sample_indices={[item.sample_index for item in windows]!r}, "
                f"decode_retries={[item.decode_retry_count for item in windows]!r}"
            )
        for name in (
            "observed_images",
            "future_world_images",
            "wan_video",
            "observed_view_valid_mask",
            "future_view_valid_mask",
            "future_world_valid_mask",
        ):
            if getattr(window, name).shape != getattr(first, name).shape:
                raise SourceContractError(
                    f"micro-batch tensor shape differs for {name}"
                )
    state_history = _cat_state_histories([window.state_history for window in windows])
    action_history = _cat_action_timelines(
        [window.action_history for window in windows]
    )
    future_action_history = _cat_action_timelines(
        [window.future_action_history for window in windows]
    )
    future_actions = _cat_action_batches([window.future_actions for window in windows])
    batch_window = OnlineRobotWindow(
        program=first.program,
        source=first.source,
        source_id=first.source_id,
        episode_id=first.episode_id,
        task_text=first.task_text,
        observed_images=torch.stack(
            [window.observed_images for window in windows], dim=0
        ),
        future_world_images=torch.stack(
            [window.future_world_images for window in windows], dim=0
        ),
        wan_video=torch.stack([window.wan_video for window in windows], dim=0),
        observed_view_valid_mask=torch.stack(
            [window.observed_view_valid_mask for window in windows], dim=0
        ),
        future_view_valid_mask=torch.stack(
            [window.future_view_valid_mask for window in windows], dim=0
        ),
        future_world_valid_mask=torch.stack(
            [window.future_world_valid_mask for window in windows], dim=0
        ),
        state_history=state_history,
        action_history=action_history,
        future_action_history=future_action_history,
        future_actions=future_actions,
        observation_times_s=torch.stack(
            [window.observation_times_s for window in windows], dim=0
        ),
        future_world_times_s=torch.stack(
            [window.future_world_times_s for window in windows], dim=0
        ),
        future_video_times_s=torch.stack(
            [window.future_video_times_s for window in windows], dim=0
        ),
        action_history_times_s=action_history.events.times_s,
        future_action_times_s=future_actions.times_s,
        quality_weight=first.quality_weight,
        anchor_time_s=first.anchor_time_s,
        target_view_index=first.target_view_index,
        valid_view_count=first.valid_view_count,
        sample_index=first.sample_index,
        decode_retry_count=sum(window.decode_retry_count for window in windows),
    )
    return OnlineTrainingSample(
        requests=tuple(request for value in values for request in value.requests),
        window=batch_window,
        task_texts=tuple(text for value in values for text in value.task_texts),
        episode_ids=tuple(episode for value in values for episode in value.episode_ids),
        quality_weights=tuple(
            weight for value in values for weight in value.quality_weights
        ),
        decode_retry_count=sum(value.decode_retry_count for value in values),
    )


def seed_online_worker(worker_id: int) -> None:
    del worker_id
    # Training constructs the loader after the FSDP model and checkpoint have
    # initialized CUDA. Workers must therefore be fresh spawned interpreters,
    # never forked copies of a live CUDA process. Hide the devices inside the
    # decode worker as an additional contract: all dataset tensors are CPU
    # tensors and only the rank process may own a CUDA context.
    if torch.cuda.is_initialized():
        raise SourceContractError(
            "online DataLoader worker inherited an initialized CUDA context"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def build_online_dataloader(
    dataset: OnlineRobotDataset,
    sampler: RecoverableHierarchicalSampler,
    *,
    num_workers: int,
    micro_batch_size: int = 1,
    prefetch_factor: int = 2,
    timeout_seconds: float = 120.0,
) -> DataLoader[OnlineTrainingSample]:
    if (
        num_workers < 0
        or micro_batch_size <= 0
        or prefetch_factor <= 0
        or timeout_seconds <= 0.0
    ):
        raise SourceContractError("invalid DataLoader worker/prefetch configuration")
    worker_generator = torch.Generator()
    worker_generator.manual_seed(int(sampler.seed) + 1_000_003 * int(sampler.rank))
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "sampler": sampler,
        "batch_size": int(micro_batch_size),
        "drop_last": True,
        "collate_fn": collate_online_training_samples,
        "num_workers": int(num_workers),
        "pin_memory": False,
        "worker_init_fn": seed_online_worker,
        "generator": worker_generator,
    }
    if num_workers:
        kwargs.update(
            {
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
                "prefetch_factor": int(prefetch_factor),
                "timeout": float(timeout_seconds),
            }
        )
    return DataLoader(**kwargs)
