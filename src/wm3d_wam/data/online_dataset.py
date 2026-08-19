"""Formal online Dataset/DataLoader over fixed episode splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from .episode_splits import is_v1_eligible_manifest_record
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
    request: WindowRequest
    window: OnlineRobotWindow


def _load_split_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise SourceContractError(f"episode split file is missing: {path}")
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
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
                source_root=Path(str(row["raw_root"])).expanduser().resolve(strict=True),
                adapter_path=Path(str(row["adapter_config"])).expanduser().resolve(strict=True),
                manifest_path=manifest_path,
                profile_weight=float(row["weight"]),
                episodes=episodes,
            )
        if not catalogs:
            raise SourceContractError("no verified source has episodes in the requested split")
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
            view_fraction = (
                request.target_view_fraction + retry * 0.4142135623730950
            ) % 1.0
            try:
                window = load_online_robot_window(
                    source_root=catalog.source_root,
                    adapter_path=catalog.adapter_path,
                    episode=episode,
                    source_contract=catalog.contract,
                    normalization=self.normalization,
                    anchor_fraction=anchor_fraction,
                    target_view_fraction=view_fraction,
                    sample_index=request.global_sample_index,
                    decode_retry_count=retry,
                    max_views=self.max_views,
                    max_groups=self.max_groups,
                    max_action_dim=self.max_action_dim,
                    max_state_dim=self.max_state_dim,
                )
                return OnlineTrainingSample(request=request, window=window)
            except (OnlineEpisodeError, OSError) as exc:
                errors.append(
                    f"{episode.get('episode_id', '<unknown>')}: {type(exc).__name__}: {exc}"
                )
        raise OnlineEpisodeError(
            f"failed {self.max_decode_retries} deterministic windows for "
            f"source={request.source}, sample={request.global_sample_index}; "
            f"last_errors={errors[-3:]}"
        )


def identity_collate(value: OnlineTrainingSample) -> OnlineTrainingSample:
    return value


def seed_online_worker(worker_id: int) -> None:
    del worker_id
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def build_online_dataloader(
    dataset: OnlineRobotDataset,
    sampler: RecoverableHierarchicalSampler,
    *,
    num_workers: int,
    prefetch_factor: int = 2,
    timeout_seconds: float = 120.0,
) -> DataLoader[OnlineTrainingSample]:
    if num_workers < 0 or prefetch_factor <= 0 or timeout_seconds <= 0.0:
        raise SourceContractError("invalid DataLoader worker/prefetch configuration")
    worker_generator = torch.Generator()
    worker_generator.manual_seed(
        int(sampler.seed) + 1_000_003 * int(sampler.rank)
    )
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "sampler": sampler,
        "batch_size": None,
        "collate_fn": identity_collate,
        "num_workers": int(num_workers),
        "pin_memory": False,
        "worker_init_fn": seed_online_worker,
        "generator": worker_generator,
    }
    if num_workers:
        kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(prefetch_factor),
                "timeout": float(timeout_seconds),
            }
        )
    return DataLoader(**kwargs)
