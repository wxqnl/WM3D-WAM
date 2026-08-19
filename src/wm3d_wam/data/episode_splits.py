"""Deterministic source-local episode/parent-trajectory split materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EpisodeSplit:
    source: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    parent_field: str | None

    @property
    def total(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)


def requested_holdout_count(eligible_episodes: int) -> int:
    """Return the per-split val/test target from the v1 data contract."""

    count = int(eligible_episodes)
    if count < 11:
        raise ValueError("at least 11 eligible episodes are required for splitting")
    if count >= 1000:
        return max(10, int(np.floor(count * 0.01 + 0.5)))
    if count >= 100:
        return 10
    return 5


def _choose_parent_field(
    records: Sequence[Mapping[str, object]],
    candidates: Sequence[str],
) -> str | None:
    present = [
        field
        for field in candidates
        if any(record.get(field) not in (None, "") for record in records)
    ]
    if len(present) > 1:
        raise ValueError(f"multiple parent trajectory fields are present: {present}")
    return present[0] if present else None


def materialize_source_split(
    records: Iterable[Mapping[str, object]],
    *,
    source: str,
    seed: int,
    parent_field_candidates: Sequence[str] = (
        "parent_trajectory_id",
        "parent_episode_id",
        "trajectory_id",
    ),
) -> EpisodeSplit:
    """Split parent groups after canonical sorting and a source-local PCG64 shuffle."""

    rows = list(records)
    if not rows:
        raise ValueError(f"source {source!r} contains no eligible episodes")
    episode_ids: list[str] = []
    for row in rows:
        if str(row.get("source")) != source:
            raise ValueError("split input contains records from another source")
        episode_id = str(row.get("episode_id", ""))
        if not episode_id:
            raise ValueError("eligible record has no episode_id")
        episode_ids.append(episode_id)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError(f"source {source!r} contains duplicate episode IDs")

    parent_field = _choose_parent_field(rows, parent_field_candidates)
    groups: dict[str, list[str]] = {}
    for row, episode_id in zip(rows, episode_ids):
        parent = (
            str(row[parent_field])
            if parent_field is not None and row.get(parent_field) not in (None, "")
            else episode_id
        )
        groups.setdefault(parent, []).append(episode_id)
    for values in groups.values():
        values.sort()

    parent_ids = sorted(groups)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    order = rng.permutation(len(parent_ids)).tolist()
    shuffled = [parent_ids[index] for index in order]
    target = requested_holdout_count(len(rows))

    validation_parents: list[str] = []
    test_parents: list[str] = []
    consumed = 0
    for parent in shuffled:
        if consumed >= target:
            break
        validation_parents.append(parent)
        consumed += len(groups[parent])
    consumed = 0
    for parent in shuffled[len(validation_parents) :]:
        if consumed >= target:
            break
        test_parents.append(parent)
        consumed += len(groups[parent])
    held_out = set(validation_parents) | set(test_parents)
    train_parents = [parent for parent in shuffled if parent not in held_out]

    def episodes(parents: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            episode
            for parent in parents
            for episode in groups[parent]
        )

    result = EpisodeSplit(
        source=source,
        train=episodes(train_parents),
        validation=episodes(validation_parents),
        test=episodes(test_parents),
        parent_field=parent_field,
    )
    all_ids = set(result.train) | set(result.validation) | set(result.test)
    if len(all_ids) != len(rows) or all_ids != set(episode_ids):
        raise RuntimeError("episode split is not a complete disjoint partition")
    return result


def is_v1_eligible_manifest_record(record: Mapping[str, object]) -> bool:
    """Eligibility used by the audited inventory: train label and 90% time span."""

    if record.get("split") != "train":
        return False
    clock = record.get("observation_clock")
    if not isinstance(clock, Mapping):
        return False
    try:
        start = float(clock["start_s"])
        end = float(clock["end_s"])
        samples = int(clock["sample_count"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(start) and np.isfinite(end) and samples >= 24 and end - start >= 4.8 * 0.9 - 1.0e-6)
