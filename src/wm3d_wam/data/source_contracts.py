"""Audited, per-source robot semantics and normalization.

The LeRobot feature name ``action`` is only a storage key.  It does not imply
that two datasets share units, frames, gripper polarity, or even the same
controller space.  This module turns the explicit YAML audit into the padded
grouped ABI consumed by WM3D-WAM.  Sources whose converter provenance is not
yet strong enough are represented in the registry, but cannot be sampled for
policy or dynamics training.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .grouped_robot import (
    ACTION_SEMANTIC_IDS,
    COMPOSITION_OPERATOR_IDS,
    STATE_SEMANTIC_IDS,
)


TRAINING_PROGRAMS = frozenset(
    {
        "geometry_pretrain",
        "action_only",
        "forward_world",
        "joint_world_action",
    }
)


class SourceContractError(ValueError):
    """Raised when source semantics are missing, ambiguous, or inconsistent."""


def _tuple(value: object, *, name: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise SourceContractError(f"{name} must be a YAML list")
    return tuple(value)


def _float_tuple(
    value: object | None,
    *,
    length: int,
    default: float,
    name: str,
) -> tuple[float, ...]:
    if value is None:
        return (float(default),) * length
    result = tuple(float(item) for item in _tuple(value, name=name))
    if len(result) != length or not np.isfinite(result).all():
        raise SourceContractError(f"{name} must contain {length} finite values")
    return result


def _bool_tuple(
    value: object | None,
    *,
    length: int,
    default: bool,
    name: str,
) -> tuple[bool, ...]:
    if value is None:
        return (bool(default),) * length
    result = tuple(bool(item) for item in _tuple(value, name=name))
    if len(result) != length:
        raise SourceContractError(f"{name} must contain {length} booleans")
    return result


@dataclass(frozen=True)
class SourceGroupContract:
    name: str
    group_id: int
    action_indices: tuple[int, ...]
    state_indices: tuple[int, ...]
    action_semantics: tuple[str, ...]
    state_semantics: tuple[str, ...]
    action_frame: str
    state_frame: str
    composition_operators: tuple[str, ...]
    action_scale: tuple[float, ...]
    action_offset: tuple[float, ...]
    state_scale: tuple[float, ...]
    state_offset: tuple[float, ...]
    normalize_action: tuple[bool, ...]
    normalize_state: tuple[bool, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], *, source: str) -> "SourceGroupContract":
        name = str(value.get("name", "")).strip()
        action_indices = tuple(
            int(item)
            for item in _tuple(value.get("action_indices"), name=f"{source}.{name}.action_indices")
        )
        state_indices = tuple(
            int(item)
            for item in _tuple(value.get("state_indices", []), name=f"{source}.{name}.state_indices")
        )
        action_semantics = tuple(
            str(item)
            for item in _tuple(value.get("action_semantics"), name=f"{source}.{name}.action_semantics")
        )
        state_semantics = tuple(
            str(item)
            for item in _tuple(value.get("state_semantics", []), name=f"{source}.{name}.state_semantics")
        )
        operators = tuple(
            str(item)
            for item in _tuple(
                value.get("composition_operators"),
                name=f"{source}.{name}.composition_operators",
            )
        )
        result = cls(
            name=name,
            group_id=int(value.get("group_id", 0)),
            action_indices=action_indices,
            state_indices=state_indices,
            action_semantics=action_semantics,
            state_semantics=state_semantics,
            action_frame=str(value.get("action_frame", "")).strip(),
            state_frame=str(value.get("state_frame", "")).strip(),
            composition_operators=operators,
            action_scale=_float_tuple(
                value.get("action_scale"),
                length=len(action_indices),
                default=1.0,
                name=f"{source}.{name}.action_scale",
            ),
            action_offset=_float_tuple(
                value.get("action_offset"),
                length=len(action_indices),
                default=0.0,
                name=f"{source}.{name}.action_offset",
            ),
            state_scale=_float_tuple(
                value.get("state_scale"),
                length=len(state_indices),
                default=1.0,
                name=f"{source}.{name}.state_scale",
            ),
            state_offset=_float_tuple(
                value.get("state_offset"),
                length=len(state_indices),
                default=0.0,
                name=f"{source}.{name}.state_offset",
            ),
            normalize_action=_bool_tuple(
                value.get("normalize_action"),
                length=len(action_indices),
                default=True,
                name=f"{source}.{name}.normalize_action",
            ),
            normalize_state=_bool_tuple(
                value.get("normalize_state"),
                length=len(state_indices),
                default=True,
                name=f"{source}.{name}.normalize_state",
            ),
        )
        result.validate(source=source)
        return result

    def validate(self, *, source: str) -> None:
        prefix = f"{source}.{self.name or '<unnamed>'}"
        if not self.name or self.group_id <= 0:
            raise SourceContractError(f"{prefix} requires a name and positive group_id")
        if not self.action_indices:
            raise SourceContractError(f"{prefix} contains no action channels")
        if len(self.action_indices) != len(self.action_semantics):
            raise SourceContractError(f"{prefix} action index/semantic cardinality differs")
        if len(self.state_indices) != len(self.state_semantics):
            raise SourceContractError(f"{prefix} state index/semantic cardinality differs")
        if len(self.composition_operators) != len(self.action_indices):
            raise SourceContractError(f"{prefix} requires one composition operator per action channel")
        if any(index < 0 for index in (*self.action_indices, *self.state_indices)):
            raise SourceContractError(f"{prefix} contains a negative raw column index")
        if len(set(self.action_indices)) != len(self.action_indices):
            raise SourceContractError(f"{prefix} repeats an action column")
        if len(set(self.state_indices)) != len(self.state_indices):
            raise SourceContractError(f"{prefix} repeats a state column")
        unknown_action = set(self.action_semantics) - set(ACTION_SEMANTIC_IDS)
        unknown_state = set(self.state_semantics) - set(STATE_SEMANTIC_IDS)
        unknown_operator = set(self.composition_operators) - set(COMPOSITION_OPERATOR_IDS)
        if unknown_action or unknown_state or unknown_operator:
            raise SourceContractError(
                f"{prefix} has unknown semantics/operators: "
                f"action={sorted(unknown_action)}, state={sorted(unknown_state)}, "
                f"operators={sorted(unknown_operator)}"
            )
        if not self.action_frame or (self.state_indices and not self.state_frame):
            raise SourceContractError(f"{prefix} must declare action/state coordinate frames")
        if any(abs(scale) < 1.0e-12 for scale in (*self.action_scale, *self.state_scale)):
            raise SourceContractError(f"{prefix} contains a non-invertible affine scale")


@dataclass(frozen=True)
class SourceContract:
    name: str
    source_id: int
    family: str
    embodiment: str
    embodiment_id: int
    source_hz: int
    raw_action_dim: int
    raw_state_dim: int
    status: str
    programs: tuple[str, ...]
    quality_weight: float
    groups: tuple[SourceGroupContract, ...]
    ignored_action_indices: tuple[int, ...]
    ignored_state_indices: tuple[int, ...]
    exclusion_reason: str
    evidence: tuple[str, ...]

    @property
    def trainable(self) -> bool:
        return self.status == "verified"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SourceContract":
        name = str(value.get("name", "")).strip()
        groups_value = value.get("groups", [])
        if not isinstance(groups_value, list):
            raise SourceContractError(f"{name}.groups must be a list")
        result = cls(
            name=name,
            source_id=int(value.get("source_id", -1)),
            family=str(value.get("family", "")).strip(),
            embodiment=str(value.get("embodiment", "")).strip(),
            embodiment_id=int(value.get("embodiment_id", 0)),
            source_hz=int(value.get("source_hz", 0)),
            raw_action_dim=int(value.get("raw_action_dim", 0)),
            raw_state_dim=int(value.get("raw_state_dim", 0)),
            status=str(value.get("status", "")).strip(),
            programs=tuple(
                str(item)
                for item in _tuple(value.get("programs", []), name=f"{name}.programs")
            ),
            quality_weight=float(value.get("quality_weight", 1.0)),
            groups=tuple(
                SourceGroupContract.from_mapping(item, source=name)
                for item in groups_value
                if isinstance(item, Mapping)
            ),
            ignored_action_indices=tuple(
                int(item)
                for item in _tuple(
                    value.get("ignored_action_indices", []),
                    name=f"{name}.ignored_action_indices",
                )
            ),
            ignored_state_indices=tuple(
                int(item)
                for item in _tuple(
                    value.get("ignored_state_indices", []),
                    name=f"{name}.ignored_state_indices",
                )
            ),
            exclusion_reason=str(value.get("exclusion_reason", "")).strip(),
            evidence=tuple(
                str(item)
                for item in _tuple(value.get("evidence", []), name=f"{name}.evidence")
            ),
        )
        if len(result.groups) != len(groups_value):
            raise SourceContractError(f"{name}.groups contains a non-mapping entry")
        result.validate()
        return result

    def validate(self) -> None:
        if not self.name or self.source_id < 0 or self.embodiment_id <= 0:
            raise SourceContractError(f"source {self.name!r} has invalid identity fields")
        if self.family not in {"oxe", "robocasa"}:
            raise SourceContractError(f"{self.name}.family must be oxe or robocasa")
        if self.source_hz not in {5, 10, 15, 20}:
            raise SourceContractError(f"{self.name}.source_hz is unsupported")
        if self.raw_action_dim <= 0 or self.raw_state_dim <= 0:
            raise SourceContractError(f"{self.name} raw dimensions must be positive")
        if self.status not in {"verified", "excluded"}:
            raise SourceContractError(f"{self.name}.status must be verified or excluded")
        if not np.isfinite(self.quality_weight) or not 0.0 < self.quality_weight <= 1.0:
            raise SourceContractError(f"{self.name}.quality_weight must be in (0,1]")
        if self.status == "excluded":
            if self.programs or self.groups or not self.exclusion_reason:
                raise SourceContractError(
                    f"excluded source {self.name} must have no programs/groups and a reason"
                )
            return
        if not self.groups or not self.programs:
            raise SourceContractError(f"verified source {self.name} requires groups and programs")
        unknown_programs = set(self.programs) - TRAINING_PROGRAMS
        if unknown_programs:
            raise SourceContractError(f"{self.name} has unknown programs {sorted(unknown_programs)}")
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise SourceContractError(f"{self.name} repeats a group_id")
        action_columns = [index for group in self.groups for index in group.action_indices]
        state_columns = [index for group in self.groups for index in group.state_indices]
        if len(action_columns) != len(set(action_columns)):
            raise SourceContractError(f"{self.name} assigns an action column twice")
        if len(state_columns) != len(set(state_columns)):
            raise SourceContractError(f"{self.name} assigns a state column twice")
        if set(action_columns) | set(self.ignored_action_indices) != set(range(self.raw_action_dim)):
            raise SourceContractError(f"{self.name} does not explicitly account for every action column")
        if set(state_columns) | set(self.ignored_state_indices) != set(range(self.raw_state_dim)):
            raise SourceContractError(f"{self.name} does not explicitly account for every state column")
        if set(action_columns) & set(self.ignored_action_indices):
            raise SourceContractError(f"{self.name} both uses and ignores an action column")
        if set(state_columns) & set(self.ignored_state_indices):
            raise SourceContractError(f"{self.name} both uses and ignores a state column")
        if max(action_columns) >= self.raw_action_dim or (state_columns and max(state_columns) >= self.raw_state_dim):
            raise SourceContractError(f"{self.name} maps beyond its declared raw dimensions")


@dataclass(frozen=True)
class SourceContractRegistry:
    sources: Mapping[str, SourceContract]

    @classmethod
    def load(cls, path: str | Path) -> "SourceContractRegistry":
        path = Path(path).expanduser().resolve(strict=True)
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        rows = value.get("sources") if isinstance(value, Mapping) else None
        if not isinstance(rows, list) or not rows:
            raise SourceContractError("source contract file contains no sources")
        contracts = [
            SourceContract.from_mapping(row)
            for row in rows
            if isinstance(row, Mapping)
        ]
        if len(contracts) != len(rows):
            raise SourceContractError("source contract list contains a non-mapping entry")
        by_name = {contract.name: contract for contract in contracts}
        if len(by_name) != len(contracts):
            raise SourceContractError("source contract names are not unique")
        ids = [contract.source_id for contract in contracts]
        embodiment_ids = [contract.embodiment_id for contract in contracts]
        if len(ids) != len(set(ids)) or len(embodiment_ids) != len(set(embodiment_ids)):
            raise SourceContractError("source_id and embodiment_id must be globally unique")
        expected = value.get("expected_source_count")
        if expected is not None and int(expected) != len(contracts):
            raise SourceContractError(
                f"expected {int(expected)} source contracts, found {len(contracts)}"
            )
        return cls(sources=by_name)

    def require(self, source: str, *, program: str | None = None) -> SourceContract:
        try:
            contract = self.sources[source]
        except KeyError as exc:
            raise SourceContractError(f"source {source!r} is absent from the audit registry") from exc
        if not contract.trainable:
            raise SourceContractError(
                f"source {source!r} is excluded: {contract.exclusion_reason}"
            )
        if program is not None and program not in contract.programs:
            raise SourceContractError(
                f"source {source!r} is not approved for program {program!r}"
            )
        return contract

    def approved(self, program: str) -> tuple[SourceContract, ...]:
        if program not in TRAINING_PROGRAMS:
            raise SourceContractError(f"unknown training program {program!r}")
        return tuple(
            contract
            for contract in self.sources.values()
            if contract.trainable and program in contract.programs
        )


@dataclass(frozen=True)
class NormalizationStat:
    offset: float
    scale: float


@dataclass(frozen=True)
class NormalizationRegistry:
    rows: Mapping[tuple[str, str, int], NormalizationStat]

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationRegistry":
        path = Path(path).expanduser().resolve(strict=True)
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("rows") if isinstance(value, Mapping) else None
        if not isinstance(rows, list) or not rows:
            raise SourceContractError("normalization file contains no rows")
        output: dict[tuple[str, str, int], NormalizationStat] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise SourceContractError("normalization row is not a mapping")
            key = (str(row["source"]), str(row["kind"]), int(row["dimension"]))
            if key in output:
                raise SourceContractError(f"duplicate normalization row {key}")
            offset = float(row["offset"])
            scale = float(row["scale"])
            if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0.0:
                raise SourceContractError(f"invalid normalization row {key}")
            output[key] = NormalizationStat(offset=offset, scale=scale)
        return cls(rows=output)

    def require(self, source: str, kind: str, dimension: int) -> NormalizationStat:
        key = (source, kind, int(dimension))
        try:
            return self.rows[key]
        except KeyError as exc:
            raise SourceContractError(f"normalization is missing row {key}") from exc


@dataclass(frozen=True)
class PackedRobotArrays:
    action_values: np.ndarray       # [N,G,D_a]
    action_value_mask: np.ndarray   # [N,G,D_a]
    state_values: np.ndarray        # [N,G,D_s]
    state_value_mask: np.ndarray    # [N,G,D_s]
    group_ids: np.ndarray           # [G]
    group_mask: np.ndarray          # [G]
    action_semantic_ids: np.ndarray # [G,D_a]
    state_semantic_ids: np.ndarray  # [G,D_s]
    composition_operator_ids: np.ndarray # [G,D_a]


def _normalize_selected(
    raw: np.ndarray,
    *,
    source: str,
    kind: str,
    indices: Sequence[int],
    affine_scale: Sequence[float],
    affine_offset: Sequence[float],
    normalize: Sequence[bool],
    registry: NormalizationRegistry,
) -> np.ndarray:
    selected = raw[:, np.asarray(indices, dtype=np.int64)].astype(np.float32, copy=True)
    scale_array = np.asarray(affine_scale, dtype=np.float32)
    offset_array = np.asarray(affine_offset, dtype=np.float32)
    selected = selected * scale_array + offset_array
    for local, (dimension, enabled) in enumerate(zip(indices, normalize)):
        if not enabled:
            continue
        stat = registry.require(source, kind, int(dimension))
        transformed_mean = stat.offset * float(affine_scale[local]) + float(affine_offset[local])
        transformed_scale = stat.scale * abs(float(affine_scale[local]))
        selected[:, local] = (selected[:, local] - transformed_mean) / transformed_scale
    if not np.isfinite(selected).all():
        raise SourceContractError(f"normalized {source}.{kind} contains NaN/Inf")
    return selected


def pack_robot_arrays(
    actions: np.ndarray,
    states: np.ndarray,
    *,
    contract: SourceContract,
    normalization: NormalizationRegistry,
    max_groups: int,
    max_action_dim: int,
    max_state_dim: int,
) -> PackedRobotArrays:
    """Map raw flat arrays to audited semantic groups and train normalization."""

    if not contract.trainable:
        raise SourceContractError(f"cannot pack excluded source {contract.name!r}")
    actions = np.asarray(actions)
    states = np.asarray(states)
    if actions.ndim != 2 or actions.shape[1] != contract.raw_action_dim:
        raise SourceContractError(
            f"{contract.name} action shape {actions.shape} does not match raw dim {contract.raw_action_dim}"
        )
    if states.ndim != 2 or states.shape != (actions.shape[0], contract.raw_state_dim):
        raise SourceContractError(
            f"{contract.name} state shape {states.shape} does not match rows/dim"
        )
    if len(contract.groups) > max_groups:
        raise SourceContractError(f"{contract.name} exceeds max_groups={max_groups}")
    row_count = actions.shape[0]
    action_values = np.zeros((row_count, max_groups, max_action_dim), dtype=np.float32)
    action_mask = np.zeros_like(action_values, dtype=np.bool_)
    state_values = np.zeros((row_count, max_groups, max_state_dim), dtype=np.float32)
    state_mask = np.zeros_like(state_values, dtype=np.bool_)
    group_ids = np.zeros((max_groups,), dtype=np.int64)
    group_mask = np.zeros((max_groups,), dtype=np.bool_)
    action_semantics = np.zeros((max_groups, max_action_dim), dtype=np.int64)
    state_semantics = np.zeros((max_groups, max_state_dim), dtype=np.int64)
    composition = np.zeros((max_groups, max_action_dim), dtype=np.int64)
    for slot, group in enumerate(contract.groups):
        action_dim = len(group.action_indices)
        state_dim = len(group.state_indices)
        if action_dim > max_action_dim or state_dim > max_state_dim:
            raise SourceContractError(
                f"{contract.name}.{group.name} exceeds grouped dimension capacity"
            )
        action_values[:, slot, :action_dim] = _normalize_selected(
            actions,
            source=contract.name,
            kind="action",
            indices=group.action_indices,
            affine_scale=group.action_scale,
            affine_offset=group.action_offset,
            normalize=group.normalize_action,
            registry=normalization,
        )
        action_mask[:, slot, :action_dim] = True
        if state_dim:
            state_values[:, slot, :state_dim] = _normalize_selected(
                states,
                source=contract.name,
                kind="state",
                indices=group.state_indices,
                affine_scale=group.state_scale,
                affine_offset=group.state_offset,
                normalize=group.normalize_state,
                registry=normalization,
            )
            state_mask[:, slot, :state_dim] = True
        group_ids[slot] = group.group_id
        group_mask[slot] = True
        action_semantics[slot, :action_dim] = [
            ACTION_SEMANTIC_IDS[name] for name in group.action_semantics
        ]
        state_semantics[slot, :state_dim] = [
            STATE_SEMANTIC_IDS[name] for name in group.state_semantics
        ]
        composition[slot, :action_dim] = [
            COMPOSITION_OPERATOR_IDS[name] for name in group.composition_operators
        ]
    return PackedRobotArrays(
        action_values=action_values,
        action_value_mask=action_mask,
        state_values=state_values,
        state_value_mask=state_mask,
        group_ids=group_ids,
        group_mask=group_mask,
        action_semantic_ids=action_semantics,
        state_semantic_ids=state_semantics,
        composition_operator_ids=composition,
    )


def denormalize_action_values(
    grouped_values: np.ndarray,
    *,
    contract: SourceContract,
    normalization: NormalizationRegistry,
) -> np.ndarray:
    """Invert grouped model values into the source's flat controller layout."""

    values = np.asarray(grouped_values, dtype=np.float32)
    if values.ndim < 2 or values.shape[-2] < len(contract.groups):
        raise SourceContractError("grouped action array has insufficient group capacity")
    output = np.zeros((*values.shape[:-2], contract.raw_action_dim), dtype=np.float32)
    for slot, group in enumerate(contract.groups):
        selected = values[..., slot, : len(group.action_indices)].copy()
        for local, (dimension, enabled) in enumerate(
            zip(group.action_indices, group.normalize_action)
        ):
            if enabled:
                stat = normalization.require(contract.name, "action", dimension)
                transformed_mean = stat.offset * group.action_scale[local] + group.action_offset[local]
                transformed_scale = stat.scale * abs(group.action_scale[local])
                selected[..., local] = selected[..., local] * transformed_scale + transformed_mean
            selected[..., local] = (
                selected[..., local] - group.action_offset[local]
            ) / group.action_scale[local]
            output[..., dimension] = selected[..., local]
    return output


def validate_profile_coverage(
    profile_sources: Iterable[Mapping[str, object]],
    registry: SourceContractRegistry,
) -> None:
    """Require the data profile and semantic registry to name the same sources."""

    names = [str(row.get("name", "")) for row in profile_sources]
    if len(names) != len(set(names)):
        raise SourceContractError("data profile repeats a source name")
    missing = set(names) - set(registry.sources)
    stale = set(registry.sources) - set(names)
    if missing or stale:
        raise SourceContractError(
            f"profile/contract source mismatch: missing={sorted(missing)}, stale={sorted(stale)}"
        )
