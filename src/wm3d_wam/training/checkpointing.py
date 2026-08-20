"""Distributed numbered checkpoints with exact rank-local runtime recovery."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    LocalStateDictConfig,
    StateDictType,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)


CANONICAL_DCP_SCHEMA = "wm3d_wam_checkpoint_v1"
LOCAL_FSDP_SCHEMA = "wm3d_wam_local_fsdp_checkpoint_v2"


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResumeState:
    checkpoint_path: Path
    global_step: int
    next_local_sample_index: int
    metadata: Mapping[str, Any]


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + f".tmp-rank{_rank():03d}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp-rank{_rank():03d}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Please use DTensor instead and we are deprecating ShardedTensor.*",
            category=FutureWarning,
        )
        torch.save(value, temporary)
    os.replace(temporary, path)


@contextmanager
def _local_state_dict_context(model: torch.nn.Module):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FSDP.state_dict_type.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Please use DTensor instead and we are deprecating ShardedTensor.*",
            category=FutureWarning,
        )
        if not FSDP.fsdp_modules(model):
            yield
        else:
            with FSDP.state_dict_type(
                model,
                StateDictType.LOCAL_STATE_DICT,
                LocalStateDictConfig(offload_to_cpu=False),
            ):
                yield


def _load_local_model_state(
    model: torch.nn.Module,
    state_dict: Mapping[str, object],
) -> None:
    # FSDP1 with use_orig_params=True emits both the canonical original
    # parameter views and a redundant ShardedTensor ``_flat_param`` per FSDP
    # boundary. PyTorch 2.7 loads every original view correctly but reports
    # those redundant flat keys as unexpected. Permit exactly that known set;
    # every missing key or any other unexpected key remains fatal.
    expected_flat_keys = {
        name for name in state_dict if name.rsplit(".", 1)[-1] == "_flat_param"
    }
    with _local_state_dict_context(model):
        incompatible = model.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]
    unexpected = set(incompatible.unexpected_keys)
    if incompatible.missing_keys or unexpected != expected_flat_keys:
        raise CheckpointError(
            "rank-local model state mismatch: "
            f"missing={sorted(incompatible.missing_keys)}, "
            f"unexpected={sorted(unexpected)}, "
            f"expected_flat={sorted(expected_flat_keys)}"
        )


def _uses_local_fsdp_checkpoint(phase: str) -> bool:
    # Stage A is small enough to retain a canonical DCP checkpoint and must be
    # reshardable into the structurally different full pipeline. The full
    # Wan/Action phases keep the same model and device mesh, so an exact local
    # flat-shard checkpoint avoids FSDP1's prohibitively expensive canonical
    # optimizer-state reconstruction for the 11B-parameter MoT.
    return str(phase) != "world_core_pretrain"


def _capture_rng() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return state


def _restore_rng(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    if torch.cuda.is_available():
        cuda_state = state.get("torch_cuda")
        if not isinstance(cuda_state, torch.Tensor):
            raise CheckpointError("rank checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state(cuda_state, torch.cuda.current_device())


def checkpoint_directory(root: str | Path, global_step: int) -> Path:
    if int(global_step) < 0:
        raise CheckpointError("global_step must be non-negative")
    return Path(root).expanduser().resolve() / f"step_{int(global_step):08d}"


def prune_completed_checkpoints(
    root: str | Path,
    *,
    keep_last: int,
) -> tuple[Path, ...]:
    """Remove only older, completed numbered checkpoints on rank zero.

    A directory is eligible only when its name and completion metadata agree.
    Incomplete checkpoint directories are intentionally left untouched for
    diagnosis. ``keep_last=0`` disables retention and performs no deletion.
    """

    if int(keep_last) < 0:
        raise CheckpointError("keep_last must be non-negative")
    if int(keep_last) == 0 or _rank() != 0:
        return ()
    checkpoint_root = Path(root).expanduser().resolve(strict=True)
    candidates: list[tuple[int, Path]] = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        suffix = child.name.removeprefix("step_")
        if len(suffix) != 8 or not suffix.isdecimal():
            continue
        metadata_path = child / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        step = int(suffix)
        if metadata.get("global_step") != step or metadata.get("schema") not in {
            CANONICAL_DCP_SCHEMA,
            LOCAL_FSDP_SCHEMA,
        }:
            continue
        candidates.append((step, child))
    candidates.sort(key=lambda item: item[0])
    stale = candidates[: max(0, len(candidates) - int(keep_last))]
    removed: list[Path] = []
    for _, path in stale:
        if path.parent != checkpoint_root:
            raise CheckpointError(f"refusing to prune checkpoint outside root: {path}")
        shutil.rmtree(path)
        removed.append(path)
    return tuple(removed)


def save_checkpoint(
    *,
    root: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    phase: str,
    global_step: int,
    next_local_sample_index: int,
    seed: int,
    gradient_accumulation_steps: int,
    micro_batch_size: int,
    extra_metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write tensor payloads first and ``metadata.json`` as completion marker."""

    if int(gradient_accumulation_steps) <= 0 or int(micro_batch_size) <= 0:
        raise CheckpointError("checkpoint batch controls must be positive")
    path = checkpoint_directory(root, global_step)
    if _rank() == 0:
        if path.exists():
            raise CheckpointError(f"refusing to overwrite checkpoint {path}")
        path.mkdir(parents=True, exist_ok=False)
    _barrier()
    rng = _capture_rng()
    local_fsdp = _uses_local_fsdp_checkpoint(phase)
    if local_fsdp:
        # LOCAL_STATE_DICT exposes the already-resident flat FSDP shards and
        # optimizer.state_dict() preserves the matching rank-local slots. This
        # path performs no canonical per-parameter reconstruction or all-gather.
        with _local_state_dict_context(model):
            model_state = model.state_dict()
        _atomic_torch_save(
            path / f"model_rank_{_rank():03d}.pt",
            {"model": model_state},
        )
        del model_state
        _atomic_torch_save(
            path / f"optimizer_rank_{_rank():03d}.pt",
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
        )
    else:
        # Stage A remains canonical and reshardable for the geometry-only to
        # full-model transition.
        options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=False,
            strict=True,
        )
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        payload: dict[str, object] = {
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler.state_dict(),
        }
        dcp.save(payload, checkpoint_id=path)
    runtime = {
        "rank": _rank(),
        "world_size": _world_size(),
        "seed": int(seed),
        "next_local_sample_index": int(next_local_sample_index),
        "rng": rng,
    }
    runtime_path = path / f"runtime_rank_{_rank():03d}.pt"
    temporary = runtime_path.with_suffix(".pt.tmp")
    torch.save(runtime, temporary)
    os.replace(temporary, runtime_path)
    _barrier()
    if _rank() == 0:
        metadata: dict[str, object] = {
            "schema": LOCAL_FSDP_SCHEMA if local_fsdp else CANONICAL_DCP_SCHEMA,
            "storage": "rank_local_fsdp" if local_fsdp else "canonical_dcp",
            "phase": str(phase),
            "global_step": int(global_step),
            "world_size": _world_size(),
            "seed": int(seed),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "micro_batch_size": int(micro_batch_size),
            "next_local_sample_index": int(next_local_sample_index),
        }
        if extra_metadata:
            overlap = set(metadata).intersection(extra_metadata)
            if overlap:
                raise CheckpointError(
                    f"extra checkpoint metadata overrides reserved keys: {sorted(overlap)}"
                )
            metadata.update(extra_metadata)
        _atomic_json(path / "metadata.json", metadata)
        latest = Path(root).expanduser().resolve() / "latest.txt"
        temporary_latest = latest.with_name("latest.txt.tmp")
        temporary_latest.write_text(path.name + "\n", encoding="utf-8")
        os.replace(temporary_latest, latest)
    _barrier()
    # Keep the uninterrupted trajectory at exactly the state represented by
    # the checkpoint.  Checkpoint I/O is not allowed to perturb RNG streams.
    _restore_rng(rng)
    return path


def resolve_checkpoint(path_or_root: str | Path) -> Path:
    path = Path(path_or_root).expanduser().resolve(strict=True)
    if (path / "metadata.json").is_file():
        return path
    latest = path / "latest.txt"
    if not latest.is_file():
        raise CheckpointError(f"no completed checkpoint or latest.txt at {path}")
    name = latest.read_text(encoding="utf-8").strip()
    if not name or Path(name).name != name:
        raise CheckpointError("latest.txt contains an unsafe checkpoint name")
    result = (path / name).resolve(strict=True)
    if result.parent != path or not (result / "metadata.json").is_file():
        raise CheckpointError("latest.txt does not point to a completed checkpoint")
    return result


def reshard_local_sample_cursor(
    *,
    saved_next_local_sample_index: int,
    saved_world_size: int,
    runtime_world_size: int,
    micro_batch_size: int,
) -> int:
    """Map a canonical checkpoint cursor onto a new distributed world size.

    The sampler has consumed the contiguous global range
    ``[0, saved_next_local_sample_index * saved_world_size)``. A resharded
    continuation is lossless only when that boundary maps to a complete local
    micro-batch on every new rank.
    """

    if (
        int(saved_next_local_sample_index) < 0
        or int(saved_world_size) <= 0
        or int(runtime_world_size) <= 0
        or int(micro_batch_size) <= 0
    ):
        raise CheckpointError("invalid sampler cursor resharding parameters")
    committed_global_samples = int(saved_next_local_sample_index) * int(
        saved_world_size
    )
    next_local_index, remainder = divmod(
        committed_global_samples, int(runtime_world_size)
    )
    if remainder:
        raise CheckpointError(
            "canonical checkpoint sampler cursor cannot be divided across the "
            f"runtime world size: committed_global_samples={committed_global_samples}, "
            f"runtime_world_size={runtime_world_size}"
        )
    if next_local_index % int(micro_batch_size):
        raise CheckpointError(
            "canonical checkpoint sampler cursor is not aligned to a runtime "
            f"micro-batch: next_local_index={next_local_index}, "
            f"micro_batch_size={micro_batch_size}"
        )
    return next_local_index


def load_checkpoint(
    *,
    path_or_root: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_phase: str,
    expected_seed: int,
    expected_gradient_accumulation_steps: int,
    expected_micro_batch_size: int,
    expected_physical_cuda_devices: Sequence[int] | None = None,
) -> ResumeState:
    path = resolve_checkpoint(path_or_root)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    schema = metadata.get("schema")
    saved_world_size = int(metadata.get("world_size", -1))
    runtime_world_size = _world_size()
    expected = {
        "phase": str(expected_phase),
        "seed": int(expected_seed),
        "gradient_accumulation_steps": int(expected_gradient_accumulation_steps),
        "micro_batch_size": int(expected_micro_batch_size),
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise CheckpointError(
                f"checkpoint {name} mismatch: saved={metadata.get(name)!r}, runtime={value!r}"
            )
    if schema == LOCAL_FSDP_SCHEMA:
        if saved_world_size != runtime_world_size:
            raise CheckpointError(
                "rank-local checkpoint world size mismatch: "
                f"saved={saved_world_size}, runtime={runtime_world_size}"
            )
        if expected_physical_cuda_devices is not None and metadata.get(
            "physical_cuda_devices"
        ) != [int(device) for device in expected_physical_cuda_devices]:
            raise CheckpointError(
                "rank-local checkpoint physical CUDA mesh mismatch: "
                f"saved={metadata.get('physical_cuda_devices')!r}, "
                f"runtime={list(expected_physical_cuda_devices)!r}"
            )
        model_path = path / f"model_rank_{_rank():03d}.pt"
        optimizer_path = path / f"optimizer_rank_{_rank():03d}.pt"
        if not model_path.is_file() or not optimizer_path.is_file():
            raise CheckpointError(
                f"rank-local checkpoint payload is missing for rank {_rank()}"
            )
        model_payload = torch.load(
            model_path,
            map_location=f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu",
            weights_only=False,
        )
        if not isinstance(model_payload, Mapping) or "model" not in model_payload:
            raise CheckpointError("rank-local model checkpoint is malformed")
        local_model_state = model_payload["model"]
        if not isinstance(local_model_state, Mapping):
            raise CheckpointError("rank-local model state is not a mapping")
        _load_local_model_state(model, local_model_state)
        del model_payload
        optimizer_payload = torch.load(
            optimizer_path,
            map_location=f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu",
            weights_only=False,
        )
        if not isinstance(optimizer_payload, Mapping):
            raise CheckpointError("rank-local optimizer checkpoint is malformed")
        optimizer.load_state_dict(optimizer_payload["optimizer"])  # type: ignore[arg-type]
        scheduler.load_state_dict(optimizer_payload["scheduler"])  # type: ignore[arg-type]
    elif schema == CANONICAL_DCP_SCHEMA:
        options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=False,
            strict=True,
        )
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        payload: dict[str, object] = {
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler.state_dict(),
        }
        dcp.load(payload, checkpoint_id=path)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=payload["model"],  # type: ignore[arg-type]
            optim_state_dict=payload["optimizer"],  # type: ignore[arg-type]
            options=options,
        )
        scheduler.load_state_dict(payload["scheduler"])  # type: ignore[arg-type]
    else:
        raise CheckpointError(f"unsupported checkpoint schema: {schema!r}")
    runtime_path = path / f"runtime_rank_{_rank():03d}.pt"
    if not runtime_path.is_file():
        raise CheckpointError(f"rank runtime checkpoint is missing: {runtime_path}")
    runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
    if not isinstance(runtime, Mapping):
        raise CheckpointError("rank runtime checkpoint is malformed")
    runtime_expected = {
        "rank": _rank(),
        "world_size": saved_world_size,
        "seed": int(expected_seed),
    }
    for name, value in runtime_expected.items():
        if int(runtime.get(name, -1)) != value:
            raise CheckpointError(
                f"rank runtime {name} mismatch: saved={runtime.get(name)!r}, runtime={value}"
            )
    rng = runtime.get("rng")
    if not isinstance(rng, Mapping):
        raise CheckpointError("rank runtime checkpoint has no RNG mapping")
    _restore_rng(rng)
    saved_next_index = int(runtime.get("next_local_sample_index", -1))
    if saved_next_index < 0 or saved_next_index != int(
        metadata["next_local_sample_index"]
    ):
        raise CheckpointError("rank and global sampler cursors differ")
    next_index = (
        saved_next_index
        if saved_world_size == runtime_world_size
        else reshard_local_sample_cursor(
            saved_next_local_sample_index=saved_next_index,
            saved_world_size=saved_world_size,
            runtime_world_size=runtime_world_size,
            micro_batch_size=expected_micro_batch_size,
        )
    )
    return ResumeState(
        checkpoint_path=path,
        global_step=int(metadata["global_step"]),
        next_local_sample_index=next_index,
        metadata=metadata,
    )


def load_model_only(
    *,
    path_or_root: str | Path,
    model: torch.nn.Module,
    expected_physical_cuda_devices: Sequence[int] | None = None,
) -> Mapping[str, Any]:
    """Load only model tensors for a deliberate cross-phase transition."""

    path = resolve_checkpoint(path_or_root)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") == LOCAL_FSDP_SCHEMA:
        if int(metadata.get("world_size", -1)) != _world_size():
            raise CheckpointError(
                "rank-local cross-phase initialization requires the saved world size"
            )
        if expected_physical_cuda_devices is not None and metadata.get(
            "physical_cuda_devices"
        ) != [int(device) for device in expected_physical_cuda_devices]:
            raise CheckpointError(
                "rank-local cross-phase initialization requires the saved "
                "ordered physical CUDA mesh"
            )
        model_path = path / f"model_rank_{_rank():03d}.pt"
        if not model_path.is_file():
            raise CheckpointError(
                f"rank-local model checkpoint is missing: {model_path}"
            )
        payload = torch.load(
            model_path,
            map_location=f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu",
            weights_only=False,
        )
        if not isinstance(payload, Mapping) or "model" not in payload:
            raise CheckpointError("rank-local model checkpoint is malformed")
        local_model_state = payload["model"]
        if not isinstance(local_model_state, Mapping):
            raise CheckpointError("rank-local model state is not a mapping")
        _load_local_model_state(model, local_model_state)
        return metadata
    if metadata.get("schema") != CANONICAL_DCP_SCHEMA:
        raise CheckpointError(
            f"unsupported checkpoint schema: {metadata.get('schema')!r}"
        )
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        strict=True,
        broadcast_from_rank0=True,
    )
    model_state = get_model_state_dict(model, options=options)
    payload: dict[str, object] = {"model": model_state}
    dcp.load(payload, checkpoint_id=path)
    set_model_state_dict(
        model,
        payload["model"],  # type: ignore[arg-type]
        options=options,
    )
    return metadata
