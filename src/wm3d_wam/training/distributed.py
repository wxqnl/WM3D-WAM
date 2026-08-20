"""NCCL/FSDP runtime contract for New-H100-2."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
import random
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from wm3d_wam.models.wan_action_mot import WanActionMoT
from wm3d_wam.vendor.fastwam.wan22.wan_video_vae import WanVideoVAE38


class DistributedRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    physical_cuda_devices: tuple[int, ...]

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def parse_visible_physical_devices() -> tuple[int, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        raise DistributedRuntimeError(
            "CUDA_VISIBLE_DEVICES must explicitly list allowed physical GPUs"
        )
    try:
        devices = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise DistributedRuntimeError(
            "WM3D-WAM requires numeric physical IDs in CUDA_VISIBLE_DEVICES"
        ) from exc
    if not devices or len(devices) != len(set(devices)):
        raise DistributedRuntimeError("CUDA_VISIBLE_DEVICES is empty or repeats a device")
    if 0 in devices:
        raise DistributedRuntimeError("physical GPU 0 is forbidden for WM3D-WAM")
    if any(device < 1 or device > 7 for device in devices):
        raise DistributedRuntimeError("this server permits only physical GPUs 1-7")
    return devices


def initialize_distributed() -> DistributedContext:
    devices = parse_visible_physical_devices()
    if not torch.cuda.is_available():
        raise DistributedRuntimeError("CUDA is required for production training")
    # NCCL 2.26's NVLS path on this node fails for process groups larger than
    # two ranks with cudaErrorInvalidValue. Regular NVLink collectives pass on
    # the same 1/5/6/7 topology, so default to that path while still allowing
    # an operator to override the setting after the node runtime is repaired.
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != len(devices):
        raise DistributedRuntimeError(
            f"WORLD_SIZE={world_size} but CUDA_VISIBLE_DEVICES names {len(devices)} GPUs"
        )
    if not 0 <= local_rank < len(devices):
        raise DistributedRuntimeError("LOCAL_RANK is outside the visible device list")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
        physical_cuda_devices=devices,
    )


def seed_everything(seed: int, *, rank_offset: bool) -> None:
    value = int(seed) + (dist.get_rank() if rank_offset and dist.is_initialized() else 0)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)


def _transformer_layer_types(_module: nn.Module) -> set[type[nn.Module]]:
    # Only wrap blocks whose complete forward owns every parameter use.
    # TransformerEncoderLayer calls MultiheadAttention.out_proj functionally,
    # while the causal VGGT resume path calls block.norm/qkv/attention directly.
    # Wrapping either block type separately therefore leaves raw weights
    # sharded when those functional paths read them. MoT likewise accesses
    # both experts' DiTBlock internals directly instead of calling each block.
    # Its safe boundary is WanActionMoT.forward, which every routed program
    # invokes exactly once. The root owns the complete geometry path.
    return {WanActionMoT}


def wrap_full_shard(
    module: nn.Module,
    *,
    context: DistributedContext,
    ignored_modules: Iterable[nn.Module] = (),
    use_device_mesh: bool = True,
) -> nn.Module:
    if context.world_size == 1:
        return module
    ignored = list(ignored_modules)
    ignored.extend(
        child for child in module.modules() if isinstance(child, WanVideoVAE38)
    )
    # Preserve order while avoiding duplicate module identities.
    unique_ignored: list[nn.Module] = []
    seen: set[int] = set()
    for child in ignored:
        if id(child) not in seen:
            unique_ignored.append(child)
            seen.add(id(child))
    ignored_parameter_ids = {
        id(parameter)
        for child in unique_ignored
        for parameter in child.parameters()
    }
    # FSDP mixed precision needs full-precision original parameters so that
    # BF16 gradients can be reduced and written back to FP32 master weights.
    # Asset loading stays BF16 to avoid a large construction-time peak; only
    # FSDP-managed parameters are promoted immediately before sharding.  The
    # ignored frozen VAE remains BF16 and is executed under autocast.
    with torch.no_grad():
        for parameter in module.parameters():
            if (
                id(parameter) not in ignored_parameter_ids
                and parameter.is_floating_point()
                and parameter.dtype != torch.float32
            ):
                parameter.data = parameter.data.float()
    auto_wrap = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=_transformer_layer_types(module),
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Reduce the full gradient in BF16, then let FSDP cast only the local
        # reduce-scattered shard back to the FP32 master-parameter dtype.
        # Casting the unsharded Wan+Action gradient first requires an extra
        # 22+ GiB allocation per rank and cannot fit on an 80GB H100.
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    # Stage A uses DeviceMesh so its canonical DCP payload is DTensor-based
    # and can be resharded into the full pipeline. PyTorch 2.7 intentionally
    # rejects LOCAL_STATE_DICT with DeviceMesh, so the much larger full phases
    # use the equivalent default process group and exact local flat shards.
    device_mesh = (
        init_device_mesh(
            "cuda",
            (context.world_size,),
            mesh_dim_names=("fsdp",),
        )
        if use_device_mesh
        else None
    )
    return FSDP(
        module,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        ignored_modules=unique_ignored or None,
        device_id=context.device,
        sync_module_states=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=False,
        limit_all_gathers=True,
        use_orig_params=True,
        device_mesh=device_mesh,
    )


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, FSDP) else module


def reduce_metrics(metrics: dict[str, float], *, device: torch.device) -> dict[str, float]:
    if not metrics:
        return {}
    names = sorted(metrics)
    values = torch.tensor(
        [float(metrics[name]) for name in names],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return {name: float(values[index].item()) for index, name in enumerate(names)}


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def shutdown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        # Successful paths synchronize explicitly before returning. A barrier
        # here can hide the original exception forever when one rank has
        # already failed or exited.
        dist.destroy_process_group()
