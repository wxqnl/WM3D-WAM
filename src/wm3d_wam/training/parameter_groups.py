"""Exact Stage A/B/C trainability and learning-rate ownership."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

import torch
import torch.nn as nn

from wm3d_wam.models.system import WM3DWAMSystem
from wm3d_wam.models.online_vggt_geometry import OnlineVGGTGeometryCore


class TrainingStage(str, Enum):
    WORLD_CORE_PRETRAIN = "world_core_pretrain"
    WAN_ACTION_WARMUP = "wan_action_warmup"
    WAN_ACTION_MAIN = "wan_action_main"
    TRI_STREAM_ALIGNMENT = "tri_stream_alignment"


def _set_trainable(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


def _unique_parameters(values: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    output: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in values:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        output.append(parameter)
    return output


def _world_core_parameters(
    geometry: OnlineVGGTGeometryCore,
) -> list[torch.nn.Parameter]:
    return _unique_parameters(
        list(geometry.state_dynamics.parameters())
        + list(geometry.history_connector.parameters())
        + list(geometry.geometry_reducer.parameters())
    )


def configure_world_core_pretraining_parameter_groups(
    geometry: OnlineVGGTGeometryCore,
    *,
    weight_decay: float = 0.01,
) -> list[dict[str, object]]:
    """Stage-A groups: WM3D state dynamics plus VGGT deep pairs."""

    geometry.requires_grad_(False)
    world_core_parameters = _world_core_parameters(geometry)
    for parameter in world_core_parameters:
        parameter.requires_grad = True
    encoder = geometry.encoder
    encoder.freeze_blocks_before(encoder.split_layer)
    for head in (encoder.dpt_head, encoder.point_head, encoder.camera_head):
        _set_trainable(head, False)
    deep_parameters = _unique_parameters(
        parameter
        for index in range(encoder.split_layer, encoder.block_count)
        for block in (
            encoder.aggregator.frame_blocks[index],
            encoder.aggregator.global_blocks[index],
        )
        for parameter in block.parameters()
    )
    groups = [
        {
            "name": "wm3d_state_dynamics",
            "params": world_core_parameters,
            "lr": 1.0e-5,
            "weight_decay": float(weight_decay),
        },
        {
            "name": "vggt_deep",
            "params": deep_parameters,
            "lr": 1.0e-5,
            "weight_decay": float(weight_decay),
        },
    ]
    assigned = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in geometry.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in assigned} != {id(parameter) for parameter in trainable}:
        raise RuntimeError("Stage-A groups do not exactly cover trainable geometry parameters")
    return groups


def configure_stage_parameter_groups(
    system: WM3DWAMSystem,
    stage: TrainingStage | str,
    *,
    weight_decay: float = 0.01,
) -> list[dict[str, object]]:
    """Freeze the full system, enable the design-specified owners, and group LR."""

    stage = TrainingStage(stage)
    system.requires_grad_(False)
    geometry = system.geometry_core
    encoder = geometry.encoder

    world_core_parameters = _world_core_parameters(geometry)
    for parameter in world_core_parameters:
        parameter.requires_grad = True

    enable_deep = stage in {
        TrainingStage.WORLD_CORE_PRETRAIN,
        TrainingStage.WAN_ACTION_MAIN,
        TrainingStage.TRI_STREAM_ALIGNMENT,
    }
    encoder.freeze_blocks_before(
        encoder.split_layer if enable_deep else encoder.block_count
    )
    # VGGT geometry heads stay fixed; gradients still propagate through them
    # into trainable deep pairs for supervised depth/point/pose objectives.
    for head in (encoder.dpt_head, encoder.point_head, encoder.camera_head):
        _set_trainable(head, False)

    groups: list[dict[str, object]] = []

    def append_group(name: str, parameters: Iterable[torch.nn.Parameter], lr: float) -> None:
        params = [parameter for parameter in _unique_parameters(parameters) if parameter.requires_grad]
        if params:
            groups.append(
                {
                    "name": name,
                    "params": params,
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                }
            )

    world_core_lr = {
        TrainingStage.WORLD_CORE_PRETRAIN: 1.0e-5,
        TrainingStage.WAN_ACTION_WARMUP: 1.0e-5,
        TrainingStage.WAN_ACTION_MAIN: 1.0e-5,
        TrainingStage.TRI_STREAM_ALIGNMENT: 5.0e-6,
    }[stage]
    append_group("wm3d_state_dynamics", world_core_parameters, world_core_lr)
    if enable_deep:
        deep_lr = (
            1.0e-5
            if stage is TrainingStage.WORLD_CORE_PRETRAIN
            else 5.0e-6
        )
        append_group(
            "vggt_deep",
            (
                parameter
                for index in range(encoder.split_layer, encoder.block_count)
                for block in (
                    encoder.aggregator.frame_blocks[index],
                    encoder.aggregator.global_blocks[index],
                )
                for parameter in block.parameters()
            ),
            deep_lr,
        )

    if stage is not TrainingStage.WORLD_CORE_PRETRAIN:
        _set_trainable(system.wan_action.action_expert, True)
        append_group(
            "action_expert",
            system.wan_action.action_expert.parameters(),
            1.0e-5
            if stage is TrainingStage.TRI_STREAM_ALIGNMENT
            else 2.0e-5,
        )
        if system.wan_action.geometry_adapters is not None:
            _set_trainable(system.wan_action.geometry_adapters, True)
            append_group(
                "geometry_adapters",
                system.wan_action.geometry_adapters.parameters(),
                1.0e-5,
            )

    enable_wan = stage in {
        TrainingStage.WAN_ACTION_MAIN,
        TrainingStage.TRI_STREAM_ALIGNMENT,
    }
    if enable_wan:
        _set_trainable(system.wan_action.video_expert, True)
        append_group(
            "wan_dit",
            system.wan_action.video_expert.parameters(),
            1.0e-6
            if stage is TrainingStage.TRI_STREAM_ALIGNMENT
            else 2.0e-6,
        )

    system.video_vae.requires_grad_(False).eval()
    assigned = [parameter for group in groups for parameter in group["params"]]
    if len({id(parameter) for parameter in assigned}) != len(assigned):
        raise RuntimeError("a trainable parameter was assigned to multiple optimizer groups")
    trainable = [parameter for parameter in system.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in assigned} != {id(parameter) for parameter in trainable}:
        raise RuntimeError("optimizer groups do not exactly cover trainable parameters")
    return groups
