"""Factories for the production-width Wan/Action MoT composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from wm3d_wam.vendor.fastwam.wan22.wan_video_dit import WanVideoDiT
from wm3d_wam.vendor.vggt_gam.vggt_encoder import VGGTEncoder

from .geometry_adapters import SparseGeometryKVAdapters
from .grouped_action_flow import GroupedActionFlowExpert, load_grouped_action_backbone
from .grouped_history import GroupedHistoryConnector
from .online_vggt_geometry import OnlineVGGTGeometryCore
from .wan_action_mot import WanActionMoT
from .wm3d_state_dynamics import (
    WM3DStateDynamicsConfig,
    WM3DStateDynamicsCore,
)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve(strict=True)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"model configuration is missing mapping {name!r}")
    result = dict(value)
    result.pop("_target_", None)
    return result


def build_wan_action_mot(
    *,
    video_expert: WanVideoDiT,
    model_config: Mapping[str, Any],
    action_backbone_path: str | Path,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> WanActionMoT:
    """Build grouped ActionDiT beside an already loaded local Wan2.2 DiT."""

    action_config = _section(model_config, "action_expert")
    geometry_config = _section(model_config, "geometry_adapters")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            action_expert = GroupedActionFlowExpert(**action_config)
            geometry_adapters = SparseGeometryKVAdapters(**geometry_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    action_expert = action_expert.to(device=device, dtype=dtype)
    geometry_adapters = geometry_adapters.to(device=device, dtype=dtype)
    load_grouped_action_backbone(action_expert, action_backbone_path)
    video_expert = video_expert.to(device=device, dtype=dtype)
    return WanActionMoT(
        video_expert=video_expert,
        action_expert=action_expert,
        geometry_adapters=geometry_adapters,
        mot_checkpoint_mixed_attn=bool(
            model_config.get("mot_checkpoint_mixed_attn", True)
        ),
    )


def build_online_geometry_core(
    *,
    geometry_config: Mapping[str, Any],
    action_codec_config: Mapping[str, Any],
    vggt_checkpoint: str | Path,
    vggt_source_root: str | Path,
    views_per_timestep: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> OnlineVGGTGeometryCore:
    """Materialize online VGGT around the original-WM3D state core."""

    encoder_config = _section(geometry_config, "encoder")
    encoder_config.update(
        {
            "ckpt_path": str(Path(vggt_checkpoint).expanduser().resolve(strict=True)),
            "source_root": str(Path(vggt_source_root).expanduser().resolve(strict=True)),
            "views_per_timestep": int(views_per_timestep),
        }
    )
    history_config = _section(geometry_config, "history_connector")
    state_config = _section(geometry_config, "state_dynamics")
    observed_keyframes = tuple(
        int(value)
        for value in geometry_config.get(
            "observed_keyframe_indices", (0, 5, 10, 15)
        )
    )
    future_steps = int(geometry_config.get("future_steps", 16))
    geometry_anchor_indices = tuple(
        int(value)
        for value in geometry_config.get(
            "geometry_anchor_indices", (3, 7, 11, 15)
        )
    )
    geometry_output_patch_grid = int(
        geometry_config.get("geometry_output_patch_grid", 4)
    )
    shallow_scene_chunk_size = int(
        geometry_config.get("shallow_scene_chunk_size", 8)
    )

    encoder = VGGTEncoder(**encoder_config).to(device=device, dtype=dtype)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            connector = GroupedHistoryConnector(
                history_config,
                action_codec_config=action_codec_config,
            )
            state_config.update(
                {
                    "observed_steps": len(observed_keyframes),
                    "future_steps": future_steps,
                    "token_count": (
                        1 + encoder.num_register_tokens + encoder.num_patches
                    ),
                    "token_dim": encoder.embed_dim,
                    "max_views": int(views_per_timestep),
                    "history_dim": connector.config.d_model,
                }
            )
            state_dynamics = WM3DStateDynamicsCore(
                WM3DStateDynamicsConfig(**state_config)
            )
            core = OnlineVGGTGeometryCore(
                encoder=encoder,
                state_dynamics=state_dynamics,
                history_connector=connector,
                observed_keyframe_indices=observed_keyframes,
                future_steps=future_steps,
                geometry_anchor_indices=geometry_anchor_indices,
                geometry_output_patch_grid=geometry_output_patch_grid,
                shallow_scene_chunk_size=shallow_scene_chunk_size,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    return core.to(device=device, dtype=dtype)
