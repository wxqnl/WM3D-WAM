#!/usr/bin/env python3
"""Run the production WM3D world-core graph on real RGB/robot data."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from wm3d_wam.assets import encode_local_wan_prompts, load_local_wan_components
from wm3d_wam.data.online_episode import first_eligible_episode, load_online_robot_window
from wm3d_wam.data.source_contracts import NormalizationRegistry, SourceContractRegistry
from wm3d_wam.models.factory import build_online_geometry_core, load_yaml_mapping
from wm3d_wam.training.geometry_pipeline import WorldCorePretrainingPipeline
from wm3d_wam.training.parameter_groups import (
    configure_world_core_pretraining_parameter_groups,
)


def _sync_elapsed(start: float) -> float:
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--source-contracts", default="configs/data/source_contracts_v1.yaml")
    parser.add_argument(
        "--normalization",
        default="/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/grouped_normalization_1b.json",
    )
    parser.add_argument("--wan-assets", required=True)
    parser.add_argument("--vggt-checkpoint", required=True)
    parser.add_argument("--vggt-source-root", required=True)
    parser.add_argument("--model-config", default="configs/model/wan_action_mot_v1.yaml")
    parser.add_argument("--geometry-config", default="configs/model/vggt_geometry_v1.yaml")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.physical_gpu == 0:
        raise ValueError("physical GPU 0 is forbidden for this project")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight expects exactly one CUDA_VISIBLE_DEVICES entry")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    contracts = SourceContractRegistry.load(args.source_contracts)
    normalization = NormalizationRegistry.load(args.normalization)
    source_name = Path(args.manifest).stem
    contract = contracts.require(source_name)
    minimum_rows = (16 + 8) * (contract.source_hz // 5) + 1
    start = time.perf_counter()
    episode = first_eligible_episode(Path(args.manifest), minimum_rows=minimum_rows)
    window = load_online_robot_window(
        source_root=Path(args.source_root),
        adapter_path=Path(args.adapter),
        episode=episode,
        source_contract=contract,
        normalization=normalization,
        program="world_core_pretrain",
    )
    data_seconds = time.perf_counter() - start
    model_config = load_yaml_mapping(args.model_config)
    geometry_config = load_yaml_mapping(args.geometry_config)

    start = time.perf_counter()
    text_components = load_local_wan_components(
        asset_root=args.wan_assets,
        dit_config={},
        device=device,
        dtype=dtype,
        load_dit=False,
        load_vae=False,
        load_text_encoder=True,
    )
    context, context_mask = encode_local_wan_prompts(
        text_components,
        [window.task_text],
        device=device,
        dtype=dtype,
    )
    text_seconds = _sync_elapsed(start)
    del text_components
    gc.collect()
    torch.cuda.empty_cache()

    start = time.perf_counter()
    core = build_online_geometry_core(
        geometry_config=geometry_config,
        action_codec_config=model_config["action_expert"]["codec_config"],
        vggt_checkpoint=args.vggt_checkpoint,
        vggt_source_root=args.vggt_source_root,
        views_per_timestep=window.view_count,
        device=device,
        dtype=dtype,
    )
    build_seconds = _sync_elapsed(start)
    groups = configure_world_core_pretraining_parameter_groups(core)
    pipeline = WorldCorePretrainingPipeline(core).train()
    window = window.to(device=device, dtype=dtype)

    start = time.perf_counter()
    output = pipeline(
        window=window,
        context=context,
        context_mask=context_mask,
        gradient_checkpointing=args.backward,
    )
    forward_seconds = _sync_elapsed(start)
    gradients: dict[str, object] | None = None
    backward_seconds = None
    if args.backward:
        start = time.perf_counter()
        output.total_loss.backward()
        backward_seconds = _sync_elapsed(start)
        gradients = {}
        for group in groups:
            values = [
                parameter.grad
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            gradients[str(group["name"])] = {
                "tensors": len(values),
                "finite": bool(
                    values and all(torch.isfinite(value).all() for value in values)
                ),
                "summed_norm": float(
                    sum(value.float().norm().item() for value in values)
                ),
            }
        for owner, module in {
            "state_prior": core.state_dynamics.state_blocks,
            "factual_dynamics": core.state_dynamics.dynamics_blocks,
            "vggt_token_decoder": core.state_dynamics.token_decoder,
        }.items():
            values = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            gradients[f"owner/{owner}"] = {
                "tensors": len(values),
                "finite": bool(
                    values and all(torch.isfinite(value).all() for value in values)
                ),
                "summed_norm": float(
                    sum(value.float().norm().item() for value in values)
                ),
            }

    result = {
        "physical_gpu": args.physical_gpu,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "data_seconds": data_seconds,
        "data": {
            "source": window.source,
            "episode_id": window.episode_id,
            "views": window.view_count,
            "history_events": int(window.action_history.events.event_mask.sum()),
            "future_events": int(window.future_actions.event_mask.sum()),
        },
        "text_seconds": text_seconds,
        "geometry_build_seconds": build_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "losses": output.detached_metrics(),
        "predicted_world_shape": list(
            output.geometry.predicted_future_shallow_tokens.shape
        ),
        "native_state_shape": list(output.geometry.native_state.shape),
        "target_shallow_detached": bool(
            output.geometry.target_future_shallow_tokens is not None
            and not output.geometry.target_future_shallow_tokens.requires_grad
        ),
        "parameter_groups": {
            str(group["name"]): sum(
                parameter.numel() for parameter in group["params"]
            )
            for group in groups
        },
        "gradients": gradients,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
