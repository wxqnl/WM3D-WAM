#!/usr/bin/env python3
"""Run a production-width WM3D-WAM program on one real online window."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from wm3d_wam.assets import encode_local_wan_prompts, load_local_wan_components
from wm3d_wam.data.online_episode import (
    first_eligible_episode,
    load_online_robot_window,
)
from wm3d_wam.data.source_contracts import NormalizationRegistry, SourceContractRegistry
from wm3d_wam.models.factory import (
    build_online_geometry_core,
    build_wan_action_mot,
    load_yaml_mapping,
)
from wm3d_wam.models.interaction_masks import InteractionProgram
from wm3d_wam.models.online_vggt_geometry import GeometryConditionMode
from wm3d_wam.models.system import WM3DWAMSystem
from wm3d_wam.training.parameter_groups import (
    TrainingStage,
    configure_stage_parameter_groups,
)
from wm3d_wam.training.pipeline import WM3DWAMTrainingPipeline


def _elapsed(start: float) -> float:
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _video_config(model_config: dict) -> dict:
    value = dict(model_config["video_expert"])
    value.pop("_target_", None)
    return value


def _grad_report(groups: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for group in groups:
        gradients = [
            parameter.grad
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        report[str(group["name"])] = {
            "tensors": len(gradients),
            "finite": bool(
                gradients and all(torch.isfinite(value).all() for value in gradients)
            ),
            "summed_norm": float(
                sum(value.float().norm().item() for value in gradients)
            ),
        }
    return report


@torch.no_grad()
def _cache_parity(
    *,
    system: WM3DWAMSystem,
    training_output,
    clean_video_latents: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
) -> dict[str, float]:
    flow = training_output.action_flow
    if flow is None:
        raise ValueError("cache parity requires an action-producing program")
    geometry = training_output.program_output.geometry
    model = system.wan_action.eval()
    batch_size = int(clean_video_latents.shape[0])
    zero = torch.zeros(
        (batch_size,),
        device=clean_video_latents.device,
        dtype=clean_video_latents.dtype,
    )
    full = model(
        program=InteractionProgram.ACTION_ONLY,
        video_latents=clean_video_latents[:, :, :1],
        video_timestep=zero,
        action_batch=flow.noisy,
        action_timestep=flow.timestep,
        context=context,
        context_mask=context_mask,
        geometry_tokens=geometry.geometry_tokens,
        geometry_token_mask=geometry.geometry_token_mask,
    )
    cache = model.prefill_observed_video(
        observed_video_latents=clean_video_latents[:, :, :1],
        context=context,
        context_mask=context_mask,
        geometry_tokens=geometry.geometry_tokens,
        geometry_token_mask=geometry.geometry_token_mask,
    )
    cached = model.action_velocity_from_cache(
        action_batch=flow.noisy,
        action_timestep=flow.timestep,
        context=context,
        context_mask=context_mask,
        cache=cache,
    )
    if full.action_velocity is None:
        raise RuntimeError("full action-only path returned no action velocity")
    difference = (full.action_velocity.float() - cached.float())
    reference_rms = full.action_velocity.float().square().mean().sqrt()
    error_rms = difference.square().mean().sqrt()
    return {
        "max_delta": float(difference.abs().max()),
        "error_rms": float(error_rms),
        "reference_rms": float(reference_rms),
        "relative_rms": float(error_rms / reference_rms.clamp_min(1.0e-12)),
    }


@torch.no_grad()
def _policy_target_leakage(
    *,
    system: WM3DWAMSystem,
    window,
    context: torch.Tensor,
    context_mask: torch.Tensor,
) -> dict[str, float | bool]:
    core = system.geometry_core.eval()
    kwargs = {
        "observed_images": system._batched_images(
            window.observed_images, name="observed_images"
        ),
        "state_history": window.state_history,
        "action_history": window.action_history,
        "future_world_times_s": window.future_world_times_s,
        "mode": GeometryConditionMode.ACTION_FREE,
        "language_features": context,
        "language_padding_mask": context_mask,
        "observed_view_valid_mask": system._batched_view_mask(
            window.observed_view_valid_mask, name="observed_view_valid_mask"
        ),
        "future_view_valid_mask": system._batched_view_mask(
            window.future_view_valid_mask, name="future_view_valid_mask"
        ),
        "future_world_valid_mask": system._batched_view_mask(
            window.future_world_valid_mask, name="future_world_valid_mask"
        ),
        "decode_geometry_heads": False,
        "compute_target_geometry": False,
    }
    without_target = core(future_target_images=None, **kwargs)
    with_target = core(
        future_target_images=system._batched_images(
            window.future_world_images, name="future_world_images"
        ),
        **kwargs,
    )
    return {
        "predicted_shallow_max_delta": float(
            (
                without_target.predicted_future_shallow_tokens
                - with_target.predicted_future_shallow_tokens
            )
            .abs()
            .max()
        ),
        "deep_visual_max_delta": float(
            (without_target.deep_visual_tokens - with_target.deep_visual_tokens)
            .abs()
            .max()
        ),
        "geometry_token_max_delta": float(
            (without_target.geometry_tokens - with_target.geometry_tokens)
            .abs()
            .max()
        ),
        "target_detached": bool(
            with_target.target_future_shallow_tokens is not None
            and not with_target.target_future_shallow_tokens.requires_grad
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--program", choices=[value.value for value in InteractionProgram], default="action_only")
    parser.add_argument("--stage", choices=[value.value for value in TrainingStage], default="wan_action_warmup")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--source-contracts", default="configs/data/source_contracts_v1.yaml")
    parser.add_argument(
        "--normalization",
        default="/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/grouped_normalization_1b.json",
    )
    parser.add_argument("--wan-assets", required=True)
    parser.add_argument("--action-backbone", required=True)
    parser.add_argument("--vggt-checkpoint", required=True)
    parser.add_argument("--vggt-source-root", required=True)
    parser.add_argument("--model-config", default="configs/model/wan_action_mot_v1.yaml")
    parser.add_argument("--geometry-config", default="configs/model/vggt_geometry_v1.yaml")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--backward-loss",
        choices=("total", "action", "video", "geometry"),
        default="total",
    )
    parser.add_argument("--cache-parity", action="store_true")
    parser.add_argument("--target-leakage", action="store_true")
    parser.add_argument("--optimizer-steps", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.physical_gpu == 0:
        raise ValueError("physical GPU 0 is forbidden for this project")
    if args.optimizer_steps < 0:
        raise ValueError("optimizer steps must be non-negative")
    if args.backward and args.optimizer_steps:
        raise ValueError("use either --backward or --optimizer-steps, not both")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight expects exactly one CUDA_VISIBLE_DEVICES entry")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    result: dict[str, object] = {
        "program": args.program,
        "stage": args.stage,
        "physical_gpu": args.physical_gpu,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
    }

    start = time.perf_counter()
    contracts = SourceContractRegistry.load(args.source_contracts)
    normalization = NormalizationRegistry.load(args.normalization)
    source_name = Path(args.manifest).stem
    contract = contracts.require(source_name)
    minimum_rows = (16 + 8) * (contract.source_hz // 5) + 1
    episode = first_eligible_episode(Path(args.manifest), minimum_rows=minimum_rows)
    window = load_online_robot_window(
        source_root=Path(args.source_root),
        adapter_path=Path(args.adapter),
        episode=episode,
        source_contract=contract,
        normalization=normalization,
        program=args.program,
    )
    result["data_seconds"] = time.perf_counter() - start
    result["data"] = {
        "source": window.source,
        "episode_id": window.episode_id,
        "views": window.view_count,
        "history_events": int(window.action_history.events.event_mask.sum()),
        "future_events": int(window.future_actions.event_mask.sum()),
        "wan_rgb_shape": list(window.wan_video.shape),
    }
    model_config = load_yaml_mapping(args.model_config)
    geometry_config = load_yaml_mapping(args.geometry_config)

    start = time.perf_counter()
    components = load_local_wan_components(
        asset_root=args.wan_assets,
        dit_config=_video_config(model_config),
        device=device,
        dtype=dtype,
        load_dit=True,
        load_vae=True,
        load_text_encoder=True,
    )
    result["wan_assets_seconds"] = _elapsed(start)
    if components.dit is None or components.vae is None:
        raise RuntimeError("Wan DiT/VAE did not load")
    start = time.perf_counter()
    context, context_mask = encode_local_wan_prompts(
        components,
        [window.task_text],
        device=device,
        dtype=dtype,
    )
    result["text_encode_seconds"] = _elapsed(start)
    result["context_shape"] = list(context.shape)
    components.text_encoder = None
    components.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

    start = time.perf_counter()
    wan_action = build_wan_action_mot(
        video_expert=components.dit,
        model_config=model_config,
        action_backbone_path=args.action_backbone,
        device=device,
        dtype=dtype,
    )
    result["action_build_seconds"] = _elapsed(start)
    start = time.perf_counter()
    geometry_core = build_online_geometry_core(
        geometry_config=geometry_config,
        action_codec_config=model_config["action_expert"]["codec_config"],
        vggt_checkpoint=args.vggt_checkpoint,
        vggt_source_root=args.vggt_source_root,
        views_per_timestep=window.view_count,
        device=device,
        dtype=dtype,
    )
    result["geometry_build_seconds"] = _elapsed(start)
    system = WM3DWAMSystem(
        geometry_core=geometry_core,
        wan_action=wan_action,
        video_vae=components.vae,
    )
    del components
    window = window.to(device=device, dtype=dtype)
    pipeline = WM3DWAMTrainingPipeline(system=system).train()
    groups = configure_stage_parameter_groups(system, args.stage)
    result["parameter_groups"] = {
        str(group["name"]): sum(parameter.numel() for parameter in group["params"])
        for group in groups
    }

    start = time.perf_counter()
    clean_video_latents = system.encode_wan_video(window.wan_video)
    result["vae_encode_seconds"] = _elapsed(start)
    result["wan_latent_shape"] = list(clean_video_latents.shape)

    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed_all(args.seed + 1)
    fixed_action_timestep = torch.full(
        (window.batch_size,), 500.0, device=device, dtype=dtype
    )
    fixed_action_noise = torch.randn_like(window.future_actions.values)
    fixed_video_timestep = torch.full(
        (window.batch_size,), 500.0, device=device, dtype=dtype
    )
    fixed_video_noise = torch.randn_like(clean_video_latents)
    start = time.perf_counter()
    output = pipeline(
        program=args.program,
        window=window,
        context=context,
        context_mask=context_mask,
        clean_video_latents=clean_video_latents,
        action_timestep=fixed_action_timestep,
        action_noise=fixed_action_noise,
        video_timestep=fixed_video_timestep,
        video_noise=fixed_video_noise,
        geometry_gradient_checkpointing=bool(args.backward or args.optimizer_steps),
    )
    result["forward_seconds"] = _elapsed(start)
    result["losses"] = output.detached_metrics()
    result["action_velocity_shape"] = (
        list(output.program_output.mot.action_velocity.shape)
        if output.program_output.mot.action_velocity is not None
        else None
    )
    result["video_velocity_shape"] = (
        list(output.program_output.mot.video_velocity.shape)
        if output.program_output.mot.video_velocity is not None
        else None
    )
    if output.video_flow is not None:
        result["observed_latent_max_delta"] = float(
            (
                output.video_flow.noisy_latents[:, :, 0]
                - output.video_flow.clean_latents[:, :, 0]
            )
            .abs()
            .max()
        )
    if args.backward:
        selected_loss = {
            "total": output.total_loss,
            "action": output.action_loss,
            "video": output.video_loss,
            "geometry": output.geometry_loss,
        }[args.backward_loss]
        if not selected_loss.requires_grad:
            raise ValueError(
                f"selected backward loss {args.backward_loss!r} has no trainable path"
            )
        start = time.perf_counter()
        selected_loss.backward()
        result["backward_seconds"] = _elapsed(start)
        result["backward_loss"] = args.backward_loss
        result["gradients"] = _grad_report(groups)
    if args.optimizer_steps:
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
        probe_name, probe_parameter = next(
            (name, parameter)
            for name, parameter in system.named_parameters()
            if parameter.requires_grad
        )
        probe_before = probe_parameter.detach().float().clone()
        losses: list[float] = []
        step_seconds: list[float] = []
        grad_norms: list[float] = []
        for step in range(args.optimizer_steps):
            if step:
                output = pipeline(
                    program=args.program,
                    window=window,
                    context=context,
                    context_mask=context_mask,
                    clean_video_latents=clean_video_latents,
                    action_timestep=fixed_action_timestep,
                    action_noise=fixed_action_noise,
                    video_timestep=fixed_video_timestep,
                    video_noise=fixed_video_noise,
                    geometry_gradient_checkpointing=True,
                )
            losses.append(float(output.total_loss.detach()))
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output.total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in system.parameters()
                    if parameter.requires_grad
                ],
                max_norm=1.0,
            )
            optimizer.step()
            step_seconds.append(_elapsed(step_start))
            grad_norms.append(float(grad_norm))
        with torch.no_grad():
            output = pipeline(
                program=args.program,
                window=window,
                context=context,
                context_mask=context_mask,
                clean_video_latents=clean_video_latents,
                action_timestep=fixed_action_timestep,
                action_noise=fixed_action_noise,
                video_timestep=fixed_video_timestep,
                video_noise=fixed_video_noise,
            )
        losses.append(float(output.total_loss.detach()))
        result["optimization"] = {
            "steps": args.optimizer_steps,
            "losses": losses,
            "step_seconds": step_seconds,
            "preclip_grad_norms": grad_norms,
            "probe_parameter": probe_name,
            "probe_max_delta": float(
                (probe_parameter.detach().float() - probe_before).abs().max()
            ),
        }
    if args.cache_parity:
        result["cache_action_parity"] = _cache_parity(
            system=system,
            training_output=output,
            clean_video_latents=clean_video_latents,
            context=context,
            context_mask=context_mask,
        )
    if args.target_leakage:
        result["policy_target_leakage"] = _policy_target_leakage(
            system=system,
            window=window,
            context=context,
            context_mask=context_mask,
        )
    result["peak_memory_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
