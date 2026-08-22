#!/usr/bin/env python3
"""Render held-out Stage-B RGB rollouts from a rank-local FSDP checkpoint.

The renderer executes the deployed ``forward_world`` program.  It conditions
on observed RGB, language, robot history, and the recorded candidate future
actions, preserves the observed Wan latent exactly, integrates only the four
future latent slots from Gaussian noise, and decodes the resulting 17-frame
clip with the frozen Wan2.2 VAE.  Held-out future RGB is used only after
generation for visualization and pixel diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = _PROJECT_ROOT / "outputs" / "runtime_cache"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_CACHE_ROOT / "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_CACHE_ROOT / "triton"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import torch  # noqa: E402

from wm3d_wam.data.hierarchical_sampler import WindowRequest  # noqa: E402
from wm3d_wam.data.online_dataset import (  # noqa: E402
    OnlineRobotDataset,
    OnlineTrainingSample,
)
from wm3d_wam.data.online_episode import OnlineRobotWindow  # noqa: E402
from wm3d_wam.models.interaction_masks import InteractionProgram  # noqa: E402
from wm3d_wam.models.system import WM3DWAMSystem  # noqa: E402
from wm3d_wam.training.checkpointing import resolve_checkpoint  # noqa: E402
from wm3d_wam.training.parameter_groups import (  # noqa: E402
    TrainingStage,
    configure_stage_parameter_groups,
)
from wm3d_wam.training.pipeline import WM3DWAMTrainingPipeline  # noqa: E402
from wm3d_wam.training.trainer import (  # noqa: E402
    PromptEncoderCache,
    TrainingPaths,
    build_training_model,
)
from wm3d_wam.vendor.fastwam.wan22.schedulers.scheduler_continuous import (  # noqa: E402
    WanContinuousFlowMatchScheduler,
)


DEFAULT_CHECKPOINT = Path(
    "outputs/train/wm3d_wam_k16_r4/stage_b_main_gpu1_4/checkpoints/step_00006000"
)
DEFAULT_SOURCES = (
    "oxe_droid",
    "oxe_bc_z",
    "robocasa_composite",
    "robocasa_mg",
)


def _training_paths(args: argparse.Namespace) -> TrainingPaths:
    return TrainingPaths.resolve(
        data_profile=args.data_profile,
        source_contracts=args.source_contracts,
        normalization=args.normalization,
        episode_splits=args.episode_splits,
        wan_assets=args.wan_assets,
        action_backbone=args.action_backbone,
        vggt_checkpoint=args.vggt_checkpoint,
        vggt_source_root=args.vggt_source_root,
        model_config=args.model_config,
        geometry_config=args.geometry_config,
    )


def _request(
    dataset: OnlineRobotDataset,
    *,
    source: str,
    demo_index: int,
    candidate: int,
    seed: int,
) -> WindowRequest:
    if source not in dataset.catalogs:
        raise RuntimeError(f"source {source!r} is absent from the validation split")
    catalog = dataset.catalogs[source]
    if InteractionProgram.FORWARD_WORLD.value not in catalog.contract.programs:
        raise RuntimeError(f"source {source!r} does not support forward_world")
    count = len(catalog.episodes)
    sequence = np.random.SeedSequence(
        [int(seed) & 0xFFFFFFFF, int(demo_index), int(candidate), 0x524742]
    )
    rng = np.random.Generator(np.random.PCG64(sequence))
    retry_stride = int(rng.integers(1, max(2, count)))
    if retry_stride % 2 == 0:
        retry_stride += 1
    return WindowRequest(
        global_sample_index=int(seed) + demo_index * 1000 + candidate,
        local_sample_index=candidate,
        program=InteractionProgram.FORWARD_WORLD.value,
        family=catalog.contract.family,
        source=source,
        episode_index=int(rng.integers(0, count)),
        anchor_fraction=float(rng.uniform(0.15, 0.85)),
        target_view_fraction=float(rng.random()),
        retry_stride=retry_stride,
    )


def _reference_motion(sample: OnlineTrainingSample) -> float:
    video = sample.window.wan_video.float()
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4 or video.shape[1] != 17:
        raise RuntimeError(f"unexpected Wan validation clip shape {tuple(video.shape)}")
    return float((video[:, 1:] - video[:, :-1]).abs().mean().item())


def _select_sample(
    dataset: OnlineRobotDataset,
    *,
    source: str,
    demo_index: int,
    seed: int,
    candidates: int,
) -> tuple[OnlineTrainingSample, WindowRequest, float]:
    choices: list[tuple[float, OnlineTrainingSample, WindowRequest]] = []
    for candidate in range(candidates):
        request = _request(
            dataset,
            source=source,
            demo_index=demo_index,
            candidate=candidate,
            seed=seed,
        )
        sample = dataset[request]
        choices.append((_reference_motion(sample), sample, request))
    motion, sample, request = max(choices, key=lambda value: value[0])
    return sample, request, motion


def _anchor_video(video: torch.Tensor) -> torch.Tensor:
    if video.ndim == 4:
        return video[:, :1]
    if video.ndim == 5:
        return video[:, :, :1]
    raise RuntimeError(f"unexpected Wan video shape {tuple(video.shape)}")


def _unbatched_video(video: torch.Tensor) -> torch.Tensor:
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise RuntimeError("RGB demo expects one sample per rank")
        video = video[0]
    if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] != 17:
        raise RuntimeError(f"unexpected decoded video shape {tuple(video.shape)}")
    return video


def _sample_video(
    *,
    system: WM3DWAMSystem,
    window: OnlineRobotWindow,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    steps: int,
    seed: int,
    demo_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    scheduler = WanContinuousFlowMatchScheduler()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        anchor_latents = system.encode_wan_video(_anchor_video(window.wan_video))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + demo_index * 100003)
    future_noise = torch.randn(
        (
            anchor_latents.shape[0],
            anchor_latents.shape[1],
            4,
            anchor_latents.shape[3],
            anchor_latents.shape[4],
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    latents = torch.cat((anchor_latents, future_noise), dim=2)
    timesteps, deltas = scheduler.build_inference_schedule(
        steps, device=device, dtype=dtype
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    with torch.no_grad():
        for index, (timestep, delta) in enumerate(zip(timesteps, deltas), start=1):
            batch_timestep = timestep.expand(latents.shape[0])
            action_timestep = torch.zeros(
                (latents.shape[0],), device=device, dtype=dtype
            )
            with torch.autocast("cuda", dtype=dtype):
                output = system.forward_program(
                    program=InteractionProgram.FORWARD_WORLD,
                    window=window,
                    video_latents=latents,
                    video_timestep=batch_timestep,
                    action_batch=window.future_actions,
                    action_timestep=action_timestep,
                    context=context,
                    context_mask=context_mask,
                    include_future_targets=False,
                    decode_geometry_heads=False,
                    compute_target_geometry=False,
                    geometry_gradient_checkpointing=False,
                )
            velocity = output.mot.video_velocity
            if velocity is None or velocity.shape != latents.shape:
                raise RuntimeError("forward_world returned invalid video velocity")
            latents = scheduler.step(velocity, delta, latents)
            latents[:, :, 0].copy_(anchor_latents[:, :, 0])
            if index == 1 or index == steps or index % 5 == 0:
                print(
                    json.dumps(
                        {
                            "kind": "denoise",
                            "demo_index": demo_index,
                            "step": index,
                            "steps": steps,
                            "sigma_t": float(timestep.float().item() / 1000.0),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    torch.cuda.synchronize(device)
    denoise_seconds = time.monotonic() - started
    del output, velocity
    torch.cuda.empty_cache()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        decoded = system.video_vae.decode(latents, device=device)
    torch.cuda.synchronize(device)
    total_seconds = time.monotonic() - started
    decoded = ((decoded.float() + 1.0) * 0.5).clamp_(0.0, 1.0).cpu()
    return decoded, {
        "denoise_seconds": float(denoise_seconds),
        "total_generation_seconds": float(total_seconds),
        "peak_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30),
    }


def _valid_future_mask(window: OnlineRobotWindow) -> torch.Tensor:
    mask = window.future_world_valid_mask.detach().cpu()
    if mask.ndim == 3:
        mask = mask[0]
    if mask.shape[0] != 16:
        raise RuntimeError("future RGB mask does not contain K=16 steps")
    view = int(window.target_view_index)
    return mask[:, view].to(dtype=torch.bool)


def _rgb_metrics(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    valid_future: torch.Tensor,
) -> dict[str, float]:
    predicted = _unbatched_video(predicted).float()
    reference = _unbatched_video(reference).float().cpu()
    anchor_mse = float((predicted[:, 0] - reference[:, 0]).square().mean().item())
    per_frame = (predicted[:, 1:] - reference[:, 1:]).square().mean((0, 2, 3))
    if bool(valid_future.any()):
        future_mse = float(per_frame[valid_future].mean().item())
    else:
        future_mse = float("nan")
    return {
        "anchor_vae_mse": anchor_mse,
        "anchor_vae_psnr_db": float(-10.0 * math.log10(max(anchor_mse, 1.0e-12))),
        "valid_future_frames": int(valid_future.sum().item()),
        "future_mse": future_mse,
        "future_psnr_db": float(-10.0 * math.log10(max(future_mse, 1.0e-12))),
    }


def _uint8_frames(video: torch.Tensor) -> np.ndarray:
    value = _unbatched_video(video).permute(1, 2, 3, 0).numpy()
    return np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _run_ffmpeg(frame_dir: Path, output: Path, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%03d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def _render_artifacts(
    *,
    directory: Path,
    predicted: torch.Tensor,
    reference: torch.Tensor,
    source: str,
    episode_id: str,
    task_text: str,
    fps: int,
) -> None:
    prediction = _uint8_frames(predicted)
    target = _uint8_frames(reference)
    anchor = target[0]
    height, width = anchor.shape[:2]
    frame_dir = directory / "generated_frames"
    frame_dir.mkdir(parents=True, exist_ok=False)
    predicted_images: list[Image.Image] = []
    comparison_images: list[Image.Image] = []
    title_font = _font(17)
    small_font = _font(14)
    for index in range(17):
        predicted_image = Image.fromarray(prediction[index], mode="RGB")
        predicted_image.save(frame_dir / f"frame_{index:03d}.png")
        predicted_images.append(predicted_image)
        canvas = Image.new("RGB", (3 * width, height + 58), "white")
        canvas.paste(Image.fromarray(anchor, mode="RGB"), (0, 58))
        canvas.paste(predicted_image, (width, 58))
        canvas.paste(Image.fromarray(target[index], mode="RGB"), (2 * width, 58))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), "Observed anchor", fill="black", font=title_font)
        draw.text((width + 8, 5), "Model prediction", fill="#0057B8", font=title_font)
        draw.text((2 * width + 8, 5), "Held-out reference", fill="#8B1E1E", font=title_font)
        draw.text(
            (8, 32),
            f"{source} | t=+{index / fps:.1f}s | {task_text[:72]}",
            fill="#333333",
            font=small_font,
        )
        comparison_images.append(canvas)
    duration_ms = int(round(1000.0 / fps))
    predicted_images[0].save(
        directory / "generated.gif",
        save_all=True,
        append_images=predicted_images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    comparison_images[0].save(
        directory / "comparison.gif",
        save_all=True,
        append_images=comparison_images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    Image.fromarray(anchor, mode="RGB").save(directory / "observed_anchor.png")
    _run_ffmpeg(frame_dir, directory / "generated.mp4", fps)
    key_indices = (0, 4, 8, 12, 16)
    sheet = Image.new("RGB", (len(key_indices) * width, 2 * height + 48), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), f"Prediction: {source} / {episode_id}", fill="black", font=title_font)
    for column, index in enumerate(key_indices):
        x = column * width
        sheet.paste(Image.fromarray(prediction[index], mode="RGB"), (x, 48))
        sheet.paste(Image.fromarray(target[index], mode="RGB"), (x, 48 + height))
        draw.text((x + 6, 27), f"+{index / fps:.1f}s", fill="#333333", font=small_font)
    sheet.save(directory / "contact_sheet.png")


def _write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    cards = []
    for sample in summary["samples"]:
        directory = html.escape(str(sample["directory"]))
        source = html.escape(str(sample["source"]))
        episode = html.escape(str(sample["episode_id"]))
        task = html.escape(str(sample["task_text"]))
        cards.append(
            f"""
<section>
  <h2>{source}</h2>
  <p><code>{episode}</code> — {task}</p>
  <img src="{directory}/comparison.gif" alt="RGB comparison for {source}">
  <p><a href="{directory}/generated.mp4">generated MP4</a> ·
     <a href="{directory}/contact_sheet.png">contact sheet</a></p>
</section>
"""
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>WM3D-WAM RGB demos</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #18212b; }}
section {{ margin: 2rem 0 3rem; }} img {{ max-width: 100%; border: 1px solid #ccd3da; }}
code {{ font-size: .9em; }} .note {{ padding: 1rem; background: #eef5fb; }}
</style></head><body>
<h1>WM3D-WAM Stage B RGB rollouts</h1>
<p class="note">Blue center: generated future RGB. Red right: held-out reference,
used only after generation. Each clip is {summary['fps']} fps over K=16 future steps.</p>
{''.join(cards)}
</body></html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rank-views-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--candidate-windows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--data-profile",
        default="/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/data_profile.yaml",
    )
    parser.add_argument("--source-contracts", default="configs/data/source_contracts_v1.yaml")
    parser.add_argument(
        "--normalization",
        default="/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/grouped_normalization_1b.json",
    )
    parser.add_argument("--episode-splits", default="outputs/data/episode_splits_v1")
    parser.add_argument("--wan-assets", default="/data/Minko/models/WM3D-WAM/Wan2.2-TI2V-5B")
    parser.add_argument(
        "--action-backbone",
        default="/data/Minko/models/WM3D-WAM/ActionDiT/ActionDiT_grouped_Wan22_1024.pt",
    )
    parser.add_argument(
        "--vggt-checkpoint",
        default="/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/assets/vggt_model.safetensors",
    )
    parser.add_argument(
        "--vggt-source-root",
        default="/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/runtime/vggt",
    )
    parser.add_argument("--model-config", default="configs/model/wan_action_mot_v1.yaml")
    parser.add_argument("--geometry-config", default="configs/model/vggt_geometry_v1.yaml")
    args = parser.parse_args()
    if args.inference_steps <= 0 or args.fps <= 0 or args.candidate_windows <= 0:
        parser.error("inference steps, fps, and candidate windows must be positive")
    return args


def _load_rank_views(
    *, rank_views_dir: Path, checkpoint: Path
) -> tuple[list[Mapping[str, torch.Tensor]], Mapping[str, Any]]:
    directory = rank_views_dir.expanduser().resolve(strict=True)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "wm3d_wam_plain_rank_views_v1":
        raise RuntimeError("rank-view directory has an unsupported schema")
    if int(metadata.get("world_size", -1)) != 4:
        raise RuntimeError("RGB deployment expects exactly four source shards")
    checkpoint_metadata = json.loads(
        (checkpoint / "metadata.json").read_text(encoding="utf-8")
    )
    if int(metadata.get("checkpoint_step", -1)) != int(
        checkpoint_metadata.get("global_step", -2)
    ):
        raise RuntimeError("rank views and checkpoint refer to different steps")
    recorded_checkpoint = Path(str(metadata.get("checkpoint", ""))).resolve()
    if recorded_checkpoint != checkpoint:
        raise RuntimeError(
            f"rank views were extracted from {recorded_checkpoint}, not {checkpoint}"
        )

    states: list[Mapping[str, torch.Tensor]] = []
    expected_keys: set[str] | None = None
    for rank in range(4):
        path = directory / f"views_rank_{rank:03d}.pt"
        payload = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "wm3d_wam_plain_rank_views_v1"
            or int(payload.get("rank", -1)) != rank
            or int(payload.get("world_size", -1)) != 4
            or int(payload.get("checkpoint_step", -1))
            != int(checkpoint_metadata["global_step"])
            or not isinstance(payload.get("model"), Mapping)
        ):
            raise RuntimeError(f"malformed plain rank-view payload {path}")
        state = payload["model"]
        keys = set(state)
        if any(key.rsplit(".", 1)[-1] == "_flat_param" for key in keys):
            raise RuntimeError(f"redundant flat FSDP keys remain in {path}")
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise RuntimeError("plain rank-view payloads have different key sets")
        states.append(state)  # type: ignore[arg-type]
    return states, checkpoint_metadata


@torch.no_grad()
def _restore_full_model_from_rank_views(
    *,
    pipeline: WM3DWAMTrainingPipeline,
    states: Sequence[Mapping[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, int]:
    target_state = pipeline.state_dict()
    target_keys = set(target_state)
    if not states or any(set(state) != target_keys for state in states):
        missing = sorted(target_keys - set(states[0])) if states else sorted(target_keys)
        extra = sorted(set(states[0]) - target_keys) if states else []
        raise RuntimeError(
            f"checkpoint/model key mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )

    replicated = 0
    partitioned = 0
    copied_numel = 0
    for index, (key, target) in enumerate(target_state.items(), start=1):
        if not isinstance(target, torch.Tensor):
            raise RuntimeError(f"target state {key!r} is not a tensor")
        pieces = [state[key].detach().reshape(-1) for state in states]
        if any(not isinstance(piece, torch.Tensor) for piece in pieces):
            raise RuntimeError(f"source state {key!r} is not tensor-only")
        nonempty = [piece for piece in pieces if piece.numel()]
        full = [piece for piece in nonempty if piece.numel() == target.numel()]
        if full:
            if any(piece.numel() != target.numel() for piece in nonempty):
                raise RuntimeError(f"ambiguous full/partial layout for {key!r}")
            source = full[0]
            replicated += 1
        else:
            if sum(piece.numel() for piece in nonempty) != target.numel():
                raise RuntimeError(
                    f"cannot reconstruct {key!r}: source numel="
                    f"{[piece.numel() for piece in pieces]}, target={target.numel()}"
                )
            source = nonempty[0] if len(nonempty) == 1 else torch.cat(nonempty)
            partitioned += 1
        converted = source.to(device=device, dtype=target.dtype)
        target.copy_(converted.reshape(target.shape))
        copied_numel += target.numel()
        del converted, source, pieces, nonempty, full
        if index == 1 or index == len(target_state) or index % 250 == 0:
            print(
                json.dumps(
                    {
                        "kind": "checkpoint_restore",
                        "keys_done": index,
                        "keys_total": len(target_state),
                        "gpu_memory_gib": round(
                            torch.cuda.memory_allocated(device) / 2**30, 3
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "keys": len(target_state),
        "replicated_keys": replicated,
        "partitioned_keys": partitioned,
        "copied_numel": copied_numel,
    }


def _build_model(
    *,
    paths: TrainingPaths,
    checkpoint: Path,
    rank_views_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    WM3DWAMTrainingPipeline,
    PromptEncoderCache,
    Mapping[str, Any],
    Mapping[str, int],
]:
    torch.manual_seed(20260822)
    torch.cuda.manual_seed(20260822)
    built = build_training_model(
        phase=TrainingStage.WAN_ACTION_MAIN.value,
        paths=paths,
        device=device,
        dtype=dtype,
    )
    if not isinstance(built.pipeline, WM3DWAMTrainingPipeline):
        raise RuntimeError("Stage-B builder returned the wrong pipeline type")
    system = built.pipeline.system
    text_components = built.text_components
    configure_stage_parameter_groups(system, TrainingStage.WAN_ACTION_MAIN)
    for head in (
        system.geometry_core.encoder.dpt_head,
        system.geometry_core.encoder.point_head,
        system.geometry_core.encoder.camera_head,
    ):
        head.float()
    prompt_cache = PromptEncoderCache(
        components=text_components,
        device=device,
        dtype=dtype,
        max_entries=8,
    )
    prompt_cache.components.text_encoder.to(device=torch.device("cpu"))
    torch.cuda.empty_cache()
    states, metadata = _load_rank_views(
        rank_views_dir=rank_views_dir,
        checkpoint=checkpoint,
    )
    restore_stats = _restore_full_model_from_rank_views(
        pipeline=built.pipeline,
        states=states,
        device=device,
    )
    del states
    gc.collect()
    built.pipeline.eval()
    torch.cuda.empty_cache()
    return built.pipeline, prompt_cache, metadata, restore_stats


def main() -> None:
    args = _parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    if visible != "7":
        raise RuntimeError("RGB demo must run with CUDA_VISIBLE_DEVICES=7")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("RGB demo requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    checkpoint = resolve_checkpoint(args.checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _training_paths(args)
    dataset = OnlineRobotDataset(
        data_profile_path=paths.data_profile,
        source_contract_path=paths.source_contracts,
        normalization_path=paths.normalization,
        split_root=paths.episode_splits,
        split="val",
        max_decode_retries=8,
    )
    dtype = torch.bfloat16
    pipeline, prompt_cache, checkpoint_metadata, restore_stats = _build_model(
        paths=paths,
        checkpoint=checkpoint,
        rank_views_dir=args.rank_views_dir,
        device=device,
        dtype=dtype,
    )
    samples: list[dict[str, Any]] = []
    for demo_index, source_value in enumerate(args.sources):
        source = str(source_value)
        sample, request, reference_motion = _select_sample(
            dataset,
            source=source,
            demo_index=demo_index,
            seed=args.seed,
            candidates=args.candidate_windows,
        )
        window = sample.window.to(device=device, dtype=dtype)
        context_features, context_mask = prompt_cache.encode_batch(sample.task_texts)
        generated, runtime = _sample_video(
            system=pipeline.system,
            window=window,
            context=context_features,
            context_mask=context_mask,
            steps=args.inference_steps,
            seed=args.seed,
            demo_index=demo_index,
            device=device,
            dtype=dtype,
        )
        reference = _unbatched_video(window.wan_video.detach().float().cpu())
        valid_future = _valid_future_mask(window)
        metrics = _rgb_metrics(generated, reference, valid_future)
        directory_name = f"demo_{demo_index + 1:02d}_{source}"
        directory = output_dir / directory_name
        directory.mkdir(parents=True, exist_ok=False)
        _render_artifacts(
            directory=directory,
            predicted=generated,
            reference=reference,
            source=source,
            episode_id=window.episode_id,
            task_text=window.task_text,
            fps=args.fps,
        )
        result = {
            "demo_index": demo_index,
            "directory": directory_name,
            "source": source,
            "episode_id": window.episode_id,
            "task_text": window.task_text,
            "sample_index": window.sample_index,
            "decode_retry_count": window.decode_retry_count,
            "anchor_time_s": float(window.anchor_time_s),
            "target_view_index": int(window.target_view_index),
            "valid_view_count": int(window.valid_view_count),
            "reference_motion_l1": reference_motion,
            "request": {
                "episode_index": request.episode_index,
                "anchor_fraction": request.anchor_fraction,
                "target_view_fraction": request.target_view_fraction,
            },
            "metrics": metrics,
            "runtime": runtime,
        }
        (directory / "metadata.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        samples.append(result)
        del window, context_features, context_mask, generated, reference
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "schema": "wm3d_wam_rgb_rollout_eval_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint_metadata["global_step"]),
        "checkpoint_restore": dict(restore_stats),
        "physical_gpu": 7,
        "split": "val",
        "program": InteractionProgram.FORWARD_WORLD.value,
        "future_target_access_during_generation": False,
        "inference_steps": int(args.inference_steps),
        "fps": int(args.fps),
        "seed": int(args.seed),
        "samples": samples,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
