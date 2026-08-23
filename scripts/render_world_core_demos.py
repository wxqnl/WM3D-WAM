#!/usr/bin/env python3
"""Render held-out Stage-A geometry demos from a canonical checkpoint.

The Stage-A world core does not contain a learned RGB decoder.  This script
therefore labels future RGB strictly as held-out reference data and visualizes
the quantities that Stage A actually predicts: shallow future features, depth,
world points, camera pose, and the factual/action-free difference.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from wm3d_wam.data.hierarchical_sampler import (  # noqa: E402
    RecoverableHierarchicalSampler,
    WindowRequest,
)
from wm3d_wam.data.online_dataset import OnlineRobotDataset  # noqa: E402
from wm3d_wam.models.online_vggt_geometry import (  # noqa: E402
    GeometryConditionMode,
    OnlineGeometryOutput,
)
from wm3d_wam.training.checkpointing import load_model_only  # noqa: E402
from wm3d_wam.training.geometry_pipeline import (  # noqa: E402
    WorldCorePretrainingPipeline,
)
from wm3d_wam.training.trainer import (  # noqa: E402
    PromptEncoderCache,
    TrainingPaths,
    build_training_model,
)


DEFAULT_STAGE_A = Path("outputs/train/wm3d_wam_k16_r5/stage_a_world_core_gpu1_4")
DEFAULT_CHECKPOINT = DEFAULT_STAGE_A / "checkpoints/step_00030000"
DEFAULT_OUTPUT = Path("outputs/eval/world_core_r5_step_00030000")
DEFAULT_SOURCES = ("oxe_bc_z", "oxe_bridge", "robocasa_composite")
ANCHOR_SECONDS = (0.4, 0.8, 1.2, 1.6)


def _finite_percentile(
    values: Iterable[np.ndarray],
    low: float = 1.0,
    high: float = 99.0,
) -> tuple[float, float]:
    flattened = [value[np.isfinite(value)].reshape(-1) for value in values]
    flattened = [value for value in flattened if value.size]
    if not flattened:
        return 0.0, 1.0
    merged = np.concatenate(flattened)
    left, right = np.percentile(merged, [low, high]).astype(float)
    if not math.isfinite(left) or not math.isfinite(right) or left == right:
        center = left if math.isfinite(left) else 0.0
        return center - 0.5, center + 0.5
    return left, right


def _to_rgb(value: torch.Tensor) -> np.ndarray:
    image = value.detach().float().cpu().clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _moving_average(values: np.ndarray, width: int = 25) -> np.ndarray:
    if values.size < width:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    padded = np.pad(values, (width - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def render_training_curves(metrics_path: Path, output_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = [row for row in rows if row.get("kind") == "train"]
    validation = [row for row in rows if row.get("kind") == "validation"]
    if not train or not validation:
        raise RuntimeError("metrics.jsonl has no complete train/validation history")

    train_steps = np.asarray([row["step"] for row in train], dtype=np.int64)
    val_steps = np.asarray([row["step"] for row in validation], dtype=np.int64)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    for key, label, color in (
        ("loss_total", "total", "#234E70"),
        ("loss_future_feature", "factual feature", "#FB8B24"),
        ("loss_action_free_future_feature", "action-free feature", "#2A9D8F"),
    ):
        values = np.asarray([row[key] for row in train], dtype=np.float64)
        axes[0, 0].plot(train_steps, values, color=color, alpha=0.12, linewidth=0.7)
        axes[0, 0].plot(
            train_steps,
            _moving_average(values),
            color=color,
            linewidth=1.8,
            label=label,
        )
    axes[0, 0].set(title="Training losses (25-log moving average)", xlabel="step")
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend()

    for key, label in (
        ("loss_depth", "depth"),
        ("loss_world_points", "world points"),
        ("loss_camera_pose", "camera pose"),
    ):
        values = np.asarray([max(float(row[key]), 1.0e-10) for row in train])
        axes[0, 1].plot(train_steps, _moving_average(values), label=label)
    axes[0, 1].set(
        title="Geometry teacher-consistency losses",
        xlabel="step",
        yscale="log",
    )
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend()

    for key, label, marker in (
        ("loss_total", "total", "o"),
        ("loss_future_feature", "factual feature", "s"),
        ("loss_action_free_future_feature", "action-free feature", "^"),
    ):
        axes[1, 0].plot(
            val_steps,
            [row[key] for row in validation],
            marker=marker,
            markersize=3,
            linewidth=1.3,
            label=label,
        )
    axes[1, 0].set(title="Held-out validation losses", xlabel="step")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend()

    throughput = np.asarray(
        [row["samples_per_second_global"] for row in train], dtype=np.float64
    )
    axes[1, 1].plot(
        train_steps,
        _moving_average(throughput),
        color="#6A4C93",
        label="global samples/s",
    )
    axes[1, 1].set(title="Training throughput", xlabel="step")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend()
    fig.suptitle("WM3D-WAM Stage A — step 30,000", fontsize=16)
    _save_figure(fig, output_path)

    first = validation[0]
    last = validation[-1]
    return {
        "first_validation_step": int(first["step"]),
        "first_validation_loss": float(first["loss_total"]),
        "final_validation_step": int(last["step"]),
        "final_validation_loss": float(last["loss_total"]),
        "validation_loss_relative_change": float(
            float(last["loss_total"]) / float(first["loss_total"]) - 1.0
        ),
    }


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


def _requests_for_sources(
    dataset: OnlineRobotDataset,
    *,
    sources: Sequence[str],
    seed: int,
) -> list[WindowRequest]:
    missing = set(sources)
    selected: dict[str, WindowRequest] = {}
    sampler = RecoverableHierarchicalSampler(
        episode_counts=dataset.episode_counts,
        contracts=dataset.contracts,
        profile_weights=dataset.profile_weights,
        program_mix={"world_core_pretrain": 1.0},
        seed=seed,
        rank=0,
        world_size=1,
        micro_batch_size=1,
    )
    for index in range(100_000):
        request = sampler.request_at(index)
        if request.source in missing:
            selected[request.source] = request
            missing.remove(request.source)
            if not missing:
                break
    if missing:
        raise RuntimeError(
            f"could not sample requested validation sources: {sorted(missing)}"
        )
    return [selected[source] for source in sources]


def _future_geometry(
    output: OnlineGeometryOutput,
    *,
    views: int,
    key: str,
    target: bool,
) -> torch.Tensor:
    values = output.target_geometry if target else output.student_geometry
    if values is None or key not in values:
        raise RuntimeError(f"geometry output has no {key!r} tensor")
    value = values[key]
    anchors = len(output.geometry_anchor_indices)
    if target:
        return value.reshape(1, anchors, views, *value.shape[1:])[0]
    total_steps = int(output.deep_visual_tokens.shape[1])
    return value.reshape(1, total_steps, views, *value.shape[1:])[0, -anchors:]


def _feature_similarity(output: OnlineGeometryOutput, view: int) -> np.ndarray:
    target = output.target_future_shallow_tokens
    if target is None:
        raise RuntimeError("held-out target shallow features are missing")
    prediction = output.predicted_future_shallow_tokens
    special = 1 + 4
    anchors = torch.as_tensor(output.geometry_anchor_indices, device=prediction.device)
    prediction = prediction.index_select(1, anchors)[0, :, view, special:]
    target = target.index_select(1, anchors)[0, :, view, special:]
    similarity = F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    patch_side = int(math.isqrt(similarity.shape[-1]))
    if patch_side * patch_side != similarity.shape[-1]:
        raise RuntimeError("VGGT patch token count is not square")
    return similarity.reshape(-1, patch_side, patch_side).detach().cpu().numpy()


def _metric_summary(
    output: OnlineGeometryOutput, *, views: int
) -> dict[str, list[float]]:
    target_features = output.target_future_shallow_tokens
    if target_features is None:
        raise RuntimeError("target future features are missing")
    anchors = torch.as_tensor(
        output.geometry_anchor_indices, device=target_features.device
    )
    feature = 1.0 - F.cosine_similarity(
        output.predicted_future_shallow_tokens.index_select(1, anchors).float(),
        target_features.index_select(1, anchors).float(),
        dim=-1,
    ).mean(dim=(0, 2, 3))
    predicted_depth = _future_geometry(output, views=views, key="depth", target=False)
    target_depth = _future_geometry(output, views=views, key="depth", target=True)
    predicted_points = _future_geometry(
        output, views=views, key="world_points", target=False
    )
    target_points = _future_geometry(
        output, views=views, key="world_points", target=True
    )
    predicted_pose = _future_geometry(output, views=views, key="pose_enc", target=False)
    target_pose = _future_geometry(output, views=views, key="pose_enc", target=True)
    return {
        "horizon_seconds": list(ANCHOR_SECONDS),
        "feature_cosine_error": feature.detach().cpu().tolist(),
        "depth_mae": (predicted_depth - target_depth)
        .abs()
        .mean(dim=(1, 2, 3))
        .cpu()
        .tolist(),
        "world_point_l2": torch.linalg.vector_norm(
            predicted_points - target_points, dim=-1
        )
        .mean(dim=(1, 2, 3))
        .cpu()
        .tolist(),
        "pose_l2": torch.linalg.vector_norm(predicted_pose - target_pose, dim=-1)
        .mean(dim=1)
        .cpu()
        .tolist(),
    }


def render_prediction_board(
    *,
    window,
    output: OnlineGeometryOutput,
    view: int,
    path: Path,
) -> None:
    anchors = output.geometry_anchor_indices
    predicted = _future_geometry(
        output, views=window.view_count, key="depth", target=False
    )
    target = _future_geometry(output, views=window.view_count, key="depth", target=True)
    predicted_np = predicted[:, view].float().cpu().numpy()
    target_np = target[:, view].float().cpu().numpy()
    errors = np.abs(predicted_np - target_np)
    depth_min, depth_max = _finite_percentile((predicted_np, target_np))
    _, error_max = _finite_percentile((errors,), low=0.0, high=99.0)
    error_max = max(error_max, 1.0e-8)

    fig, axes = plt.subplots(5, 4, figsize=(13, 15), constrained_layout=True)
    observed_indices = (0, 5, 10, 15)
    for column, (observed_index, anchor) in enumerate(zip(observed_indices, anchors)):
        axes[0, column].imshow(_to_rgb(window.observed_images[column, view]))
        axes[0, column].set_title(
            f"observed t={float(window.observation_times_s[observed_index]):+.1f}s"
        )
        axes[1, column].imshow(_to_rgb(window.future_world_images[anchor, view]))
        axes[1, column].set_title(f"held-out RGB t=+{ANCHOR_SECONDS[column]:.1f}s")
        axes[2, column].imshow(
            predicted_np[column], cmap="turbo", vmin=depth_min, vmax=depth_max
        )
        axes[3, column].imshow(
            target_np[column], cmap="turbo", vmin=depth_min, vmax=depth_max
        )
        axes[4, column].imshow(errors[column], cmap="magma", vmin=0.0, vmax=error_max)
    labels = (
        "observed RGB",
        "future RGB (reference only)",
        "predicted future depth",
        "VGGT teacher depth",
        "absolute depth error",
    )
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label, fontsize=11)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"{window.source} | {window.episode_id} | view {view} | {window.task_text}",
        fontsize=14,
    )
    _save_figure(fig, path)


def render_action_conditioning_board(
    *,
    factual: OnlineGeometryOutput,
    action_free: OnlineGeometryOutput,
    views: int,
    view: int,
    path: Path,
) -> dict[str, list[float]]:
    factual_depth = (
        _future_geometry(factual, views=views, key="depth", target=False)[:, view]
        .float()
        .cpu()
        .numpy()
    )
    action_free_depth = (
        _future_geometry(action_free, views=views, key="depth", target=False)[:, view]
        .float()
        .cpu()
        .numpy()
    )
    difference = np.abs(factual_depth - action_free_depth)
    similarity = _feature_similarity(factual, view)
    depth_min, depth_max = _finite_percentile((factual_depth, action_free_depth))
    _, difference_max = _finite_percentile((difference,), low=0.0, high=99.0)
    difference_max = max(difference_max, 1.0e-8)

    fig, axes = plt.subplots(4, 4, figsize=(13, 12), constrained_layout=True)
    for column, seconds in enumerate(ANCHOR_SECONDS):
        axes[0, column].imshow(similarity[column], cmap="viridis", vmin=0.0, vmax=1.0)
        axes[0, column].set_title(f"t=+{seconds:.1f}s")
        axes[1, column].imshow(
            factual_depth[column], cmap="turbo", vmin=depth_min, vmax=depth_max
        )
        axes[2, column].imshow(
            action_free_depth[column], cmap="turbo", vmin=depth_min, vmax=depth_max
        )
        axes[3, column].imshow(
            difference[column], cmap="magma", vmin=0.0, vmax=difference_max
        )
    labels = (
        "feature cosine similarity",
        "factual (real action)",
        "action-free prior",
        "|factual - action-free|",
    )
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label, fontsize=11)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Action conditioning changes the future world state (not a counterfactual GT test)",
        fontsize=14,
    )
    _save_figure(fig, path)
    return {
        "horizon_seconds": list(ANCHOR_SECONDS),
        "mean_absolute_depth_delta": difference.mean(axis=(1, 2)).tolist(),
    }


def _camera_centers(pose: np.ndarray) -> np.ndarray:
    translation = pose[..., :3]
    quaternion = pose[..., 3:7]
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion = quaternion / np.maximum(norm, 1.0e-8)
    # VGGT uses scalar-last XYZW quaternions.
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    rotation = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    return -np.einsum("...ji,...j->...i", rotation, translation)


def _point_cloud_payload(
    output: OnlineGeometryOutput,
    *,
    views: int,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted = (
        _future_geometry(output, views=views, key="world_points", target=False)[-1]
        .float()
        .cpu()
        .numpy()
    )
    target = (
        _future_geometry(output, views=views, key="world_points", target=True)[-1]
        .float()
        .cpu()
        .numpy()
    )
    predicted = predicted[:, ::stride, ::stride].reshape(-1, 3)
    target = target[:, ::stride, ::stride].reshape(-1, 3)
    predicted = predicted[np.isfinite(predicted).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    predicted_pose = (
        _future_geometry(output, views=views, key="pose_enc", target=False)
        .float()
        .cpu()
        .numpy()
    )
    target_pose = (
        _future_geometry(output, views=views, key="pose_enc", target=True)
        .float()
        .cpu()
        .numpy()
    )
    return (
        predicted,
        target,
        _camera_centers(predicted_pose),
        _camera_centers(target_pose),
    )


def _cloud_limits(
    predicted: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    merged = np.concatenate((predicted, target), axis=0)
    low = np.percentile(merged, 1.0, axis=0)
    high = np.percentile(merged, 99.0, axis=0)
    center = (low + high) * 0.5
    radius = max(float((high - low).max()) * 0.55, 1.0e-3)
    return center - radius, center + radius


def _draw_cloud(
    axis,
    points: np.ndarray,
    cameras: np.ndarray,
    *,
    title: str,
    limits: tuple[np.ndarray, np.ndarray],
) -> None:
    low, high = limits
    color = points[:, 2]
    axis.scatter(
        points[:, 0], points[:, 2], -points[:, 1], c=color, cmap="turbo", s=0.6
    )
    for view in range(cameras.shape[1]):
        trajectory = cameras[:, view]
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 2],
            -trajectory[:, 1],
            "o-",
            color="black",
            markersize=3,
            linewidth=1,
        )
    axis.set(
        title=title,
        xlabel="world x",
        ylabel="world z",
        zlabel="-world y",
        xlim=(low[0], high[0]),
        ylim=(low[2], high[2]),
        zlim=(-high[1], -low[1]),
    )
    axis.set_box_aspect((1, 1, 1))


def render_point_cloud(
    *,
    output: OnlineGeometryOutput,
    views: int,
    image_path: Path,
    gif_path: Path,
) -> None:
    predicted, target, predicted_cameras, target_cameras = _point_cloud_payload(
        output, views=views
    )
    limits = _cloud_limits(predicted, target)
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    axes = (
        fig.add_subplot(121, projection="3d"),
        fig.add_subplot(122, projection="3d"),
    )
    _draw_cloud(
        axes[0],
        predicted,
        predicted_cameras,
        title="predicted world points @ +1.6s",
        limits=limits,
    )
    _draw_cloud(
        axes[1],
        target,
        target_cameras,
        title="VGGT teacher world points @ +1.6s",
        limits=limits,
    )
    fig.suptitle("Depth-colored geometry; black path is camera-center trajectory")
    _save_figure(fig, image_path)

    frames: list[Image.Image] = []
    for azimuth in np.linspace(0.0, 340.0, 18):
        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        axes = (
            fig.add_subplot(121, projection="3d"),
            fig.add_subplot(122, projection="3d"),
        )
        _draw_cloud(
            axes[0], predicted, predicted_cameras, title="prediction", limits=limits
        )
        _draw_cloud(axes[1], target, target_cameras, title="teacher", limits=limits)
        for axis in axes:
            axis.view_init(elev=22.0, azim=float(azimuth))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        frames.append(Image.fromarray(rgba).convert("RGB"))
        plt.close(fig)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )


def _run_action_free(
    pipeline: WorldCorePretrainingPipeline,
    *,
    window,
    context: torch.Tensor,
    context_mask: torch.Tensor,
) -> OnlineGeometryOutput:
    return pipeline.geometry_core(
        observed_images=window.observed_images.unsqueeze(0),
        state_history=window.state_history,
        action_history=window.action_history,
        future_world_times_s=window.future_world_times_s,
        mode=GeometryConditionMode.ACTION_FREE,
        future_action_history=None,
        future_target_images=None,
        language_features=context,
        language_padding_mask=context_mask,
        observed_view_valid_mask=window.observed_view_valid_mask.unsqueeze(0),
        future_view_valid_mask=window.future_view_valid_mask.unsqueeze(0),
        future_world_valid_mask=window.future_world_valid_mask.unsqueeze(0),
        decode_geometry_heads=True,
        compute_target_geometry=False,
        gradient_checkpointing=False,
    )


def _write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    cards: list[str] = []
    for sample in summary["samples"]:
        directory = sample["directory"]
        title = html.escape(f"{sample['source']} — {sample['episode_id']}")
        cards.append(
            f"""
            <section>
              <h2>{title}</h2>
              <p>{html.escape(sample["task_text"])}</p>
              <img src="{directory}/prediction_board.png" alt="prediction board">
              <img src="{directory}/action_conditioning.png" alt="action conditioning">
              <img src="{directory}/point_cloud.png" alt="point cloud">
              <img src="{directory}/point_cloud_orbit.gif" alt="point cloud orbit">
            </section>
            """
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>WM3D-WAM Stage A Eval</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:0 auto;padding:24px;background:#f4f6f8;color:#18212b}}
header,section{{background:white;border-radius:14px;padding:20px;margin:18px 0;box-shadow:0 2px 14px #0001}}
img{{max-width:100%;height:auto;display:block;margin:14px auto;border:1px solid #d9dee5;border-radius:8px}}
.warning{{border-left:5px solid #e76f51;padding-left:14px}}
</style></head><body>
<header><h1>WM3D-WAM Stage A — held-out world-core evaluation</h1>
<p class="warning"><strong>边界：</strong>Stage A 没有 RGB decoder。图中的 future RGB 是验证集参考帧，
不是模型生成结果；模型输出是 future feature、depth、world points 与 camera pose。</p>
<p>Checkpoint: {html.escape(str(summary["checkpoint"]))} | split: val | step: {summary["checkpoint_step"]}</p>
<img src="training_curves.png" alt="training curves"></header>
{"".join(cards)}
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--metrics", type=Path, default=DEFAULT_STAGE_A / "metrics.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--seed", type=int, default=20270819)
    parser.add_argument(
        "--data-profile",
        default="/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/data_profile.yaml",
    )
    parser.add_argument(
        "--source-contracts", default="configs/data/source_contracts_v1.yaml"
    )
    parser.add_argument(
        "--normalization",
        default="/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/grouped_normalization_1b.json",
    )
    parser.add_argument("--episode-splits", default="outputs/data/episode_splits_v1")
    parser.add_argument(
        "--wan-assets", default="/data/Minko/models/WM3D-WAM/Wan2.2-TI2V-5B"
    )
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
    parser.add_argument(
        "--model-config", default="configs/model/wan_action_mot_v1.yaml"
    )
    parser.add_argument(
        "--geometry-config", default="configs/model/vggt_geometry_v1.yaml"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("world-core evaluation requires one CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to exactly one permitted physical GPU"
        )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite non-empty eval directory {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_summary = render_training_curves(
        args.metrics.expanduser().resolve(strict=True),
        output_dir / "training_curves.png",
    )

    paths = _training_paths(args)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    built = build_training_model(
        phase="world_core_pretrain", paths=paths, device=device, dtype=dtype
    )
    if not isinstance(built.pipeline, WorldCorePretrainingPipeline):
        raise RuntimeError("Stage-A builder returned the wrong pipeline type")
    for head in (
        built.pipeline.geometry_core.encoder.dpt_head,
        built.pipeline.geometry_core.encoder.point_head,
        built.pipeline.geometry_core.encoder.camera_head,
    ):
        head.float()
    checkpoint_metadata = load_model_only(
        path_or_root=args.checkpoint, model=built.pipeline
    )
    built.pipeline.eval()
    prompt_cache = PromptEncoderCache(
        components=built.text_components,
        device=device,
        dtype=dtype,
        max_entries=max(8, len(args.sources)),
    )
    dataset = OnlineRobotDataset(
        data_profile_path=paths.data_profile,
        source_contract_path=paths.source_contracts,
        normalization_path=paths.normalization,
        split_root=paths.episode_splits,
        split="val",
        max_decode_retries=4,
    )
    requests = _requests_for_sources(dataset, sources=args.sources, seed=args.seed)

    samples: list[dict[str, Any]] = []
    for index, request in enumerate(requests, start=1):
        sample = dataset[request]
        window = sample.window.to(device=device, dtype=dtype)
        context, context_mask = prompt_cache.encode_batch(sample.task_texts)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            factual = built.pipeline(
                window=window,
                context=context,
                context_mask=context_mask,
                gradient_checkpointing=False,
            )
            action_free = _run_action_free(
                built.pipeline,
                window=window,
                context=context,
                context_mask=context_mask,
            )
        directory_name = f"demo_{index:02d}_{request.source}"
        directory = output_dir / directory_name
        directory.mkdir(parents=True, exist_ok=False)
        view = int(window.target_view_index)
        render_prediction_board(
            window=window,
            output=factual.geometry,
            view=view,
            path=directory / "prediction_board.png",
        )
        action_summary = render_action_conditioning_board(
            factual=factual.geometry,
            action_free=action_free,
            views=window.view_count,
            view=view,
            path=directory / "action_conditioning.png",
        )
        render_point_cloud(
            output=factual.geometry,
            views=window.view_count,
            image_path=directory / "point_cloud.png",
            gif_path=directory / "point_cloud_orbit.gif",
        )
        samples.append(
            {
                "directory": directory_name,
                "source": window.source,
                "episode_id": window.episode_id,
                "task_text": window.task_text,
                "view": view,
                "valid_view_count": window.valid_view_count,
                "sample_index": window.sample_index,
                "decode_retry_count": window.decode_retry_count,
                "losses": factual.detached_metrics(),
                "per_horizon": _metric_summary(
                    factual.geometry, views=window.view_count
                ),
                "action_conditioning": action_summary,
            }
        )
        del window, factual, action_free, context, context_mask
        torch.cuda.empty_cache()

    summary = {
        "schema": "wm3d_wam_world_core_eval_v1",
        "checkpoint": str(args.checkpoint.expanduser().resolve(strict=True)),
        "checkpoint_step": int(checkpoint_metadata["global_step"]),
        "split": "val",
        "seed": int(args.seed),
        "training_curves": curve_summary,
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
