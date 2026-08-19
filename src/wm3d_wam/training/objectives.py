"""Losses whose masks match the online geometry/action contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from wm3d_wam.data.action_events import GroupedActionBatch
from wm3d_wam.models.geometry_action_heads import GroupedAuxiliaryActionOutput
from wm3d_wam.models.online_vggt_geometry import OnlineGeometryOutput


@dataclass(frozen=True)
class GeometryObjectiveLoss:
    total: torch.Tensor
    future_feature: torch.Tensor
    depth: torch.Tensor
    world_points: torch.Tensor
    camera_pose: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach()),
            "loss_future_feature": float(self.future_feature.detach()),
            "loss_depth": float(self.depth.detach()),
            "loss_world_points": float(self.world_points.detach()),
            "loss_camera_pose": float(self.camera_pose.detach()),
        }


@dataclass(frozen=True)
class GroupedAuxiliaryActionLoss:
    total: torch.Tensor
    direct: torch.Tensor
    refined: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_aux_action": float(self.total.detach()),
            "loss_aux_action_direct": float(self.direct.detach()),
            "loss_aux_action_refined": float(self.refined.detach()),
        }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=value.device, dtype=value.dtype)
    while weights.ndim < value.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(value)
    denominator = weights.sum().clamp_min(1.0)
    return (value * weights).sum() / denominator


def _future_student_rows(
    value: torch.Tensor,
    *,
    batch_size: int,
    total_steps: int,
    future_steps: int,
    views: int,
) -> torch.Tensor:
    expected = batch_size * total_steps * views
    if value.shape[0] != expected:
        raise ValueError(
            f"student geometry row count {value.shape[0]} != B*T*V={expected}"
        )
    return value.reshape(batch_size, total_steps, views, *value.shape[1:])[
        :, -future_steps:
    ]


def geometry_objective_loss(
    output: OnlineGeometryOutput,
    *,
    future_view_valid_mask: torch.Tensor,
    feature_weight: float = 1.0,
    geometry_weight: float = 0.3,
) -> GeometryObjectiveLoss:
    """Stage-A/forward-world geometry loss with detached online targets."""

    target = output.target_future_shallow_tokens
    if target is None:
        raise ValueError("future shallow target is required for geometry training")
    prediction = output.predicted_future_shallow_tokens
    if prediction.shape != target.shape:
        raise ValueError("predicted and target shallow token shapes do not match")
    b, future_steps, views = prediction.shape[:3]
    view_mask = future_view_valid_mask.to(
        device=prediction.device, dtype=torch.bool
    )
    if view_mask.shape != (b, future_steps, views):
        raise ValueError("future view mask shape mismatch")
    if target.requires_grad:
        raise ValueError("online VGGT target tensors must be detached")

    cosine = 1.0 - F.cosine_similarity(
        prediction.float(), target.float(), dim=-1
    )
    feature_loss = _masked_mean(cosine, view_mask)
    zero = feature_loss.new_zeros(())
    depth_loss = zero
    points_loss = zero
    pose_loss = zero
    geometry_terms: list[torch.Tensor] = []

    target_geometry = output.target_geometry
    if target_geometry is not None:
        total_steps = int(output.deep_visual_tokens.shape[1])
        student = output.student_geometry
        for key in ("depth", "world_points", "pose_enc"):
            if key in target_geometry and target_geometry[key].requires_grad:
                raise ValueError(f"target geometry {key} must be detached")
        if "depth" in student and "depth" in target_geometry:
            depth_prediction = _future_student_rows(
                student["depth"],
                batch_size=b,
                total_steps=total_steps,
                future_steps=future_steps,
                views=views,
            )
            depth_target = target_geometry["depth"].reshape_as(depth_prediction)
            valid_depth = view_mask
            depth_loss = _masked_mean(
                F.smooth_l1_loss(
                    depth_prediction.float(), depth_target.float(), reduction="none"
                ),
                valid_depth,
            )
            geometry_terms.append(depth_loss)
        if "world_points" in student and "world_points" in target_geometry:
            point_prediction = _future_student_rows(
                student["world_points"],
                batch_size=b,
                total_steps=total_steps,
                future_steps=future_steps,
                views=views,
            )
            point_target = target_geometry["world_points"].reshape_as(
                point_prediction
            )
            points_loss = _masked_mean(
                F.smooth_l1_loss(
                    point_prediction.float(), point_target.float(), reduction="none"
                ),
                view_mask,
            )
            geometry_terms.append(points_loss)
        if "pose_enc" in student and "pose_enc" in target_geometry:
            pose_prediction = _future_student_rows(
                student["pose_enc"],
                batch_size=b,
                total_steps=total_steps,
                future_steps=future_steps,
                views=views,
            )
            pose_target = target_geometry["pose_enc"].reshape_as(pose_prediction)
            pose_loss = _masked_mean(
                F.smooth_l1_loss(
                    pose_prediction.float(), pose_target.float(), reduction="none"
                ),
                view_mask,
            )
            geometry_terms.append(pose_loss)

    geometry = torch.stack(geometry_terms).mean() if geometry_terms else zero
    total = float(feature_weight) * feature_loss + float(geometry_weight) * geometry
    return GeometryObjectiveLoss(
        total=total,
        future_feature=feature_loss,
        depth=depth_loss,
        world_points=points_loss,
        camera_pose=pose_loss,
    )


def grouped_auxiliary_action_loss(
    prediction: GroupedAuxiliaryActionOutput,
    target: GroupedActionBatch,
) -> GroupedAuxiliaryActionLoss:
    """Normalize direct/refined action losses by valid native-rate scalars."""

    if prediction.direct.shape != target.values.shape or prediction.refined.shape != target.values.shape:
        raise ValueError("auxiliary action predictions must match grouped targets")
    valid = (
        target.value_mask.to(dtype=torch.bool)
        & target.event_mask[:, :, None, None].to(dtype=torch.bool)
        & target.group_mask[:, None, :, None].to(dtype=torch.bool)
    )
    counts = valid.flatten(1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("every auxiliary action sample needs a valid scalar")

    def loss(value: torch.Tensor) -> torch.Tensor:
        elementwise = F.smooth_l1_loss(
            value.float(), target.values.float(), reduction="none"
        )
        return ((elementwise * valid).flatten(1).sum(dim=1) / counts).mean()

    direct = loss(prediction.direct)
    refined = loss(prediction.refined)
    return GroupedAuxiliaryActionLoss(
        total=0.5 * (direct + refined),
        direct=direct,
        refined=refined,
    )
