"""Stage-A online VGGT-GAM objective without loading Wan video generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from wm3d_wam.data.online_episode import OnlineRobotWindow
from wm3d_wam.models.geometry_action_heads import GroupedAuxiliaryActionOutput
from wm3d_wam.models.online_vggt_geometry import (
    GeometryConditionMode,
    OnlineGeometryOutput,
    OnlineVGGTGeometryCore,
)

from .objectives import (
    GeometryObjectiveLoss,
    GroupedAuxiliaryActionLoss,
    geometry_objective_loss,
    grouped_auxiliary_action_loss,
)


@dataclass(frozen=True)
class GeometryPretrainingOutput:
    geometry: OnlineGeometryOutput
    auxiliary_actions: GroupedAuxiliaryActionOutput
    geometry_objective: GeometryObjectiveLoss
    auxiliary_action_objective: GroupedAuxiliaryActionLoss
    total_loss: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        metrics = self.geometry_objective.detached_metrics()
        metrics.update(self.auxiliary_action_objective.detached_metrics())
        metrics["loss_total"] = float(self.total_loss.detach())
        return metrics


class GeometryPretrainingPipeline(nn.Module):
    """Run Stage A entirely online from RGB and grouped robot history."""

    def __init__(self, geometry_core: OnlineVGGTGeometryCore) -> None:
        super().__init__()
        self.geometry_core = geometry_core

    @staticmethod
    def _images(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0) if value.ndim == 5 else value

    @staticmethod
    def _view_mask(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0) if value.ndim == 2 else value

    def forward(
        self,
        *,
        window: OnlineRobotWindow,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = False,
    ) -> GeometryPretrainingOutput:
        observed_mask = self._view_mask(window.observed_view_valid_mask)
        future_mask = self._view_mask(window.future_view_valid_mask)
        geometry = self.geometry_core(
            observed_images=self._images(window.observed_images),
            state_history=window.state_history,
            action_history=window.action_history,
            mode=GeometryConditionMode.POLICY,
            future_action_history=None,
            future_target_images=self._images(window.future_anchor_images),
            language_features=context,
            language_padding_mask=context_mask,
            observed_view_valid_mask=observed_mask,
            future_view_valid_mask=future_mask,
            decode_geometry_heads=True,
            compute_target_geometry=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        auxiliary_actions = self.geometry_core.predict_auxiliary_actions(
            geometry,
            action_template=window.future_actions,
            future_view_valid_mask=future_mask,
        )
        geometry_objective = geometry_objective_loss(
            geometry,
            future_view_valid_mask=future_mask,
            feature_weight=1.0,
            geometry_weight=0.3,
        )
        auxiliary_objective = grouped_auxiliary_action_loss(
            auxiliary_actions,
            window.future_actions,
        )
        total = geometry_objective.total + 0.1 * auxiliary_objective.total
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Stage-A geometry loss is non-finite")
        return GeometryPretrainingOutput(
            geometry=geometry,
            auxiliary_actions=auxiliary_actions,
            geometry_objective=geometry_objective,
            auxiliary_action_objective=auxiliary_objective,
            total_loss=total,
        )
