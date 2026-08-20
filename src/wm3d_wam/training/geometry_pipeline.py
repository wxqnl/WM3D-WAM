"""World-core pretraining without loading Wan or ActionDiT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from wm3d_wam.data.online_episode import OnlineRobotWindow
from wm3d_wam.models.online_vggt_geometry import (
    GeometryConditionMode,
    OnlineGeometryOutput,
    OnlineVGGTGeometryCore,
)

from .objectives import GeometryObjectiveLoss, geometry_objective_loss


@dataclass(frozen=True)
class WorldCorePretrainingOutput:
    geometry: OnlineGeometryOutput
    geometry_objective: GeometryObjectiveLoss
    total_loss: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        metrics = self.geometry_objective.detached_metrics()
        metrics["loss_total"] = float(self.total_loss.detach())
        return metrics


class WorldCorePretrainingPipeline(nn.Module):
    """Train the dense K=16 WM3D state prior and sparse VGGT geometry path."""

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
    ) -> WorldCorePretrainingOutput:
        observed_mask = self._view_mask(window.observed_view_valid_mask)
        future_view_mask = self._view_mask(window.future_view_valid_mask)
        future_world_mask = self._view_mask(window.future_world_valid_mask)
        geometry = self.geometry_core(
            observed_images=self._images(window.observed_images),
            state_history=window.state_history,
            action_history=window.action_history,
            future_world_times_s=window.future_world_times_s,
            mode=GeometryConditionMode.FACTUAL,
            future_action_history=window.future_action_history,
            future_target_images=self._images(window.future_world_images),
            language_features=context,
            language_padding_mask=context_mask,
            observed_view_valid_mask=observed_mask,
            future_view_valid_mask=future_view_mask,
            future_world_valid_mask=future_world_mask,
            decode_geometry_heads=True,
            compute_target_geometry=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        geometry_objective = geometry_objective_loss(
            geometry,
            future_world_valid_mask=future_world_mask,
            feature_weight=1.0,
            action_free_feature_weight=0.25,
            geometry_weight=0.3,
        )
        total = geometry_objective.total
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("world-core pretraining loss is non-finite")
        return WorldCorePretrainingOutput(
            geometry=geometry,
            geometry_objective=geometry_objective,
            total_loss=total,
        )
