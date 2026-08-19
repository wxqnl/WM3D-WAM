"""Training-program orchestration for the full WM3D-WAM computation graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from wm3d_wam.data.online_episode import OnlineRobotWindow
from wm3d_wam.models.interaction_masks import InteractionProgram
from wm3d_wam.models.system import WM3DWAMProgramOutput, WM3DWAMSystem
from wm3d_wam.vendor.fastwam.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)

from .flow_matching import (
    GroupedActionFlowSample,
    VideoFlowSample,
    sample_grouped_action_flow,
    sample_video_flow,
    weighted_grouped_action_flow_loss,
    weighted_video_flow_loss,
)
from .objectives import (
    GeometryObjectiveLoss,
    GroupedAuxiliaryActionLoss,
    geometry_objective_loss,
    grouped_auxiliary_action_loss,
)


@dataclass(frozen=True)
class WM3DWAMTrainingOutput:
    program_output: WM3DWAMProgramOutput
    total_loss: torch.Tensor
    action_loss: torch.Tensor
    video_loss: torch.Tensor
    geometry_loss: torch.Tensor
    auxiliary_action_loss: torch.Tensor
    action_flow: Optional[GroupedActionFlowSample]
    video_flow: Optional[VideoFlowSample]
    geometry_objective: Optional[GeometryObjectiveLoss]
    auxiliary_action_objective: Optional[GroupedAuxiliaryActionLoss]

    def detached_metrics(self) -> dict[str, float]:
        metrics = {
            "loss_total": float(self.total_loss.detach()),
            "loss_action": float(self.action_loss.detach()),
            "loss_video": float(self.video_loss.detach()),
            "loss_geometry": float(self.geometry_loss.detach()),
            "loss_auxiliary_action": float(self.auxiliary_action_loss.detach()),
        }
        if self.geometry_objective is not None:
            for name, value in self.geometry_objective.detached_metrics().items():
                if name != "loss_total":
                    metrics[f"geometry_{name}"] = value
        if self.auxiliary_action_objective is not None:
            metrics.update(self.auxiliary_action_objective.detached_metrics())
        return metrics


class WM3DWAMTrainingPipeline(nn.Module):
    """Create flow targets and execute one of the three deployed programs."""

    def __init__(
        self,
        *,
        system: WM3DWAMSystem,
        action_scheduler: WanContinuousFlowMatchScheduler | None = None,
        video_scheduler: WanContinuousFlowMatchScheduler | None = None,
    ) -> None:
        super().__init__()
        self.system = system
        self.action_scheduler = action_scheduler or WanContinuousFlowMatchScheduler()
        self.video_scheduler = video_scheduler or WanContinuousFlowMatchScheduler()

    def forward(
        self,
        *,
        program: InteractionProgram | str,
        window: OnlineRobotWindow,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        clean_video_latents: Optional[torch.Tensor] = None,
        action_timestep: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
        video_timestep: Optional[torch.Tensor] = None,
        video_noise: Optional[torch.Tensor] = None,
        geometry_gradient_checkpointing: bool = False,
    ) -> WM3DWAMTrainingOutput:
        mode = InteractionProgram(program)
        if clean_video_latents is None:
            clean_video_latents = self.system.encode_wan_video(window.wan_video)
        if clean_video_latents.shape[0] != window.batch_size:
            raise ValueError("Wan latent batch does not match the robot window")
        device = clean_video_latents.device
        dtype = clean_video_latents.dtype
        clean_actions = window.future_actions.to(device=device, dtype=dtype)
        batch_size = int(clean_video_latents.shape[0])
        zero_timestep = torch.zeros((batch_size,), device=device, dtype=dtype)

        action_flow = None
        video_flow = None
        if mode is not InteractionProgram.FORWARD_WORLD:
            action_flow = sample_grouped_action_flow(
                clean_actions,
                self.action_scheduler,
                timestep=action_timestep,
                noise=action_noise,
            )
        if mode is not InteractionProgram.ACTION_ONLY:
            video_flow = sample_video_flow(
                clean_video_latents,
                self.video_scheduler,
                timestep=video_timestep,
                noise=video_noise,
            )

        if mode is InteractionProgram.ACTION_ONLY:
            assert action_flow is not None
            model_video = clean_video_latents[:, :, :1]
            model_video_timestep = zero_timestep
            model_actions = action_flow.noisy
            model_action_timestep = action_flow.timestep
            include_future_targets = False
            decode_geometry_heads = False
            compute_target_geometry = False
        elif mode is InteractionProgram.FORWARD_WORLD:
            assert video_flow is not None
            model_video = video_flow.noisy_latents
            model_video_timestep = video_flow.timestep
            model_actions = clean_actions
            model_action_timestep = zero_timestep
            include_future_targets = True
            decode_geometry_heads = True
            compute_target_geometry = True
        else:
            assert action_flow is not None and video_flow is not None
            model_video = video_flow.noisy_latents
            model_video_timestep = video_flow.timestep
            model_actions = action_flow.noisy
            model_action_timestep = action_flow.timestep
            include_future_targets = True
            decode_geometry_heads = False
            compute_target_geometry = False

        program_output = self.system.forward_program(
            program=mode,
            window=window,
            video_latents=model_video,
            video_timestep=model_video_timestep,
            action_batch=model_actions,
            action_timestep=model_action_timestep,
            context=context,
            context_mask=context_mask,
            include_future_targets=include_future_targets,
            decode_geometry_heads=decode_geometry_heads,
            compute_target_geometry=compute_target_geometry,
            geometry_gradient_checkpointing=geometry_gradient_checkpointing,
        )
        zero = clean_video_latents.new_zeros((), dtype=torch.float32)
        action_loss = zero
        if action_flow is not None:
            if program_output.mot.action_velocity is None:
                raise RuntimeError("action program returned no action velocity")
            action_loss = weighted_grouped_action_flow_loss(
                program_output.mot.action_velocity, action_flow
            )
        video_loss = zero
        if video_flow is not None:
            if program_output.mot.video_velocity is None:
                raise RuntimeError("video program returned no video velocity")
            video_loss = weighted_video_flow_loss(
                program_output.mot.video_velocity, video_flow
            )

        geometry_objective = None
        geometry_loss = zero
        if mode is InteractionProgram.FORWARD_WORLD:
            geometry_objective = geometry_objective_loss(
                program_output.geometry,
                future_view_valid_mask=self.system._batched_view_mask(
                    window.future_view_valid_mask,
                    name="future_view_valid_mask",
                ),
                feature_weight=1.0,
                geometry_weight=0.3,
            )
            geometry_loss = geometry_objective.total
        elif mode is InteractionProgram.JOINT_WORLD_ACTION:
            geometry_objective = geometry_objective_loss(
                program_output.geometry,
                future_view_valid_mask=self.system._batched_view_mask(
                    window.future_view_valid_mask,
                    name="future_view_valid_mask",
                ),
                feature_weight=0.5,
                geometry_weight=0.0,
            )
            geometry_loss = geometry_objective.total
        auxiliary_action_objective = None
        auxiliary_action_loss = zero
        if mode is InteractionProgram.ACTION_ONLY:
            if program_output.auxiliary_actions is None:
                raise RuntimeError("action-only program returned no auxiliary actions")
            auxiliary_action_objective = grouped_auxiliary_action_loss(
                program_output.auxiliary_actions,
                clean_actions,
            )
            auxiliary_action_loss = 0.1 * auxiliary_action_objective.total
        total = action_loss + video_loss + geometry_loss + auxiliary_action_loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("WM3D-WAM training loss is non-finite")
        return WM3DWAMTrainingOutput(
            program_output=program_output,
            total_loss=total,
            action_loss=action_loss,
            video_loss=video_loss,
            geometry_loss=geometry_loss,
            auxiliary_action_loss=auxiliary_action_loss,
            action_flow=action_flow,
            video_flow=video_flow,
            geometry_objective=geometry_objective,
            auxiliary_action_objective=auxiliary_action_objective,
        )
