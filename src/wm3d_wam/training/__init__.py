"""Training objectives and orchestration for WM3D-WAM."""

from .flow_matching import (
    GroupedActionFlowSample,
    VideoFlowSample,
    sample_grouped_action_flow,
    sample_video_flow,
    weighted_grouped_action_flow_loss,
    weighted_video_flow_loss,
)
from .geometry_pipeline import GeometryPretrainingOutput, GeometryPretrainingPipeline
from .objectives import (
    GeometryObjectiveLoss,
    GroupedAuxiliaryActionLoss,
    geometry_objective_loss,
    grouped_auxiliary_action_loss,
)
from .parameter_groups import (
    TrainingStage,
    configure_geometry_pretraining_parameter_groups,
    configure_stage_parameter_groups,
)
from .pipeline import WM3DWAMTrainingOutput, WM3DWAMTrainingPipeline

__all__ = [
    "GeometryObjectiveLoss",
    "GeometryPretrainingOutput",
    "GeometryPretrainingPipeline",
    "GroupedActionFlowSample",
    "GroupedAuxiliaryActionLoss",
    "TrainingStage",
    "VideoFlowSample",
    "WM3DWAMTrainingOutput",
    "WM3DWAMTrainingPipeline",
    "configure_stage_parameter_groups",
    "configure_geometry_pretraining_parameter_groups",
    "geometry_objective_loss",
    "grouped_auxiliary_action_loss",
    "sample_grouped_action_flow",
    "sample_video_flow",
    "weighted_grouped_action_flow_loss",
    "weighted_video_flow_loss",
]
