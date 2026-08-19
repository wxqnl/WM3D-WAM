import torch

from wm3d_wam.data.action_events import GroupedActionBatch
from wm3d_wam.training.flow_matching import (
    sample_grouped_action_flow,
    sample_video_flow,
    weighted_grouped_action_flow_loss,
    weighted_video_flow_loss,
)
from wm3d_wam.vendor.fastwam.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def _action_batch() -> GroupedActionBatch:
    values = torch.tensor([[[[1.0, 2.0]], [[3.0, 0.0]]]])
    value_mask = torch.tensor([[[[True, True]], [[True, False]]]])
    return GroupedActionBatch(
        values=values,
        value_mask=value_mask,
        event_mask=torch.tensor([[True, True]]),
        times_s=torch.tensor([[0.0, 0.2]]),
        event_dt_s=torch.tensor([[0.0, 0.2]]),
        group_ids=torch.tensor([[1]]),
        group_mask=torch.tensor([[True]]),
        action_semantic_ids=torch.tensor([[[1, 1]]]),
        composition_operator_ids=torch.tensor([[[1, 1]]]),
        embodiment_ids=torch.tensor([1]),
    )


def test_grouped_action_flow_never_noises_padded_scalars():
    scheduler = WanContinuousFlowMatchScheduler()
    batch = _action_batch()
    noise = torch.full_like(batch.values, 5.0)
    sample = sample_grouped_action_flow(
        batch,
        scheduler,
        timestep=torch.tensor([500.0]),
        noise=noise,
    )
    assert sample.noisy.values[0, 1, 0, 1].item() == 0.0
    assert sample.target_velocity[0, 1, 0, 1].item() == 0.0
    assert torch.allclose(
        sample.noisy.values[batch.value_mask],
        0.5 * batch.values[batch.value_mask] + 0.5 * noise[batch.value_mask],
    )
    assert weighted_grouped_action_flow_loss(
        sample.target_velocity, sample
    ).item() == 0.0


def test_video_flow_preserves_observed_latent_and_scores_only_future():
    scheduler = WanContinuousFlowMatchScheduler()
    clean = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 1, 2)
    noise = torch.full_like(clean, -2.0)
    sample = sample_video_flow(
        clean,
        scheduler,
        timestep=torch.tensor([250.0]),
        noise=noise,
    )
    assert torch.equal(sample.noisy_latents[:, :, 0], clean[:, :, 0])
    assert torch.equal(sample.target_velocity[:, :, 0], torch.zeros_like(clean[:, :, 0]))
    assert sample.prediction_mask.tolist() == [[False, True, True]]
    assert weighted_video_flow_loss(sample.target_velocity, sample).item() == 0.0
