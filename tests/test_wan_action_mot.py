from __future__ import annotations

import torch

from wm3d_wam.models.geometry_adapters import SparseGeometryKVAdapters
from wm3d_wam.models.interaction_masks import InteractionProgram
from wm3d_wam.models.wan_action_mot import WanActionMoT
from wm3d_wam.vendor.fastwam.wan22.wan_video_dit import WanVideoDiT

from test_grouped_action_flow import _batch, _expert


def _model(*, with_geometry: bool = False) -> WanActionMoT:
    video = WanVideoDiT(
        hidden_dim=16,
        in_dim=4,
        ffn_dim=32,
        out_dim=4,
        text_dim=12,
        freq_dim=8,
        eps=1.0e-6,
        patch_size=(1, 1, 1),
        num_heads=2,
        attn_head_dim=4,
        num_layers=2,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
        video_attention_mask_mode="first_frame_causal",
    )
    action = _expert(hidden_dim=8, num_layers=2)
    geometry_adapters = None
    if with_geometry:
        geometry_adapters = SparseGeometryKVAdapters(
            geometry_dim=6,
            num_heads=2,
            attention_head_dim=4,
            fusion_layers=(0, 1),
        )
    return WanActionMoT(
        video_expert=video,
        action_expert=action,
        geometry_adapters=geometry_adapters,
        mot_checkpoint_mixed_attn=False,
    )


def test_experts_each_have_one_registered_module_path() -> None:
    model = _model()

    video_paths = [
        name
        for name, module in model.named_modules(remove_duplicate=False)
        if module is model.video_expert
    ]
    action_paths = [
        name
        for name, module in model.named_modules(remove_duplicate=False)
        if module is model.action_expert
    ]

    assert video_paths == ["mot.mixtures.video"]
    assert action_paths == ["mot.mixtures.action"]
    assert not any(name.startswith("video_expert.") for name in model.state_dict())
    assert not any(name.startswith("action_expert.") for name in model.state_dict())


def test_action_only_runs_through_wan_mot_and_omits_video_decoder() -> None:
    torch.manual_seed(11)
    model = _model().eval()
    batch = _batch(max_events=8)
    observed_latents = torch.randn(2, 4, 1, 2, 2)
    context = torch.randn(2, 5, 12)
    context_mask = torch.ones(2, 5, dtype=torch.bool)

    output = model(
        program=InteractionProgram.ACTION_ONLY,
        video_latents=observed_latents,
        video_timestep=torch.zeros(2),
        action_batch=batch,
        action_timestep=torch.tensor([100.0, 600.0]),
        context=context,
        context_mask=context_mask,
    )

    assert output.video_velocity is None
    assert output.action_velocity is not None
    assert output.action_velocity.shape == batch.values.shape
    assert not output.action_velocity[0, 2:].any()


def test_observed_video_cache_matches_full_action_only_path() -> None:
    torch.manual_seed(17)
    model = _model().eval()
    batch = _batch(max_events=8)
    observed_latents = torch.randn(2, 4, 1, 2, 2)
    context = torch.randn(2, 5, 12)
    context_mask = torch.ones(2, 5, dtype=torch.bool)
    action_timestep = torch.tensor([150.0, 550.0])

    full = model(
        program=InteractionProgram.ACTION_ONLY,
        video_latents=observed_latents,
        video_timestep=torch.zeros(2),
        action_batch=batch,
        action_timestep=action_timestep,
        context=context,
        context_mask=context_mask,
    ).action_velocity
    cache = model.prefill_observed_video(
        observed_video_latents=observed_latents,
        context=context,
        context_mask=context_mask,
    )
    cached = model.action_velocity_from_cache(
        action_batch=batch,
        action_timestep=action_timestep,
        context=context,
        context_mask=context_mask,
        cache=cache,
    )

    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-5)


def test_predicted_geometry_uses_sparse_kv_path_and_preserves_cache_parity() -> None:
    torch.manual_seed(23)
    model = _model(with_geometry=True).eval()
    batch = _batch(max_events=8)
    observed_latents = torch.randn(2, 4, 1, 2, 2)
    context = torch.randn(2, 5, 12)
    context_mask = torch.ones(2, 5, dtype=torch.bool)
    action_timestep = torch.tensor([200.0, 500.0])
    geometry = torch.randn(2, 3, 6)
    geometry_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)

    full = model(
        program=InteractionProgram.ACTION_ONLY,
        video_latents=observed_latents,
        video_timestep=torch.zeros(2),
        action_batch=batch,
        action_timestep=action_timestep,
        context=context,
        context_mask=context_mask,
        geometry_tokens=geometry,
        geometry_token_mask=geometry_mask,
    ).action_velocity
    cache = model.prefill_observed_video(
        observed_video_latents=observed_latents,
        context=context,
        context_mask=context_mask,
        geometry_tokens=geometry,
        geometry_token_mask=geometry_mask,
    )
    cached = model.action_velocity_from_cache(
        action_batch=batch,
        action_timestep=action_timestep,
        context=context,
        context_mask=context_mask,
        cache=cache,
    )

    assert set(cache.geometry_kv) == {0, 1}
    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-5)
