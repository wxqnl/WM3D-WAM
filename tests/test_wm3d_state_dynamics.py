from __future__ import annotations

import torch

from wm3d_wam.models.wm3d_state_dynamics import (
    WM3DStateDynamicsConfig,
    WM3DStateDynamicsCore,
)


def _config() -> WM3DStateDynamicsConfig:
    return WM3DStateDynamicsConfig(
        observed_steps=2,
        future_steps=16,
        token_count=4,
        token_dim=16,
        max_views=2,
        history_dim=12,
        language_dim=20,
        state_hidden=32,
        state_layers=2,
        state_heads=4,
        state_ff_mult=2.0,
        dynamics_layers=1,
        view_hidden=16,
        view_heads=4,
        view_ff_mult=2.0,
        time_fourier_dim=8,
        activation_checkpointing=False,
    )


def _batch(config: WM3DStateDynamicsConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    batch = 2
    observed = torch.tensor([-0.2, 0.0]).view(1, 2).expand(batch, -1)
    future = (
        torch.arange(1, config.future_steps + 1).float().view(1, -1) * 0.1
    ).expand(batch, -1)
    return {
        "observed_tokens": torch.randn(
            batch,
            config.observed_steps,
            config.max_views,
            config.token_count,
            config.token_dim,
        ),
        "observed_view_mask": torch.tensor(
            [[[True, True], [True, True]], [[True, False], [True, False]]]
        ),
        "world_times_s": torch.cat((observed, future), dim=1),
        "history_state_tokens": torch.randn(
            batch, config.observed_steps, config.history_dim
        ),
        "history_action_tokens": torch.randn(
            batch, config.observed_steps, config.history_dim
        ),
        "language_context": torch.randn(batch, 5, config.language_dim),
        "language_mask": torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        ),
    }


def test_k16_state_prior_and_factual_dynamics_are_separate() -> None:
    config = _config()
    model = WM3DStateDynamicsCore(config).eval()
    batch = _batch(config)
    action = torch.randn(2, config.future_steps, config.history_dim)
    action_mask = torch.ones(2, config.future_steps, dtype=torch.bool)

    first = model(
        **batch,
        factual_action_tokens=action,
        factual_action_mask=action_mask,
    )
    changed = model(
        **batch,
        factual_action_tokens=action + 10.0,
        factual_action_mask=action_mask,
    )

    assert first.action_free_tokens.shape == (
        2,
        16,
        2,
        4,
        16,
    )
    torch.testing.assert_close(
        first.action_free_native_state,
        changed.action_free_native_state,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.action_free_tokens,
        changed.action_free_tokens,
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(first.native_state, changed.native_state)
    assert not torch.allclose(first.factual_tokens, changed.factual_tokens)


def test_world_loss_reaches_state_prior_dynamics_and_vggt_token_decoder() -> None:
    config = _config()
    model = WM3DStateDynamicsCore(config).train()
    action = torch.randn(2, config.future_steps, config.history_dim)
    output = model(
        **_batch(config),
        factual_action_tokens=action,
        factual_action_mask=torch.ones(2, 16, dtype=torch.bool),
    )
    output.factual_tokens.square().mean().backward()

    named = dict(model.named_parameters())
    for prefix in ("state_blocks.0", "dynamics_blocks.0", "token_decoder.output"):
        gradients = [
            parameter.grad
            for name, parameter in named.items()
            if name.startswith(prefix)
        ]
        assert gradients
        assert any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
            for gradient in gradients
        )

