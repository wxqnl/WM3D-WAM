import pytest
import torch

from wm3d_wam.vendor.vggt_gam.future_predictor import GAMFuturePredictor


def test_gam_predictor_accepts_grouped_preembedded_history():
    predictor = GAMFuturePredictor(
        d_da3=32,
        d_model=32,
        depth=1,
        num_heads=4,
        ffn_ratio=2.0,
        num_patches_per_view=4,
        num_register_tokens=1,
        use_language=False,
        proprio_dim=3,
        action_dim=2,
        action_chunk_size=1,
        input_proj_norm="ln",
    )
    visual = torch.randn(1, 2, 1, 6, 32)
    state_tokens = torch.randn(1, 2, 32)
    action_tokens = torch.randn(1, 2, 32)
    output = predictor(
        past_visual_tokens=visual,
        proprio_token_embeddings=state_tokens,
        past_action_token_embeddings=action_tokens,
    )
    assert output["predicted_next_visual_tokens"].shape == visual.shape
    assert output["predicted_action_tokens"].shape == (1, 2, 1, 32)
    assert output["predicted_next_proprio_tokens"].shape == (1, 2, 32)
    assert output["predicted_next_action_history_tokens"].shape == (1, 2, 32)

    with pytest.raises(ValueError, match="not both"):
        predictor(
            past_visual_tokens=visual,
            proprio_history=torch.randn(1, 2, 3),
            proprio_token_embeddings=state_tokens,
            past_action_token_embeddings=action_tokens,
        )
