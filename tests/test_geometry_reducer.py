import torch

from wm3d_wam.models.online_vggt_geometry import GeometryTokenReducer


def test_geometry_reducer_preserves_view_mask_and_width():
    reducer = GeometryTokenReducer(
        token_dim=16,
        num_register_tokens=1,
        source_patch_grid=4,
        output_patch_grid=2,
    )
    tokens = torch.randn(2, 3, 2, 18, 16)
    valid = torch.tensor(
        [
            [[True, True], [True, False], [True, True]],
            [[True, True], [False, True], [True, True]],
        ]
    )
    reduced, mask = reducer(tokens, valid)
    assert reduced.shape == (2, 3 * 2 * 6, 16)
    assert mask.shape == (2, 3 * 2 * 6)
    assert torch.count_nonzero(reduced[~mask]) == 0
