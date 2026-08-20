import torch
import torch.nn as nn

from wm3d_wam.models.online_vggt_geometry import (
    GeometryTokenReducer,
    OnlineVGGTGeometryCore,
)


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


class _FakeShallowEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scene_batches: list[int] = []

    def encode_shallow_visual_slots(self, images, T, V):
        assert images.ndim == 5
        assert images.shape[1] == V
        assert T == 1
        self.scene_batches.append(int(images.shape[0]))
        value = images.mean(dim=(2, 3, 4), keepdim=False)
        visual = value[:, None, :, None, None].expand(-1, 1, -1, 2, 3)
        return {"visual_tokens": visual}


def test_shallow_scene_chunking_preserves_batch_time_and_view_layout() -> None:
    core = object.__new__(OnlineVGGTGeometryCore)
    nn.Module.__init__(core)
    core.encoder = _FakeShallowEncoder()
    core.shallow_scene_chunk_size = 2
    images = torch.arange(2 * 3 * 2 * 3 * 2 * 2, dtype=torch.float32).reshape(
        2, 3, 2, 3, 2, 2
    )

    encoded = core._encode_shallow(images)

    assert encoded.shape == (2, 3, 2, 2, 3)
    assert core.encoder.scene_batches == [2, 2, 2]
    expected = images.mean(dim=(3, 4, 5))
    torch.testing.assert_close(encoded[:, :, :, 0, 0], expected)
