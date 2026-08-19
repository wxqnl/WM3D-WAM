import torch

from wm3d_wam.vendor.fastwam.wan22.wan_video_vae import WanVideoVAE38
from wm3d_wam.vendor.fastwam.wan22.wan_video_dit import precompute_freqs_cis_3d


def test_wan_vae_scale_stays_materialized_during_meta_construction():
    with torch.device("meta"):
        vae = WanVideoVAE38()
    assert next(vae.model.parameters()).is_meta
    assert not vae.mean.is_meta
    assert not vae.std.is_meta
    assert torch.isfinite(vae.mean).all()
    assert torch.isfinite(vae.std).all()


def test_wan_rope_tables_stay_materialized_during_meta_construction():
    with torch.device("meta"):
        tables = precompute_freqs_cis_3d(128)
    assert all(value.device.type == "cpu" for value in tables)
    assert all(torch.isfinite(value).all() for value in tables)
