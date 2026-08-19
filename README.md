# WM3D-WAM

WM3D-WAM combines an online VGGT geometry core, a Wan2.2 Video Expert, and a
source-native Grouped Action Flow Expert. Video supervision uses 5 Hz / 9
frames over 1.6 seconds. Robot actions remain at their original 5, 10, 15, or
20 Hz control clock.

The repository is a composition root rather than a fork of one upstream
project:

- FastWAM supplies the Wan2.2 VideoDiT, ActionDiT, layer-wise MoT attention,
  flow scheduler, and observed-video K/V prefill.
- the previous WM3D supplies timestamp selection and the grouped robot ABI;
- the node-41 VGGT-GAM work supplies VGGT split-and-resume and the geometry
  future-predictor baseline.

Current implementation includes the lossless grouped action path, batch-aware
interaction masks, Wan/Action MoT execution, one-call video K/V reuse, sparse
geometry K/V adapters, and the VGGT split encoder with camera, depth, and point
heads. The grouped history connector between the robot ABI and the GAM future
predictor is the next hard gate. No flattened fallback is enabled.

Documents:

- [v1 design](docs/WM3D_WAM_V1_DESIGN.md)
- [implementation base and current status](docs/IMPLEMENTATION_BASE.md)
- [third-party notices](THIRD_PARTY_NOTICES.md)

Run the current contract and model-component tests on New-H100-2:

```bash
cd /data/Minko/WM3D-WAM
PYTHONPATH=src /data/Minko/.venvs/wm3d/bin/python -m pytest -q
```
