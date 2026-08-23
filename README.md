# WM3D-WAM

WM3D-WAM is a world-action model built from three large, reusable components:

- the original WM3D state prior and factual dynamics, adapted to VGGT tokens;
- online VGGT-1B shallow encoding and deep geometry propagation;
- Wan2.2 TI2V-5B plus a Wan-coupled Grouped Action Flow Expert.

The world grid is fixed at `K=16` over 1.6 seconds, or 10 Hz. Wan receives the
current frame plus all 16 future frames, so a video window contains 17 RGB
frames and encodes to five temporal latent positions. Robot commands keep their
recorded 5/10/15/20 Hz clock. They are assigned to 16 physical-time bins
without interpolation, repetition, or rate conversion.

VGGT runs inside every model forward. Training reads raw Parquet and MP4 and
does not require a VGGT, depth, point, pose, or Wan-latent cache.

## Architecture contract

- Frozen VGGT pairs 0–3 encode four observed multi-view keyframes.
- The 876M-parameter WM3D core fuses real views, robot history, task language,
  and physical time, then predicts 16 action-free future world states.
- A separate factual-dynamics block refines that completed prior with recorded
  future actions when the route permits factual conditioning.
- A per-view decoder maps every dense world state back to the VGGT shallow-token
  ABI.
- Trainable VGGT pairs 4–23 resume at 0.4/0.8/1.2/1.6-second anchors and produce
  geometry tokens for Wan/Action attention.
- Wan2.2 owns RGB velocity prediction. Grouped ActionDiT owns action velocity
  prediction. They reuse FastWAM's mixed-attention layers with route-specific
  directionality: clean, time-aligned actions causally condition RGB in
  `forward_world`, while observed/noisy video conditions action prediction in
  `action_only` and `joint_world_action`.

GAM is not the world-model core and has no active policy or action head in the
factory graph. The repository keeps its vendored VGGT adapter as implementation
provenance for the shallow/deep split.

## Data contract

Seven audited sources enter training; fourteen sources stay excluded until
their stored controller semantics are proven. Splits are materialized at the
episode or parent-trajectory level before window sampling.

For sources at 10 Hz or faster, all 16 world targets are real recorded frames.
A genuine 5 Hz source provides eight real targets at 0.2-second intervals;
those supervise slots `1,3,...,15`, while the missing 0.1-second slots are
masked. The loader never fills a missing world step by copying a frame.

The active routes are:

- `world_core_pretrain`: factual world rollout with online feature and geometry
  supervision; Wan and ActionDiT are not loaded;
- `action_only`: action flow from the observed frame, task, robot history, and
  action-free world state; the action loss cannot update Wan or VGGT;
- `forward_world`: video flow and factual world dynamics conditioned on clean
  recorded actions through an exact group-diagonal mapping from 16 physical bins
  to four future Wan latent groups;
- `joint_world_action`: video and action flow are trained together, but video
  cannot read noisy action tokens; video supplies context to action, matching
  mature FastWAM joint directionality and preventing target leakage.

## Repository layout

```text
src/wm3d_wam/data/       source contracts, time windows, grouped robot ABI, RGB I/O
src/wm3d_wam/models/     WM3D state core, online VGGT, Wan/Action MoT
src/wm3d_wam/training/   objectives, parameter ownership, FSDP, checkpoints
src/wm3d_wam/vendor/     pinned FastWAM and VGGT adapter code
scripts/                 split materialization, preflights, formal trainer
configs/                 production data, model, and training contracts
```

## Tests

Run only on New-H100-2 from `/data/Minko/WM3D-WAM`:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
  /data/Minko/.venvs/wm3d/bin/python -m pytest -q
```

GPU 0 is reserved. GPU work may use physical devices 1–7 only.

## Documents

- [complete design](docs/WM3D_WAM_V1_DESIGN.md)
- [implementation base and status](docs/IMPLEMENTATION_BASE.md)
- [training runbook](docs/TRAINING.md)
- [validation experiments](docs/EXPERIMENTS.md)
- [third-party notices](THIRD_PARTY_NOTICES.md)
