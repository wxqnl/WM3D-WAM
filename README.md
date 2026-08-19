# WM3D-WAM

WM3D-WAM is a world-action model that combines three production-width paths:

- online VGGT-1B split-and-resume geometry;
- Wan2.2 TI2V-5B video flow;
- a source-native Grouped Action Flow Expert coupled to Wan at every transformer layer.

Video uses one 5 Hz, 9-frame, 1.6-second bucket. Robot commands keep their
recorded 5/10/15/20 Hz clock. A nominal future chunk contains 8/16/24/32
action events; timestamp jitter may retain one boundary event, so padded
capacity is computed with the exact integer horizon as `source_hz * 8 // 5 + 1`.
Training reads raw Parquet and MP4
data online. It does not require a VGGT, depth, point, pose, or Wan-latent
cache.

## Current implementation

- explicit contracts for all 21 sources: 7 verified sources enter training
  and 14 sources remain excluded until their stored controller semantics are
  proven;
- deterministic episode-level train/val/test materialization, with
  parent-trajectory grouping when a manifest supplies it;
- recorded-timestamp window selection, exact native-rate action events, and a
  grouped state/action history connector;
- online frozen VGGT pairs 0–3, autoregressive four-anchor GAM prediction,
  trainable pairs 4–23, geometry heads, and direct/refined grouped auxiliary
  actions;
- local-only Wan2.2 DiT, VAE, UMT5, and tokenizer loading;
- a 30-layer 1024-wide grouped ActionDiT initialized from Wan2.2 with the
  FastWAM interpolation rule;
- `action_only`, `forward_world`, and `joint_world_action` flow objectives;
- deployment-identical observed-video K/V prefill for action training and
  inference;
- exact Stage A/B/C optimizer ownership and learning rates.
- a deterministic hierarchical program/family/source sampler, multi-process
  DataLoader, multi-GPU FSDP, validation, numbered checkpoints, exact resume,
  and deliberate cross-stage initialization.

Production-width preflights have run on real OXE Bridge and RoboCasa Atomic
windows with the official local model weights. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for results and
[docs/TRAINING.md](docs/TRAINING.md) for commands.

## Repository layout

```text
src/wm3d_wam/data/       timestamp, split, grouped robot, and online RGB I/O
src/wm3d_wam/models/     online VGGT-GAM, Wan/Action MoT, and system composition
src/wm3d_wam/training/   flow objectives, Stage A pipeline, and optimizer groups
src/wm3d_wam/vendor/     licensed FastWAM and VGGT-GAM components
scripts/                 asset preparation and reproducible preflight programs
configs/                 production model/data/training contracts
```

## Tests

Run on New-H100-2 from the project directory:

```bash
cd /data/Minko/WM3D-WAM
PYTHONPATH=src /data/Minko/.venvs/wm3d/bin/python -m pytest -q
```

The current suite has 60 passing tests. GPU 0 is forbidden by the project
runtime contract; GPU experiments must use physical devices 1–7.

## Formal-training status

The checked-in formal profile uses physical GPUs `1,2,3,4,5,6,7`. Stage A/B/C
first passed production-width FSDP canaries and exact checkpoint recovery on
the five-card mesh `1,2,5,6,7`; the seven-card Stage A formal run started on
2026-08-20. Full-phase rank-local resume and Stage B-to-C initialization must
keep the same ordered seven-card mesh.
The sampler admits only `oxe_bridge`, `oxe_droid`, `oxe_furniture_bench`,
`oxe_bc_z`, `robocasa_atomic`, `robocasa_composite`, and `robocasa_mg`.

The code path is ready for formal training on those seven sources. Adding any
of the other fourteen sources still requires a payload-level unit, frame,
composition, and gripper-polarity audit; the loader rejects them instead of
guessing.

## Documents

- [v1 design](docs/WM3D_WAM_V1_DESIGN.md)
- [implementation base and status](docs/IMPLEMENTATION_BASE.md)
- [training and asset runbook](docs/TRAINING.md)
- [preliminary experiments](docs/EXPERIMENTS.md)
- [third-party notices](THIRD_PARTY_NOTICES.md)
