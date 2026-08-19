# WM3D-WAM

WM3D-WAM is a world-action model that combines three production-width paths:

- online VGGT-1B split-and-resume geometry;
- Wan2.2 TI2V-5B video flow;
- a source-native Grouped Action Flow Expert coupled to Wan at every transformer layer.

Video uses one 5 Hz, 9-frame, 1.6-second bucket. Robot commands keep their
recorded 5/10/15/20 Hz clock, so one future chunk contains 8/16/24/32 action
events. Training reads raw Parquet and MP4 data online. It does not require a
VGGT, depth, point, pose, or Wan-latent cache.

## Current implementation

- deterministic episode-level train/val/test materialization for all 21 data
  sources, with parent-trajectory grouping when a manifest supplies it;
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

The current suite has 48 passing tests. GPU 0 is forbidden by the project
runtime contract; GPU experiments must use physical devices 1–7.

## Remaining gate before formal training

The present v4 source adapters preserve raw controller channels but still use
generic `controller_command` / `controller_state` semantics for several
sources. Formal cross-embodiment training remains blocked until units,
coordinate frames, composition operators, and gripper polarity are audited per
source. The production-width single-GPU graph is verified; the seven-GPU FSDP
canary and checkpoint-resume run are the next runtime milestone.

## Documents

- [v1 design](docs/WM3D_WAM_V1_DESIGN.md)
- [implementation base and status](docs/IMPLEMENTATION_BASE.md)
- [training and asset runbook](docs/TRAINING.md)
- [preliminary experiments](docs/EXPERIMENTS.md)
- [third-party notices](THIRD_PARTY_NOTICES.md)
