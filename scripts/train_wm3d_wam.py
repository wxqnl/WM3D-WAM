#!/usr/bin/env python3
"""Launch one production WM3D-WAM training phase under torchrun."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Whole-MoT FSDP briefly allocates and releases multi-GiB unsharded buffers.
# Expandable segments prevent those variable-size collectives from stranding
# enough reserved address space to block the reduced FP32 gradient shard.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Root filesystem space is intentionally tight on New-H100-2. Keep generated
# Inductor/Triton artifacts with the project outputs on /data, not under /tmp
# or /root. The variables must be set before importing the model stack.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_COMPILER_CACHE_ROOT = _PROJECT_ROOT / "outputs" / "runtime_cache"
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(_COMPILER_CACHE_ROOT / "torchinductor"),
)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    str(_COMPILER_CACHE_ROOT / "triton"),
)

from wm3d_wam.training.distributed import (  # noqa: E402
    initialize_distributed,
    shutdown_distributed,
)
from wm3d_wam.training.trainer import (  # noqa: E402
    PHASES,
    TrainerOptions,
    TrainingPaths,
    train,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--initialize-from")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-samples-per-rank", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=0,
        help="Retain this many completed checkpoints; zero disables pruning.",
    )
    parser.add_argument("--prompt-cache-entries", type=int, default=64)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Gracefully checkpoint and exit at this phase-local global step.",
    )
    parser.add_argument(
        "--data-profile",
        default="/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/data_profile.yaml",
    )
    parser.add_argument(
        "--source-contracts",
        default="configs/data/source_contracts_v1.yaml",
    )
    parser.add_argument(
        "--normalization",
        default="/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/grouped_normalization_1b.json",
    )
    parser.add_argument(
        "--episode-splits",
        default="outputs/data/episode_splits_v1",
    )
    parser.add_argument(
        "--wan-assets",
        default="/data/Minko/models/WM3D-WAM/Wan2.2-TI2V-5B",
    )
    parser.add_argument(
        "--action-backbone",
        default="/data/Minko/models/WM3D-WAM/ActionDiT/ActionDiT_grouped_Wan22_1024.pt",
    )
    parser.add_argument(
        "--vggt-checkpoint",
        default="/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/assets/vggt_model.safetensors",
    )
    parser.add_argument(
        "--vggt-source-root",
        default="/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/runtime/vggt",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/wan_action_mot_v1.yaml",
    )
    parser.add_argument(
        "--geometry-config",
        default="configs/model/vggt_geometry_v1.yaml",
    )
    args = parser.parse_args()
    context = initialize_distributed()
    try:
        paths = TrainingPaths.resolve(
            data_profile=args.data_profile,
            source_contracts=args.source_contracts,
            normalization=args.normalization,
            episode_splits=args.episode_splits,
            wan_assets=args.wan_assets,
            action_backbone=args.action_backbone,
            vggt_checkpoint=args.vggt_checkpoint,
            vggt_source_root=args.vggt_source_root,
            model_config=args.model_config,
            geometry_config=args.geometry_config,
        )
        options = TrainerOptions(
            phase=args.phase,
            max_steps=args.max_steps,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
            validation_interval=args.validation_interval,
            validation_samples_per_rank=args.validation_samples_per_rank,
            checkpoint_interval=args.checkpoint_interval,
            keep_last_checkpoints=args.keep_last_checkpoints,
            output_dir=Path(args.output_dir),
            resume=Path(args.resume) if args.resume else None,
            initialize_from=(
                Path(args.initialize_from) if args.initialize_from else None
            ),
            prompt_cache_entries=args.prompt_cache_entries,
            stop_after_step=args.stop_after_step,
        )
        result = train(context=context, paths=paths, options=options)
        if context.is_main:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
