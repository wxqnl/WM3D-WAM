#!/usr/bin/env python3
"""Materialize plain CPU parameter views from a rank-local FSDP checkpoint.

FSDP1 ``LOCAL_STATE_DICT`` checkpoints written with ``use_orig_params=True``
contain the canonical original-parameter views plus a redundant ShardedTensor
``_flat_param`` at each FSDP boundary.  Loading the files therefore requires
the original four-rank process group, even though the original views can be
reassembled later without FSDP.  This utility removes only those redundant
flat keys and clones every original view into independent CPU storage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _atomic_torch_save(path: Path, value: object, rank: int) -> None:
    temporary = path.with_name(path.name + f".tmp-rank{rank:03d}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("materialization must run with CUDA hidden")
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        metadata_path = checkpoint / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != "wm3d_wam_local_fsdp_checkpoint_v2":
            raise RuntimeError("checkpoint is not rank-local FSDP v2")
        if int(metadata.get("world_size", -1)) != world_size:
            raise RuntimeError(
                f"checkpoint world size is {metadata.get('world_size')}, "
                f"but runtime world size is {world_size}"
            )
        output_dir = args.output_dir.expanduser().resolve()
        if rank == 0:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise RuntimeError(f"refusing to overwrite non-empty {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        source_path = checkpoint / f"model_rank_{rank:03d}.pt"
        if not source_path.is_file():
            raise RuntimeError(f"missing rank payload {source_path}")
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("model"), Mapping
        ):
            raise RuntimeError(f"malformed rank payload {source_path}")

        plain: dict[str, torch.Tensor] = {}
        flat_keys: list[str] = []
        source_numel = 0
        for key, value in payload["model"].items():
            if str(key).rsplit(".", 1)[-1] == "_flat_param":
                flat_keys.append(str(key))
                continue
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"original view {key!r} has unexpected type {type(value)!r}"
                )
            cloned = value.detach().cpu().contiguous().clone()
            plain[str(key)] = cloned
            source_numel += cloned.numel()
        if not flat_keys:
            raise RuntimeError("rank payload contains no redundant _flat_param key")
        del payload

        output_path = output_dir / f"views_rank_{rank:03d}.pt"
        _atomic_torch_save(
            output_path,
            {
                "schema": "wm3d_wam_plain_rank_views_v1",
                "checkpoint": str(checkpoint),
                "checkpoint_step": int(metadata["global_step"]),
                "rank": rank,
                "world_size": world_size,
                "model": plain,
                "flat_keys_removed": sorted(flat_keys),
                "view_numel": source_numel,
            },
            rank,
        )
        print(
            json.dumps(
                {
                    "rank": rank,
                    "keys": len(plain),
                    "flat_keys_removed": len(flat_keys),
                    "view_numel": source_numel,
                    "output": str(output_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        dist.barrier()
        if rank == 0:
            expected = [
                output_dir / f"views_rank_{index:03d}.pt"
                for index in range(world_size)
            ]
            if not all(path.is_file() for path in expected):
                raise RuntimeError("not every rank produced a plain-view payload")
            completion = {
                "schema": "wm3d_wam_plain_rank_views_v1",
                "checkpoint": str(checkpoint),
                "checkpoint_step": int(metadata["global_step"]),
                "world_size": world_size,
                "files": [path.name for path in expected],
            }
            temporary = output_dir / "metadata.json.tmp"
            temporary.write_text(
                json.dumps(completion, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, output_dir / "metadata.json")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
