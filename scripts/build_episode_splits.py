#!/usr/bin/env python3
"""Materialize the fixed WM3D-WAM train/val/test episode partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from wm3d_wam.data.episode_splits import (
    is_v1_eligible_manifest_record,
    materialize_source_split,
    requested_holdout_count,
)
from wm3d_wam.data.online_episode import iter_episode_manifest


def _write_ids(path: Path, values: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    profile_path = Path(args.data_profile).expanduser().resolve(strict=True)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    sources = profile.get("sources") if isinstance(profile, dict) else None
    if not isinstance(sources, list) or not sources:
        raise ValueError("data profile contains no sources")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    family_counts = {
        "oxe": {"train": 0, "val": 0, "test": 0},
        "robocasa": {"train": 0, "val": 0, "test": 0},
    }
    index_path = output / "episode_split_index.jsonl"
    with index_path.open("w", encoding="utf-8") as index_handle:
        for source_config in sources:
            if not isinstance(source_config, dict):
                raise ValueError("source profile entry must be a mapping")
            source = str(source_config["name"])
            manifest = Path(source_config["manifest"]).expanduser().resolve(strict=True)
            eligible = [
                row
                for row in iter_episode_manifest(manifest)
                if is_v1_eligible_manifest_record(row)
            ]
            split = materialize_source_split(
                eligible,
                source=source,
                seed=args.seed,
            )
            values_by_name = {
                "train": split.train,
                "val": split.validation,
                "test": split.test,
            }
            for name, values in values_by_name.items():
                _write_ids(output / name / f"{source}.txt", values)
                for episode_id in values:
                    index_handle.write(
                        json.dumps(
                            {
                                "source": source,
                                "episode_id": episode_id,
                                "split": name,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            family = "robocasa" if source.startswith("robocasa_") else "oxe"
            for name, values in values_by_name.items():
                family_counts[family][name] += len(values)
            summaries.append(
                {
                    "source": source,
                    "eligible": split.total,
                    "requested_val_test_each": requested_holdout_count(split.total),
                    "train": len(split.train),
                    "val": len(split.validation),
                    "test": len(split.test),
                    "parent_field": split.parent_field,
                }
            )

    totals = {
        name: sum(int(source[name]) for source in summaries)
        for name in ("eligible", "train", "val", "test")
    }
    summary = {
        "schema": "wm3d_wam_episode_splits_v1",
        "seed": args.seed,
        "unit": "parent_trajectory_when_present_else_episode",
        "eligibility": "upstream_train_and_90pct_of_4.8s_recorded_clock_span",
        "task_ood": "not_materialized_without_audited_discrete_task_ids",
        "totals": totals,
        "families": family_counts,
        "sources": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
