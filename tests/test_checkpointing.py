from __future__ import annotations

import json

import pytest
import torch

from wm3d_wam.training.checkpointing import (
    CheckpointError,
    LOCAL_FSDP_SCHEMA,
    load_checkpoint,
    load_model_only,
    prune_completed_checkpoints,
    save_checkpoint,
)


def _training_objects():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 / float(step + 1),
    )
    loss = model(torch.randn(3, 4)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def test_rank_local_checkpoint_restores_model_optimizer_scheduler_and_cursor(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(31)
    model, optimizer, scheduler = _training_objects()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    checkpoint = save_checkpoint(
        root=tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        phase="wan_action_warmup",
        global_step=3,
        next_local_sample_index=8,
        seed=123,
        gradient_accumulation_steps=2,
        micro_batch_size=2,
        extra_metadata={"phase_total_steps": 9},
    )
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == LOCAL_FSDP_SCHEMA
    assert metadata["storage"] == "rank_local_fsdp"

    restored_model, restored_optimizer, restored_scheduler = _training_objects()
    resumed = load_checkpoint(
        path_or_root=tmp_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_phase="wan_action_warmup",
        expected_seed=123,
        expected_gradient_accumulation_steps=2,
        expected_micro_batch_size=2,
    )

    assert resumed.global_step == 3
    assert resumed.next_local_sample_index == 8
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.state_dict()["param_groups"] == optimizer.state_dict()[
        "param_groups"
    ]
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected[name])

    model_only, _, _ = _training_objects()
    load_model_only(path_or_root=checkpoint, model=model_only)
    for name, value in model_only.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_prune_completed_checkpoints_keeps_newest_and_ignores_incomplete(tmp_path) -> None:
    for step in (1, 2, 3, 4):
        path = tmp_path / f"step_{step:08d}"
        path.mkdir()
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "schema": LOCAL_FSDP_SCHEMA,
                    "global_step": step,
                }
            ),
            encoding="utf-8",
        )
    incomplete = tmp_path / "step_00000005"
    incomplete.mkdir()
    malformed = tmp_path / "step_00000006"
    malformed.mkdir()
    (malformed / "metadata.json").write_text(
        json.dumps({"schema": LOCAL_FSDP_SCHEMA, "global_step": 7}),
        encoding="utf-8",
    )

    removed = prune_completed_checkpoints(tmp_path, keep_last=2)

    assert [path.name for path in removed] == ["step_00000001", "step_00000002"]
    assert not (tmp_path / "step_00000001").exists()
    assert not (tmp_path / "step_00000002").exists()
    assert (tmp_path / "step_00000003").is_dir()
    assert (tmp_path / "step_00000004").is_dir()
    assert incomplete.is_dir()
    assert malformed.is_dir()


def test_rank_local_checkpoint_rejects_a_different_physical_mesh(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, optimizer, scheduler = _training_objects()
    save_checkpoint(
        root=tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        phase="wan_action_main",
        global_step=1,
        next_local_sample_index=1,
        seed=5,
        gradient_accumulation_steps=1,
        micro_batch_size=1,
        extra_metadata={"physical_cuda_devices": [1]},
    )

    with pytest.raises(CheckpointError, match="physical CUDA mesh mismatch"):
        load_checkpoint(
            path_or_root=tmp_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_phase="wan_action_main",
            expected_seed=5,
            expected_gradient_accumulation_steps=1,
            expected_micro_batch_size=1,
            expected_physical_cuda_devices=[2],
        )
