import os
from types import SimpleNamespace

import pytest
import torch

from wm3d_wam.data.hierarchical_sampler import WindowRequest
from wm3d_wam.data.online_dataset import (
    CompactEpisodeRecord,
    OnlineRobotDataset,
    build_online_dataloader,
    seed_online_worker,
)
from wm3d_wam.data.online_episode import OnlineEpisodeError
from wm3d_wam.data.source_contracts import SourceContractError


class _WindowStub:
    batch_size = 1
    task_text = "move the object"
    episode_id = "source:retry-success"
    quality_weight = 1.0
    decode_retry_count = 1


class _EmptySampler:
    seed = 17
    rank = 0

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


def _compact_episode(episode_id: str) -> CompactEpisodeRecord:
    return CompactEpisodeRecord(
        source="source",
        episode_id=episode_id,
        task_text="move the object",
        observation_samples=64,
        payload="data/chunk/file.parquet",
        payload_row_start=0,
        payload_row_stop=64,
        assets=(("rgb/head", "videos/head.mp4"),),
        views=(("head", "rgb/head", 0.0, 6.4),),
    )


def test_decode_retry_preserves_the_routed_target_view(monkeypatch) -> None:
    dataset = object.__new__(OnlineRobotDataset)
    dataset.catalogs = {
        "source": SimpleNamespace(
            contract=SimpleNamespace(programs={"world_core_pretrain"}),
            source_root="/unused",
            adapter_path="/unused/adapter.yaml",
            episodes=(
                _compact_episode("source:first"),
                _compact_episode("source:second"),
            ),
        )
    }
    dataset.normalization = object()
    dataset.max_decode_retries = 2
    dataset.max_views = 3
    dataset.max_groups = 8
    dataset.max_action_dim = 16
    dataset.max_state_dim = 32
    request = WindowRequest(
        global_sample_index=17,
        local_sample_index=4,
        program="world_core_pretrain",
        family="oxe",
        source="source",
        episode_index=0,
        anchor_fraction=0.2,
        target_view_fraction=0.49,
        retry_stride=1,
    )
    calls: list[dict[str, object]] = []

    def fake_load_online_robot_window(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise OnlineEpisodeError("transient decode failure")
        return _WindowStub()

    monkeypatch.setattr(
        "wm3d_wam.data.online_dataset.load_online_robot_window",
        fake_load_online_robot_window,
    )

    sample = dataset[request]

    assert sample.decode_retry_count == 1
    assert [call["target_view_fraction"] for call in calls] == [0.49, 0.49]
    assert calls[0]["anchor_fraction"] != calls[1]["anchor_fraction"]
    assert calls[0]["episode"] != calls[1]["episode"]


def test_online_workers_use_a_fresh_spawn_context() -> None:
    loader = build_online_dataloader(
        [],
        _EmptySampler(),
        num_workers=2,
        micro_batch_size=1,
    )

    assert loader.multiprocessing_context is not None
    assert loader.multiprocessing_context.get_start_method() == "spawn"


def test_online_worker_hides_cuda_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3,4")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    seed_online_worker(0)

    assert not torch.cuda.is_initialized()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_online_worker_rejects_an_inherited_cuda_context(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(SourceContractError, match="inherited an initialized CUDA"):
        seed_online_worker(0)
