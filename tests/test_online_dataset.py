from types import SimpleNamespace

from wm3d_wam.data.hierarchical_sampler import WindowRequest
from wm3d_wam.data.online_dataset import OnlineRobotDataset
from wm3d_wam.data.online_episode import OnlineEpisodeError


class _WindowStub:
    batch_size = 1
    task_text = "move the object"
    episode_id = "source:retry-success"
    quality_weight = 1.0
    decode_retry_count = 1


def test_decode_retry_preserves_the_routed_target_view(monkeypatch) -> None:
    dataset = object.__new__(OnlineRobotDataset)
    dataset.catalogs = {
        "source": SimpleNamespace(
            contract=SimpleNamespace(programs={"world_core_pretrain"}),
            source_root="/unused",
            adapter_path="/unused/adapter.yaml",
            episodes=(
                {"episode_id": "source:first"},
                {"episode_id": "source:second"},
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
