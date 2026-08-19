from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def test_production_model_config_preserves_fastwam_scale_and_geometry_layers() -> None:
    config = OmegaConf.load(ROOT / "configs/model/wan_action_mot_v1.yaml")

    assert config.video_expert.num_layers == 30
    assert config.video_expert.hidden_dim == 3072
    assert config.action_expert.num_layers == 30
    assert config.action_expert.hidden_dim == 1024
    assert config.action_expert.codec_config.max_groups == 8
    assert config.action_expert.codec_config.max_action_dim == 16
    assert list(config.geometry_adapters.fusion_layers) == [5, 11, 17, 23, 29]


def test_data_config_keeps_renderer_and_action_clocks_separate() -> None:
    config = OmegaConf.load(ROOT / "configs/data/grouped_robot_v1.yaml")

    assert config.window.renderer_hz == 5
    assert config.window.video_frames == 9
    assert config.action.preserve_source_native_clock
    assert OmegaConf.to_container(config.action.events_per_window) == {
        5: 8,
        10: 16,
        15: 24,
        20: 32,
    }
    assert config.action.interpolation == "forbidden"
    assert config.action.overflow_policy == "reject_window"


def test_runtime_forbids_gpu_zero_and_geometry_bridge_is_a_hard_gate() -> None:
    runtime = OmegaConf.load(ROOT / "configs/train/wm3d_wam_v1.yaml")
    geometry = OmegaConf.load(ROOT / "configs/model/vggt_geometry_v1.yaml")

    assert list(runtime.runtime.visible_cuda_devices) == [1, 2, 3, 4, 5, 6, 7]
    assert list(runtime.runtime.forbidden_cuda_devices) == [0]
    assert (
        geometry.future_predictor.integration_status
        == "grouped_history_bridge_implemented"
    )
    assert geometry.history_connector._target_ == "wm3d_wam.models.GroupedHistoryConnector"
