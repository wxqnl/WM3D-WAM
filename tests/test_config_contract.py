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

    assert config.window.world_hz == 10
    assert config.window.future_steps == 16
    assert config.window.video_frames == 17
    assert list(config.window.vggt_geometry_anchor_indices) == [3, 7, 11, 15]
    assert list(config.window.five_hz_world_valid_indices) == [
        1, 3, 5, 7, 9, 11, 13, 15
    ]
    assert config.action.preserve_source_native_clock
    assert OmegaConf.to_container(config.action.events_per_window) == {
        5: 8,
        10: 16,
        15: 24,
        20: 32,
    }
    assert config.action.max_events == 33
    assert config.action.max_history_events == 65
    assert (
        config.action.padded_capacity_rule
        == "ceil_duration_times_source_hz_plus_one"
    )
    assert config.action.interpolation == "forbidden"
    assert config.action.overflow_policy == "reject_window"


def test_runtime_forbids_gpu_zero_and_geometry_bridge_is_a_hard_gate() -> None:
    runtime = OmegaConf.load(ROOT / "configs/train/wm3d_wam_v1.yaml")
    geometry = OmegaConf.load(ROOT / "configs/model/vggt_geometry_v1.yaml")

    assert list(runtime.runtime.permitted_cuda_devices) == [1, 2, 3, 4, 5, 6, 7]
    assert list(runtime.runtime.visible_cuda_devices) == [1, 2, 3, 4, 5, 6, 7]
    assert list(runtime.runtime.forbidden_cuda_devices) == [0]
    assert runtime.runtime.compiler_cache_root == "outputs/runtime_cache"
    assert runtime.runtime.micro_batch_per_gpu == 4
    assert runtime.runtime.gradient_accumulation_steps == 1
    assert runtime.runtime.effective_global_batch == 28
    assert runtime.stages.world_core_pretrain.k16_memory_canary == "passed"
    assert runtime.stages.world_core_pretrain.rejected_micro_batch_per_gpu == 8
    assert geometry.future_steps == 16
    assert list(geometry.geometry_anchor_indices) == [3, 7, 11, 15]
    assert geometry.state_dynamics.state_hidden == 1600
    assert geometry.state_dynamics.state_layers == 18
    assert not hasattr(geometry, "future_predictor")
    assert geometry.history_connector._target_ == "wm3d_wam.models.GroupedHistoryConnector"
