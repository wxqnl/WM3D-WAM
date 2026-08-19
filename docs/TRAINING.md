# WM3D-WAM 训练与资产 Runbook

本文档只描述当前仓库已经实现的入口。所有命令在 New-H100-2 的
`/data/Minko/WM3D-WAM` 执行。GPU 0 保留给其他任务；单卡命令中的
`CUDA_VISIBLE_DEVICES=1` 指物理 GPU 1，程序内部看到的是逻辑 `cuda:0`。

## 1. 环境与本地资产

```bash
ssh New-H100-2
cd /data/Minko/WM3D-WAM

export PYTHONPATH=/data/Minko/WM3D-WAM/src
export WM3D_PYTHON=/data/Minko/.venvs/wm3d/bin/python
export WAN_ASSETS=/data/Minko/models/WM3D-WAM/Wan2.2-TI2V-5B
export ACTION_BACKBONE=/data/Minko/models/WM3D-WAM/ActionDiT/ActionDiT_grouped_Wan22_1024.pt
export VGGT_CKPT=/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/assets/vggt_model.safetensors
export VGGT_SOURCE_ROOT=/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/runtime/vggt
export DATA_PROFILE=/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/data_profile.yaml
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

训练 worker 对 Wan bundle、Action backbone、VGGT checkpoint 和 VGGT source
使用严格本地路径。缺文件、checkpoint key 不一致或 meta-device runtime
constant 未物化时直接失败，不在 worker 内下载模型。训练读取原始 Parquet 和
MP4；不要求 VGGT、depth、point、pose 或 Wan latent cache。

当前 Wan bundle 必须同时包含：

- Wan2.2 DiT safetensor shards；
- `Wan2.2_VAE.pth`；
- `models_t5_umt5-xxl-enc-bf16.pth`；
- `google/umt5-xxl/` tokenizer 目录。

## 2. 一次性准备

ActionDiT backbone 已存在时不要重复生成。需要重建时，先确认物理 GPU 1
空闲，再执行：

```bash
nvidia-smi -i 1 --query-gpu=index,memory.used,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=1 "$WM3D_PYTHON" scripts/prepare_grouped_action_backbone.py \
  --asset-root "$WAN_ASSETS" \
  --model-config configs/model/wan_action_mot_v1.yaml \
  --output "$ACTION_BACKBONE" \
  --device cuda \
  --dtype bfloat16
```

该脚本只迁移 Wan 与 ActionDiT 共享的 30 层 backbone。shape 相同的 tensor
直接复制，shape 不同的 tensor 使用 FastWAM 的逐维线性插值与 alpha scaling；
grouped codec 和输出层由本模型初始化。

固定 episode split 的物化命令：

```bash
"$WM3D_PYTHON" scripts/build_episode_splits.py \
  --data-profile "$DATA_PROFILE" \
  --output outputs/data/episode_splits_v1 \
  --seed 20260819
```

输出包括 `train/`、`val/`、`test/` 下逐 source ID 列表、全局
`episode_split_index.jsonl` 和 `summary.json`。同一个 parent trajectory（若
manifest 提供）不会跨 split；否则以 episode 为不可分单位。没有可信离散 task
ID 时不物化严格 task-OOD split。

## 3. CPU 合同测试

```bash
PYTHONPATH=src "$WM3D_PYTHON" -m pytest -q
```

当前预期为 48 项通过。测试覆盖张量合同、时间选择、split、mask、flow、cache
路径和参数归属；它不替代真实权重、真实视频上的 GPU preflight。

## 4. 真实 Bridge preflight

先定义同一个真实 source 的参数：

```bash
export SOURCE_ROOT=/shared/Minko/datasets/gam_stage1/openx_lerobot/BrunoM42/bridge_orig_lerobot
export SOURCE_MANIFEST=/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/manifests/oxe_bridge.jsonl
export SOURCE_ADAPTER=/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/contracts/adapters/oxe_bridge.yaml
```

每次启动前都应再次检查所用物理卡；下面只使用物理 GPU 1：

```bash
nvidia-smi -i 1 --query-gpu=index,memory.used,utilization.gpu --format=csv
```

Stage A 在线 Geometry-GAM forward/backward：

```bash
CUDA_VISIBLE_DEVICES=1 "$WM3D_PYTHON" scripts/run_geometry_preflight.py \
  --physical-gpu 1 \
  --source-root "$SOURCE_ROOT" \
  --manifest "$SOURCE_MANIFEST" \
  --adapter "$SOURCE_ADAPTER" \
  --source-hz 5 \
  --embodiment-id 0 \
  --wan-assets "$WAN_ASSETS" \
  --vggt-checkpoint "$VGGT_CKPT" \
  --vggt-source-root "$VGGT_SOURCE_ROOT" \
  --backward \
  --output outputs/preflight/final_bridge_geometry_stage_a.json
```

完整 action-only online path、反向、部署 cache parity 与 target leakage：

```bash
CUDA_VISIBLE_DEVICES=1 "$WM3D_PYTHON" scripts/run_pipeline_preflight.py \
  --physical-gpu 1 \
  --program action_only \
  --stage wan_action_warmup \
  --source-root "$SOURCE_ROOT" \
  --manifest "$SOURCE_MANIFEST" \
  --adapter "$SOURCE_ADAPTER" \
  --source-hz 5 \
  --embodiment-id 0 \
  --wan-assets "$WAN_ASSETS" \
  --action-backbone "$ACTION_BACKBONE" \
  --vggt-checkpoint "$VGGT_CKPT" \
  --vggt-source-root "$VGGT_SOURCE_ROOT" \
  --backward \
  --cache-parity \
  --target-leakage \
  --output outputs/preflight/final_bridge_action_only.json
```

`run_pipeline_preflight.py` 还接受：

| program | 输入噪声与主监督 | 用途 |
|---|---|---|
| `action_only` | noisy future action；observed video 只做 K/V prefill | 部署策略路径 |
| `forward_world` | clean candidate action + noisy future video | action-conditioned world model |
| `joint_world_action` | noisy future action + noisy future video | 双流联合对齐 |

`--backward-loss action` 或 `--backward-loss video` 可隔离单个 objective 做梯度
归因；`--optimizer-steps 3` 在固定真实窗口、固定 timestep/noise 上执行短拟合。
完整生产训练不能用单样本 preflight 替代。

## 5. Stage 参数归属

`configure_stage_parameter_groups` 先冻结全系统，再显式启用下表参数，并检查每个
trainable parameter 恰好属于一个 optimizer group。

| stage | trainable group | learning rate |
|---|---|---:|
| `geometry_gam` | geometry predictor/history/aux heads | 1e-5 |
| `geometry_gam` | VGGT deep pairs 4–23 | 1e-5 |
| `wan_action_warmup` | geometry predictor/history/aux heads | 1e-5 |
| `wan_action_warmup` | Action Expert | 2e-5 |
| `wan_action_warmup` | sparse geometry adapters | 1e-5 |
| `wan_action_main` | 上述三组 | 1e-5 / 2e-5 / 1e-5 |
| `wan_action_main` | VGGT deep pairs 4–23 | 5e-6 |
| `wan_action_main` | Wan DiT | 2e-6 |
| `tri_stream_alignment` | geometry / Action / adapter / VGGT deep / Wan | 5e-6 / 1e-5 / 1e-5 / 5e-6 / 1e-6 |

Wan VAE、UMT5、tokenizer、VGGT pairs 0–3 和 VGGT geometry heads 不进入
optimizer。全局 optimizer 配置是 AdamW、betas `(0.9, 0.95)`、weight decay
`0.01`、gradient clipping `1.0`。

## 6. 正式训练门禁

单卡生产宽度 forward/backward、三种 interaction program、短拟合和双向梯度
耦合已经通过，数值见 [EXPERIMENTS.md](EXPERIMENTS.md)。在跨 source 正式长训
之前仍需完成两件事：

1. 逐 source 确认 action/state 的单位、坐标系、composition operator 与
   gripper polarity；当前若干 v4 adapter 仍只有通用
   `controller_command/controller_state` 语义。
2. 实现并运行七卡 FSDP canary，验证吞吐、编号 checkpoint 以及 model、
   optimizer、scheduler、sampler cursor 和 RNG 的精确恢复。

因此 `configs/train/wm3d_wam_v1.yaml` 中的七卡参数是冻结的目标合同，不应在
缺少 launcher/resume canary 时被解释为已完成的正式训练入口。
