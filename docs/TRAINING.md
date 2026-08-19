# WM3D-WAM v1 训练 Runbook

更新日期：2026-08-20。本文档只写仓库中已经实现并通过真实数据验证的入口。
所有命令在 New-H100-2 的 `/data/Minko/WM3D-WAM` 执行。物理 GPU 0 禁用。

## 1. 环境与本地资产

```bash
ssh New-H100-2
cd /data/Minko/WM3D-WAM

export WM3D_ROOT=/data/Minko/WM3D-WAM
export WM3D_PYTHON=/data/Minko/.venvs/wm3d/bin/python
export WM3D_TORCHRUN=/data/Minko/.venvs/wm3d/bin/torchrun
export PYTHONPATH="$WM3D_ROOT/src"
export WAN_ASSETS=/data/Minko/models/WM3D-WAM/Wan2.2-TI2V-5B
export ACTION_BACKBONE=/data/Minko/models/WM3D-WAM/ActionDiT/ActionDiT_grouped_Wan22_1024.pt
export VGGT_CKPT=/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/assets/vggt_model.safetensors
export VGGT_SOURCE_ROOT=/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/runtime/vggt
export DATA_PROFILE=/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/data_profile.yaml
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Worker 只从本地读取 Wan2.2 DiT、VAE、UMT5、tokenizer、ActionDiT backbone、
VGGT checkpoint、Parquet 和 MP4。缺失资产、checkpoint key 不匹配、未物化的
meta-device 常量都会立即报错。训练不需要 VGGT、depth、point、pose 或 Wan
latent 缓存。

## 2. 数据门禁

`configs/data/source_contracts_v1.yaml` 为 21 个 source 提供显式合同。正式
sampler 只接纳 7 个 `verified` source：

| source | Hz | train | val | test |
|---|---:|---:|---:|---:|
| oxe_bridge | 5 | 20,690 | 211 | 211 |
| oxe_droid | 15 | 83,209 | 849 | 849 |
| oxe_furniture_bench | 10 | 2,365 | 24 | 24 |
| oxe_bc_z | 10 | 14,056 | 143 | 143 |
| robocasa_atomic | 20 | 5,999 | 61 | 61 |
| robocasa_composite | 20 | 15,921 | 162 | 162 |
| robocasa_mg | 20 | 385,407 | 3,933 | 3,933 |
| 合计 |  | 527,647 | 5,383 | 5,383 |

其余 14 个 source 保留 split 与统计，但 `status: excluded`，不会进入训练。
加入它们之前必须从当前本地 payload 证明 action/state 的单位、坐标系、
composition operator 和 gripper polarity。代码不会用通用语义代替缺失合同。

重新物化固定 episode split：

```bash
"$WM3D_PYTHON" scripts/build_episode_splits.py \
  --data-profile "$DATA_PROFILE" \
  --output outputs/data/episode_splits_v1 \
  --seed 20260819
```

同一 parent trajectory 在 manifest 提供该字段时不会跨 split；否则 episode 是
最小划分单位。window 只继承 episode split，训练时不临时重划。

## 3. CPU 合同测试

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
  "$WM3D_PYTHON" -m pytest -q
```

当前预期为 59 项通过。覆盖 source gate、时间窗、原生 action event、grouped
ABI、三种 interaction mask、在线 VGGT-GAM、Wan/Action MoT、分层 sampler、
FSDP checkpoint 合同、精确恢复和 checkpoint 保留策略。

## 4. 分布式运行合同

当前验证过的完整模型 mesh 是物理 GPU `1,2,5,6,7`：

```bash
nvidia-smi -i 1,2,5,6,7 \
  --query-gpu=index,memory.used,utilization.gpu \
  --format=csv
export CUDA_VISIBLE_DEVICES=1,2,5,6,7
```

运行时允许显式列出物理 GPU 1–7 的任意非空子集，拒绝 GPU 0、重复 ID、UUID
写法和越界 ID。当前五卡配置使用 BF16 forward/reduce、FP32 master weights、
FSDP FULL_SHARD、每卡 micro-batch 1、gradient accumulation 4，有效 global
batch 为 20。node 42 的 NCCL NVLS 多 rank 路径存在驱动错误，launcher 默认
设置 `NCCL_NVLS_ENABLE=0`，使用已验证的普通 NVLink collective。

每个 rank 在进入 FSDP forward 前交换数据读取状态。某个 worker 解码失败时，
所有 rank 一起退出并报告原始数据错误，不会把其他 rank 留在 NCCL collective
中。

## 5. Checkpoint 合同

| phase | 存储 | 恢复约束 |
|---|---|---|
| `geometry_gam` | canonical DTensor DCP | 可重分片，并可初始化完整模型 |
| Stage B/C | rank-local FSDP flat shard | world size 与有序物理 GPU mesh 必须相同 |

checkpoint 包含 model、optimizer、scheduler、每 rank 的 Python/NumPy/CPU/CUDA
RNG 和已提交 sampler cursor。程序先写全部 tensor 与 runtime payload，最后写
`metadata.json` 完成标记，再原子更新 `latest.txt`。恢复不会使用 DataLoader
预取位置，因此不会跳样本。

完整 Stage B-main/C checkpoint 在五卡上约 91 GiB。以下正式命令每 5,000 步
保存一次，并显式设置 `--keep-last-checkpoints 2`。清理只删除更老且拥有合法
完成标记的编号目录；默认值 0 不自动删除任何 checkpoint。

## 6. 正式训练命令

以下四个 output 目录在首次启动前必须不存在。Stage A、Stage B warmup、Stage B
main 和 Stage C 的 `max_steps` 都是各自 phase 内的步数。

### 6.1 Stage A：Geometry-GAM，30,000 步

```bash
"$WM3D_TORCHRUN" --standalone --nproc_per_node=5 \
  scripts/train_wm3d_wam.py \
  --phase geometry_gam \
  --max-steps 30000 \
  --warmup-steps 500 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --log-interval 10 \
  --validation-interval 500 \
  --validation-samples-per-rank 8 \
  --checkpoint-interval 5000 \
  --keep-last-checkpoints 2 \
  --output-dir outputs/train/wm3d_wam_v1/stage_a_geometry
```

### 6.2 Stage B warmup：Action/geometry，2,000 步

```bash
"$WM3D_TORCHRUN" --standalone --nproc_per_node=5 \
  scripts/train_wm3d_wam.py \
  --phase wan_action_warmup \
  --max-steps 2000 \
  --warmup-steps 200 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --log-interval 10 \
  --validation-interval 500 \
  --validation-samples-per-rank 8 \
  --checkpoint-interval 1000 \
  --keep-last-checkpoints 2 \
  --initialize-from outputs/train/wm3d_wam_v1/stage_a_geometry/checkpoints \
  --output-dir outputs/train/wm3d_wam_v1/stage_b_warmup
```

### 6.3 Stage B main：解冻 Wan/VGGT deep，38,000 步

```bash
"$WM3D_TORCHRUN" --standalone --nproc_per_node=5 \
  scripts/train_wm3d_wam.py \
  --phase wan_action_main \
  --max-steps 38000 \
  --warmup-steps 500 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --log-interval 10 \
  --validation-interval 500 \
  --validation-samples-per-rank 8 \
  --checkpoint-interval 5000 \
  --keep-last-checkpoints 2 \
  --initialize-from outputs/train/wm3d_wam_v1/stage_b_warmup/checkpoints \
  --output-dir outputs/train/wm3d_wam_v1/stage_b_main
```

### 6.4 Stage C：Tri-stream alignment，20,000 步

```bash
"$WM3D_TORCHRUN" --standalone --nproc_per_node=5 \
  scripts/train_wm3d_wam.py \
  --phase tri_stream_alignment \
  --max-steps 20000 \
  --warmup-steps 500 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --log-interval 10 \
  --validation-interval 500 \
  --validation-samples-per-rank 8 \
  --checkpoint-interval 5000 \
  --keep-last-checkpoints 2 \
  --initialize-from outputs/train/wm3d_wam_v1/stage_b_main/checkpoints \
  --output-dir outputs/train/wm3d_wam_v1/stage_c_tri_stream
```

Stage B warmup、main 和 Stage C 必须使用相同的
`CUDA_VISIBLE_DEVICES=1,2,5,6,7` 顺序。参数 stage 的学习率与可训练模块由
`src/wm3d_wam/training/parameter_groups.py` 唯一决定；CLI 不接受临时覆盖。

## 7. 精确恢复与计划停机

恢复时复用原 phase、max steps、seed、gradient accumulation、output 目录和有序
GPU mesh，把 `--initialize-from` 改成 `--resume`：

```bash
"$WM3D_TORCHRUN" --standalone --nproc_per_node=5 \
  scripts/train_wm3d_wam.py \
  --phase wan_action_main \
  --max-steps 38000 \
  --warmup-steps 500 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --log-interval 10 \
  --validation-interval 500 \
  --validation-samples-per-rank 8 \
  --checkpoint-interval 5000 \
  --keep-last-checkpoints 2 \
  --resume outputs/train/wm3d_wam_v1/stage_b_main/checkpoints \
  --output-dir outputs/train/wm3d_wam_v1/stage_b_main
```

`--stop-after-step N` 会在 phase-local step N 做验证、保存完整 checkpoint 并以
`paused.json` 退出。它适合维护窗口，不改变 `max_steps`，之后按上面的 resume
命令继续。

## 8. 已完成门禁

真实权重、真实 MP4/Parquet 上已经完成：单卡三种 program 与梯度归因；Stage A
双卡 canonical DCP 保存及精确恢复；Stage B warmup 五卡保存、恢复和三种 route；
Stage B main 五卡深层解冻；Stage C 五卡 train、validation 和完整 checkpoint。
数值与输出目录见 [EXPERIMENTS.md](EXPERIMENTS.md)。这些门禁证明训练 pipeline
能运行和恢复，不代表下游策略质量已经达标。
