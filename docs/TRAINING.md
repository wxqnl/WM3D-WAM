# WM3D-WAM Revision 5 训练 Runbook

更新日期：2026-08-23。所有命令在 New-H100-2 的
`/data/Minko/WM3D-WAM` 执行。当前完整训练 mesh 固定为物理 GPU 1–4；GPU
0、5、6、7 均保留给其他任务。

## 1. 环境

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
export CUDA_VISIBLE_DEVICES=1,2,3,4
```

训练只从本地读取模型、Parquet 和 MP4。缺少资产或 checkpoint key 不匹配时立即
报错。训练不需要离线 geometry 或 latent cache。

## 2. 数据门禁

正式 sampler 接纳：

```text
oxe_bridge
oxe_droid
oxe_furniture_bench
oxe_bc_z
robocasa_atomic
robocasa_composite
robocasa_mg
```

合计 527,647 train、5,383 val、5,383 test episode。其余十四个 source 的
contract 状态为 excluded。

重新物化 split：

```bash
"$WM3D_PYTHON" scripts/build_episode_splits.py \
  --data-profile "$DATA_PROFILE" \
  --output outputs/data/episode_splits_v1 \
  --seed 20260819
```

## 3. K=16 合同

- world grid：10 Hz，未来 0.1 到 1.6 秒，共 16 步；
- Wan clip：当前帧加 16 个未来帧，共 17 帧；
- action：保留 source-native 5/10/15/20 Hz，未来容量最多 33 event；
- 5 Hz RGB：只有索引 `1,3,...,15` 是真实监督，其余位置 masked；
- VGGT deep：未来索引 `3,7,11,15`；
- source 低于 10 Hz 时禁止 forward/joint video route。

修改这些数字必须同时更新 data config、geometry config、Wan VAE contract 和相关
测试，不能只改一个 YAML。

## 4. CPU 测试

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
  "$WM3D_PYTHON" -m pytest -q
```

测试覆盖数据时钟、5 Hz mask、原生 action event、source gate、WM3D state/
dynamics、在线 VGGT、Wan/Action MoT、interaction mask、sampler 和 checkpoint。

## 5. 分布式合同

```bash
nvidia-smi -i 1,2,3,4 \
  --query-gpu=index,memory.used,utilization.gpu \
  --format=csv
```

launcher 校验 `CUDA_VISIBLE_DEVICES`，拒绝 GPU 0、重复 ID、UUID 写法和越界 ID。
正式配置使用 BF16 forward/reduce、FP32 master weights、FSDP FULL_SHARD 和
`NCCL_NVLS_ENABLE=0`。TorchInductor 与 Triton 生成物写入
`outputs/runtime_cache`，不占用空间紧张的根盘 `/tmp` 或 `/root`。

K=16 world-core 的正式 batch 是每卡 4、累积 1、global batch 16。完整
Wan/Action 的真实四卡 canary 使用每卡 1、累积 1、global batch 4，训练、验证和
rank-local checkpoint 均通过，首步峰值约 68.6 GiB。累积 4 会在后续 micro-step
同时保留 FP32 gradient shard 和新一轮 full-parameter all-gather，80 GiB H100
实测 OOM，因此禁止用于当前四卡 mesh。

冻结的 UMT5 只在 prompt cache miss 时临时进入对应 rank 的 GPU。文本特征产生后，
encoder 必须立即回到 CPU，并在 FSDP forward/backward 前释放 CUDA allocator cache。
UMT5 不参与 optimizer 或训练图；若让其约 11 GiB 的 BF16 权重常驻，每卡可用空间
不足以承载解冻 Wan VideoDiT 后约 11.3 GiB 的 FSDP full-parameter all-gather。
cache miss step 会包含一次 CPU/GPU 权重搬运，命中已有 prompt 时不再搬运或重新
编码。

## 6. Checkpoint

| phase | 格式 | 恢复条件 |
|---|---|---|
| world_core_pretrain | canonical DTensor DCP | 可重分片；可初始化完整模型 |
| Wan/Action phases | rank-local FSDP shard | world size 与有序 GPU mesh 必须相同 |

checkpoint 保存 model、optimizer、scheduler、各 rank RNG 和已提交 sampler cursor。
完成目录最后写 `metadata.json`，随后原子更新 `latest.txt`。`--keep-last-checkpoints`
只清理更老且带合法完成标记的 checkpoint。

Revision 5 修复了 grouped state/action codec 的字段—数值绑定，并改变了 MoT 的
跨流 attention 合同。Revision 4 的 Stage A/B checkpoint 虽然文件完整，但参数空间
和训练语义都已过期，不能 resume 或作为后续阶段初始化点。Revision 5 必须从官方
VGGT/Wan/FastWAM 基础权重重新训练 Stage A；旧输出只保留作审计证据。

`stage_a_geometry`、`geometry_gam` 和更早 checkpoint 同样不包含当前 WM3D state
core，不能用于 Revision 5。

## 7. 正式训练

每个 output 目录在首次启动前必须不存在。

### 7.1 Stage A：WM3D world core，30,000 步

```bash
"$WM3D_TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_wm3d_wam.py \
  --phase world_core_pretrain \
  --max-steps 30000 \
  --micro-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --num-workers 4 \
  --warmup-steps 500 \
  --log-interval 10 \
  --validation-interval 500 \
  --validation-samples-per-rank 8 \
  --checkpoint-interval 500 \
  --keep-last-checkpoints 3 \
  --output-dir outputs/train/wm3d_wam_k16_r5/stage_a_world_core_gpu1_4
```

这个阶段不加载 Wan 或 ActionDiT。optimizer 只包含 WM3D state dynamics、history
connector、geometry reducer 和 VGGT deep pairs 4–23。

### 7.2 Stage B warmup，2,000 步

```bash
"$WM3D_TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_wm3d_wam.py \
  --phase wan_action_warmup \
  --max-steps 2000 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 4 \
  --warmup-steps 200 \
  --log-interval 10 \
  --validation-interval 500 \
  --validation-samples-per-rank 8 \
  --checkpoint-interval 500 \
  --keep-last-checkpoints 3 \
  --initialize-from outputs/train/wm3d_wam_k16_r5/stage_a_world_core_gpu1_4/checkpoints/step_00030000 \
  --output-dir outputs/train/wm3d_wam_k16_r5/stage_b_warmup_gpu1_4
```

warmup 训练 WM3D core、Action Expert 和 geometry adapters；Wan VideoDiT 与 VGGT
deep 暂时冻结。

### 7.3 Stage B main，38,000 步

```bash
"$WM3D_TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_wm3d_wam.py \
  --phase wan_action_main \
  --max-steps 38000 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 4 \
  --warmup-steps 500 \
  --log-interval 10 \
  --validation-interval 500 \
  --validation-samples-per-rank 8 \
  --checkpoint-interval 500 \
  --keep-last-checkpoints 3 \
  --initialize-from outputs/train/wm3d_wam_k16_r5/stage_b_warmup_gpu1_4/checkpoints \
  --output-dir outputs/train/wm3d_wam_k16_r5/stage_b_main_gpu1_4
```

main 解冻 Wan VideoDiT 和 VGGT deep，继续训练 Action Expert、WM3D core 与
geometry adapters。默认 route mix 为 `forward_world=0.65`、`action_only=0.25`、
`joint_world_action=0.10`，使多数更新直接学习 clean action-conditioned RGB。
`action_only` 的视频/VGGT conditioner 在 no-grad 区域构建；`joint_world_action`
只允许 video→action，不允许视频读取 noisy action。

单卡 full-pipeline canary 必须分别覆盖 `forward_world`、
`joint_world_action` 和 `action_only` 的真实 forward/backward 与梯度归属；四卡 main
canary 还必须完成真实 FSDP optimizer step 和完整 rank-local checkpoint。仅成功构建
模型不算通过显存门禁。

### 7.4 Stage C：tri-stream alignment，20,000 步

```bash
"$WM3D_TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_wm3d_wam.py \
  --phase tri_stream_alignment \
  --max-steps 20000 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 4 \
  --warmup-steps 500 \
  --log-interval 10 \
  --validation-interval 500 \
  --validation-samples-per-rank 8 \
  --checkpoint-interval 500 \
  --keep-last-checkpoints 3 \
  --initialize-from outputs/train/wm3d_wam_k16_r5/stage_b_main_gpu1_4/checkpoints \
  --output-dir outputs/train/wm3d_wam_k16_r5/stage_c_tri_stream_gpu1_4
```

Stage B warmup、main 和 Stage C 必须保持相同的有序四卡 mesh（物理 1、2、3、4）。学习率和参数归属
由 `src/wm3d_wam/training/parameter_groups.py` 决定，CLI 不提供临时覆盖。

## 8. 精确恢复

恢复时复用 phase、max steps、seed、micro-batch、gradient accumulation、output
目录和 GPU mesh，将初始化参数改为 `--resume`：

```bash
"$WM3D_TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_wm3d_wam.py \
  --phase world_core_pretrain \
  --max-steps 30000 \
  --micro-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --num-workers 4 \
  --warmup-steps 500 \
  --log-interval 10 \
  --validation-interval 500 \
  --validation-samples-per-rank 8 \
  --checkpoint-interval 500 \
  --keep-last-checkpoints 3 \
  --resume outputs/train/wm3d_wam_k16_r5/stage_a_world_core_gpu1_4/checkpoints \
  --output-dir outputs/train/wm3d_wam_k16_r5/stage_a_world_core_gpu1_4
```

`--stop-after-step N` 在 phase-local step N 验证、保存完整 checkpoint，并写
`paused.json` 后退出。它不改变 `max_steps`。

## 9. 日志检查

训练 JSONL 每个 log interval 至少包含：

- `loss_total` 与各分项 loss；
- `grad_norm_preclip`；
- 每个 optimizer group 的 learning rate；
- `samples_per_second_global` 和 `step_seconds`；
- `peak_memory_gib`；
- `decode_retries_per_step`；
- rank 0 的 source/program 计数。

正式启动后先检查至少十个 optimizer step。任何非有限 loss、持续 decode retry、
GPU 0 占用、错误 source 或 checkpoint 缺少完成标记都应停止该次运行并诊断。
