# WM3D-WAM Revision 4 前期实验

更新日期：2026-08-20。机器为 New-H100-2，GPU 为 H100 80GB。实验使用生产宽度
模型、本地正式权重、真实 Parquet/MP4 和在线 VGGT。没有缩小 hidden/layer、使用
合成样本或读取派生 geometry cache。

## 1. CPU 合同测试

```text
62 passed in 5.45s
```

测试在 GPU 完全不可见时运行，覆盖数据、模型 shape、梯度不变量、sampler 和
checkpoint 合同。vendored GAM predictor 和已删除的 auxiliary action head 不再属于
active 测试面。

## 2. 数据布局

三种真实 source/program probe：

| source / route | observed views | K=16 真实 RGB slot | Wan frames | 原生 action events |
|---|---:|---:|---:|---:|
| oxe_bridge / world-core | 2 | 8 | 1 | 8 |
| oxe_droid / forward-world | 3 | 16 | 17 | 24 |
| robocasa_atomic / action-only | source view bucket | 不解码 future target | 1 | 32 |

Bridge 的八帧映射到索引 `1,3,...,15`，其余 slot 的 supervision mask 为 false。
DROID 15 Hz 数据为 10 Hz world grid 选择 16 个互不重复的真实 frame。三种样本的
future action timeline 都有 16 个物理时间 bin，但 event 数保持原生频率。

PyAV 遇到损坏 bitstream 时，loader 将 `av.FFmpegError` 转为带 episode/source
上下文的可重试 `OnlineEpisodeError`。它不会让某个 rank 无声退出并把其他 rank
留在 collective 中。

## 3. WM3D core 单卡门禁

真实 DROID window、生产配置、物理 GPU 1：

| 项目 | 结果 |
|---|---|
| native future state | `[1,16,261,1600]` |
| per-view shallow prediction | `[1,16,3,261,1024]` |
| WM3D core 参数 | 875,935,233 |
| trainable VGGT deep 参数 | 503,941,120 |
| forward peak memory | 21.43 GiB |

factual backward 的 total loss 为 1.2869。以下参数组都得到 finite、非零梯度：

| 参数组 | gradient norm sum |
|---|---:|
| action-free state prior | 24.65 |
| factual dynamics | 1.99 |
| per-view token decoder | 1.09 |
| VGGT deep | 2.086 |

activation checkpointing 下 backward peak memory 为 11.81 GiB。测试还单独扰动
future action，并确认 action-free prior 完全不变；factual output 会随动作变化。

## 4. 完整模型单卡门禁

### 4.1 action-only

真实 RoboCasa 样本：

| 项目 | 结果 |
|---|---|
| ActionDiT output | `[1,33,8,16]` |
| Wan anchor latent | `[1,48,1,16,16]` |
| peak memory | 21.27 GiB |
| observed-video cache parity max delta | 0 |
| future-target leakage max delta | 0 |

WM3D state core、Action Expert 和 geometry adapters 都得到 finite、非零梯度。
clean future RGB 的有无不改变 action-only geometry 或 action output。

### 4.2 forward-world

17 帧 RGB 经 Wan VAE 得到 `[1,48,5,16,16]` latent，Video Expert 输出同 shape
velocity。WM3D state、VGGT deep、Action Expert、geometry adapters 和 Wan
VideoDiT 的梯度均 finite、非零。peak memory 为 33.36 GiB。

Action Expert 在 forward-world 中不输出 action velocity，但仍通过逐层 mixed
attention 参与 video 表征，因此得到梯度符合设计。

### 4.3 joint-world-action

joint route 同时输出 action 和 video velocity。WM3D state、VGGT deep、Action
Expert、geometry adapters 和 Wan VideoDiT 的梯度均 finite、非零。peak memory
为 31.00 GiB。

## 5. 七卡 K=16 FSDP

物理 mesh 为 `1,2,3,4,5,6,7`，GPU 0 空闲。所有实验执行真实 train、validation
和 canonical checkpoint。

### 5.1 micro-batch 1

目录：`outputs/canary/k16_world_core_fsdp7_step1_20260820`

| 指标 | 数值 |
|---|---:|
| global batch | 7 |
| total loss | 1.2688 |
| pre-clip grad norm | 2.9999 |
| step time | 14.7046 s |
| global throughput | 0.4760 sample/s |
| peak memory | 25.2169 GiB |

step 1 checkpoint 完整写入。

### 5.2 micro-batch 4

最终干净运行目录：
`outputs/canary/k16_r4_world_core_fsdp7_mb4_final_v2_20260820`

| 指标 | 数值 |
|---|---:|
| global batch | 28 |
| total loss | 1.266875 |
| pre-clip grad norm | 2.943112 |
| step time | 13.4848 s |
| global throughput | 2.07641 sample/s |
| train peak memory | 48.6996 GiB |
| validation total | 1.256213 |

rank 0 的 micro-batch 来自四个真实 DROID window。validation 每 rank 八个样本，
随后成功写入 step 1 canonical checkpoint。

### 5.3 micro-batch 6

目录：`outputs/canary/k16_world_core_fsdp7_mb6_v2_step1_20260820`

| 指标 | 数值 |
|---|---:|
| global batch | 42 |
| total loss | 1.267013 |
| pre-clip grad norm | 2.908211 |
| step time | 20.9887 s |
| global throughput | 2.00108 sample/s |
| PyTorch train peak memory | 64.3464 GiB |
| validation total | 1.256229 |

训练一步和 checkpoint 均通过，但 validation/checkpoint 周期中驱动侧板卡占用最高
约 78.7GB。该值没有足够余量覆盖视角 bucket、allocator 碎片和长期波动，所以不
作为正式配置。

### 5.4 micro-batch 8

batch 8 在 frozen future shallow target 编码和后续完整图阶段超过 80GB，进程退出，
输出目录只有 `run.json`，没有完成 checkpoint。将 shallow teacher 按 scene 分块后
仍不能为完整 K=16 图提供足够余量。模型、分辨率和 K 均未缩水。

正式选择 micro-batch 4、gradient accumulation 1。编译缓存预热后的最终干净
canary 把全局吞吐从 batch-1 canary 的 0.476 提高到 2.076 sample/s，并保留约
30GB 级板卡余量。

## 6. 损失行为

world-core canary 的 factual shallow loss 与 action-free shallow loss 都约为 1.0，
符合随机初始化 decoder 对 detached VGGT target 的起点。depth、world point 和
camera pose loss 均 finite。所有 optimizer owner 在 backward 后有非零梯度，说明
loss 不是只在 frozen teacher 上计算。

一步 canary 使用 `max_steps=1` 时，cosine scheduler 在更新后记录的下一步学习率
为 0。这是单步 schedule 的端点，不是 optimizer 没有更新。正式 30,000 步训练有
500-step warmup，不存在这个端点问题。

## 7. 结论与边界

现有证据支持启动 Revision 4 world-core 正式训练：K=16 数据、WM3D prior、事实
动作动力学、在线 VGGT、真实多视角、FSDP、validation 和 canonical checkpoint
已经闭合。

这些实验没有证明策略成功率、长 rollout 稳定性或相对 baseline 收益。进入
Wan/Action 阶段前仍需执行新架构的七卡完整模型 canary；旧 Geometry-GAM 的多卡
结果不能替代它。
