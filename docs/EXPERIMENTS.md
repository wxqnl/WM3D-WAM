# WM3D-WAM v1 前期实验记录

更新日期：2026-08-20。机器为 New-H100-2，NVIDIA H100 80GB HBM3。实验使用
生产宽度模型、官方本地权重、真实 Parquet/MP4 和在线 VGGT，没有缩小层数或
hidden，也没有用合成数据或派生 geometry cache 替代正式路径。

## 1. CPU 合同测试

```text
59 passed
```

测试覆盖 source contract、split、recorded timestamp、原生 action clock、grouped
ABI、动态视角、三种 interaction mask、flow matching、online VGGT-GAM、
Wan/Action MoT、参数归属、分层 sampler、checkpoint 恢复、物理 GPU mesh 门禁和
checkpoint 保留策略。

## 2. 正式数据范围

21 个 source 都有显式合同和固定 episode split。7 个 source 通过当前本地
payload 的控制语义审计，14 个 source 保持 `excluded`。训练 loader 只暴露以下
数据：

| family | train | val | test | source 数 |
|---|---:|---:|---:|---:|
| OXE | 120,320 | 1,227 | 1,227 | 4 |
| RoboCasa | 407,327 | 4,156 | 4,156 | 3 |
| 合计 | 527,647 | 5,383 | 5,383 | 7 |

全量 21-source split 仍覆盖 547,382 个 eligible episode，但被排除 source 不会
因为存在 split 文件而进入 sampler。source contract 同时限制允许的 program、
raw dimension、group mapping、单位、frame、composition operator、gripper
polarity 和 quality weight。

分层 sampler 的选择顺序为 program → family → source → episode → window →
view。program、family、source 和 view bucket 按 local sample index 在各 rank
同步，episode 与 window 按 global sample index 分开，既满足 FSDP route 一致性，
又不让不同 rank 重复同一窗口。sampler cursor 只记录已经完成 optimizer step 的
样本，不受 DataLoader 预取影响。

## 3. 时间窗与在线解码

视频统一为 5 Hz、9 帧、1.6 秒。Action 保留 5/10/15/20 Hz 原始时钟；名义
future event 数为 8/16/24/32。recorded timestamp 的边界抖动可能多保留一个
合法事件，所以张量容量按 `ceil(duration * source_hz) + 1` 计算。历史 3.2 秒
最大容量为 65，未来 1.6 秒最大容量为 33，不会把边界事件截断。

MP4 loader 根据 manifest PTS seek 到首个目标之前的 keyframe，只解码覆盖目标
行的 GOP 和约 2.35 秒片段。对同一真实样本，新旧路径生成的 observed RGB、
future RGB 和 Wan 输入逐元素一致；单样本解码从 2.5575 秒降到 0.5975 秒。

一个真实 RoboCasa validation 请求原先因为 65 个 history event 触发四次重试，
耗时 24.9 秒。动态容量后第一次读取即成功，用时 0.6026 秒，保留 65 个 history
event 和 32 个 future event。trainer 在 FSDP forward 前 collective 检查所有
rank 的读取状态，后续数据错误会同步失败，不再形成 NCCL 等待。

## 4. 单卡生产宽度门禁

以下结果使用物理 GPU 1。

### 4.1 Stage A online Geometry-GAM

结果：`outputs/preflight/final_bridge_geometry_stage_a.json`

| 指标 | 数值 |
|---|---:|
| total loss | 1.013668 |
| future shallow feature | 1.000575 |
| depth / world point / camera pose | 0.072368 / 0.020363 / 0.027891 |
| direct/refined auxiliary action | 0.010309 |
| forward / backward | 5.650 s / 1.185 s |
| peak memory | 10.632 GiB |

geometry predictor/history/aux heads 与 VGGT deep pairs 4–23 都得到 finite 非零
梯度。future shallow target 已 detach，direct/refined action 保持 grouped shape
`[1,32,8,16]`。

### 4.2 Action-only 完整路径

结果：`outputs/preflight/final_bridge_action_only.json`

raw RGB → UMT5/VAE → online VGGT-GAM → sparse geometry K/V → observed-video
prefill → 30-layer grouped Action flow 完成 forward/backward。action flow loss 为
1.914100，峰值显存 21.273 GiB。Action Expert、geometry predictor 和 geometry
adapter 都得到 finite 非零梯度。

训练 action-only 与部署 prefill cache 的输出逐元素一致。给 policy 分支加入或
移除 clean future RGB 时，predicted shallow、deep visual 和 geometry token 的
最大差值都为 0，说明 future target 没有泄漏到动作条件图。

### 4.3 Forward-world 与双向耦合

真实 RoboCasa Atomic 20 Hz 样本的 video flow loss 为 1.759363，geometry loss
为 1.030978，输出 velocity shape 为 `[1,48,3,16,16]`，observed latent 保持
逐元素不变。

隔离 objective 后，action loss 对 Wan DiT 产生 801 个有梯度 tensor，gradient
norm sum 为 22.9960；joint video loss 对 Action Expert 产生 818 个有梯度
tensor，gradient norm sum 为 5.7852。Action 使用了 Wan 的逐层表征，video 也能
在 joint program 中更新 Action Expert。

### 4.4 三步短拟合

固定真实 Bridge window、flow timestep 和 noise，Stage B warmup 的 loss 为：

```text
1.961929 → 1.918631 → 1.885765 → 1.839303
```

三次 AdamW step 后 probe 参数最大变化为 `4.58e-5`，峰值显存 25.414 GiB。

## 5. 多卡正式 pipeline canary

完整模型使用物理 GPU `1,2,5,6,7`。Stage A 的恢复实验使用 GPU `1,5`。这些
实验都读取正式七源 sampler 与真实数据。

### 5.1 Stage A：canonical DCP 与精确恢复

目录：`outputs/canary/stage_a_fsdp2_resume_final_v3`

| 步骤 | route/source | total loss | grad norm | validation |
|---:|---|---:|---:|---:|
| 1 | geometry / oxe_bridge | 1.049200 | 1.125761 | 1.048720 |
| 2，恢复后 | geometry / oxe_bc_z | 1.045226 | 0.862645 | 1.042834 |

step 1 保存 canonical DTensor DCP；新进程从 model、optimizer、scheduler、RNG 和
sampler cursor 恢复后完成 step 2。最终 cursor 为 2，两个 checkpoint 都有完成
标记。训练峰值显存为 29.689 GiB，进程最终记录峰值为 33.147 GiB。

### 5.2 Stage B warmup：三种 route 与 rank-local 恢复

目录：`outputs/canary/stage_b_warmup_fsdp5_local_ckpt_v9`

| 步骤 | program/source | total | action | video | geometry | grad norm |
|---:|---|---:|---:|---:|---:|---:|
| 1 | forward_world / oxe_droid | 1.414732 | 0 | 0.391411 | 1.023321 | 0.874933 |
| 2，恢复后 | joint / oxe_furniture_bench | 2.682766 | 1.221465 | 0.959269 | 0.502033 | 3.206210 |
| 3 | action_only / oxe_furniture_bench | 0.959309 | 0.933515 | 0 | 0 | 5.268638 |

step 1 保存五个 model shard、五个 optimizer/runtime shard 和完成元数据。新进程
精确恢复后完成 joint 与 action-only route；step 3 validation 为 1.613138，cursor
为 3，并写出第二个完整 checkpoint。step 1 平均峰值显存 64.016 GiB。

### 5.3 Stage B main：解冻 Wan 与 VGGT deep

目录：`outputs/canary/stage_b_main_fsdp5_v2`

从 warmup rank-local checkpoint 初始化后，Wan DiT 和 VGGT deep 进入 optimizer。
真实 forward-world step 的 total/video/geometry loss 为
1.378100/0.361505/1.016595，pre-clip grad norm 为 2.395052。validation 为
1.071116，平均峰值显存 72.217 GiB，rank 0 最终峰值 70.938 GiB。完整 checkpoint
成功写入。

这个 canary 的 `max_steps=1`，scheduler 在 optimizer step 后把记录的下一步 LR
降到 0；本次更新使用的是配置中的初始 LR。正式多步 schedule 不存在这一边界
现象。

### 5.4 Stage C：tri-stream alignment

目录：`outputs/canary/stage_c_tri_stream_fsdp5_v2`

seed 20260822 的首个 route 为 `joint_world_action / robocasa_composite`：

| 指标 | 数值 |
|---|---:|
| total / action / video / geometry | 2.024735 / 0.936038 / 0.591476 / 0.497221 |
| pre-clip grad norm | 6.769821 |
| decode retries | 0 |
| step time | 9.824 s |
| average peak memory | 66.325 GiB |
| validation total | 1.870091 |

validation 后成功写入完整 rank-local checkpoint，rank 0 最终峰值为 65.045 GiB。

## 6. Checkpoint 体积与磁盘

| 目录 | 体积 |
|---|---:|
| Stage A，两个 checkpoint | 25 GiB |
| Stage B warmup，两个 checkpoint | 100 GiB |
| Stage B main，一个 checkpoint | 91 GiB |
| Stage C，一个 checkpoint | 91 GiB |

实验结束时 `/data` 可用空间约 1.6 TiB。正式配置每 5,000 步保存并只保留最近
两个已完成 checkpoint；Stage B warmup 每 1,000 步保存。未完成目录不参与自动
清理，便于诊断写盘或进程故障。

## 7. 结论与边界

当前证据支持启动七源正式训练：在线数据读取、VGGT 核心路径、Wan/Action 逐层
耦合、三种 program、FSDP、validation、跨阶段初始化和精确恢复都已在真实模型
上闭合。

这些 canary 没有证明策略成功率、长 rollout 稳定性、task-OOD 或相对 baseline
收益。14 个 excluded source 也不能从现有七源结果推断为可用。它们需要独立的
payload-level 控制合同审计，策略质量需要正式训练后的统一评测。
