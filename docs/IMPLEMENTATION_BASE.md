# WM3D-WAM 实现基线与状态

| 项目 | 内容 |
|---|---|
| 日期 | 2026-08-23 |
| 分支 | `v1` |
| 设计版本 | Revision 5，WM3D core + K=16 + motion-conditioning fix |
| 训练机器 | New-H100-2，物理 GPU 1–4 |
| 数据合同 | `configs/data/grouped_robot_v1.yaml`、`configs/data/source_contracts_v1.yaml` |
| 模型合同 | `configs/model/vggt_geometry_v1.yaml`、`configs/model/wan_action_mot_v1.yaml` |
| 训练合同 | `configs/train/wm3d_wam_v1.yaml` |

## 基线选择

WM3D-WAM 使用独立仓库作为集成边界，没有把一个旧项目整体复制成主干。选择如下：

| 子系统 | 基线 | 采用内容 |
|---|---|---|
| 世界状态 | 原版 WM3D 3D | 多视角 token fusion、factorized state trunk、连续时间、action-free prior、factual dynamics |
| RGB 和动作耦合 | FastWAM / Wan2.2 | VideoDiT、VAE、UMT5、ActionDiT 初始化、逐层 MoT、flow matching、prefill cache |
| 几何 | 官方 VGGT + node 41 adapter | pairs 0–3 shallow split、pairs 4–23 deep resume、DPT heads |
| 数据组织 | 原 WM3D + Worldscape-MoE 经验 | recorded timestamp、grouped robot ABI、显式 source contract、分层 sampler |

原版 WM3D 最适合承担 world model core，因为它的状态先验与事实动力学就是本项目
需要的世界状态抽象。FastWAM 最适合承担 Wan/Action 的工程基线。node 41 项目只
提供经过实测的 VGGT 切分适配器。GAM future predictor 和 GAM action heads 已从
active graph、optimizer、loss 和公共导出中移除。

## Active 模型图

```text
raw observed RGB
    -> frozen VGGT pairs 0-3
    -> observed shallow tokens
    -> WM3D action-free state prior, K=16
         -> optional factual-action dynamics
         -> per-view shallow decoder
    -> trainable VGGT pairs 4-23 at four anchors
    -> reduced geometry K/V
    -> Wan2.2 Video Expert <- clean grouped action (forward_world)
    -> Wan2.2 Video Expert -> Grouped Action Expert (action/joint)
         -> RGB velocity
         -> grouped action velocity
```

`src/wm3d_wam/models/factory.py` 只构造 `WM3DStateDynamicsCore`、
`GroupedHistoryConnector`、`VGGTEncoder` 和 reducer。没有 GAMFuturePredictor、
policy token 或 auxiliary action head 的构造路径。

## 数据实现

`scripts/build_episode_splits.py` 为 21 个 source 物化固定 split。source contract
允许七个 verified source 进入 sampler，合计 527,647 train、5,383 val、5,383
test episode。十四个 source 保持 excluded。

在线 loader 提供：

- 16 个历史 grouped state step 和全部原生历史 action event；
- 四个 observed VGGT keyframe；
- K=16 的 10 Hz future world grid；
- 10 Hz 以上 source 的 17 帧 Wan clip；
- 5 Hz source 的八个真实 target 与八个 masked slot；
- 5/10/15/20 Hz 原生 future action event，按 16 个时间 bin 建索引；
- 一到三个真实 camera view，不补假视角；
- manifest PTS seek 和覆盖目标行的稀疏 MP4 decode。

模型不读取 VGGT、depth、point、pose 或 Wan latent 派生缓存。

## 模型实现

`src/wm3d_wam/models/wm3d_state_dynamics.py` 包含约 875.9M 参数的生产状态核心。
它先生成不依赖未来动作的 K=16 prior，再由独立 dynamics block 使用事实动作。
`ViewTokenDecoder` 将融合状态恢复到 `[B,16,V,261,1024]`。

`src/wm3d_wam/models/online_vggt_geometry.py` 负责在线 shallow targets、VGGT deep
resume 和 geometry reduction。所有 16 步有 shallow supervision；0.4、0.8、1.2、
1.6 秒有额外 deep geometry supervision。future teacher 使用 chunked no-grad
编码，clean target tensor 始终 detach。

`src/wm3d_wam/models/wan_action_mot.py` 保留 30 层 Wan Video Expert 与 30 层
Grouped Action Expert。两个 expert 复用逐层 mixed attention，但 route 方向显式受
mask 约束：`forward_world` 用 16→4 group-diagonal clean action→video，
`action_only`/joint 用 video→action，noisy action 不进入 RGB query。动作输出仍遵守
grouped robot ABI，geometry 在五层稀疏注入。

grouped state/action codec 在 pooling 前通过非线性 `phi(value, field)` 绑定数值和
轴/关节语义，避免字段间数值置换被错误编码成同一个 token。action-only 的 frozen
conditioner 在 no-grad 路径构建，因此 policy loss 只更新 Action Expert。

## 训练与恢复

`scripts/train_wm3d_wam.py` 支持四个 phase：

- `world_core_pretrain`；
- `wan_action_warmup`；
- `wan_action_main`；
- `tri_stream_alignment`。

运行时使用 BF16 forward/reduce、FP32 master weights、FSDP FULL_SHARD、activation
checkpointing、cosine schedule、validation、编号 checkpoint 和已提交 sampler
cursor。world-core checkpoint 使用 canonical DCP。完整模型使用 rank-local shard，
恢复时校验有序 GPU mesh、micro-batch 和 gradient accumulation。

四卡 K=16 world-core 的正式值是 micro-batch 4、accumulation 1、global batch 16。
完整 Wan/Action 使用 micro-batch 1、accumulation 1、global batch 4，最终
group-diagonal FSDP canary 峰值约 68.53 GiB。完整模型 accumulation 4 会因 gradient
shard 与下一轮 all-gather 并存而 OOM，因此禁止使用。

## 本地资产

```text
/data/Minko/models/WM3D-WAM/Wan2.2-TI2V-5B
/data/Minko/models/WM3D-WAM/ActionDiT/ActionDiT_grouped_Wan22_1024.pt
/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/assets/vggt_model.safetensors
/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/runtime/vggt
```

路径中的 `gam_node42_v1` 是已存在的 VGGT checkpoint/source 资产目录名，不代表
运行时使用 GAM policy。

## 当前边界

Revision 5 已完成 75 项测试、真实单卡三 route backward、四卡 world-core DCP 和
最终 group-diagonal full-model optimizer/checkpoint 门禁。Revision 4 的 Stage A/B
权重因 codec 参数空间和 attention 语义改变而不兼容，只保留作审计证据；Revision 5
从基础权重重新训练 Stage A。正式 checkpoint 上的 motion ratio、field-swap
sensitivity 和 held-out RGB demo 仍是训练中的质量门禁。
