# WM3D-WAM v1 完整设计

版本：Revision 5

日期：2026-08-23

状态：Revision 4 正式训练已停止并保留；Revision 5 结构修复已通过完整测试、真实
单卡三 route backward、四卡 Stage A/Stage B FSDP 与 checkpoint 门禁

## 1. 设计结论

WM3D-WAM 使用原版 WM3D 的世界状态先验和事实动力学作为 world model core。
VGGT 是模型内部的视觉几何编码器，Wan2.2 负责 RGB 生成，Grouped ActionDiT
负责动作生成。GAM 不再承担世界状态预测，也不提供动作头。

所有未来世界量统一到 `K=16`：

| 合同 | 数值 |
|---|---:|
| 世界时间网格 | 10 Hz |
| 未来跨度 | 1.6 s |
| 未来状态数 K | 16 |
| 未来时刻 | 0.1, 0.2, ..., 1.6 s |
| Wan RGB 帧数 | 当前帧 1 + 未来帧 16 = 17 |
| Wan latent 时间长度 | 5 |
| VGGT deep 未来锚点 | 0.4, 0.8, 1.2, 1.6 s |

K=16 统一的是世界状态、时间边界和视频目标。它不把所有数据的控制器强行改成
10 Hz，也不要求每个 source 伪造 16 个动作。动作保留 5/10/15/20 Hz 原生事件，
按实际秒数归入 16 个 0.1 秒区间。

## 2. 模块职责

| 模块 | 唯一职责 | 不负责 |
|---|---|---|
| WM3D state prior | 从观测、历史状态、历史动作、语言和时间预测 K=16 世界状态 | 不输出动作，不解码 RGB |
| WM3D factual dynamics | 用已知未来动作修正完整的 action-free prior | 不读取未来 RGB |
| VGGT | 在线编码观测、提供在线监督、传播深层几何 | 不预测策略动作 |
| Wan2.2 Video Expert | 预测未来 RGB latent flow | 不承担 grouped action decoder |
| Grouped ActionDiT | 预测 source-native grouped action flow | 不承担 RGB decoder |
| Wan/Action MoT | 按 route 执行有方向的逐层 mixed attention | 不让 noisy action 反向污染 RGB |

这份职责划分解决了旧方案中的两个混淆。第一，Geometry-GAM 不是 world model
core；新的 Stage A 直接训练 WM3D 世界状态。第二，动作没有绕开 Wan 能力。
ActionDiT 从 Wan2.2 初始化，并在 30 层 mixed attention 中持续读取 Wan 表征；
动作最终由结构化 ActionDiT 头输出，因为 Wan 的视频头没有 robot group、语义、
控制频率和 embodiment ABI。

## 3. 代码基线

仓库采用组合式基线，不整体 fork 任一旧项目：

| 来源 | 复用内容 |
|---|---|
| 原版 WM3D 3D | 多视角状态融合、factorized state blocks、连续物理时间、action-free prior、factual dynamics |
| FastWAM / Wan2.2 | VideoDiT、VAE、UMT5、flow scheduler、ActionDiT 初始化、逐层 mixed attention、observed-video K/V prefill |
| node 41 VGGT-GAM 原型 | VGGT pair 0–3 / 4–23 的浅层切分与深层恢复适配器 |
| Worldscape-MoE | 多源数据隔离、family/source 分层采样和显式 source contract 的组织经验 |

GAM 的 future predictor、policy token、direct/refined action heads 均不在 active
factory graph。`src/wm3d_wam/vendor/vggt_gam/` 中仍保留来源代码和许可证，用于
追溯 VGGT adapter，不代表运行时加载 GAM policy。

## 4. 数据划分

### 4.1 划分单位

训练、验证、测试先在 episode 层物化，再生成窗口。manifest 有 parent trajectory
时，同一 parent 的全部 episode 必须进入同一 split。没有 parent 字段时，episode
是最小不可拆单位。window 不参与随机重划，因此相邻窗口不会跨 split 泄漏。

当前只开放七个已经核对控制语义的 source：

| source | 原生 Hz | train | val | test | 可用视频 route |
|---|---:|---:|---:|---:|---|
| oxe_bridge | 5 | 20,690 | 211 | 211 | action-only；world-core 只做稀疏监督 |
| oxe_droid | 15 | 83,209 | 849 | 849 | 全部 |
| oxe_furniture_bench | 10 | 2,365 | 24 | 24 | 全部 |
| oxe_bc_z | 10 | 14,056 | 143 | 143 | 全部 |
| robocasa_atomic | 20 | 5,999 | 61 | 61 | 全部 |
| robocasa_composite | 20 | 15,921 | 162 | 162 | 全部 |
| robocasa_mg | 20 | 385,407 | 3,933 | 3,933 | 全部 |
| 合计 |  | 527,647 | 5,383 | 5,383 |  |

其余十四个 source 保留统计与 split，但 source contract 标记为 `excluded`。在确认
单位、坐标系、composition operator 和 gripper polarity 前，sampler 不会选择
这些数据。

### 4.2 采样层级

sampler 的顺序是：

```text
program -> family -> source -> episode -> window -> real-view bucket
```

program/family/source/view schema 按 rank-local sample index 同步，保证各 FSDP
rank 经过相同模块路径和 tensor shape。episode/window 使用 rank-specific global
index，避免不同 rank 反复读同一个窗口。checkpoint 保存已经提交的 sampler
cursor，不使用 DataLoader 的预取位置。

### 4.3 K=16 的帧选择

世界目标时刻固定为 `t0 + [0.1, 0.2, ..., 1.6]` 秒。loader 根据 recorded
timestamp 选真实帧，并执行 PTS 误差门禁。

- 10/15/20 Hz source 提供 16 个互不重复的真实 future frame。
- 5 Hz source 只能在 0.2、0.4、...、1.6 秒提供八个真实 frame。
- 5 Hz 的八帧写入零基索引 `1,3,5,7,9,11,13,15`。
- 其余八个位置的监督 mask 为 false，RGB tensor 中的占位值不进入损失。
- loader 禁止复制末帧、插帧或把同一真实帧重复绑定到多个 world slot。

Wan 的 forward/joint route 必须有完整 17 帧，因此只接纳至少 10 Hz 的 source。
5 Hz 数据仍可用于 world-core 的八个真实监督时刻和 action-only route。

### 4.4 动作时间

未来动作区间是 `[t0, t0+1.6s)`。原生事件数量为：

| source rate | 名义事件数 | K=16 每个 bin 的事件关系 |
|---:|---:|---|
| 5 Hz | 8 | 约每两个 bin 一个事件 |
| 10 Hz | 16 | 约每个 bin 一个事件 |
| 15 Hz | 24 | 部分 bin 有两个事件 |
| 20 Hz | 32 | 每个 bin 约两个事件 |

recorded timestamp 可能保留一个边界事件，所以 padded capacity 为 33。每个事件
继续携带 value、semantic、group、composition、embodiment、timestamp 和 delta。
world core 只按物理时间分 bin，不改变事件值和顺序。

## 5. WM3D world model core

### 5.1 输入与输出

生产配置的主要 tensor：

| 名称 | shape |
|---|---|
| observed RGB | `[B,4,V,3,224,224]` |
| observed VGGT shallow | `[B,4,V,261,1024]` |
| fused native state | `[B,20,261,1600]`，4 observed + 16 future |
| future native state | `[B,16,261,1600]` |
| future per-view shallow | `[B,16,V,261,1024]` |
| language | `[B,L,4096]` |

`V` 是同一 micro-batch 的真实 camera bucket，取 1、2 或 3。模型只融合 mask 为
真的 camera，不插入假 view。state core 生产配置为 hidden 1600、18 个 state
block、16 heads、一个 factual dynamics block，总参数约 875.9M。

### 5.2 action-free state prior

1. VGGT pairs 0–3 在 `no_grad` 下独立编码四个多视角观测。
2. MultiViewTokenFuser 在每个时刻、每个 patch 坐标上融合真实 camera。
3. GroupedHistoryConnector 编码 16 个历史 state step 和全部原生历史动作事件。
4. 模型加入语言、连续秒数 Fourier embedding、空间 token embedding 和 16 个
   future query。
5. 每个 FactorizedStateBlock 先做同一时刻的 spatial attention，再做同一 token
   的 causal temporal attention，最后执行 SwiGLU。
6. 18 层输出完整的 K=16 action-free future prior。

这个 prior 不接收 future action。测试会对 future action 做随机扰动，并要求
`action_free_native_state` 和 `action_free_tokens` 保持逐元素不变。

### 5.3 factual dynamics

需要预测“执行给定动作后的世界”时，GroupedHistoryConnector 将原生 future
events 聚合到 16 个物理时间 bin。Factual dynamics 对每个 future state 读取该
bin 的真实动作 token；没有事件的 bin 使用显式 null action。它在 action-free
prior 完成后做条件修正，不反向污染 prior 的定义。

以下路径允许 factual action：

- `world_core_pretrain`，使用数据中的真实未来动作；
- `forward_world`，使用调用方给定或数据中的事实动作。

`action_only` 与 `joint_world_action` 使用 action-free state。joint 中的动作本身
处在 flow 去噪过程，不能把 clean future action 偷渡给 world state。

### 5.4 per-view shallow decoder

WM3D state 是跨视角融合的内部状态。ViewTokenDecoder 给每个真实 view 加入可学习
view embedding，再映射回 1024 维 VGGT shallow token。它一次输出全部 16 个
future step。这个 decoder 只恢复 VGGT ABI，不解码 RGB。

## 6. 在线 VGGT

### 6.1 模型内计算

VGGT 不做离线 cache。每个训练 forward 包含：

- frozen shallow teacher：观测和 clean future RGB 经过 pairs 0–3；
- WM3D student：从观测预测 K=16 shallow tokens；
- trainable deep propagation：预测 token 经过 pairs 4–23；
- frozen geometry heads：输出 depth、world points 和 camera pose 用于监督。

clean future RGB 只进入 detached target branch，不进入部署图。shallow target 分块
在线计算，`shallow_scene_chunk_size=8`，用于限制 batch 4 的瞬时显存；它不改变
模型、分辨率或样本。

### 6.2 dense state 与 sparse deep anchors

WM3D 始终预测 16 个 dense world state。VGGT deep stack 只在未来索引
`3,7,11,15` 运行，对应 0.4、0.8、1.2、1.6 秒。deep 输入还包含四个 observed
keyframe，因此 deep 时序长度为 8。

稀疏 deep anchor 是计算安排，不改变 K。所有 16 个 step 都接受 shallow feature
监督；四个 anchor 额外接受 depth、point、pose 监督，并向 Wan/Action 提供
geometry K/V。5 Hz source 在这四个时刻都有真实 RGB，因此 anchor supervision
完整。

### 6.3 geometry tokens

每个 view 的 VGGT deep token 包含 5 个 special token 和 16×16 patch。Reducer
保留 5 个 special token，把 patch grid 池化到 4×4。每个 anchor、每个 view 最终
提供 21 个 geometry token。SparseGeometryKVAdapters 在 Wan/Action 第
5/11/17/23/29 层注入这些 token。

## 7. Wan2.2 和 ActionDiT

### 7.1 RGB 路径

Wan2.2 TI2V-5B Video Expert 保留 30 层、hidden 3072、FFN 14336。VAE 接收
`[B,3,17,H,W]` RGB 并输出 `[B,48,5,H/16,W/16]` latent。训练对完整 latent
采样 continuous flow timestep 和 noise，Video Expert 预测 velocity。VAE 和
UMT5 永久冻结。

### 7.2 动作路径

Grouped Action Expert 有 30 层、hidden 1024、FFN 4096。Action codec 不把机器人
压成一个固定 7D 向量，而是保留最多 8 个 group、每组 16 个标量及其语义 mask。
输出 velocity 与 grouped action tensor 同 shape。

每个物理标量先将 `value feature` 与 joint/axis/semantic/group/dimension field
metadata 拼接并经过非线性 `phi(value, field)`，再进行 masked set pooling。禁止先做
`value_encoder(value) + field_metadata` 后直接求和：该写法对不同字段之间的数值
置换严格不敏感，会把 x/y、不同关节或 gripper 值编码为同一个 token。state history
codec 使用同一字段—数值绑定合同。

Action Expert 的通用 transformer 权重由 Wan2.2 初始化。shape 一致的 tensor
直接迁移，shape 不一致的 tensor 使用 FastWAM 的逐维线性插值和 alpha scaling。
grouped codec 与输出层单独初始化。

### 7.3 Wan 能力如何进入动作

Wan 和 Action Expert 在每一层分别生成 Q/K/V，随后执行一次受 interaction mask
约束的 mixed attention，再回到各自 projection 和 FFN。action-only 部署使用
observed-video K/V prefill；训练调用同一 cache 路径。动作因此持续读取 Wan 的
视觉表征，同时保留适合 robot control 的输出 ABI。action-only loss 只更新 Action
Expert；Wan、VGGT 和 world-state conditioner 在该 route 中是 detached 条件，避免
policy-only 样本把视频生成器推向静态捷径。

`forward_world` 使用 clean factual action 条件 RGB。16 个 0.1 秒物理 action bin
按照 `future_action_history.step_indices` 精确映射到四个未来 Wan latent group；anchor
不读 future action，第 q 个未来 latent group 只读本组 action。该 mask 对齐 FastWAM
正式配置的 `action_group_causal_mask_mode=group_diagonal`，同时不假设每组 event
数量相等或数据源 action rate。

`joint_world_action` 遵循成熟 FastWAM 的 joint 方向：noisy video 可以为 action
提供上下文，video 不读取 noisy action。它同时优化两个 flow，但不是双向泄漏。

## 8. 四种训练 route

| route | future action 输入 | future video 输入 | 跨流方向 | 输出与损失 |
|---|---|---|---|---|
| world_core_pretrain | clean factual | clean RGB 仅作 detached target | 不加载 MoT | factual shallow、action-free shallow、geometry |
| action_only | noisy action | current RGB anchor | video→action cache | action flow；只更新 Action Expert |
| forward_world | clean factual | noisy 17-frame latent | group-diagonal action→video | video flow、factual shallow、geometry |
| joint_world_action | noisy action | noisy 17-frame latent | video→action | action flow、video flow、action-free shallow |

三个 MoT route 使用独立的严格 attention mask。target RGB 永远不进入 action-only
policy graph。cache parity 和 target-leakage 测试要求两条部署等价路径输出完全
一致。Stage B warmup/main 的默认采样比例为 `action_only=0.25`、
`forward_world=0.65`、`joint_world_action=0.10`；Stage C 为 0.20/0.60/0.20。

## 9. 损失

### 9.1 world-core pretrain

```text
L = 1.0 * L_factual_shallow
  + 0.25 * L_action_free_shallow
  + 0.30 * mean(L_depth, L_world_point, L_camera_pose)
```

shallow 使用 token cosine distance，geometry 使用 masked Smooth L1。所有损失只
统计 `future_world_valid_mask` 为真的真实 frame/view。5 Hz 的八个缺失 world
slot 不进入分母。

### 9.2 full model

- action flow：只统计 event/group/value 三层 mask 都为真的原生动作标量；
- video flow：只训练 future latent，observed anchor 保持不变；
- forward-world geometry：feature weight 1.0，geometry weight 0.3；
- joint geometry：action-free feature weight 0.5，不执行昂贵 geometry head target。

项目不再计算 GAM auxiliary action loss。ActionDiT 是唯一动作训练出口。

## 10. 训练阶段和参数所有权

| 阶段 | 步数 | 训练参数 | 默认 batch |
|---|---:|---|---|
| world_core_pretrain | 30,000 | WM3D core、history connector、reducer、VGGT pairs 4–23 | 4/GPU × 4，accum 1 |
| wan_action_warmup | 2,000 | WM3D core、Action Expert、geometry adapters | 1/GPU × 4，accum 1 |
| wan_action_main | 38,000 | 上述参数 + VGGT deep + Wan VideoDiT | 1/GPU × 4，accum 1 |
| tri_stream_alignment | 20,000 | 与 main 相同，降低部分学习率 | 1/GPU × 4，accum 1 |

VGGT shallow pairs、VGGT DPT heads、Wan VAE 和 UMT5 始终冻结。optimizer group
必须完整覆盖所有 `requires_grad=True` 参数，重复归属或漏参会立即报错。

world-core 使用 canonical DTensor DCP，可在不同 world size 之间重分片。完整
Wan/Action 阶段使用 rank-local FSDP shard，精确恢复要求相同 world size 和相同
有序物理 GPU mesh。

## 11. batch 选择

当前有序 mesh 固定为四张 H100 80GB（物理 1、2、3、4）。真实结果：

| phase | micro-batch/GPU | accum | global batch | 结果 | 训练峰值 |
|---|---:|---:|---:|---|---:|
| world-core | 4 | 1 | 16 | 正式训练 30,000 步完成 | 长期稳定 |
| full Wan/Action | 1 | 1 | 4 | train、validation、rank-local checkpoint 通过 | 68.6 GiB |
| full Wan/Action | 1 | 4 | 16 | backward OOM，无完整 checkpoint | 超过 80 GiB |

完整模型在 accumulation 4 时会同时保留 FP32 gradient shard 和下一轮
full-parameter all-gather；失败 rank 还需申请约 11.29 GiB，但只剩约 3.48 GiB。
因此当前正式配置固定为 micro-batch 1、accumulation 1。Stage B warmup、main 和
Stage C 的 rank-local checkpoint 必须继续使用同一个有序四卡 mesh。

## 12. 推理接口

### 12.1 action-only

输入 observed RGB、机器人历史和任务文本。模型先计算 action-free K=16 world
state，再通过 observed-video cache 对 grouped action flow 去噪。输出恢复到 source
contract 定义的原生事件时刻和 group 语义。

### 12.2 forward-world

输入 observed RGB、机器人历史、任务和给定未来动作。factual dynamics 生成动作
条件世界状态，Wan 对四个 future latent group 去噪并解码 17 帧 RGB。VGGT
geometry 可作为 rollout 诊断输出。

### 12.3 joint rollout

动作和视频同时从 noise 开始；视频流保持自回归/文本/几何条件，动作流逐层读取
当前 noisy video 表征。world state 使用 action-free prior，避免把尚未完成的 noisy
action 当事实动力学。联合采样器可在每轮 denoise 后更新两个流。

## 13. 已验证门禁

- K=16 数据布局覆盖 5/10/15/20 Hz，不重复 5 Hz frame；
- corrupt MP4 被转换为可重试的 episode error；
- WM3D core shape、action-free invariance 和 state/dynamics/decoder 梯度通过；
- 真实 DROID world-core forward/backward 和所有梯度 owner 通过；
- 真实 RoboCasa action-only cache parity 与 future-target leakage delta 均为 0；
- grouped action/state codec 对 x/y、关节和字段间 value swap 均敏感；
- 16 个物理 action bin 到四个 future latent group 的 group-diagonal mask 通过；
- 单卡三 route 真实 backward 的梯度归属通过，action-only 对 Wan/VGGT/WM3D 梯度
  为 0；
- 四卡 Revision 5 world-core 完成两个真实 optimizer step 和 canonical DCP；
- 四卡 Revision 5 full model 的最终 group-diagonal 路径完成一个真实
  `forward_world` optimizer step 和完整 rank-local checkpoint，峰值约 68.5 GiB；
  action-only/joint 的最终梯度归属由单卡 full-model backward 覆盖。

这些门禁证明 pipeline 可训练，不证明下游任务成功率。策略质量、长 rollout
稳定性和 OOD 泛化必须在正式 checkpoint 上独立评测。

## 14. 迁移规则

Revision 4 的 grouped state/action codec 在 set pooling 前将 value feature 与 field
metadata 相加，导致字段间 value swap 不改变 token；同时旧 MoT route 使用了过宽的
action→video 可见性。其 checkpoint 文件仍完整保留，但这些权重不能 resume
Revision 5，也不能作为 Revision 5 后续阶段初始化点。

Revision 5 从官方 VGGT 权重和新初始化的 WM3D core 重新开始 Stage A。后续阶段只
允许从同一 revision 的完整 checkpoint 初始化。`geometry_gam` 等更早 checkpoint
同样不兼容。

## 15. 最终回答

数据划分采用 episode/parent 级固定 split，当前七源门禁合理。K 统一为 16 更好，
因为状态、视频和动作时间边界共享同一 10 Hz 世界坐标；5 Hz 数据通过 mask 保留
真实信息，不做伪造。动作输出使用 Wan2.2 的逐层表征，但由独立的 grouped
ActionDiT 头生成。VGGT 是在线模型组件，原版 WM3D 是 world model core，GAM
只留下 shallow/deep adapter 的实现来源。
