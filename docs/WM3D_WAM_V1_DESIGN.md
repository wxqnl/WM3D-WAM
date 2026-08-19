# WM3D-WAM v1 设计方案（Revision 2）

| 项目 | 内容 |
|---|---|
| 状态 | Revision 2，M1 实现中 |
| 日期 | 2026-08-19 |
| 目标仓库 | wxqnl/WM3D-WAM |
| 当前分支 | codex/implement-wm3d-wam-v1 |
| 核心模型 | 在线 VGGT 几何主干 + Wan2.2 视频专家 + Grouped Action Flow Expert |
| 预测范围 | 1.6 秒 |
| 视频时钟 | 全部 source 统一为 5 Hz / 9 帧 |
| 动作时钟 | 保留 source-native 5 / 10 / 15 / 20 Hz |
| 训练设备 | New-H100-2 的 GPU 1–7；GPU 0 禁用 |

实现基线、已完成模块和硬门禁见
[IMPLEMENTATION_BASE.md](IMPLEMENTATION_BASE.md)。本文件继续作为 v1 的
架构与数据合同，阶段性实现不得通过弱化这些合同来换取可运行状态。

## 0. 本次修订解决什么

Revision 1 有三个需要改正的设计点：

1. 数据按 5 Hz / 9 帧和 10 Hz / 17 帧分成两个视频桶。这个划分在时间上成立，但会增加 Wan 与 VGGT 的在线计算、分布式 shape 组合和训练比较变量。现有四种原始频率都能整数降采样到 5 Hz，因此 v1 改为单一视频时钟。
2. Action 由独立 Native3D policy head 预测，不能读取 Wan 中间表征。FastWAM 的公开代码和消融表明，视频联合训练及逐层视频/动作交互会提高动作能力。v1 的主 Action 输出因此改为 Wan 耦合的 Action Flow Expert。
3. VGGT 被当成离线特征生产器，训练依赖 episode feature cache。这样会把几何主干游离到系统之外。v1 改为 GAM 式 split-and-resume：VGGT 在训练 forward 内在线运行，浅层冻结，深层参与学习，不再要求任何 VGGT 派生缓存。

这三项不是互相独立的小改动。统一 5 Hz 降低在线双主干的成本；在线 VGGT 提供几何状态；Wan 和 Action 逐层耦合把视频先验真正传给策略。三者合起来才是一套闭合设计。

## 1. 最终结论

### 1.1 数据

- 当前 21 个 source、547,382 个上游 train-labeled 可用 episode、11,034,768 个候选窗口的统计可信，可继续作为容量基线。
- 上游 train 标签不是 WM3D-WAM 的最终 train/val/test。最终划分必须在 episode 或 parent trajectory 层完成，绝不在 window 层划分。
- 全部 source 的训练视频统一成 5 Hz、9 帧、1.6 秒。5/10/15/20 Hz 分别使用 stride 1/2/3/4，全部选择真实帧，不插帧。
- Action 不随视频降采样。1.6 秒内分别保留 8/16/24/32 个原生控制事件及其真实时间戳。
- 训练集物理划分与训练采样比例分开管理。物理划分固定；OXE 与 RoboCasa 的采样权重按 objective 改变。

### 1.2 Action

- v1 主 Action 输出走 Wan2.2 表征路径，但不由 Wan 视频输出头解码。
- 具体实现是 FastWAM 风格的 Mixture-of-Transformers：Wan Video Expert 与 Grouped Action Flow Expert 各有自己的投影和 FFN，每一层共享一次混合 attention。
- Action token 因而能读取 Wan 的观测视频表征、任务文本和 VGGT 几何 token，同时保留适合机器人控制的独立容量与输出合同。
- GAM 风格的 VGGT direct action head 仍保留，但只作为辅助监督、校准和故障诊断，不是发布时的主动作输出。

### 1.3 VGGT

- VGGT 是在线核心几何主干，不是预处理工具。
- VGGT frame/global block pair 0–3 作为冻结 shallow encoder，在训练 forward 内用 no-grad 计算观测与未来目标 token。
- Continuous Geometry Future Predictor 预测未来 shallow tokens；VGGT block pair 4–23 从分割点恢复计算并参与反向传播。
- 深层几何 token、相机、深度和 geometry-refined action seed 同时供 Wan Video Expert 与 Action Flow Expert 使用。
- 训练不依赖 VGGT token、depth、point 或 pose 的磁盘 cache。Wan VAE 默认也在线运行。

### 1.4 代码基线

新仓库 WM3D-WAM 是唯一集成入口，不把旧项目整体复制过来：

- 以官方 Wan2.2 和 FastWAM 的 DiT / ActionDiT / MoT 交互方式作为视频动作骨架；
- 从 node 41 的 gam-vggt-prototype 迁入已经验证过的 VGGT split-and-resume 思路；
- 从原 WM3D 迁入 grouped robot ABI、数据 adapter、真实时间窗、多视角合同、分布式运行和评测组件；
- 不迁入旧缓存依赖、实验配置堆积和独立于 Wan 的最终 policy head。

## 2. 证据与设计依据

### 2.1 FastWAM 对 Action 路径的启示

FastWAM 不是把 Action 直接塞进 Wan 的视频输出层。它使用两个不同宽度的专家：

| 分支 | 层数 | hidden | FFN | attention |
|---|---:|---:|---:|---:|
| Wan2.2 TI2V-5B Video Expert | 30 | 3072 | 14336 | 24 heads × 128 |
| ActionDiT Expert | 30 | 1024 | 4096 | 24 heads × 128 |

每一层先由两个专家分别产生 Q/K/V，再把视频和动作的 K/V 放入同一次 attention，最后回到各自的输出投影与 FFN。它是两条 token stream 的逐层交互，不是 top-k router，也不是让视频头回归机器人动作。

FastWAM 公开消融中的关键结果如下：

| 设置 | RoboTwin | LIBERO |
|---|---:|---:|
| FastWAM | 91.8 | 97.6 |
| joint video-action variant | 90.6 | 98.5 |
| inverse-dynamics variant | 91.3 | 98.0 |
| no-video training | 83.8 | 93.5 |

这些结果不能直接保证 WM3D-WAM 得到同样增益，但足以否定“Action 永久不读 Wan 表征也不会损失能力”这一默认假设。Revision 2 选择耦合路径，并用等数据、等参数量的 detached-head ablation 验证收益。

本机审计的官方代码位置：

- /data/Minko/external/FastWAM/configs/model/fastwam.yaml
- /data/Minko/external/FastWAM/src/fastwam/models/wan22/fastwam.py
- /data/Minko/external/FastWAM/src/fastwam/models/wan22/action_dit.py
- /data/Minko/external/FastWAM/src/fastwam/models/wan22/mot.py

### 2.2 GAM 对 VGGT 路径的启示

GAM 的核心不是把完整 GFM 输出先做成缓存，而是在中间层分割几何基础模型：

1. 浅层把观测图像编码成 shallow visual tokens；
2. future predictor 在 shallow token 空间预测未来；
3. 预测 token 从分割点进入剩余几何主干；
4. 深层几何推理同时改善未来几何和动作。

node 41 上的 gam-vggt-prototype 已经实现了这一结构：

- VGGT 共 24 对 frame/global blocks，hidden 1024；
- split layer 为 4；
- frame/global block pair 0–3 在线冻结；
- block pair 4–23 恢复计算并训练；
- future predictor 同时输出未来 shallow visual tokens 和 action tokens；
- action token 插入 VGGT 深层序列，再由 direct 与 refined 两条头解码。

本机审计位置：

- /data/Minko/experiments/gam-vggt-prototype/src/robot/modeling/vggt_encoder.py
- /data/Minko/experiments/gam-vggt-prototype/src/robot/modeling/future_predictor.py
- /data/Minko/experiments/gam-vggt-prototype/src/robot/losses/unified_loss.py
- /data/Minko/experiments/gam-vggt-prototype/configs/training/pretraining/gam_vggt_h4.yaml

该原型的 OpenArm 数据路径中存在 RGB / pseudo-depth cache，但它是特定监督与 I/O 选择，不是 split-and-resume 的必要条件。WM3D-WAM 只采用在线分层主干，不采用强制特征缓存。

### 2.3 其他近期 WAM 工作的边界

OpenWAM 与 Tapestry 强调同一模型内的多种 interaction program；Faster-WAM 说明未来表征对 OOD 策略可能有帮助，并采用稀疏融合降低成本。这些工作发布时间较新，Revision 2 只采用可单独验证的结构原则：

- 明确 action-only、forward-world、joint-world-action 三种训练程序；
- 用 attention mask 保证干净未来目标不会泄漏给 policy；
- 默认推理只做观测 prefill 与动作去噪；
- 未来视频或未来表征只作为可选慢路径，不成为实时动作的硬依赖。

## 3. 当前数据审计

### 3.1 权威输入

- 数据 profile：/data/Minko/wm3d_formal_1b_raw_100k_3f056a4_20260816/data_profile.yaml
- episode/window 统计：/data/Minko/wm3d_async_cache_pilot_20260819/episode_window_counts.jsonl
- 原 WM3D 代码：/data/Minko/wm3d_rgb_chunk16_code_20260819/wm3d

### 3.2 总量

表中的 eligible episode 是上游标为 train 且通过当前窗口条件的 episode，不是本项目最终 train 数。

| 家族 | source 数 | 原始 episode | eligible episode | 候选窗口 | 小时 | 旧 profile 权重 |
|---|---:|---:|---:|---:|---:|---:|
| OXE | 18 | 204,860 | 131,743 | 2,930,466 | 933.669 | 83.2% |
| RoboCasa365 | 3 | 568,073 | 415,639 | 8,104,302 | 2,018.787 | 16.8% |
| 合计 | 21 | 772,933 | 547,382 | 11,034,768 | 2,952.456 | 100% |

RoboCasa 占 68.4% 物理小时，旧 profile 却只占 16.8% 采样权重。旧权重适合给 OXE 强上采样，但不能同时代表几何学习、动作学习和视频动力学的最优比例。

### 3.3 source 明细与统一时间方案

分辨率是 H×W。Action events 是半开区间 [t0, t0+1.6s) 中的原生控制事件数。视频包含 t0 与 t0+1.6s 两端，因此是 9 帧。

| source | 原生 Hz | 分辨率 | action/state | eligible episode | 窗口 | 小时 | Action events | 视频 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oxe_bridge | 5 | 256×256 | 7/8 | 21,112 | 128,148 | 105.168 | 8 | 5 Hz / 9f |
| oxe_droid | 15 | 180×320 | 8/8 | 84,907 | 2,107,539 | 511.674 | 24 | 5 Hz / 9f |
| oxe_utaustin_mutex | 20 | 128×128 | 7/8 | 1,458 | 17,737 | 5.026 | 32 | 5 Hz / 9f |
| oxe_stanford_hydra | 10 | 240×320 | 7/8 | 545 | 65,305 | 9.951 | 16 | 5 Hz / 9f |
| oxe_berkeley_autolab_ur5 | 5 | 480×640 | 7/8 | 973 | 20,301 | 5.441 | 8 | 5 Hz / 9f |
| oxe_austin_sailor | 20 | 128×128 | 7/8 | 234 | 31,787 | 4.904 | 32 | 5 Hz / 9f |
| oxe_austin_sirius | 20 | 84×84 | 7/8 | 543 | 14,964 | 3.888 | 32 | 5 Hz / 9f |
| oxe_berkeley_fanuc | 10 | 224×224 | 7/8 | 349 | 6,428 | 1.739 | 16 | 5 Hz / 9f |
| oxe_jaco_play | 10 | 224×224 | 7/8 | 900 | 5,454 | 2.166 | 16 | 5 Hz / 9f |
| oxe_fmb | 10 | 256×256 | 7/8 | 1,753 | 35,466 | 9.394 | 16 | 5 Hz / 9f |
| oxe_berkeley_cable | 10 | 128×128 | 7/8 | 113 | 303 | 1.176 | 16 | 5 Hz / 9f |
| oxe_roboturk | 10 | 480×640 | 7/8 | 1,383 | 15,712 | 5.209 | 16 | 5 Hz / 9f |
| oxe_dlr_edan | 5 | 360×640 | 7/7 | 99 | 1,793 | 0.496 | 8 | 5 Hz / 9f |
| oxe_austin_buds | 5 | 128×128 | 7/24 | 49 | 12,968 | 1.895 | 8 | 5 Hz / 9f |
| oxe_nyu_franka | 5 | 128×128 | 15/13 | 438 | 9,348 | 2.493 | 8 | 5 Hz / 9f |
| oxe_cmu_stretch | 5 | 128×128 | 8/4 | 132 | 6,251 | 1.390 | 8 | 5 Hz / 9f |
| oxe_furniture_bench | 10 | 224×224 | 7/8 | 2,413 | 263,198 | 109.668 | 16 | 5 Hz / 9f |
| oxe_bc_z | 10 | 171×213 | 7/8 | 14,342 | 187,764 | 151.991 | 16 | 5 Hz / 9f |
| robocasa_atomic | 20 | 256×256 | 12/16 | 6,121 | 64,104 | 20.768 | 32 | 5 Hz / 9f |
| robocasa_composite | 20 | 256×256 | 12/16 | 16,245 | 1,658,404 | 383.485 | 32 | 5 Hz / 9f |
| robocasa_mg | 20 | 256×256 | 12/16 | 393,273 | 6,381,794 | 1,614.534 | 32 | 5 Hz / 9f |

### 3.4 数据真正的风险

1. 多数 adapter 仍把 action/state 笼统标为 controller_command/controller_state，scale 与 offset 还是 identity。Wan 或 VGGT 都不能修复单位、坐标系和 gripper 极性错误。
2. RoboCasa 有 left external、right external、eye-in-hand 三路视频；旧路径没有稳定使用 right external。
3. RoboCasa 本地数据只有 MP4 与 Parquet，不能把离线视频指标写成 simulator success。
4. DROID、BC-Z 和 128/84 像素 source 数量大，不能让低分辨率像素 loss 主导 Wan。
5. 窗口数不是数据价值。RoboCasa MG 占绝大多数 episode 与小时，但 composite 的任务类型更丰富；OXE 的 embodiment 与真实场景多样性更高。

## 4. 最终数据划分

### 4.1 物理 train/val/test

划分单位依次为：

    source
      → parent trajectory（若存在）
      → episode
      → windows

同一个 parent trajectory 的所有 episode 必须在同一 split；同一个 episode 的所有 window 必须在同一 split。split 列表物化成可读的 episode ID 文件，训练时只读列表，不按 window 临时决定。

每个 source 使用以下规则：

| eligible episode 数 N | val | test | train |
|---:|---:|---:|---:|
| N >= 1000 | round(1% × N)，至少 10 | 与 val 相同 | 其余 |
| 100 <= N < 1000 | 10 | 10 | 其余 |
| N < 100 | 5 | 5 | 其余 |

在尚未额外 carve out 严格 task-OOD 的情况下，目标总量为：

| 家族 | train | val | test | 合计 |
|---|---:|---:|---:|---:|
| OXE | 128,995 | 1,374 | 1,374 | 131,743 |
| RoboCasa | 407,327 | 4,156 | 4,156 | 415,639 |
| 合计 | 536,322 | 5,530 | 5,530 | 547,382 |

具体 ID 的选择流程固定为：先按 canonical episode ID 排序，再用固定种子的 PCG64 对每个 source 独立打乱，最后按上表数量切分并写出显式列表。不能用文件名自然顺序直接取尾部，因为很多数据集按采集日期或任务成块排列。

### 4.2 task-OOD

严格 task-OOD 只对有可信离散 task ID 的 source 建立，不对 DROID 这类自由文本直接做伪精确分组。

对 task 数不少于 50 的可信 source：

1. 先按 task ID 聚合全部 parent trajectories；
2. 留出 min(50, max(5, round(2% × task_count))) 个 task；
3. 这些 task 的全部 episode 进入 task_ood_test；
4. 再对剩余 episode 执行 4.1 的 IID 划分；
5. task_ood_test 单独报告，不与 IID test 混成一个分数。

task-OOD 列表必须在第一次正式训练前冻结。自然语言归一化得到的近似分组只能标为 unseen-text probe，不能写成严格未见任务。

### 4.3 训练采样不是数据划分

物理 split 固定后，每种 interaction program 使用不同家族比例：

| objective / program | OXE | RoboCasa | 原因 |
|---|---:|---:|---|
| geometry_pretrain | 70% | 30% | 优先真实场景、embodiment 与任务多样性 |
| action_only | 70% | 30% | 策略训练继续强调跨 embodiment 泛化 |
| forward_world | 40% | 60% | 利用 RoboCasa 的长时数、统一分辨率和多视角 |
| joint_world_action | 60% | 40% | 在动作泛化与视频动力学间折中 |

RoboCasa 内部固定 atomic/composite/MG = 10/60/30。MG 虽然物理量最大，但不允许按 episode 数自然占满 batch。OXE 内部从现有 profile 权重起步，任何单一 source 在 OXE batch 中封顶 20%，其余权重重新归一化。

抽样层级固定为：

    interaction program
      → data family
      → source
      → episode
      → valid window
      → target view

先选 episode，再选 window，避免单条长 episode 因窗口多而被无限放大。

### 4.4 画质权重

| tier | 原始短边 | Wan video loss | VGGT / Action |
|---|---:|---:|---|
| A | >= 224 | 1.0 | 正常使用 |
| B | 160–223 | 0.5 | 正常使用，几何指标分层报告 |
| C | < 160 | 0.0 | 可做 Action 与低权重几何训练，不监督正式 Wan 视频 |

## 5. 时间与窗口合同

### 5.1 三条时钟

系统不能用一个 fps 同时解释图像、控制器和 Wan latent：

| 时钟 | 含义 | v1 处理 |
|---|---|---|
| observation clock | 原视频 PTS | 保留真实时间 |
| action clock | 控制器事件时间 | 保留 source-native 频率 |
| renderer clock | 视频监督查询 | 统一 5 Hz |

### 5.2 固定物理范围

| 路径 | 历史 | 未来 | 数量 |
|---|---:|---:|---:|
| proprio / action history | 3.2 s | 无 | 16 个 5 Hz 状态位置，加全部原生 action events |
| VGGT observed keyframes | 3.2 s | 无 | 从 16 个观测位置取 index 0/5/10/15 |
| VGGT future anchors | 无 | 1.6 s | 0.4/0.8/1.2/1.6 s，共 4 个 |
| Wan video | t0 reference | 1.6 s | 9 帧，5 Hz |
| policy action | 历史真实动作 | 1.6 s | 8/16/24/32 个原生事件 |

VGGT keyframe 与 future anchor 都保存实际选中帧的 PTS。index 只表示从已经验证单调的观测列表中选哪一项，不代替真实时间编码。

### 5.3 视频取帧

| source Hz | stride | 选择结果 |
|---:|---:|---:|
| 5 | 1 | 9 个真实帧 |
| 10 | 2 | 9 个真实帧 |
| 15 | 3 | 9 个真实帧 |
| 20 | 4 | 9 个真实帧 |

规则：

1. anchor 是原始观测，记为 t0。
2. 目标时刻为 t0 + j/5，j 为 0…8。
3. nominal rate 只给出首选 stride，最终用 PTS 验证单调性、唯一性和时间误差。
4. 选中帧与目标时刻的误差不超过 50 ms，末帧覆盖必须到达 1.6 s 的 90%。
5. 不满足时整条 window 对 video 标为 invalid，但仍可用于合法的 Action 或 proprio 训练。
6. 不做 RGB 插帧，不重复末帧，不伪造 action event。

### 5.4 为什么 v1 统一 5 Hz

- 四种原始频率都能整数降采样，不产生 15→10 Hz 的抖动选择。
- 9 帧满足 Wan VAE 的 4n+1 时间长度要求。
- 1.6 秒内仍覆盖完整动作结果，而 Action 保留原生高频事件。
- FastWAM 公开配置使用 32 action events 配 9 video frames，说明控制与视频时钟无需一一对应。
- 在线 VGGT 与在线 Wan VAE 会明显增加单步计算，单桶先把变量压到最少。

10 Hz / 17 帧不被永久删除，而是 high-motion/contact ablation。只有它在 10/20 Hz source 的接触时序、gripper transition 和 temporal metric 上稳定优于 5 Hz，且收益足以覆盖训练吞吐下降，才进入后续版本。

## 6. 数据合同

### 6.1 episode manifest

episode manifest 只保存原始事实与访问索引：

~~~yaml
schema: wm3d_wam_episode_v2
source: robocasa_mg
episode_id: episode_012345
parent_trajectory_id: trajectory_00421
split: train
task_id: robocasa_task_00660
task_text: pick up the mug and place it in the cabinet
embodiment: robocasa_panda_omron

views:
  primary_external:
    path: ...
    pts_path: ...
  secondary_external:
    path: ...
    pts_path: ...
  wrist:
    path: ...
    pts_path: ...

robot:
  table_path: ...
  action_clock_hz: 20
  action_contract: robocasa_panda_omron_v1
  normalization_group: robocasa_panda_omron_action_v1
~~~

### 6.2 window plan

window plan 保存轻量索引，不保存模型派生特征：

~~~yaml
schema: wm3d_wam_window_v2
window_id: robocasa_mg/episode_012345/window_0007
source: robocasa_mg
episode_id: episode_012345
anchor_row: 126

observation:
  context_rows: [...]
  context_times_s: [...]
  vggt_keyframe_positions: [0, 5, 10, 15]
  available_view_roles: [primary_external, secondary_external, wrist]

future:
  video_frame_rows: [126, 130, 134, 138, 142, 146, 150, 154, 158]
  video_frame_times_s: [...]
  vggt_target_frame_positions: [2, 4, 6, 8]
  action_row_start: 126
  action_row_stop: 158
  action_times_s: [...]

render:
  video_hz: 5
  video_frames: 9
  spatial_bucket: square_256
  quality_tier: A
~~~

### 6.3 Grouped Robot ABI

每个 source 必须明确：

- action 与 state 的列、维度和 mask；
- 每一维的 semantic ID；
- absolute、delta、velocity、binary 或 gripper 类型；
- 单位、坐标系、旋转表示、组合算子和 gripper 极性；
- observation/action 的 leading 或 trailing 对齐；
- per-source 或 per-embodiment 归一化统计。

模型输入保留：

- fine_action_values[L,G,D]
- fine_action_mask[L,G,D]
- fine_action_times_s[L]
- fine_action_dt_s[L]
- semantic_ids[G,D]
- group_ids[G]
- embodiment_id

连续量只在模型输入副本上做稳健归一化；原值永久保留。binary/gripper 使用独立类型和 loss。语义未确认的 source 可做同源预测，但不能进入跨 embodiment 主指标。

### 6.4 多视角

视角槽位固定为：

1. primary_external
2. secondary_external
3. wrist

VGGT 同时读取最多三路可用视角和四个历史 keyframe，并携带 view-role、时间与缺失 mask。Wan 每个样本渲染一个目标视角：

- primary_external：50%
- secondary_external：25%
- wrist：25%

缺失角色只重新归一化概率，不复制其他画面。空间桶保留 256×256、192×256、160×288；同一窗口的全部时间和视角使用一致的确定性 resize/crop。

## 7. 在线数据路径与缓存原则

### 7.1 正式训练路径

~~~mermaid
flowchart LR
  M["episode manifest + window plan"] --> D["按 PTS 在线解码 RGB"]
  D --> V["在线 VGGT shallow/deep"]
  D --> E["在线 Wan VAE"]
  R["Parquet / robot table"] --> A["Grouped action pack"]
  V --> S["WM3D-WAM forward"]
  E --> S
  A --> S
~~~

训练的必要前置产物只有：

- episode split 列表；
- window 行号、PTS、crop、view mask；
- action 行范围和合同；
- 文本及 task ID 索引。

不生成或要求以下磁盘输入：

- VGGT shallow/deep tokens；
- VGGT depth、point、camera pose；
- 每个 window 的 Wan latent；
- 600K window sidecar；
- 预先固定的 future feature target。

future shallow target 在同一次训练 forward 中由冻结 VGGT shallow block pairs 从未来真实 RGB 在线计算，并立即用于 loss；不落盘。

### 7.2 允许的短生命周期复用

- DataLoader worker 的解码预取队列；
- 单个 optimizer step 内重复视角的 tensor 复用；
- 单次推理调用内的 Wan per-layer video K/V；
- 进程内有界 task text embedding LRU。

这些复用不改变数据合同，进程退出即可丢弃。

### 7.3 可选 Wan VAE 优化

默认先跑在线 VAE。只有真实七卡 profile 证明 VAE 占据主要 wall time，才允许实验有界、可删除的 Wan VAE shard cache。它必须满足：

- 仅缓存 Wan latent，不缓存 VGGT 派生量；
- 在线路径始终可独立训练；
- cache miss 回到同一在线实现，不换样本；
- crop、PTS 和 VAE 权重版本显式记录；
- 容量和吞吐收益经 benchmark 后再决定。

因此 cache 是性能选项，不是架构依赖，更不是开始训练前必须完成的数据工程。

## 8. 模型架构

### 8.1 总体结构

~~~mermaid
flowchart TD
  RGB["历史多视角 RGB"] --> VS["VGGT shallow pairs 0–3<br/>冻结、在线"]
  FUT["未来真实 RGB<br/>仅 target branch"] --> VT["Frozen VGGT target<br/>在线、no-grad"]
  VT --> TGT["shallow + geometry targets<br/>只进入 loss"]
  VS --> GFP["Continuous Geometry<br/>Future Predictor"]
  TXT["任务文本"] --> GFP
  PRO["proprio + past actions"] --> GFP

  GFP --> FV["未来 shallow token anchors"]
  GFP --> GS["geometry action seed"]
  FV --> VD["VGGT deep pairs 4–23<br/>低学习率、可训练"]
  GS --> VD
  VD --> GEO["deep geometry tokens<br/>depth / camera / point"]
  VD --> GACT["geometry-refined action seed"]

  FIRST["目标视角首帧"] --> WV["Wan2.2 Video Expert"]
  GEO --> FUSE["稀疏 geometry K/V adapters"]
  FUSE --> WV
  FUSE --> AF["Grouped Action Flow Expert"]
  GACT --> AF
  TXT --> WV
  TXT --> AF

  WV <-->|"每层 mixed attention"| AF
  WV --> VIDEO["未来视频"]
  AF --> ACTION["主 Action 输出"]
  VD --> AUX["VGGT direct action<br/>仅辅助"]
~~~

### 8.2 VGGT Geometry Core

Geometry Core 包含一个 student 计算路径和一个冻结 target 路径。student 是推理时保留的核心模型；target 路径只在需要几何监督的训练 program 中在线运行，不保存派生数据，也不进入任何条件 attention。

输入：

- 四个历史 keyframe；
- 每个时刻最多三路视角；
- view role、实际时间和有效 mask；
- 任务、proprio 与历史动作条件。

执行：

1. VGGT frame/global block pair 0–3 在线编码历史图像，参数冻结且不保存反向激活。
2. Continuous Geometry Future Predictor 读取历史 shallow tokens、任务、proprio 和过去动作。
3. predictor 输出四个未来 shallow token anchors，对应 0.4/0.8/1.2/1.6 秒；同时输出 geometry action seed。
4. 观测 token、预测未来 token 与 action seed 从 block 4 进入 VGGT deep sequence。
5. block pair 4–23 以低学习率训练，使用严格因果 mask。
6. VGGT heads 直接输出 camera、depth 和 point；由 depth/camera 反投影的点云只用于一致性检查，不替代 point head。
7. geometry_pretrain、forward_world 与 joint_world_action 需要几何 target 时，冻结 VGGT target 路径对未来真实 RGB 做 no-grad forward。没有对应 loss 的 program 不运行这段计算。

训练 target：

- 同一窗口的未来真实 RGB 经过冻结 target block pairs 0–3，在线得到 future shallow target；
- feature loss 比较 student 预测与 detached target；
- depth/camera/point 优先使用数据集真值；缺失时使用冻结 target 路径的在线输出；
- target 路径不保存反向激活，只计算当前 program 权重非零的输出；
- clean future RGB、target token 和 target geometry 始终只在 loss 一侧，不能输入 policy、Wan 或 student VGGT deep。

### 8.3 Continuous Geometry Future Predictor

默认配置：

| 项目 | 值 |
|---|---:|
| hidden | 1024 |
| transformer blocks | 12 |
| observed geometry keyframes | 4 |
| future geometry anchors | 4 |
| future horizon | 1.6 s |
| deep causal | true |

它提供两种显式模式：

- policy mode：不读取 future factual action，预测 observation-conditioned future geometry prior 与 action seed；
- factual mode：读取候选或真实 future actions，预测 action-conditioned geometry，用于 forward world。

两种模式共享参数但使用不同 mode embedding 和 attention mask。训练代码不能用 detach 临时修补泄漏，mask contract 必须在模块 API 中固定。

### 8.4 Grouped Action Flow Expert

Action 序列长度 L 随 source-native 控制频率变化：

| 原生 Hz | L |
|---:|---:|
| 5 | 8 |
| 10 | 16 |
| 15 | 24 |
| 20 | 32 |

batch 内 pad 到 32，并使用 event、group、dimension 三层 mask。每个 scalar action 先结合 semantic ID、group ID、embodiment、真实时间和 flow timestep 编码，再汇聚成 event token。输出端用 semantic dimension queries 解码回 grouped tensor，不假设所有机器人的第 d 维有同一含义。

主 Action 模型：

| 项目 | 值 |
|---|---:|
| blocks | 30 |
| hidden | 1024 |
| FFN | 4096 |
| attention | 24 heads × 128 |
| max events | 32 |
| objective | continuous flow matching |

连续 action 使用 flow matching。binary/gripper 先映射为 -1/+1 并参与同一条 action flow，以保持整段动作的联合建模；在预测的 clean endpoint 上再增加 typed BCE，评测时不把它计入连续误差。首版正确性评测使用 20 个动作去噪步，只有消融确认后才减少步数或做蒸馏。

### 8.5 Wan2.2 与 Action 的逐层耦合

Wan Video Expert 保留 30 层、hidden 3072、FFN 14336。Action Expert 每一层与对应 Wan 层执行：

1. 两个分支分别做 normalization 和 Q/K/V projection；
2. Q 保持各自 query，K/V 按 interaction mask 组成共享上下文；
3. attention 输出经各自 output projection 返回不同 hidden；
4. 两个分支执行各自的 FFN；
5. geometry adapters 在指定层提供额外 K/V。

默认 geometry 融合层为 0-based [5, 11, 17, 23, 29]。先用五个稀疏层控制成本；若 geometry utilization 门禁失败，再比较全层融合，不在 v1 直接复制 30 套大 adapter。

这条路径的准确表述是“Action 参与 Wan 的逐层世界表征与 mixed attention”，不是“Wan 视频 decoder 直接输出 Action”。

### 8.6 这不是 routed MoE

v1 没有 token router、top-k expert 或按 source 选择 FFN。Wan 与 Action 是两个固定 stream，所有有效 token 都按明确 mask 参与交互。这里采用的是 Mixture-of-Transformers 式并行专家，不是 Worldscape 的 control-routed MoE。

只有新增 camera trajectory、dense action map 等结构完全不同的控制接口，或观察到稳定的 source 梯度冲突，才重新评估 routed MoE。

## 9. Interaction Programs 与因果边界

### 9.1 三种部署相关训练程序

Stage A 的 geometry_pretrain 是几何主干预训练 objective，不是部署时的交互模式。它用历史观测预测未来几何与辅助动作，未来真实 RGB 只由冻结 VGGT target 路径读取并产生 loss target。Stage B/C 使用以下三种交互程序：

| program | 输入 | 预测 | policy loss |
|---|---|---|---|
| action_only | 历史 RGB、任务、proprio、past actions、noisy future action | future action | 开 |
| forward_world | 历史 RGB、任务、proprio、clean candidate action、noisy future video | future geometry/video | 关 |
| joint_world_action | 历史 RGB、任务、proprio、noisy action、noisy future video | action + geometry + video | 开 |

### 9.2 可见性

| Query | 可见 observed video | 可见 clean future video | 可见 noisy future video | 可见 clean future action | 可见 predicted geometry |
|---|---:|---:|---:|---:|---:|
| action_only Action | 是 | 否 | 否 | 否 | 是，policy mode |
| forward_world Video | 是 | 否 | 是 | 是 | 是，factual mode |
| joint Action | 是 | 否 | 是 | 否 | 是，policy mode |
| joint Video | 是 | 否 | 是 | 否，只见 noisy action | 是 |

任何产生 policy loss 的 Action query 都不能读取：

- future factual action；
- clean future RGB；
- clean future Wan latent；
- 由 clean future target 计算、未经过预测器的 VGGT token。

future target 只出现在 loss 一侧。这样既允许视频与动作共享表征，也不会用教师答案泄漏动作。

### 9.3 推理程序

默认 fast policy：

1. 在线解码四个历史 keyframe；
2. 在线运行 VGGT shallow、future predictor 与 deep；
3. Wan 对目标视角首帧做一次 observed-video prefill；
4. 每层 video K/V 只在本次调用内复用；
5. Action Flow Expert 去噪并输出 1.6 秒原生频率动作；
6. 不生成未来视频。

forward world：

- 输入一个候选动作；
- 运行 factual geometry mode；
- 生成未来几何和 9 帧视频；
- 用于候选动作评分、rollout 和可视化。

future-aware slow policy：

- 只在 OOD、低置信度或候选重排时启用；
- 预测一次未来 representation，并在五个 geometry adapter 层稀疏融合；
- 是否进入正式部署由延迟与策略收益共同决定。

## 10. 训练方案

### 10.1 Phase 0：数据与资产

必须完成：

1. 物化 episode、parent trajectory 与 task-OOD split 列表。
2. 完成 21 个 source 的 action semantic、单位、坐标系、组合算子与 gripper 极性合同。
3. 完成三路 view role 审计。
4. 建立 5 Hz / 9 帧 window plan 与 PTS invalid reason 报告。
5. 把 VGGT-1B 与 Wan2.2 TI2V-5B 放入服务器只读模型目录。
6. 用正式在线路径跑单卡与七卡 memory / throughput profile。

42 上已经存在 facebook/VGGT-1B 资产，约 4.7 GB。Wan2.2 权重是否齐备必须在实现开始时再次检查；训练脚本不允许运行中临时下载。

### 10.2 Stage A：Geometry-GAM

| 项目 | 值 |
|---|---|
| steps | 30K |
| program | geometry_pretrain |
| family mix | OXE:RoboCasa = 70:30 |
| 训练 | future predictor、VGGT deep pairs 4–23、VGGT auxiliary action heads |
| 冻结 | VGGT shallow pairs 0–3 |
| 目标 | future shallow feature、depth/camera/point、direct/refined auxiliary action |

Stage A 先确认在线 VGGT 路径、因果 future prediction 和 grouped action 合同，不加载 Wan 视频生成。

### 10.3 Stage B：Wan-Action

| 项目 | 值 |
|---|---|
| steps | 40K |
| program mix | action_only 50%、forward_world 30%、joint_world_action 20% |
| 前 2K | 只训练 Action Expert、future predictor、geometry/Wan adapters 与新投影 |
| 后 38K | 加入完整 Wan DiT，使用低学习率 |
| 冻结 | Wan VAE、text encoder、VGGT shallow |
| 目标 | action flow、video flow、future geometry |

FastWAM 的正式训练会更新 MoT 主干。永久冻结 Wan 会限制视频先验向 Action 适配，因此 Revision 2 只做短 warmup 冻结，之后以低学习率训练完整 Wan DiT。

### 10.4 Stage C：Tri-stream Alignment

| 项目 | 值 |
|---|---|
| steps | 20K |
| program mix | action_only 40%、forward_world 30%、joint_world_action 30% |
| 训练 | Action Expert、Wan DiT、VGGT deep、future predictor、全部 adapters |
| 冻结 | Wan VAE、text encoder、VGGT shallow |
| 目标 | 动作、几何、视频统一对齐 |

### 10.5 学习率

| 参数组 | Stage A | Stage B warmup | Stage B main | Stage C |
|---|---:|---:|---:|---:|
| Action Flow Expert | 无 | 2e-5 | 2e-5 | 1e-5 |
| geometry K/V / Wan adapters | 无 | 1e-5 | 1e-5 | 1e-5 |
| geometry future predictor | 1e-5 | 1e-5 | 1e-5 | 5e-6 |
| VGGT deep pairs 4–23 | 1e-5 | 0 | 5e-6 | 5e-6 |
| Wan DiT | 无 | 0 | 2e-6 | 1e-6 |
| VGGT shallow / Wan VAE / text encoder | 0 | 0 | 0 | 0 |

统一使用 AdamW、betas (0.9, 0.95)、BF16、global grad norm 1.0。新模块 warmup 500 steps，之后 cosine decay。各 loss 先按有效 token 数归一化，再乘权重，避免 32-event source 天然拥有四倍 Action 权重。

### 10.6 loss

| program | loss | 权重 |
|---|---|---:|
| geometry_pretrain | future shallow feature | 1.0 |
| geometry_pretrain | depth/camera/point | 0.3 |
| geometry_pretrain | VGGT direct/refined auxiliary action | 0.1 |
| action_only | action flow | 1.0 |
| action_only | VGGT direct/refined auxiliary action | 0.1 |
| forward_world | Wan video flow | 1.0 |
| forward_world | future shallow feature | 1.0 |
| forward_world | depth/camera/point | 0.3 |
| joint_world_action | action flow | 1.0 |
| joint_world_action | Wan video flow | 1.0 |
| joint_world_action | future shallow feature | 0.5 |

视频与 Action 使用各自 scheduler 的 flow matching。binary/gripper 的 -1/+1 flow target 另加 clean-endpoint BCE。没有反事实真值时不添加鼓励任意运动差异的伪 counterfactual loss。

### 10.7 七卡运行

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
    nproc_per_node=7
    micro_batch_per_gpu=1
    gradient_accumulation_steps=4
    effective_global_batch=28
    precision=bf16
    fsdp=full_shard
    activation_checkpointing=true

GPU 0 在 launcher 与 preflight 中都列为 forbidden device。正式训练使用同一模型路径完成短 canary，不维护简化模型：

1. 单卡 20 steps，检查 shape、mask 与 gradient ownership；
2. 七卡 200 steps，记录 decode、VGGT、VAE、DiT、通信占比；
3. 七卡 300 steps，检查 loss、吞吐、显存和恢复；
4. 通过后进入正式 steps。

## 11. 评测与关键消融

### 11.1 数据切片

固定报告：

- OXE / RoboCasa；
- 每个 source 的 macro average；
- 5/10/15/20 Hz 原生 Action；
- primary / secondary / wrist；
- Tier A / B / C；
- 静止、低运动、高运动、gripper transition、接触事件；
- IID、unseen-text probe、严格 task-OOD；
- 7D、8D、12D、15D action contract。

### 11.2 指标

| 能力 | 指标 |
|---|---|
| Action | normalized continuous error、binary/gripper F1、trajectory ADE/FDE、per-source macro |
| 几何 | future feature cosine、depth AbsRel、camera rotation/translation、point Chamfer |
| 单帧视频 | PSNR、SSIM、LPIPS、DINO/JEPA similarity |
| 时序 | FVD、temporal LPIPS、光流 EPE、contact timing error |
| 动作跟随 | true action 对 zero/time-shift/gripper-toggle 的 GT-distance margin |
| 长滚动 | 4×1.6s 的 identity、背景、geometry 和 motion drift |
| 性能 | samples/s、step time、peak memory、decode/VGGT/VAE/DiT 占比 |

### 11.3 必做消融

| 对照 | 要回答的问题 |
|---|---|
| Wan-coupled Action vs 等容量 detached Action head | Wan 联合表征是否真正提高 Action |
| online VGGT split-resume vs 不使用 VGGT deep | 深层几何是否改善动作和世界预测 |
| geometry adapters 5 层 vs 30 层 | 稀疏融合是否足够 |
| 5 Hz / 9f vs 10 Hz / 17f | 高频视频是否值得额外成本 |
| action_only vs 三程序联合 | 视频联合训练是否贡献而非仅增加参数 |
| fast policy vs future-aware slow policy | 未来表征是否改善 OOD |
| full Wan low-LR vs Wan permanently frozen | 动作适配是否需要更新视频主干 |

Wan-coupled Action 是主方案，不等待消融才实现。detached head 只是回答收益大小。若耦合版整体平均不差，但在 OOD、高运动或接触切片明显更好，仍保留耦合；若全面无收益，则先检查 mask、Action 语义和视频训练是否有效，再决定是否简化。

### 11.4 阶段门禁

Stage A → B：

- 所有 source 的 online VGGT forward 与 loss finite；
- policy mode 无法读取 future factual action 或 future target token；
- future feature、depth/camera 中至少一项稳定优于 hold-last；
- direct 与 refined auxiliary action 均有有效梯度。

Stage B → C：

- Action 优于同数据的 action-history baseline；
- Wan-coupled Action 不劣于等容量 detached head；
- true-action margin 在 OXE、RoboCasa、高运动、gripper transition 四类切片为正；
- 视频优于 first-frame hold 与未适配 Wan baseline；
- Wan 与 Action mixed-attention、geometry adapter 均有非零利用率。

Stage C → v1：

- action、geometry、video 三类主指标均不比各自 Stage B 最优 checkpoint 明显退化；
- per-source macro 与 family weighted 两套结果同时报告；
- task-OOD 和 high-motion 不出现系统性动作提前、静止或末帧复制；
- 4 段 rollout 没有持续 geometry collapse；
- 七卡编号 checkpoint 能恢复 model、optimizer、scheduler、sampler cursor 与 RNG。

## 12. 仓库结构

~~~text
WM3D-WAM/
├── README.md
├── docs/
│   ├── WM3D_WAM_V1_DESIGN.md
│   ├── DATA_CONTRACT.md
│   ├── MODEL_ARCHITECTURE.md
│   ├── TRAINING.md
│   └── EVALUATION.md
├── configs/
│   ├── data/current_oxe_robocasa.yaml
│   ├── temporal/video_5hz_9f.yaml
│   ├── model/vggt_geometry_core.yaml
│   ├── model/wan_action_mot.yaml
│   └── train/
│       ├── stage_a_geometry.yaml
│       ├── stage_b_wan_action.yaml
│       └── stage_c_alignment.yaml
├── wm3d_wam/
│   ├── data/
│   │   ├── episode_manifest.py
│   │   ├── window_plan.py
│   │   ├── online_video.py
│   │   ├── grouped_robot.py
│   │   ├── source_adapters.py
│   │   └── hierarchical_sampler.py
│   ├── models/
│   │   ├── vggt_geometry_core.py
│   │   ├── geometry_future_predictor.py
│   │   ├── grouped_action_flow.py
│   │   ├── wan_video_expert.py
│   │   ├── mot_attention.py
│   │   ├── geometry_kv_adapter.py
│   │   ├── interaction_masks.py
│   │   └── system.py
│   ├── training/
│   │   ├── objectives.py
│   │   ├── distributed.py
│   │   └── checkpoint.py
│   └── evaluation/
│       ├── action_metrics.py
│       ├── geometry_metrics.py
│       ├── video_metrics.py
│       └── counterfactual.py
├── scripts/
│   ├── audit_data.py
│   ├── build_splits.py
│   ├── build_window_plan.py
│   ├── train.py
│   └── evaluate.py
└── third_party/NOTICE.md
~~~

文件只在对应功能进入当前里程碑时创建，不生成空壳实现。

## 13. 迁入、参考与重写

### 13.1 从原 WM3D 迁入

- grouped robot schema、mask 与 tensor packing；
- 21 个 source adapter 和 inventory；
- PTS-aware window selection；
- 多视角 view-role 基础；
- FSDP runtime、activation checkpointing、编号 checkpoint；
- 原值保留、归一化与离线评测基础。

### 13.2 从 node 41 原型迁入或重写

- VGGT shallow/deep split；
- shallow target 的在线计算；
- future predictor；
- action seed 插入 deep VGGT sequence；
- direct/refined auxiliary action loss。

迁入前清理原型中的数据集特例和缓存假设，保留核心计算图。

### 13.3 参考 FastWAM / Wan2.2 重写

- Wan Video Expert 封装；
- ActionDiT 形态的 Grouped Action Flow Expert；
- 每层 mixed attention；
- observed-frame prefill 与单次调用 K/V 复用；
- video/action 双 flow scheduler。

Grouped Robot ABI、连续时间和 geometry adapter 是本项目新增，不能直接照搬固定 action_dim 的实现。

### 13.4 不迁入

- 旧 episode VGGT feature cache 作为模型输入的路径；
- 600K Wan sidecar 前置流程；
- 独立于 Wan 的发布级 policy head；
- 5/10 Hz 双视频桶；
- Worldscape control router 与多套大 FFN expert；
- 历史实验配置和数据集 retry 特例。

若复制开源代码片段，必须保留相应许可证并在 third_party/NOTICE.md 标出来源。

## 14. 实施里程碑

### M0：Revision 2 冻结

- 三项评审结论确认；
- 确认统一 5 Hz、Wan-coupled Action、online VGGT；
- 确认新仓库为唯一实现入口。

### M1：数据合同

- 物化 episode/parent/task split；
- 完成 21 个 source 的 action 与 view audit；
- 生成 5 Hz window plan 和 invalid reason 报告；
- 以少量真实 window 跑在线 decode，不生成派生 cache。

### M2：Online VGGT

- 迁入 split-and-resume；
- 实现 policy/factual 两种 geometry mode；
- 完成 Stage A canary 与正式训练；
- 验证 geometry-refined auxiliary action。

### M3：Wan-Action MoT

- 接入 Wan2.2；
- 实现 Grouped Action Flow Expert 与 mixed attention；
- 实现 interaction masks 与 geometry K/V adapter；
- 完成 Stage B。

### M4：联合与评测

- 完成三种 interaction program；
- 跑 Stage C；
- 完成 Action coupling、5/10 Hz、future-aware 等关键消融；
- 冻结 v1 inference program。

## 15. 主要风险

| 风险 | 影响 | 处理 |
|---|---|---|
| Action 语义或单位错误 | 跨 source 学到互相矛盾的控制 | Phase 0 合同先行；未确认 source 不进跨 embodiment 主指标 |
| mixed attention 发生目标泄漏 | Action 指标虚高，部署失效 | 模块级 interaction mask；target 只在 loss 侧 |
| 在线 VGGT + VAE 吞吐过低 | 正式训练成本过高 | 单桶 5 Hz、no-grad shallow、prefetch；profile 后只优化真实瓶颈 |
| Wan 忽略 Action | 视频好看但不受控 | action counterfactual margin、joint program、低学习率更新 Wan |
| Action 忽略 Wan | 耦合只有名义没有效果 | detached ablation、attention utilization 与 gradient audit |
| RoboCasa MG 主导 | 过拟合单一模拟分布 | family/source/episode 分层，RC 内 10/60/30 |
| 低分辨率大源拖低视频 | 模糊和伪影 | Tier B 半权重，Tier C 不做 Wan loss |
| VGGT 深层破坏预训练几何 | depth/camera 退化 | shallow 冻结、deep 低学习率、Stage A 门禁 |
| 七卡显存不足 | OOM 或吞吐极低 | FSDP full shard、checkpointing、micro batch 1、稀疏 geometry adapters |
| Wan 权重缺失 | Stage B 无法开始 | M1 前放入服务器只读资产目录并做 shape 检查 |

## 16. 默认配置摘要

~~~yaml
project: wm3d_wam_v1_r2

time:
  context_horizon_s: 3.2
  future_horizon_s: 1.6
  renderer_hz: 5
  video_frames: 9
  vggt_observed_keyframes: 4
  vggt_future_anchors_s: [0.4, 0.8, 1.2, 1.6]
  action_native_hz: [5, 10, 15, 20]
  action_events: [8, 16, 24, 32]
  interpolation: forbidden

geometry:
  backbone: VGGT-1B
  frozen_target_branch: VGGT-1B
  target_branch_online: true
  shallow_frozen_pairs: [0, 1, 2, 3]
  deep_trainable_pair_range: [4, 23]
  predictor_blocks: 12
  hidden: 1024
  online: true
  persistent_feature_cache: false

video_action:
  video_expert: Wan2.2-TI2V-5B
  action_expert_blocks: 30
  action_hidden: 1024
  action_ffn: 4096
  action_max_events: 32
  mot_mixed_attention: true
  geometry_adapter_layers: [5, 11, 17, 23, 29]
  freeze_vae: true
  freeze_text_encoder: true
  full_wan_low_lr_after_warmup: true

sampling:
  geometry_pretrain: {oxe: 0.70, robocasa: 0.30}
  action_only: {oxe: 0.70, robocasa: 0.30}
  forward_world: {oxe: 0.40, robocasa: 0.60}
  joint_world_action: {oxe: 0.60, robocasa: 0.40}
  robocasa: {atomic: 0.10, composite: 0.60, mg: 0.30}
  oxe_source_cap: 0.20

runtime:
  visible_gpus: [1, 2, 3, 4, 5, 6, 7]
  forbidden_gpus: [0]
  precision: bf16
  fsdp: full_shard
  micro_batch_per_gpu: 1
  gradient_accumulation_steps: 4
  effective_global_batch: 28

training:
  stage_a_steps: 30000
  stage_b_steps: 40000
  stage_c_steps: 20000
  persistent_vggt_cache: false
  persistent_wan_cache: false
~~~

## 17. 参考

- [VGGT 官方论文](https://arxiv.org/abs/2503.11651)
- [VGGT 官方代码](https://github.com/facebookresearch/vggt)
- [Geometric Action Model 论文](https://arxiv.org/abs/2606.17046)
- [Geometric Action Model 官方代码](https://github.com/cvlab-kaist/Geometric-Action-Model)
- [FastWAM 论文](https://arxiv.org/abs/2603.16666)
- [FastWAM 官方代码](https://github.com/yuantianyuan01/FastWAM)
- [Wan2.2 官方代码](https://github.com/Wan-Video/Wan2.2)
- [Worldscape-MoE 论文](https://arxiv.org/abs/2607.03964)
- [Worldscape-MoE 官方代码](https://github.com/EmbodiedCity/Worldscape-MoE.code)
- [OpenWAM 项目页](https://openwam.stanford.edu/)
- [Tapestry-WAM 项目页](https://tapestry-wam.github.io/)
- [Faster-WAM 论文](https://arxiv.org/abs/2608.04404)

## 18. 对本轮三个问题的直接回答

### 18.1 数据划分合理吗

旧方案的数据总量与分层抽样思路合理，但 dual-rate 视频桶和把上游 train 当最终 train 的表述不够严谨。Revision 2 已改为 episode/parent 层的显式 train/val/test、objective-specific family sampling，以及统一 5 Hz / 9 帧视频。Action 仍保留原生频率。这是当前数据与七卡预算下更稳的 v1。

### 18.2 Action 要不要走 Wan2.2

要。不是让视频 decoder 输出动作，而是让 Action Flow Expert 与 Wan Video Expert 在 30 层中逐层 mixed attention。FastWAM 的公开消融支持视频联合训练对动作能力有实际贡献。完全独立的 Action head 仍可做消融，但不再作为主方案。

### 18.3 VGGT 要不要依赖 cache

不要。VGGT 改为训练 forward 内的在线核心主干，采用 node 41 原型和 GAM 的 shallow split、future prediction、deep resume。磁盘只保存原始数据索引和窗口计划，不保存 VGGT 派生 feature。这样几何能力在模型里面接受监督、参与 Action 与视频，而不是先做完一套缓存再把结果喂给另一个模型。

## 19. 一句话设计

WM3D-WAM v1 用在线 VGGT 建立可训练的未来几何，用 Wan2.2 与 Grouped Action Flow Expert 的逐层 mixed attention 产生动作和视频，并以统一 5 Hz 视频、原生高频 Action、三种因果 interaction program 训练同一个世界模型。
