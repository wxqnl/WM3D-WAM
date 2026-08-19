# WM3D-WAM 实现基线与状态

| 项目 | 内容 |
|---|---|
| 日期 | 2026-08-20 |
| 分支 | `codex/implement-wm3d-wam-v1` |
| 当前里程碑 | M4，七源五卡 Stage A/B/C canary 完成，七卡 Stage A 正式训练已启动 |
| 生产模型配置 | `configs/model/wan_action_mot_v1.yaml` |
| 数据合同 | `configs/data/grouped_robot_v1.yaml`、`configs/data/source_contracts_v1.yaml` |
| 训练配置 | `configs/train/wm3d_wam_v1.yaml` |

## 代码基线选择

WM3D-WAM 是唯一集成入口，没有整体 fork 某个旧仓库。三个基线各自只负责已经
验证过的部分：

| 范围 | 基线 | 本仓库采用的实现 |
|---|---|---|
| 视频与动作 | FastWAM | Wan2.2 VideoDiT、ActionDiT、30 层 MoT、flow scheduler、observed-video prefill |
| 数据与时钟 | 原 WM3D | recorded timestamp、grouped robot ABI、adapter、manifest、episode split |
| 几何 | node 41 VGGT-GAM | VGGT pair 0–3 / 4–23 split、future predictor、deep action token |

Worldscape-MoE 用于核对多源数据组织、采样层级与 source 隔离方式。它不包含本
项目需要的 Wan2.2 ActionDiT、grouped action ABI 和 VGGT split-and-resume，
所以没有作为代码主干。

## 数据实现

`scripts/build_episode_splits.py` 为 21 个 source 物化固定 split。
`configs/data/source_contracts_v1.yaml` 再做训练门禁：7 个 verified source 进入
sampler，14 个 excluded source 保留统计但不能训练。当前有效划分为 527,647
train、5,383 val、5,383 test episode。

在线 loader 执行以下工作：

- 根据 recorded timestamp 选择 16 个历史 state、4 个 VGGT history keyframe、
  4 个未来 geometry anchor 和 9 个 Wan frame；
- 保留实际 V=1/2/3 camera bucket，不插入伪造 view；
- 通过 manifest PTS seek 和稀疏 row decode 读取 MP4，不解码完整 episode；
- past action/state 以 policy anchor 为时间原点，future action 使用半开区间
  `[0,1.6s)`；
- 保留 5/10/15/20 Hz 原生命令及真实时间戳，不插值、不复制末值；
- 用 `ceil(duration * source_hz) + 1` 分配 event 容量，容纳 recorded-time
  边界抖动；
- 不读取 VGGT、depth、point、pose 或 Wan latent 派生缓存。

分层 sampler 先选 program、family 和 source，再选 episode、window 与 view。
FSDP 要求各 rank 走相同模块路径，因此 route 和 tensor schema 按 local sample
index 同步；具体 episode/window 仍按 rank-specific global sample index 选择。

## 模型实现

### Online VGGT-GAM

- VGGT pair 0–3 在 forward 内以 `no_grad` 编码历史和 target RGB；
- grouped history connector 使用全部 16 个 state step 与原生 past-action event；
- 12 层 future predictor 预测 0.4/0.8/1.2/1.6 秒 shallow anchors；
- pair 4–23 接收预测 shallow token 与 action seed，输出 deep token、depth、
  world point 和 camera pose；
- direct 与 refined auxiliary head 解码 grouped action；
- clean future RGB 只进入 detached target branch。

VGGT 是训练计算图中的核心主干。训练不要求先生成 VGGT cache。

### Wan2.2 与 Grouped Action MoT

- Wan2.2 TI2V-5B Video Expert 保留 30 层、hidden 3072、FFN 14336；
- Grouped Action Expert 使用 30 层、hidden 1024、FFN 4096；
- action codec 编码 value、semantic、group、composition、embodiment、event
  timestamp 和 delta；
- 两个 expert 每层各自产生 Q/K/V，共享一次 mixed attention，再回到各自的
  projection 与 FFN；
- geometry 在层 5/11/17/23/29 作为额外 K/V；
- `action_only`、`forward_world`、`joint_world_action` 使用各自严格 mask；
- action-only 训练与部署复用 observed-video prefill cache 路径。

ActionDiT backbone 从本地 Wan2.2 权重生成：shape 相同的 tensor 直接迁移，
shape 不同的 tensor 使用 FastWAM 的逐维线性插值与 alpha scaling；grouped
codec 和输出层单独初始化。

## 训练与恢复实现

`scripts/train_wm3d_wam.py` 是 Stage A/B/C 的正式入口，提供：

- 多进程在线 Dataset/DataLoader 和可恢复分层 sampler；
- BF16 forward/reduce、FP32 master weights、FSDP FULL_SHARD；
- Stage-specific optimizer group、cosine schedule、gradient accumulation、
  clipping、JSONL metrics 和周期 validation；
- 数据读取失败的跨 rank readiness handshake；
- Stage A canonical DTensor DCP；
- Stage B/C rank-local model、optimizer、scheduler、RNG、cursor checkpoint；
- exact resume、Stage A→B canonical initialization、B warmup→main→C local
  initialization；
- source/view-schema 一致的真实 micro-batch collate；
- 完成标记、原子 latest 指针和显式 checkpoint retention。

完整阶段的 local shard 要求相同 world size 和相同有序物理 GPU mesh。checkpoint
元数据记录并校验该列表。Stage A canonical checkpoint 可以重分片到不同 world
size。精确恢复同时校验 micro-batch size 与 gradient accumulation。

FSDP 只在 root 与完整 `WanActionMoT` 边界切分。VGGT functional block path 和
MoT 内部 expert 不能单独 auto-wrap，否则 forward 会在参数 gather 边界之外读取
shard。冻结的 VGGT heads 保持 FP32 并排除出 FSDP；Wan VAE 与 UMT5 永久冻结。

## 本地资产

```text
/data/Minko/models/WM3D-WAM/Wan2.2-TI2V-5B
/data/Minko/models/WM3D-WAM/ActionDiT/ActionDiT_grouped_Wan22_1024.pt
/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/assets/vggt_model.safetensors
/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/runtime/vggt
```

worker 只接受完整本地 bundle，不调用 Hub 或镜像下载。loader 已处理 Wan VAE
normalization constant 与 Wan RoPE table 这两类 checkpoint 之外的 runtime
constant。

## 当前验证结论

- 60 个 CPU 合同测试通过；
- 单卡真实数据的 Stage A、三种 interaction program、cache parity、target
  leakage、双向梯度归因和三步短拟合通过；
- Stage A 双卡保存并精确恢复到下一步；
- Stage B warmup 五卡完成 forward-world、joint、action-only，且跨进程恢复；
- Stage B main 五卡解冻 Wan/VGGT deep 并保存完整 checkpoint；
- Stage C 五卡完成 train、validation 和 checkpoint；
- 五卡 canary mesh `1,2,5,6,7` 的峰值显存低于 80 GiB；正式 mesh 已扩展为
  `1,2,3,4,5,6,7`，Stage A 于 2026-08-20 启动。

详细数值见 [EXPERIMENTS.md](EXPERIMENTS.md)，启动和恢复命令见
[TRAINING.md](TRAINING.md)。代码已具备七源正式训练条件。尚未完成的是正式长训
后的策略评测，以及 14 个 excluded source 的 payload-level 控制合同审计。
