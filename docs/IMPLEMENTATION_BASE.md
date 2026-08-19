# WM3D-WAM 实现基线与当前状态

| 项目 | 内容 |
|---|---|
| 日期 | 2026-08-19 |
| 分支 | `codex/implement-wm3d-wam-v1` |
| 当前里程碑 | M2/M3 单卡生产宽度 preflight 完成 |
| 生产模型配置 | `configs/model/wan_action_mot_v1.yaml` |
| 数据配置 | `configs/data/grouped_robot_v1.yaml` |

## 代码基线

WM3D-WAM 作为集成入口，按模块复用三套已有实现。没有整体 fork 任意旧
项目，因为三者都不同时具备 grouped robot ABI、Wan2.2 动作耦合和在线
VGGT。

| 范围 | 基线 | 本项目保留或修改的部分 |
|---|---|---|
| 视频与动作 | FastWAM | Wan2.2 VideoDiT、ActionDiT、30 层 MoT、flow scheduler、observed-video K/V prefill |
| 数据与时钟 | 原 WM3D | recorded timestamp、grouped robot ABI、source adapter、manifest |
| 几何 | node 41 VGGT-GAM | VGGT pair 0–3 / 4–23 split、future predictor、action token 插入 deep path |

Worldscape-MoE 用于参考多源数据组织，不作为代码主干。它缺少本项目采用
的 Wan2.2 ActionDiT、grouped ABI 和 VGGT split-and-resume 组合。

## 已实现的数据路径

`scripts/build_episode_splits.py` 已对 21 个 source 物化固定 split。选择单位
优先使用 manifest 中的 parent trajectory；当前 manifest 没有该字段，所以
实际单位是 episode。所有 window 都继承 episode split。

| family | train | val | test | eligible |
|---|---:|---:|---:|---:|
| OXE | 128,995 | 1,374 | 1,374 | 131,743 |
| RoboCasa | 407,327 | 4,156 | 4,156 | 415,639 |
| 合计 | 536,322 | 5,530 | 5,530 | 547,382 |

在线 loader 只读取 manifest、Parquet 和 MP4：

- 按 recorded timestamp 选择 16 个历史 state、4 个 VGGT 历史 keyframe、
  4 个未来 geometry anchor 和 9 个 Wan frame；
- endpoint 允许设计规定的 90% coverage，所有选中图像仍是实际记录帧；
- past state 与 past action 都以 policy anchor 为时间原点；
- future action 的半开区间固定为 `[0, 1.6s)`；
- 5/10/15/20 Hz 分别保留 8/16/24/32 个未来 event，不插值、不重复末值；
- 不读取任何 VGGT、depth、point、pose 或 Wan latent 派生缓存。

固定 ordinal stride 曾漏掉 320 个满足时间覆盖门禁的短 episode。当前
timestamp selector 已能读取这类 episode；真实 DROID 边界样本保留 48 个
历史 event、24 个未来 event，最后一帧时间为 1.5333 秒。

## 已实现的模型路径

### Online VGGT-GAM

- VGGT pair 0–3 在 forward 内以 `no_grad` 编码历史与 target RGB；
- grouped history connector 使用全部 16 个 state step 和原生 past-action
  event，再提取 `[0,5,10,15]` 四个 keyframe summary；
- 12 层 future predictor 自回归预测 0.4/0.8/1.2/1.6 秒四个 shallow anchor；
- pair 4–23 接收预测 shallow token 与 action seed，输出 deep token、depth、
  world point 和 camera pose；
- direct predictor seed 与 refined deep action token 都通过 leakage-safe
  grouped auxiliary head 解码到 `[B,E,G,D]`；
- clean future RGB 只进入 loss target branch。policy API 拒绝 future factual
  action，factual API 则要求 candidate future action。

VGGT deep 当前按实际 view 数分桶。带 padding 的 view mask 会报错，避免无效
camera token 静默进入 deep attention。

### Wan2.2 与 Grouped Action

- Wan2.2 TI2V-5B Video Expert 保留 30 层、hidden 3072、FFN 14336；
- Grouped Action Expert 使用 30 层、hidden 1024、FFN 4096；
- Action codec 显式编码 value、semantic、group、composition、embodiment、
  event timestamp 与 event delta；
- 两个 expert 每层分别产生 Q/K/V，再执行同一次 mixed attention；
- geometry 在层 `[5,11,17,23,29]` 作为额外 K/V，不成为 query stream；
- `action_only` 训练与部署共用 observed-video prefill 路径，cache 数值逐元素
  一致；
- `forward_world` 使用 clean candidate action 与 noisy future video；
- `joint_world_action` 同时使用 noisy action 和 noisy future video。

ActionDiT backbone 由本地 Wan2.2 权重离线准备：820 个共享张量中 300 个直接
复制，520 个按 FastWAM 的逐维线性插值与 alpha scaling 规则得到。grouped
codec 与输出层独立初始化。

### Loss 与参数归属

- Action 与 video 使用各自的 continuous flow-matching sample、timestep
  weight 和有效 token 归一化；
- video latent frame 0 始终保持 clean，video loss 只计算未来 latent；
- Stage A 使用 future feature、depth/point/pose 和 direct/refined action；
- Stage B/C 的三个 interaction program 已接入相应 action/video/geometry loss；
- optimizer group 精确覆盖每个 stage 的 trainable 参数，Wan VAE、UMT5 与
  VGGT shallow 永久冻结。

## 本地资产

训练 worker 只接受完整的本地 bundle，不会调用 Hub 或镜像下载：

```text
/data/Minko/models/WM3D-WAM/Wan2.2-TI2V-5B
/data/Minko/models/WM3D-WAM/ActionDiT/ActionDiT_grouped_Wan22_1024.pt
/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/assets/vggt_model.safetensors
/data/Minko/world_model/wm3d_v8_action_experiments/gam_node42_v1/runtime/vggt
```

meta-device loader 已处理 Wan VAE normalization constant 与 Wan RoPE table
这两类不在 checkpoint 中的 runtime constant。相关回归测试覆盖真实
materialization，而不只检查 state-dict key。

## 当前验证

- 48 个 CPU 单元与合同测试通过；
- 真实 OXE Bridge Stage A forward/backward 通过；
- 真实 OXE Bridge action-only forward/backward、target leakage 与 cache parity
  通过；
- 真实 RoboCasa Atomic 20 Hz forward-world 通过；
- 单独反传 Action loss 时 Wan DiT 获得非零梯度；
- joint program 单独反传 video loss 时 Action Expert 获得非零梯度；
- 固定真实窗口与固定 flow noise 的 3-step AdamW loss 从 1.9619 降到
  1.8393。

完整数值见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 尚未完成的正式训练门禁

1. 多个 v4 adapter 仍使用通用 `controller_command/controller_state`，尚未逐
   source 确认单位、坐标系、composition operator 与 gripper polarity。
2. 当前 preflight 是单卡生产宽度计算图；七卡 FSDP launcher、编号 checkpoint、
   optimizer/scheduler/sampler cursor 与 RNG 恢复还未做 canary。
3. source/objective 分层 sampler、完整吞吐 profile、baseline 和正式评测尚未运行。
4. 严格 task-OOD 需要可信离散 task ID。当前只保留 unseen-text probe，不用
   自由文本归一化冒充严格 task-OOD。

这些门禁不会通过扁平 action fallback、派生 geometry cache 或缩小模型绕过。
