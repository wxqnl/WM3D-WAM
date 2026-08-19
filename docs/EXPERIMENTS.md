# WM3D-WAM 前期实验记录

日期：2026-08-19；机器：New-H100-2，NVIDIA H100 80GB HBM3；允许设备：
物理 GPU 1–7。以下 GPU 实验均使用物理 GPU 1。

这些实验使用生产配置、真实本地权重和真实数据窗口。没有缩小 hidden、减少
layer、生成假数据或读取派生 geometry cache。JSON 结果保存在服务器仓库的
`outputs/preflight/`。

## 1. CPU 合同测试

```text
48 passed in 6.32s
```

覆盖项包括 grouped ABI、原生 event clock、recorded-timestamp window、split、
interaction mask、ActionDiT、MoT cache、geometry K/V、grouped history、flow
matching、auxiliary action target 隔离，以及 meta-device runtime constant
materialization。

## 2. 数据 split 与时间窗口

`scripts/build_episode_splits.py` 读取 21 份 source manifest，并写出每个 source
的显式 episode ID 文件。

| family | train | val | test | eligible |
|---|---:|---:|---:|---:|
| OXE | 128,995 | 1,374 | 1,374 | 131,743 |
| RoboCasa | 407,327 | 4,156 | 4,156 | 415,639 |
| 合计 | 536,322 | 5,530 | 5,530 | 547,382 |

实际输出：`outputs/data/episode_splits_v1/summary.json`。partition 在函数内部检查
全集覆盖与互斥。当前 manifest 没有 parent trajectory 字段，因此这批 split
按 episode 执行；代码已支持 parent 字段出现时整组分配。

真实窗口抽查：

| case | views | history action | future action | future video endpoint |
|---|---:|---:|---:|---:|
| OXE Bridge 5 Hz | 2 | 16 | 8 | 1.6000 s |
| DROID 边界 episode 15 Hz | 3 | 48 | 24 | 1.5333 s |

DROID 样本只有 90% 以上 endpoint coverage。loader 仍选出 9 个真实记录帧，
没有补帧；未来 Action 继续覆盖物理 `[0,1.6s)` 并保留 24 个原生命令。

## 3. 本地模型资产

| 组件 | 结果 |
|---|---|
| Wan2.2 VideoDiT | 30 blocks，4,999,787,712 参数，BF16，本地加载成功 |
| Wan2.2 VAE | 704,688,668 参数，真实 9-frame RGB 编码成功 |
| UMT5 | 5,680,910,336 参数，输出 `[1,128,4096]` |
| VGGT-1B | split pair 4，depth/point/camera heads 可运行 |
| Grouped ActionDiT | 30 blocks，820 个 Wan backbone tensor 完成初始化 |

VAE 对真实 `[1,3,9,256,256]` RGB 的输出为
`[1,48,3,16,16]`。

## 4. Stage A：在线 Geometry-GAM

结果文件：`outputs/preflight/final_bridge_geometry_stage_a.json`

真实 OXE Bridge 窗口完成一次 forward/backward：

| 指标 | 数值 |
|---|---:|
| total loss | 1.013668 |
| future shallow feature | 1.000575 |
| depth | 0.072368 |
| world point | 0.020363 |
| camera pose | 0.027891 |
| direct/refined auxiliary action | 0.010309 |
| forward | 5.650 s |
| backward | 1.185 s |
| peak memory | 10.632 GiB |

梯度检查：

| 参数组 | 有梯度 tensor | finite | gradient norm sum |
|---|---:|---:|---:|
| predictor/history/aux heads | 375 | 是 | 3.0043 |
| VGGT deep pairs 4–23 | 720 | 是 | 3.7180 |

future shallow target 为 detached tensor。direct/refined action 都输出 grouped
shape `[1,32,8,16]`，padding scalar 保持零。

## 5. Action-only：完整在线路径

结果文件：`outputs/preflight/final_bridge_action_only.json`

路径为 raw RGB → UMT5/VAE → online VGGT-GAM → sparse geometry K/V → Wan
observed-video prefill → 30-layer grouped Action flow。

| 指标 | 数值 |
|---|---:|
| main action flow loss | 1.914100 |
| weighted auxiliary action loss | 0.001027 |
| forward | 5.134 s |
| backward | 1.457 s |
| peak memory | 21.273 GiB |

Action Expert 有 842 个梯度 tensor，geometry predictor/aux heads 有 377 个，
geometry adapters 有 20 个；三组梯度全部 finite 且非零。

部署 cache 与训练 action-only 使用同一执行路径：

```text
max delta = 0
RMS error = 0
```

同一 policy 输入分别带/不带 clean future target RGB 时：

```text
predicted shallow max delta = 0
deep visual max delta       = 0
geometry token max delta    = 0
```

这项检查说明 target branch 没有进入 policy 条件图。

## 6. Forward-world：20 Hz RoboCasa

结果文件：`outputs/preflight/final_robocasa_forward_world.json`

| 指标 | 数值 |
|---|---:|
| future action events | 32 |
| video flow loss | 1.759363 |
| geometry loss | 1.030978 |
| video velocity shape | `[1,48,3,16,16]` |
| observed latent max delta | 0 |
| forward | 5.402 s |
| peak memory | 24.855 GiB |

首个 Wan latent 在加噪前后逐元素一致，video loss 只监督后两个 latent frame。

## 7. Wan/Action 耦合检查

两个实验只反传一个 objective，用梯度方向确认耦合不是接口层声明。

| program 与 backward loss | 被检查模块 | 有梯度 tensor | gradient norm sum |
|---|---|---:|---:|
| action-only / action loss | Wan DiT | 801 | 22.9960 |
| joint / video loss | Action Expert | 818 | 5.7852 |

Action loss 会更新 Wan，video loss 也会在 joint program 中更新 Action Expert。
对应结果文件为 `final_bridge_action_to_wan_gradient.json` 和
`final_bridge_video_to_action_gradient.json`；两次检查中的相关梯度均 finite。

## 8. 三步短拟合

真实 Bridge 窗口固定 flow timestep 和 noise，Stage B warmup 使用 AdamW、
global grad clipping 1.0：

```text
1.961929 → 1.918631 → 1.885765 → 1.839303
```

三次 optimizer step 后 probe 参数最大变化为 `4.58e-5`。峰值显存为
25.414 GiB。这个实验只验证 forward、
backward、optimizer ownership 和更新链路，不把单样本下降解释为泛化能力。
结果文件为 `final_bridge_action_short_fit.json`。

## 9. 当前结论与未覆盖项

已验证：

- 数据 split、timestamp、native action clock 与 online RGB I/O 能闭合；
- VGGT 是训练图中的核心模块，不依赖磁盘 feature cache；
- Action 经过 Wan 的逐层 K/V，并能反向更新 Wan；
- 三种 interaction program 的输出/loss shape 正确且 finite；
- policy target 隔离、observed latent 固定和 cache 一致性满足合同。

尚未由这些 preflight 证明：

- action/geometry/video 相对 baseline 的任务指标；
- 21 个 source 的控制语义、单位和 gripper 极性全部正确；
- 七卡 FSDP 的吞吐、长时稳定性与 checkpoint 恢复；
- task-OOD、high-motion、contact 和 long-rollout 指标。

这些项目属于正式 canary、adapter audit 与后续训练评测，不应从单窗口实验外推。
