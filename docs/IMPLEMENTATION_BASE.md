# WM3D-WAM 实现基线与当前状态

| 项目 | 内容 |
|---|---|
| 日期 | 2026-08-19 |
| 实现分支 | `codex/implement-wm3d-wam-v1` |
| 当前里程碑 | M1：数据合同与 Wan-Action 主链路 |
| 生产模型配置 | `configs/model/wan_action_mot_v1.yaml` |
| 数据配置 | `configs/data/grouped_robot_v1.yaml` |

## 基线结论

没有一个现有仓库同时满足 grouped robot ABI、Wan2.2 动作耦合和在线
VGGT。直接 fork 任意一个仓库都会带入大量错误假设。WM3D-WAM 因此保留
为集成入口，并按模块复用三套已验证实现。

| 范围 | 采用的基线 | 原因 |
|---|---|---|
| 视频与动作主干 | FastWAM | 已有 Wan2.2 TI2V-5B、ActionDiT、30 层 MoT 和首帧 K/V prefill |
| 数据与时钟 | 原 WM3D | 已有 PTS 窗口、fine/coarse 监督隔离和 grouped robot ABI |
| 几何主干 | node 41 的 VGGT-GAM | 已有 VGGT pair 0 至 3 浅层冻结、pair 4 至 23 恢复计算和因果 deep path |
| 项目组织 | 新仓库 WM3D-WAM | 避免继承旧缓存依赖、固定 action vector 和实验目录 |

Worldscape-MoE 适合学习数据组织和多数据源训练方式，不适合作为代码
主干。它没有本项目需要的 Wan2.2 ActionDiT、Grouped Robot ABI 和 VGGT
split-and-resume 组合。

## 代码来源边界

```text
src/wm3d_wam/
  data/                         原 WM3D 合同，加 source-native event 转换
  models/                       WM3D-WAM 新增的组合模块
  vendor/fastwam/               FastWAM 的 Wan2.2 / ActionDiT / MoT
  vendor/vggt_gam/              node 41 的 VGGT-GAM split 与 predictor
```

`vendor` 目录只做必要改动。FastWAM MoT 新增 batch-aware mask 和额外
geometry K/V；VGGT-GAM encoder 恢复 camera 与 point heads。来源、修改和
许可证写在 `THIRD_PARTY_NOTICES.md`。

## 已实现的纵向链路

### 1. 数据到 Action event

`GroupedRobotWindow` 继续保存 `[interval, group, substep, dimension]`。
`extract_grouped_action_events` 根据 `world_boundaries_s + fine_action_dt`
恢复真实事件：

- 5、10、15、20 Hz 在 1.6 秒内分别产生 8、16、24、32 个 event；
- 同一时间戳的多组命令共享 event，不同时间戳不合并；
- 不降采样，不插值，不重复末值；
- 超过 32 个 event 时拒绝样本，并要求提高容量；
- 时间 offset 使用 float64，避免跨 interval 的 float32 边界漂移。

batch 只在 event 维 pad 到 32。event、group、dimension 三层 mask 始终
随张量传递。

### 2. Grouped Action Flow Expert

`GroupedActionCodec` 为每个 scalar 组合以下信息：

- 当前 noisy action value；
- action semantic ID 与 composition operator；
- physical group ID、group slot 与 dimension slot；
- embodiment ID；
- 相对时间与真实 event delta。

codec 将有效 scalar 汇聚成一个 event token。30 层 ActionDiT 使用这些
token，输出端再用相同的 grouped metadata 解码到 `[B,E,G,D]`。模型没有
假设不同机器人的第 d 维语义相同。

ActionDiT 的 transformer blocks、text/time embedding 和 flow timestep
路径来自 FastWAM。WM3D-WAM 只替换固定 action vector 的输入输出投影。

### 3. Action 走 Wan2.2

`WanActionMoT` 同时准备 Wan video tokens 与 grouped action event tokens。
每层由两个专家分别产生 Q/K/V，然后进入同一次 mixed attention。两个
stream 保留各自的 hidden、output projection 和 FFN。

三种 program 的 mask 已实现：

| program | Video query 读取 Action | Action query 读取 Video |
|---|---:|---:|
| action_only | 否 | 只读 observed first-frame tokens |
| forward_world | 读取 clean candidate action | 否 |
| joint_world_action | 读取 noisy action | 读取 noisy video |

mask 的形状为 `[B,1,Sq,Sk]`，所以 8、16、24、32 event 可以共处一个
batch。padding token 不参与有效 query 或 key。clean future target 不在
这些 API 的参数中。

动作推理使用 `prefill_observed_video`。Wan 对 observed latent frame
计算一次 per-layer K/V，后续 action flow steps 只重算 Action Expert。
cache 只存在于一次调用的内存中，不落盘。

### 4. 在线 VGGT 接口

迁入的 encoder 在 forward 内运行 VGGT：

- pair 0 至 3 用 no-grad 产生 shallow tokens；
- pair 4 至 23 从预测 shallow tokens 恢复；
- deep causal mode 使用按时间块的严格因果 attention；
- camera、depth、point heads 全部保留；
- track head 不在 v1 合同中。

`SparseGeometryKVAdapters` 在层 `[5,11,17,23,29]` 把预测 geometry
tokens 投影成共享 attention K/V。geometry 不作为第三个 query stream。
adapter 同时支持完整 MoT forward 和 action-only K/V cache 路径。

## 当前没有伪装完成的部分

以下工作尚未实现：

1. grouped proprio / past-action history 到 `GAMFuturePredictor` 的 token
   connector；
2. 四个未来 anchor 的连续 rollout 与 policy/factual mode API；
3. 21 个 source 的 manifest、split 物化、view-role adapter 和在线 RGB
   decoder；
4. Wan VAE、text encoder、VGGT 和 Wan2.2 权重的统一只读 asset loader；
5. Stage A/B/C loss orchestration、FSDP trainer、checkpoint 和正式评测。

`configs/model/vggt_geometry_v1.yaml` 把 grouped history bridge 标成硬门禁。
训练入口完成前必须连接真实 grouped history；代码不会退回 source-specific
flattened action，也不会用零向量假装条件已经接入。

## 配置约束

生产 Wan-Action 配置保留 FastWAM 规模：

| 分支 | blocks | hidden | FFN | attention |
|---|---:|---:|---:|---:|
| Wan2.2 Video Expert | 30 | 3072 | 14336 | 24 × 128 |
| Grouped Action Expert | 30 | 1024 | 4096 | 24 × 128 |

训练运行配置只允许 GPU 1 至 7，GPU 0 明确列入 forbidden devices。当前
组件测试在 CPU 上使用小尺寸实例执行同一批生产类，没有维护第二套简化
模型实现。

## 服务器资产审计

42 上已有可直接使用的 VGGT source tree 和 4.7 GB VGGT-1B safetensors。
encoder 已支持从该本地 safetensors 加载，不调用 Hugging Face 下载。

当前没有找到 FastWAM 期望的 Wan2.2 TI2V-5B DiT、Wan2.2 VAE、UMT5
text encoder 和 ActionDiT backbone 文件。M2 可以继续完成 geometry 与
数据路径；Stage B 前必须把这些资产放入只读目录并登记明确路径。正式
loader 只能读取本地文件，不能在训练进程中临时下载。

## 当前验证

服务器路径：`/data/Minko/WM3D-WAM`

```bash
PYTHONPATH=src /data/Minko/.venvs/wm3d/bin/python -m pytest -q
```

当前覆盖：

- grouped ABI 的 lossless pack 与半开时间窗；
- 5/10/15/20 Hz event 数量和逐值一致性；
- batch-aware interaction masks；
- Grouped ActionDiT 前向、反向和有效 scalar loss 归一化；
- Wan-Action MoT action-only 前向；
- 完整 MoT 与 observed-video K/V cache 数值一致；
- sparse geometry K/V 与 cache 路径数值一致；
- 物理 GPU 1 上加载服务器的 VGGT-1B 正式权重，完成真实
  shallow pair 0 至 3、deep pair 4 至 23、depth、point 和 camera 前向。

真实 VGGT smoke 的输出为 shallow `[1,1,1,261,1024]`、depth
`[1,224,224]`、world points `[1,224,224,3]`、pose encoding `[1,9]`。
当前还没有加载 Wan2.2 TI2V-5B 正式权重，也不能替代七卡 canary。

## 下一实现门槛

M2 先完成 grouped history connector 和 `OnlineVGGTGeometryCore`，用服务器
上的真实 VGGT 权重跑 shallow、future predictor、deep 与三个 geometry
heads。M2 通过后再接在线 Wan VAE 和 Stage A trainer。这样每个阶段都
使用最终模块，不引入临时 cache 或扁平 action fallback。

## 上游链接

- FastWAM: <https://github.com/AgibotTech/FastWAM>
- VGGT: <https://github.com/facebookresearch/vggt>
- GAM: <https://github.com/cvlab-kaist/Geometric-Action-Model>
- VGGT-GAM adaptation: <https://github.com/wxqnl/vggt-gam>
