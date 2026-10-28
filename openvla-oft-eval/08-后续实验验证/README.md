# 后续工作：为什么这样做 + 验证脚本

> 本次 INT8 并行解码实验的集中版验收标准与当前证据，见 [`INT8实验验收标准与文档总结.md`](./INT8实验验收标准与文档总结.md)。

> 面向 5 个失败案例（230 / 233 / 286 / 321 / 378）的后续实验包。
> 全部脚本为 Linux bash，当前目标平台为阿里云 PWI。默认使用 `/mnt/workspace`，优先激活
> `activate_oft.sh` / `venv_oft`；若实际挂载在 `/root/openvla_oft_workspace` 会自动识别。

## 证据基础（来自失败分析报告 + 推理侧诊断）

| Episode | 任务/场景              | 失败模式              |
| ------- | ------------------ | ----------------- |
| 230     | top drawer         | 夹爪进抽屉卡住，未形成有效抓取   |
| 233     | top drawer         | 空抓后仍执行搬运/放置轨迹     |
| 286     | on ramekin         | 连带抓起支撑物整组搬运       |
| 321     | next to cookie box | 抓取退化为推挤，目标被推移未重定位 |
| 378     | on stove           | 空抓后仍执行搬运/放置轨迹     |

共同点：全部死在 `grasp` 阶段，且“抓取成功后没有确认、开环动作仍继续执行”。
推理侧已证：open-loop 8→4 无效（233/378 仍失败）；腕部相机权重 1.0→1.25 无效（230 仍失败）；
仅“清空队列重规划”无效（230 触发 19 次重规划仍失败）。

## 为什么这样做（逐实验）

### 先跑 `00_env_check.sh` —— 为什么

所有结论必须基于同一 checkpoint（`openvla-7b-oft-finetuned-libero-spatial`）与已应用
`0001-Add-full-LIBERO-failure-study-pipeline.patch` 的环境。脚本一次性校验 PWI 当前 Python 运行时、
ckpt、GPU、patch 工具，避免“环境没对齐就出结果”。

### `01_reproduce_inference.sh` —— 推理侧复现

**为什么**：仿真侧增益必须和推理侧基线对照。先把归档的 4 组推理实验（chunk8 / chunk4 /
wrist w1 / w1.25）在 full-patch 环境重跑一遍，确认成功率与历史记录一致，证明环境可复现。

### `02_boundary_check.sh` —— 边界检查（233/378）

**为什么**：233/378 空抓后继续跑完整搬运轨迹，疑似“开环 chunk 尾段继续执行”所致。
核心问题是 grasp→transfer 切换发生在 **chunk 内部还是边界**：

- 在边界 → 只要在固定边界插入检测就能拦截；

- 在内部 → 边界方案证伪，必须“首次闭合动作取出时”立即干预（转 03）。
  旧日志线索：233 闭合在 c5/a1、378 在 c4/a6（内部），需新实现按 chunk 保存 8×7 动作 +
  gripper 命令 + 特权 sim 快照重新确证。

### `03_grasp_recovery.sh` —— 抓取确认 + 恢复状态机（233/378/230）

**为什么**：推理侧的 grasp\_replan 实验证明“重新规划”本身救不了——策略会重复同一错误。
需要真闭环：首次闭合 → 仿真真值确认（双指接触/夹爪-目标距离）→ 失败则
强制张开 → 重新观察 → 重试（≤3 次）。
**注意**：这是唯一需要先补代码的实验。full patch 只有 `--target_motion_replan_threshold_m`
（目标位移触发），**没有** verify\_grasp/retry 状态机。脚本检测未实现时会明确提示。

### `04_relocalize.sh` —— 推移后重新定位（321）

**为什么**：321 中碗被推离原位，策略未重新估计目标中心。用仿真特权信号检测
“目标位移 >1cm 且夹爪张开”时丢弃剩余 chunk、用**新观测**重新规划（配合 open-loop=4
缩短接触→重规划延迟）。全局 episode 321 对应 LIBERO 的 `task_id=6,
initial_state_index=20`；对照组（阈值 0）与处理组（0.01m）只改变位移重定位开关，
其余 checkpoint、控制频率 20Hz、open-loop=4 保持一致。脚本默认每组运行 10 个配对 trial，
trial `i` 的两组使用同一个 seed（默认 7～16），结果分别写入 `trial-01`～`trial-10`，避免覆盖。

验收脚本 `scripts/verify_relocalize_321.py` 会同时检查：两组是否各有 ≥10 个完整 trial、
control/treatment 是否按 seed 配对、处理组是否产生 `target_motion_forced_replan`、事件是否满足
“开爪 + 位移达到阈值 + 丢弃剩余动作”，以及处理组成功率是否高于对照组并出现
`control=False → treatment=True` 翻转。只有这些条件全部满足，才能下“重定位恢复 321”的结论；
若事件触发但成功率未翻转，只能说明机制触发，不能说明任务得到修复。

### `05_freq_ablation.sh` —— 高频控制 20Hz vs 50Hz

**为什么**：OFT 并行解码允许 50Hz 控制，把固定 open-loop=4 的单 chunk 开环窗口从 200ms 缩到 80ms，
减少开环漂移。但 checkpoint 在 20Hz 动作分布上训练，50Hz 存在分布偏移 → 必须同时观测
`chunk_deadline_met`（推断延迟是否真的跟上 80ms/4步 的时限）才能下结论。

当前脚本固定 `open-loop=4`，所以实际预算是 20Hz=200ms、50Hz=80ms。
`scripts/verify_frequency_ablation.py` 会读取三组固定状态（233/321/378）的
`chunks.jsonl`，按所有 chunk 加权计算 deadline 达标率，并计算各状态 P95 推理延迟。
工程验收门槛为：50Hz deadline 达标率 ≥95%、P95 延迟 ≤80ms，且固定状态成功数不低于 20Hz。
任一条件不满足，都不能把 50Hz 作为当前模型的有效高频控制配置。

### `06_top_drawer_sweep.sh` —— top drawer 全状态扫描

**为什么**：230/233 都发生在 top drawer。先问“失败是随机还是集中在特定碗位姿”。
把 task 4 的 50 个初始状态全扫，得 state 级成败分布，失败 state 集合就是 DAgger 的
目标集。产物同时生成 `dagger_collection_manifest.csv`（07 的输入）。

### `07_dagger_loop.sh` —— DAgger 针对性增训

**为什么**：只有改权重才能根除“抓不到/空抓”这类策略缺陷。DAgger 思路：
在策略失败的状态上补专家修正数据 → 与原始 LIBERO 数据混合 → LoRA 微调。
流程：06 定位失败 state → 录制 expert demo（LIBERO `scripts/collect_demonstration.py`）
→ 转 RLDS → 正确加载 150K 动作头后续训（约 24–25GB 显存，`batch=1 + grad accumulation`）→ 用新权重回跑
01\~04/06 判定“失败态提升且无灾难性遗忘”。

上次 0% 的首要根因与重训门槛见 [`再次微调决策.md`](./再次微调决策.md)。`07_dagger_loop.sh`
现已默认执行 200-step 安全续训，并在缺少原动作头、原始 spatial 回放数据、50 条专家示教或旋转覆盖时拒绝启动。

## 执行顺序（推荐）

```bash
CHECK_TRAINING=1 bash 00_env_check.sh     # 1. PWI 训练栈前置自检，全部通过再往下
bash 01_reproduce_inference.sh            # 2. 推理侧复现（对照基线）
bash 02_boundary_check.sh all             # 3. 边界判断（决定 03 是否必要）
REPEATS=10 bash 04_relocalize.sh all      # 4. 321 对照/处理各 10 个配对 trial + 自动验收
bash 05_freq_ablation.sh all              # 5. 20/50Hz 消融 + 自动实时性验收
# 03 需要先补 verify_grasp 状态机代码，实现后执行：
bash 03_grasp_recovery.sh all
# 训练侧（需要先有 06 的失败态清单）：
TRIALS=50 bash 06_top_drawer_sweep.sh
# 录制 expert demo 并整理 RLDS 后：
DAGGER_RLDS_ROOT=/path/to/rlds SOURCE_HDF5=/path/to/expert.hdf5 bash 07_dagger_loop.sh
```

## 每项实验的“结论分支”

| 实验            | 若结论是…                                 | 则下一步                       |
| ------------- | ------------------------------------- | -------------------------- |
| 02 边界检查       | `boundary_check_before_transfer=True` | 优先做边界插入检测                  |
| 02 边界检查       | 全 `False`（闭合在 chunk 内）                | 必须做 03 恢复状态机               |
| 03 恢复状态机      | retry3 成功率明显>no\_retry                | 把真值验证换成真实感知（真机部署）          |
| 03 恢复状态机      | 无增益（策略反复同错）                           | 早进训练侧（07），别再堆仿真修           |
| 04 relocalize | 321 成功                                | 移植成非特权信号（相机+距离）落地          |
| 04 relocalize | 无增益                                   | 321 直接进 DAgger 目标集         |
| 05 50Hz       | deadline 达标且成功率≥20Hz                  | 正式采用 50Hz                  |
| 05 50Hz       | deadline 达标但成功率降                      | 证实分布偏移 → 需 50Hz 数据重训（转 07） |
| 05 50Hz       | deadline 不达标                          | 50Hz 只是宣称，真机按 20Hz 标定      |
| 06 sweep      | 失败集中少数 state                          | 只对它们做 07                   |
| 07 DAgger     | 失败态提升 + 无回退                           | 扩到其余 suite 复评              |

## 待实现清单（第三方开发前必须知道的）

1. `run_libero_eval.py` 增加 `verify_grasp / grasp_recovery_max_retries / grasp_verify_hold_steps`
   与“首次闭合动作取出时”的拦截逻辑（复用 `collect_sim_snapshot` 的 contacts 真值）。
2. `analyze_action_chunks.py` 输出 **grasp\_verified / empty\_grasp\_forced\_replan /
   recovery\_retry\_n** 事件计数（03 验收依赖）。
3. 训练侧 RLDS 转换脚本（LIBERO docs 参考）与 expert demo 录制 SOP。
