# 03 · 失败 Case 分析方法论

目标：把 500×4 个 episode 里的失败案例变成**结构化、可统计、可归因**的分析产出。前提：已跑完评测（03 脚本），已生成失败清单（05 脚本）。

## 0. 输入材料

| 材料 | 位置 | 说明 |
|---|---|---|
| 失败视频 | `$OFT_ROOT/rollouts/<日期>/*--success=False--*.mp4` | 30fps，256×256 第三人称视角 |
| 失败清单 + 标注模板 | `results/failure_cases/annotation_template.csv` | 05 脚本生成，预填定位信息 |
| 每任务成败统计 | `results/failure_cases/failures_by_task.md` | 找系统性弱项 |
| 评测日志 | `$OFT_ROOT/experiments/logs/EVAL-*.txt` | 每条 episode 的 Success: True/False |

## 1. 五步工作流

**Step 1 · 全局统计定位**：看 `failures_by_task.md`，回答三个问题——
- 失败集中在少数任务（系统性弱项）还是均匀散布（普遍性退化）？
- 哪个 suite 失败最多？（预期 LIBERO-Goal 最多，参考值 95.8% 最低）
- 同一任务失败率是否异常高（如某任务 20/50 失败）→ 优先深挖。

**Step 2 · 观看失败视频**：从失败最多的 top 3 任务开始，每个任务看 3-5 条失败视频。带着问题看：卡在哪个阶段？动作行为像"看错了"还是"做不准"？

**Step 3 · 填写标注模板**：每条失败 case 一行，字段见第 3 节。标注时只记录客观现象，假设写在单独列——现象和归因分开，避免先入为主。

**Step 4 · 统计归因**：把标注完的 CSV 按分类汇总（透视表或几行 pandas），看失败阶段/根因维度的分布；同一类别反复出现 → 系统性问题。

**Step 5 · 写分析报告**：按第 5 节模板产出，每个结论都要能指到具体视频证据。

## 2. 失败分类体系

两个正交维度：**发生阶段**（可观测）× **根因维度**（推测）。

### 发生阶段（failure_stage，按操纵任务的典型分解）

| 阶段 | 判断标志 |
|---|---|
| `reach` | 末端从未正确到达目标物体附近（方向错、距离误判、绕圈、抖动不收敛） |
| `grasp` | 到达了但抓空 / 抓偏 / 抓错物体 / gripper 时机不对 / 抓起又捏飞 |
| `transfer` | 抓住了但运输途中掉落 / 碰倒其他物体 / 路径穿过障碍物失败 |
| `place` | 运到了但放错位置 / 放置太重弹开 / 没松开 gripper / 放完又碰倒 |
| `long_horizon` | （主要在 LIBERO-10）子任务顺序错、中途停滞、做完第一步忘了第二步目标 |
| `timeout` | 到达步数上限但无明显错误动作（常有"接近成功但差一点"与"完全跑偏"两种，看视频区分） |

### 根因维度（category，推测 + 证据）

| 维度 | 典型证据 |
|---|---|
| `感知` | 视角里明明有正确物体却伸向别处；对物体位姿的接近方向系统性偏移；遮挡场景集中失败 |
| `控制` | 方向对但幅值过冲/欠冲；末端震荡；gripper 开合时机偏差半秒导致抓空 |
| `任务理解` | 抓对了物体放错了目标位置（Goal 套件常见）；换了目标物（往另一件类似物体上放） |
| `待定` | 证据不足，标注 `待定` 而不是硬猜 |

> 提示：prompt 是模型看到的唯一任务信息（见 docs/02 环节 3）。Goal 套件里"抓对物体、放错目标"的失败，值得回头逐字对比任务描述与视频行为。

## 3. 标注字段说明（annotation_template.csv）

| 字段 | 填法 |
|---|---|
| `failure_stage` | 第 2 节六选一 |
| `category` | 第 2 节四选一 |
| `what_happened` | 客观现象 + 时间点，如"第 12s 伸向红碗左侧 3cm 抓空，随后重复同轨迹 3 次" |
| `root_cause_hypothesis` | 归因假设，如"腕部相机视角下红碗被黄油盒遮挡，第三人称视角也偏暗" |
| `evidence_timestamp` | 视频内证据时间点，如"12.3s, 15.1s" |
| `followup_experiment` | 验证实验，如"用同任务其他 seed 重跑看是否复现同一轨迹" |

## 4. 深挖手段（从"看视频"到"看数据"）

视频只能看现象，要归因到模型输出，需要 dump 中间量。建议在 `run_libero_eval.py` 主循环里加几行（不影响官方逻辑，建议 `git diff` 留档）：

```python
# episode 循环外初始化
import json as _json; _dbg = {"actions": [], "proprio": [], "task": task_description}

# get_action 调用处（拿到 chunk 后）
_dbg["actions"].append(np.asarray(action_chunk).tolist())
_dbg["proprio"].append(obs["robot0_proprio-state"].tolist())

# episode 结束处（success 判定后）
_json.dump(_dbg, open(f"dbg/{task}_{episode_idx}_{'ok' if success else 'fail'}.json", "w"))
```

拿到 action 序列后判断：
- **action 是否周期性大幅跳变**（相邻 chunk 同维差 > 数据集 q99-q01 的量级）→ 模型输出不稳定，查预处理一致性；
- **失败前 action 是否长时间恒定/极小** → 模型"呆滞"（常见于 timeout 类失败）；
- **gripper 维是否始终 ±1、开合时机** → grasp 失败的直接线索；
- 对照同任务**成功** episode 的 action 轨迹，看分歧点从哪一步开始。

进阶（可选）：把失败 episode 的首帧图像 + prompt 单独抠出来，手工做反事实实验——改 prompt 措辞再让模型出 chunk，观察 action 是否显著变化，直接验证"任务理解"归因。

## 5. 失败分析报告模板（最终产出之二）

```markdown
# OpenVLA-OFT LIBERO Baseline 失败 Case 分析

## 1. 概览
- 评测设置：seed=7，50 trials/task，A10，checkpoint=...
- 总 episodes / 失败数 / 各 suite 失败数（引用 baseline_success_rates.md）
- 失败阶段分布：reach X% / grasp Y% / transfer Z% / place ... / long_horizon ... / timeout ...
- 根因维度分布：感知 ... / 控制 ... / 任务理解 ... / 待定 ...

## 2. 各 suite 特征
### LIBERO-Spatial（失败 N 条）
- 集中任务：...（失败率 top 任务）
- 主导模式：...
### LIBERO-Object / Goal / 10（同结构）
- Goal 套件注意"放错目标"类失败占比（任务理解 vs 感知）

## 3. 代表性 Case 详述（挑 3-5 个）
### Case 1: <suite> / <任务> / episode <i>
- 视频：rollouts/.../xxx.mp4，失败时刻 t=..s
- 现象：...
- 归因：...（阶段 + 维度）
- 证据：action dump 显示 ... / prompt 为 ...
- 验证实验与结果：...

## 4. 结论与建议
- 系统性问题清单（附出现频次）
- 对后续改进的启示（数据增强方向 / prompt 设计 / chunk 长度 / 后处理规则）
```

## 6. 实用技巧

- **批量看视频**：`xdg-open` 太慢，用 `mpv rollouts/<日期>/*success=False*.mp4` 连续播放，或下载到本地用播放器列表。
- **失败视频很多时**：按 `failures_by_task.md` 的 top 任务优先；timeout 类可以用 2 倍速快进。
- **对照成功案例**：每个失败任务挑 1 条成功视频对照看，分歧点一目了然。
- **标注纪律**：现象列禁止出现归因词汇；拿不准就标"待定"，宁缺毋滥——分析的价值在准确不在数量。
