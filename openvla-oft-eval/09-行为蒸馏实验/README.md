# 基座教师蒸馏：第一轮保守实验

目标不是替换 LIBERO-Spatial 基座模型，而是生成一个只供 task 4（抽屉内抓碗）调用的候选 action head。其他任务继续加载原基座 checkpoint，所以不会受候选参数影响。

## 实验约束

- VLA 视觉、语言、proprio 和 Transformer 全冻结。
- 基座 action head 全冻结并作为在线教师。
- 学生 action head 从基座逐位初始化。
- 只训练学生最后一层的 6 个位姿输出；夹爪输出行强制零梯度。
- 损失为专家 L1 + 选择性教师蒸馏 + 参数锚定。
- RLDS 内部最初使用 top-drawer 统计，但 batch transform 会先反归一化，再把 action、proprio 和动作 token 一并转换到基座 `libero_spatial_no_noops` 坐标系；蒸馏损失两侧不得处于不同统计坐标系。
- 训练从 step 0 开始，最多 100 个 optimizer steps，不继承 150000 的全局编号。

## 云实例安装

把 `scripts` 下七个文件复制到远端：

```bash
cd /mnt/workspace/openvla-oft
cp /上传目录/finetune_teacher_distill.py vla-scripts/
cp /上传目录/verify_distill_checkpoint.py tools/
cp /上传目录/make_eval_checkpoint.sh tools/
cp /上传目录/compare_paired_results.py tools/
cp /上传目录/run_distill_train.sh tools/
cp /上传目录/run_distill_acceptance.sh tools/
cp /上传目录/run_distill_robustness.sh tools/
chmod +x tools/run_distill_train.sh tools/make_eval_checkpoint.sh tools/run_distill_acceptance.sh
chmod +x tools/run_distill_robustness.sh

source /mnt/workspace/venv_oft/bin/activate
export PYTHONPATH="/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
python -m py_compile vla-scripts/finetune_teacher_distill.py \
  tools/verify_distill_checkpoint.py tools/compare_paired_results.py
```

## 执行

```bash
cd /mnt/workspace/openvla-oft
bash tools/run_distill_train.sh
bash tools/run_distill_acceptance.sh
# 仅在上一步 PASS 后执行：
bash tools/run_distill_robustness.sh
```

## 验收标准

### Gate 0：训练产物完整性

- 产生 step 25、50、100 三个 action-head checkpoint。
- 上游 action-head 层最大差异为 0。
- 夹爪权重和偏置最大差异均为 0。
- 最后一层前六维相对权重漂移不超过 2%。
- 任一项失败，不进入仿真。

### Gate 1：6 状态筛选

- state 5、6、11、38 必须全部成功；这些是基座成功、旧 head-only 容易退化的状态。
- state 29、32 至少修复一个；这两个是基座 48/50 中的失败状态。
- 在通过者中先选成功数最多者；同分选择训练步数更少者。

### Gate 2：同条件 50 状态配对

- 基座和候选必须由同一评估器、seed=7、open-loop=8、state 0–49 各跑一次。
- 候选至少 48/50，且不得低于本次重跑基座。
- “基座成功、候选失败”的新增回归不超过 1 个状态。
- “候选修复基座失败”的状态数必须大于新增回归数。

### Gate 3：稳健性确认（Gate 2 通过后再运行）

- 对 state 0–49 分别使用 seed 1–5，共 250 个配对 episode。
- 候选成功率至少 97.2%，且比同批基座高至少 1 个百分点。
- 配对新增失败不超过 2/250。
- 若要把“提升”写成统计结论，而不只是工程趋势，McNemar 精确检验需 `p < 0.05`。

### 抓取链路验收

对 state 11、38 以及被修复的 state 29/32 保留现有抓取诊断：必须出现双指接触、持续抬升至少 2 cm、最大抬升至少 12 cm；最终碗中心距基座成功放置参考点不超过 3 cm。若成功率通过但这些事件不通过，按偶然成功处理，不晋级。

## 收益口径

基座已经是 48/50，因此单次 50 状态测试的理论上限只有 +4 个百分点。第一轮的合格收益是 49/50 或 50/50，并且没有新的稳定失败。相对失败的旧 head-only（41/50 或 37/50）出现十几个百分点的恢复，不应写成对基座的模型提升；那主要是恢复被微调破坏的能力。

## 定向 failure-correction 实验

若第一轮仍为 48/50，执行：

```bash
bash tools/run_failure_correction_train.sh
bash tools/run_failure_correction_acceptance.sh
```

该实验只截取专家 `demo_29`、`demo_32` 的 `[45, 91)` 窗口，对应靠近抽屉、接触和抬升阶段。筛选时要求 state 5、6、11、38 保持成功，并且 state 29、32 至少修复一个；否则停止，不进入全量评估。

定向实验的最终标准是全量至少 `49/50`，新增回归不超过 1 个，并且被修复状态同时出现双指接触、抬升超过 2cm、最大抬升超过 12cm、最终距放置参考点不超过 3cm。
