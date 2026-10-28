# OpenVLA-OFT LIBERO-Spatial 三侧改进实验操作手册

## 1. 目标与边界

本实验包围绕五个已确认失败状态开展三组实验：

| 全局 Episode | Task ID | 初始状态下标 | 场景 | 主要实验 |
|---:|---:|---:|---|---|
| 230 | 4 | 29 | 顶部抽屉黑碗 | 相机遮挡诊断、抽屉专项统计、DAgger采集 |
| 233 | 4 | 32 | 顶部抽屉黑碗 | 8→4步重规划、chunk边界、抓取命令检查 |
| 286 | 5 | 35 | ramekin上的黑碗 | 腕部相机输入与特征权重 |
| 321 | 6 | 20 | cookie box旁黑碗 | 推移后重新观测、目标位姿变化触发重规划 |
| 378 | 7 | 27 | 炉灶黑碗 | 8→4步重规划、chunk边界、抓取命令检查 |

需要区分三件事：

1. `NUM_ACTIONS_CHUNK=8` 是模型每次预测的动作数量，本实验不改变模型输出形状。
2. `num_open_loop_steps=4` 表示只执行预测chunk的前4步，随后重新获取图像并推理。
3. `control_freq=50` 表示仿真控制器以50 Hz推进；它不保证模型真的能实时达到50 Hz，必须看实测推理时延。

## 2. 修改后的关键链路

```text
LIBERO观察
  ├─ 第三人称图像
  ├─ 腕部图像
  ├─ proprio
  └─ 仿真目标位姿（只用于诊断）
       ↓
OpenVLA-OFT一次预测8×7动作
       ↓
保存完整8×7 chunk、原始/处理后gripper、推理时延
       ↓
只把前N步放入执行队列（N=8或4）
       ↓
env.step
       ↓
chunk边界重新读取最新观察
  ├─ 正常重规划
  └─ 诊断消融：目标被推移超过阈值且夹爪仍打开时，丢弃余下动作并提前重规划
```

代码证据：

- `run_libero_eval.py::run_episode`：动作队列、chunk日志、目标位姿触发重规划。
- `openvla_utils.py::get_vla_action`：两路图片预处理和模型输入dump。
- `modeling_prismatic.py::PrismaticVisionBackbone.forward`：腕部patch特征缩放。
- `libero_utils.py::get_libero_env`：向LIBERO传递 `control_freq`。

## 3. 运行前准备

### 3.1 激活正确的Miniconda环境

不要直接 `source common.sh && activate_env`，因为旧函数失败时会关闭当前终端。使用：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate oft

which python
python -c "import draccus, torch, transformers, libero; print('runtime ok')"
```

### 3.2 基本检查

```bash
cd /mnt/workspace/openvla-oft

python -m py_compile \
  experiments/robot/libero/run_libero_eval.py \
  experiments/robot/openvla_utils.py \
  experiments/robot/libero/failure_study/analyze_action_chunks.py \
  experiments/robot/libero/failure_study/summarize_targeted_eval.py \
  prismatic/extern/hf/modeling_prismatic.py

for SCRIPT in experiments/robot/libero/failure_study/*.sh; do
  bash -n "$SCRIPT"
done
```

所有实验默认输出到：

```text
/mnt/workspace/openvla-oft/experiments/failure_study_runs/<时间>/
```

需要把多条命令写入同一目录时，先设置固定标签：

```bash
export RUN_TAG=study-01
```

## 4. 推理侧实验

### 4.1 8步基线对照

```bash
cd /mnt/workspace/openvla-oft
bash experiments/robot/libero/failure_study/run_inference_study.sh chunk8_control
```

运行状态：233、321、378。

### 4.2 执行前4步后提前重规划

```bash
bash experiments/robot/libero/failure_study/run_inference_study.sh chunk4
```

验收日志必须出现：

```text
Open-loop execution steps: 4/8
queued first 4
```

不能出现“保留最后4步”的行为。实现中使用无界 `deque` 并显式加入 `actions[:4]`。

### 4.3 相机链路基线

```bash
bash experiments/robot/libero/failure_study/run_inference_study.sh camera_control
```

运行状态：230、286。每个重规划点产生：

```text
00_agent_raw.png
01_wrist_raw.png
02_agent_resized.png
03_wrist_resized.png
04_agent_model_input.png
05_wrist_model_input.png
06_metadata.json
```

先检查 `05_wrist_model_input.png`：

- 是否全黑、翻转、颜色异常或裁剪错误；
- 230中黑碗是否被抽屉边缘或夹爪遮挡；
- 286中黑碗与ramekin边界是否可分；
- 主相机清楚但腕部相机不清楚时，不应提高腕部权重。

### 4.4 腕部特征权重1.25

仅在图片正常后执行：

```bash
bash experiments/robot/libero/failure_study/run_inference_study.sh camera_w125
```

这里的“权重”是把第二路及后续图片的视觉patch特征乘以1.25。它是推理消融，不是重新训练出的注意力权重，必须与1.0对照，不能直接当最终改进。

## 5. 仿真与控制侧实验

### 5.1 Action chunk边界

先比较233、378的8步和4步：

```bash
bash experiments/robot/libero/failure_study/run_simulation_study.sh boundary8
bash experiments/robot/libero/failure_study/run_simulation_study.sh boundary4
```

每个状态输出：

```text
actions/task_XX/state_YY/
├── chunks.jsonl
├── steps.jsonl
├── events.jsonl
└── episode_summary.json
```

`chunks.jsonl` 保存：

- 完整8×7原始动作；
- 处理后的8×7环境动作；
- 原始和最终gripper命令；
- 实际计划执行的前N个动作下标；
- chunk开始时的目标物位姿和接触对；
- 推理时延及当前控制频率下的deadline。

自动分析结果在每次运行的 `analysis/`：

```text
episode_summary.csv
chunk_summary.csv
analysis.md
```

重点字段：

- `first_close_chunk/action_index`：第一次闭合发生在哪个chunk和动作位置；
- `discarded_close_candidates`：闭合命令是否落在4步截断后的尾部；
- `first_transfer_at_boundary`：抓取后的移动是否从chunk边界开始；
- `contact_pairs_at_boundary`：边界处是否仍在接触；
- `deadline_met_rate`：推理是否满足实时deadline。

这些是日志启发式指标，不等于真实抓取成功标签。真实结论还要结合视频和物体位姿。

### 5.2 Episode 321推移后重新定位

对照组只使用固定4步重规划：

```bash
bash experiments/robot/libero/failure_study/run_simulation_study.sh relocalize_control
```

诊断组在夹爪仍打开且黑碗相对chunk起点移动超过1 cm时，丢弃剩余队列并立即用新观察重规划：

```bash
bash experiments/robot/libero/failure_study/run_simulation_study.sh relocalize_motion
```

成功触发时日志出现：

```text
Target moved ... m while gripper was open; discarded ... queued actions and will replan
```

注意：该检测使用MuJoCo真实物体位姿，属于“特权仿真信息”。它用于证明“推移后重规划是否有帮助”，不能直接作为真实机器人部署方案。若有效，下一步应换成视觉检测或腕部相机目标跟踪。

### 5.3 空抓确认与强制重规划

先运行不干预的固定状态对照组：

```bash
bash experiments/robot/libero/failure_study/run_simulation_study.sh grasp_control
```

再开启仿真真值抓取确认：

```bash
bash experiments/robot/libero/failure_study/run_simulation_study.sh grasp_recovery
```

该模式只干预仿真执行层，不修改模型、图像或预测chunk。检测到夹爪首次闭合时：

1. 立即清空chunk中尚未执行的动作；
2. 执行当前策略给出的闭合动作，再追加2个零位移闭合控制步；
3. 使用robosuite检查左右指垫接触，且目标中心与夹爪抓取点距离不超过12 cm；
4. 抓取有效则从新观测重新推理；
5. 空抓则原地张开4个控制步，再从新观测重新接近；
6. 连续3次空抓后记录 `grasp_retry_exhausted` 并结束该episode。

`events.jsonl`记录 `grasp_close_intercepted`、`grasp_verified`、
`empty_grasp_forced_replan`、`grasp_recovery_opened` 和 `grasp_retry_exhausted`。
汇总表包含 `Grasp checks`、`Empty-grasp replans` 与 `Retry exhausted`。这是使用
MuJoCo/robosuite真值的仿真消融，不能直接迁移到真实机器人。

### 5.4 20 Hz与50 Hz

```bash
bash experiments/robot/libero/failure_study/run_simulation_study.sh freq20
bash experiments/robot/libero/failure_study/run_simulation_study.sh freq50
```

脚本按频率同比例放大最大步数和稳定等待步数，使20 Hz和50 Hz覆盖近似相同的仿真时间。

4步、50 Hz时，每个chunk只有：

```text
4 / 50 = 0.08秒 = 80 ms
```

因此必须满足：

```text
mean_inference_latency_ms < 80 ms
deadline_met_rate 接近100%
```

否则只能说“仿真控制器设置为50 Hz”，不能说“系统实现实时50 Hz控制”。此外，checkpoint在20 Hz动作分布上训练，直接改为50 Hz存在控制分布偏移；若成功率下降，需要50 Hz数据重新训练，而不是继续提高频率。

## 6. 训练侧：抽屉专项与DAgger

### 6.1 运行50个抽屉状态

```bash
export RUN_TAG=top-drawer-50
TRIALS=50 bash experiments/robot/libero/failure_study/run_top_drawer_sweep.sh
```

如果只做快速检查：

```bash
export RUN_TAG=top-drawer-20
TRIALS=20 bash experiments/robot/libero/failure_study/run_top_drawer_sweep.sh
```

推荐完整50次，因为已知失败状态29和32不在前20个状态内。

输出：

```text
summary/targeted_results.csv
summary/dagger_collection_manifest.csv
```

### 6.2 失败状态分组

人工填写manifest中的：

```text
failure_category
expert_demo_status
expert_demo_path
notes
```

建议分类：

```text
DRAWER_OCCLUSION
GRIPPER_COLLISION
EMPTY_GRASP
STALL_IN_DRAWER
WRONG_GRASP_HEIGHT
OTHER
```

只有失败集中在特定碗位姿或遮挡模式时，才进入针对性增训。如果50次失败很分散，应优先修复通用抓取反馈，而不是只补抽屉数据。

### 6.3 收集专家修正轨迹

对manifest中 `needs_expert_correction=True` 的状态，用LIBERO遥操作、脚本专家或可信专家策略复现同一 `task_id/state_index`，从失败前的状态开始完成正确抓取。

每条专家轨迹必须包含与原LIBERO-Spatial一致的字段：

```text
image
wrist_image
EEF_state
gripper_state
action
language_instruction
```

然后转换成RLDS/TFDS数据集，目录名使用：

```text
libero_spatial_top_drawer_dagger
```

本补丁已经注册：

- 单独修正数据：`libero_spatial_top_drawer_dagger_only`
- 原Spatial数据+5倍采样修正数据：`libero_spatial_plus_top_drawer_dagger`

不要把失败策略产生的动作当专家数据。DAgger的关键是“策略访问到的困难状态 + 专家给出的正确修正动作”。

### 6.4 启动微调

数据根目录需要同时包含原始Spatial RLDS和新增修正RLDS：

```bash
export DAGGER_RLDS_ROOT=/mnt/workspace/rlds
export VLA_PATH=/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial
export RUN_ROOT=/mnt/workspace/openvla-oft-training-runs/top-drawer-dagger
export DATASET_NAME=libero_spatial_plus_top_drawer_dagger

bash experiments/robot/libero/failure_study/run_dagger_finetune.sh
```

默认增训参数是实验起点，不是论文确认的最优参数：

```text
batch_size=1
grad_accumulation=8
learning_rate=1e-4
max_steps=5000
save_freq=1000
LoRA rank=32
W&B offline
```

官方估算batch size 1约需25 GB显存，因此24 GB A10很可能OOM。推荐至少32 GB，优先A100 40/80 GB。脚本会阻止低于25,000 MiB的GPU；只有明确接受OOM风险时才设置 `ALLOW_LOW_VRAM=1`。

## 7. 完整回归

小规模实验选出候选方案后，至少执行：

1. 五个已知失败状态；
2. 每个相关任务20次；
3. 完整LIBERO-Spatial 500 episodes；
4. 最终候选至少3个seed。

完整评测前将参数固化到命令和 `run_config.txt`。比较：

```text
总成功率
每任务成功率
空抓后继续放置次数
推动替代抓取次数
支撑物一起搬运次数
平均推理时延
deadline_met_rate
单episode耗时
新增失败类型
```

## 8. 结论规则

- 只运行1次成功，写“该固定状态下成功”，不能写“显著提升”。
- 50次任务统计后才能描述该任务的失败率变化。
- 完整500次评测后才能描述suite成功率。
- 3个seed后才能判断改进是否稳定。
- 腕部图片异常时，权重实验无效，应先修复图像链路。
- 50 Hz的deadline未满足时，不得宣称实时50 Hz。
- 特权仿真位姿触发重规划有效，只能证明机制方向，不能直接外推真实机器人。

## 9. 理解度自测

1. 为什么不能把 `NUM_ACTIONS_CHUNK` 直接从8改成4？
2. 为什么旧的 `deque(maxlen=4).extend(8步)` 不是执行前4步？
3. 50 Hz控制频率与模型每秒能完成多少次推理有什么区别？
4. 为什么321中的位姿触发重规划不能直接用于真实机器人？
5. DAgger数据为什么不能使用失败策略自己的动作作为标签？

答案要点：模型输出结构仍是8×7；有限deque会丢弃左侧保留后4步；控制器频率不等于推理吞吐；仿真真实位姿属于部署时不可见信息；DAgger标签必须来自专家修正。
