# OpenVLA-OFT × LIBERO-Spatial 评测 Pipeline 详解

> 目标：把一次 LIBERO-Spatial episode 从“环境给出观测”到“机械臂执行动作”的完整数据流讲清楚。
>
> 代码基准：`moojink/openvla-oft` 官方仓库 `main` 分支，核对日期 2026-08-28。实际 GPU 实例上的代码可能对应不同 commit，运行前请用 `git rev-parse HEAD` 留档。

## 1. 一句话理解

每隔 8 个控制步，系统从 LIBERO 读取第三人称图像、腕部图像和机器人末端状态，把任务语言包装成固定 prompt；OpenVLA-OFT 一次预测未来 8 个七维动作，使用 checkpoint 内的数据统计量把动作反归一化，再按时间顺序送入 LIBERO 环境执行，直到任务成功或达到 220 个有效控制步。

## 2. 完整调用链

```text
03_run_libero_eval.sh
  │
  └─ run_libero_eval.py :: eval_libero(cfg)
       ├─ 固定随机种子
       ├─ 加载 VLA、L1 action head、proprio projector、processor
       ├─ 创建 libero_spatial task suite
       │
       └─ 对 10 个 task 调用 run_task(...)
            ├─ 读取 task.language 和固定初始状态
            ├─ 创建 OffScreenRenderEnv
            │
            └─ 每个 task 运行 50 个 run_episode(...)
                 ├─ reset + set_init_state
                 ├─ 前 10 步执行 dummy action，让物体稳定
                 │
                 └─ 有效控制循环（Spatial 最多 220 步）
                      ├─ prepare_observation(obs)
                      │    ├─ 第三人称图像
                      │    ├─ 腕部图像
                      │    └─ 8维 proprio
                      │
                      ├─ action_queue 为空时调用 get_action(...)
                      │    └─ get_vla_action(...)
                      │         ├─ 图像预处理
                      │         ├─ prompt 构造
                      │         ├─ proprio 归一化
                      │         └─ vla.predict_action(...)
                      │              ├─ 视觉/语言/proprio 融合
                      │              ├─ L1 head 输出归一化 8×7 action chunk
                      │              └─ 使用 dataset_statistics 反归一化
                      │
                      ├─ action_queue.popleft()
                      ├─ gripper 二值化和符号翻转
                      ├─ env.step(action)
                      └─ 得到下一帧 obs / done
```

官方入口和控制循环见：

- [`run_libero_eval.py`](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py)
- [`openvla_utils.py`](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/openvla_utils.py)
- [`modeling_prismatic.py`](https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py)

## 3. 关键配置与常量

LIBERO 的平台常量定义在 `prismatic/vla/constants.py`：

| 常量 | 值 | 含义 |
|---|---:|---|
| `NUM_ACTIONS_CHUNK` | 8 | 一次模型前向预测 8 个未来动作 |
| `ACTION_DIM` | 7 | 每个动作 7 维 |
| `PROPRIO_DIM` | 8 | 机器人本体状态 8 维 |
| 归一化方式 | `BOUNDS_Q99` | 使用训练数据 q01/q99 做缩放 |

LIBERO-Spatial 的默认评测配置：

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `num_images_in_input` | 2 | 第三人称 + 腕部相机 |
| `use_proprio` | `True` | 使用本体状态 |
| `center_crop` | `True` | 测试预处理与训练增强对齐 |
| `num_open_loop_steps` | 8 | 每次重新推理前执行 8 个动作 |
| `num_steps_wait` | 10 | episode 开头物理稳定步数 |
| `num_trials_per_task` | 50 | 每任务 50 条轨迹 |
| `env_img_res` | 256 | 环境相机输出 256×256 |
| `seed` | 7 | Python、NumPy、PyTorch 随机种子 |
| Spatial 最大有效步数 | 220 | 不含开头 10 个稳定步 |

证据：[`constants.py`](https://github.com/moojink/openvla-oft/blob/main/prismatic/vla/constants.py) 和 [`GenerateConfig`](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py#L73-L115)。

## 4. 环节 0：模型与组件初始化

### 4.1 加载 VLA 主体

`get_vla(cfg)` 使用 `AutoModelForVision2Seq.from_pretrained(...)` 加载 checkpoint：

- 推理 dtype：`torch.bfloat16`；
- `trust_remote_code=True`；
- 当前官方代码没有启用 `flash_attention_2`，对应参数处于注释状态；
- `vla.vision_backbone.set_num_images_in_input(2)` 告诉视觉骨干会输入两路图像；
- `_load_dataset_stats(...)` 读取 checkpoint 内的 `dataset_statistics.json`。

这里的 `dataset_statistics.json` 不是普通日志，而是 proprio 归一化和 action 反归一化的标尺。使用错误 checkpoint 的统计量，即使模型前向没有报错，动作尺度也可能完全错误。

### 4.2 加载额外组件

Spatial OFT checkpoint 默认还会加载：

- `L1RegressionActionHead`：从语言模型 action 位置的隐藏状态回归连续动作；
- `ProprioProjector`：把 8 维 proprio 映射到 LLM 隐空间；
- Hugging Face processor：把 prompt 和 PIL 图像转换成 `input_ids`、`attention_mask`、`pixel_values`。

### 4.3 确定 `unnorm_key`

评测首先尝试：

```text
libero_spatial
```

如果 checkpoint 的 `norm_stats` 中没有该键，但存在：

```text
libero_spatial_no_noops
```

则自动使用后者。这个键决定从 `dataset_statistics.json` 的哪一组统计量读取 action 和 proprio 范围。

验证命令：

```bash
python - <<'PY'
import json
path = "/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial/dataset_statistics.json"
with open(path) as f:
    stats = json.load(f)
print("available unnorm keys:", list(stats.keys()))
for key in stats:
    print(key, "action fields:", stats[key]["action"].keys())
    print(key, "proprio fields:", stats[key]["proprio"].keys())
PY
```

## 5. 环节 1：任务、环境和初始状态

### 5.1 创建 task suite

`eval_libero` 通过：

```python
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[cfg.task_suite_name]()
```

创建 `libero_spatial`。随后依次遍历 suite 内的 10 个任务。

### 5.2 获取任务语言

`get_libero_env` 直接读取：

```python
task_description = task.language
```

因此 prompt 中的任务内容来自 LIBERO task，而不是从视频、文件名或人工输入中推断。

### 5.3 固定初始状态

默认初始状态来自：

```python
task_suite.get_task_init_states(task_id)
```

第 `episode_idx` 条评测使用 `initial_states[episode_idx]`。每个 episode 先 `env.reset()`，再调用 `env.set_init_state(initial_state)`。

环境创建后还会调用：

```python
env.seed(0)
```

这与全局 `cfg.seed=7` 是两件事：全局 seed 固定 Python/NumPy/PyTorch；环境内部又固定为 0。

### 5.4 前 10 步稳定期

episode 开头执行 10 次：

```python
[0, 0, 0, 0, 0, 0, -1]
```

前六维为零，最后一维保持夹爪打开。稳定期的图像不会写入 rollout 视频，也不计入 Spatial 的 220 个有效控制步。

代码证据：[`libero_utils.py`](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/libero_utils.py) 和 [`run_episode`](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py#L249-L324)。

## 6. 环节 2：从 LIBERO 观测构造策略输入

LIBERO 的原始 `obs` 是一个字典。当前链路使用三类信息：

```text
obs["agentview_image"]
obs["robot0_eye_in_hand_image"]
obs["robot0_eef_pos"]
obs["robot0_eef_quat"]
obs["robot0_gripper_qpos"]
```

`prepare_observation` 把它们转换成：

```python
observation = {
    "full_image":  ...,   # 第三人称图像
    "wrist_image": ...,   # 腕部图像
    "state": ...,         # 8维 proprio
}
```

这一字典才是 `get_vla_action` 的直接输入。

## 7. 环节 3：图像观测与预处理

### 7.1 提取两路相机

第三人称相机：

```python
img = obs["agentview_image"]
```

腕部相机：

```python
wrist_img = obs["robot0_eye_in_hand_image"]
```

两者的环境分辨率默认都是 `256×256×3`，dtype 为 `uint8`。

### 7.2 旋转 180°

两路图像都执行：

```python
img = img[::-1, ::-1]
```

这等价于同时反转高和宽两个轴，即旋转 180°。官方注释说明，这是为了匹配训练数据的图像方向。

如果忘记这一处理，模型不会报 shape 错误，但视觉分布会与训练时不一致。

### 7.3 第一次 resize：256 → 224

`resize_image_for_policy` 不只是普通缩放，而是：

```text
uint8 图像
  → JPEG encode
  → JPEG decode
  → Lanczos3 resize 到 224×224
  → round / clip 到 [0,255]
  → uint8
```

JPEG round-trip 用来尽量复现 RLDS 训练数据构建时的图像路径。

### 7.4 中心 90% 裁剪

进入 `get_vla_action` 后，`prepare_images_for_vla` 检查图像为 `H×W×3 uint8`。如果 `center_crop=True`，则：

```text
224×224
  → 取中心 90% 视野
  → resize 回 224×224
```

这里不是把最终输入变成约 202×202；裁剪后仍会恢复为 224×224。

### 7.5 processor 转换

第三人称图像先执行：

```python
inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
```

processor 输出至少包含：

```text
input_ids
attention_mask
pixel_values
```

腕部图像用同一个 prompt 单独经过 processor，但随后只取它的 `pixel_values`。两路 `pixel_values` 沿模型约定的 `dim=1` 拼接，语言 token 只保留一份。

视觉骨干随后为每路图像产生 256 个 patch feature；两路图像共 512 个视觉 patch feature，再映射到 LLM 隐空间。

代码证据：[`resize_image_for_policy`、`prepare_images_for_vla`、`get_vla_action`](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/openvla_utils.py)。

## 8. 环节 4：prompt 构造

任务文本会先转成小写，然后进入固定模板：

```python
prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
```

例如任务描述为：

```text
pick up the black bowl on the stove and place it on the plate
```

模型实际看到的文本是：

```text
In: What action should the robot take to pick up the black bowl on the stove and place it on the plate?
Out:
```

`predict_action` 还会检查 `Out:` 后是否存在训练时使用的特殊空 token；不存在时补入 token id `29871`。随后追加 56 个 action placeholder token 和 1 个 stop token：

```text
56 = 8 个动作 × 每个动作 7 维
```

容易混淆的地方：prompt 只描述“要做什么”，不携带当前物体坐标；物体位置来自图像，机器人当前位置来自 proprio。

## 9. 环节 5：8维 proprio 构造与归一化

### 9.1 8维状态的真实组成

当前官方 LIBERO 评测不是使用“7个关节角+1个夹爪值”，而是：

```text
robot0_eef_pos                 3维：末端执行器 x/y/z
quat2axisangle(eef_quat)      3维：末端姿态的轴角表示
robot0_gripper_qpos           2维：两个夹爪关节位置
--------------------------------------------------
合计                          8维
```

构造代码：

```python
state = np.concatenate((
    obs["robot0_eef_pos"],
    quat2axisangle(obs["robot0_eef_quat"]),
    obs["robot0_gripper_qpos"],
))
```

### 9.2 q01/q99 归一化

从：

```python
vla.norm_stats[cfg.unnorm_key]["proprio"]
```

读取每一维的 `q01`、`q99` 和可选 `mask`，然后计算：

```text
p_norm = 2 × (p - q01) / (q99 - q01 + 1e-8) - 1
```

再裁剪到 `[-1,1]`。若某一维的 `mask=False`，则该维不做缩放。

### 9.3 投影为一个 proprio token

`ProprioProjector` 是两层 MLP：

```text
8维 proprio
  → Linear(8, D)
  → GELU
  → Linear(D, D)
  → 1个 proprio feature/token
```

该 token 被追加在视觉 patch feature 后面。因此，默认双图像输入时，进入多模态序列的感知部分可理解为：

```text
512 个图像 patch feature + 1 个 proprio feature
```

## 10. 环节 6：多模态融合与 action chunk 预测

### 10.1 输入序列的概念布局

为便于理解，可以把进入语言模型的序列看成：

```text
<BOS>
  + 第三人称图像 patch
  + 腕部图像 patch
  + proprio token
  + prompt tokens
  + 56 个 action placeholder
  + stop token
```

实际实现中，图像和 proprio feature 在 `<BOS>` 后插入；action placeholder 的 embedding 置零，但保留位置，让模型在这些位置产生上下文化隐藏状态。

### 10.2 一次 Transformer 前向

OFT 不需要逐个生成 56 个离散 action token。一次语言模型前向后，代码取最后一层中对应 56 个 action 位置的隐藏状态：

```text
actions_hidden_states: [B, 56, D]
```

### 10.3 L1 action head

`L1RegressionActionHead` 先把每连续 7 个 action token 的隐藏状态合并：

```text
[B, 56, D]
  → reshape
[B, 8, 7D]
  → MLPResNet
[B, 8, 7]
```

得到归一化 action chunk：

```text
normalized_actions.shape = (8, 7)
```

这就是 OFT 的核心推理路径：一次前向并行得到未来 8 个连续动作，而不是自回归生成动作 token。

代码证据：[`action_heads.py`](https://github.com/moojink/openvla-oft/blob/main/prismatic/models/action_heads.py) 和 [`predict_action`](https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py)。

## 11. 环节 7：action 反归一化

### 11.1 为什么必须反归一化

action head 输出处于训练使用的归一化空间；LIBERO 环境需要真实控制尺度。模型内部调用 `_unnormalize_actions`，从：

```python
vla.norm_stats[cfg.unnorm_key]["action"]
```

读取 action 的 `q01`、`q99` 和 `mask`。

### 11.2 公式

对 `mask=True` 的维度逐维计算：

```text
a_real = 0.5 × (a_norm + 1) × (q99 - q01 + 1e-8) + q01
```

对应关系：

```text
a_norm = -1  →  q01
a_norm =  0  →  (q01 + q99) / 2
a_norm = +1  →  q99
```

对 `mask=False` 的维度，代码直接保留 `a_norm`。

输出仍然是：

```text
actions.shape = (8, 7)
```

### 11.3 七维动作的语义边界

在 LIBERO 默认末端控制器语义下，通常可解释为：

```text
[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
```

但评测脚本本身只明确把最后一维当作 gripper；前六维到控制器的精确尺度和坐标系由 LIBERO/robosuite 控制器配置决定。需要研究动作单位时，应同时检查环境 controller 配置，不应只凭这一个 Python 文件下结论。

## 12. 环节 8：gripper 后处理

模型返回 action 后，`process_action` 只对最后一维做两步处理。

### 12.1 从 `[0,1]` 映射到 `[-1,+1]`

```text
g1 = 2 × g0 - 1
```

然后二值化：

```text
g2 = sign(g1)
```

所以夹爪最终只有开/关两档，而不是连续控制。

### 12.2 符号翻转

OpenVLA 的 RLDS 数据约定与 robosuite 环境相反，因此再执行：

```text
g_env = -g2
```

官方代码注释给出的环境约定为：

```text
-1 = open
+1 = close
```

要点：action 反归一化发生在模型内部；gripper 二值化和符号翻转发生在评测循环中，二者不是同一步。

代码证据：[`robot_utils.py`](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/robot_utils.py#L137-L184)。

## 13. 环节 9：action queue 与环境步进

### 13.1 默认执行顺序

默认：

```python
action_queue = deque(maxlen=8)
```

队列为空时，模型生成：

```text
[a0, a1, a2, a3, a4, a5, a6, a7]
```

随后：

```python
action_queue.extend(actions)
action = action_queue.popleft()
```

因此实际执行顺序是：

```text
a0 → a1 → a2 → a3 → a4 → a5 → a6 → a7
```

这 8 步期间不会重新读取新图像做策略推理。环境当然会每步产生新 `obs`，但只有队列清空后，这个最新 `obs` 才会送回模型。

### 13.2 开环与闭环的边界

默认控制循环是“8步一闭环”：

```text
观测 O0 → 推理 A0...A7 → 连续执行8步
观测 O8 → 推理 A8...A15 → 连续执行8步
...
```

优点：模型前向次数少，评测更快。

代价：如果第 2 步就抓空，队列中剩余动作仍可能继续把空夹爪移动到放置区；模型要等到第 8 步后才有机会看到失败并重新规划。

### 13.3 一个非常重要的实现陷阱

当前代码使用：

```python
deque(maxlen=cfg.num_open_loop_steps)
action_queue.extend(all_8_actions)
```

如果直接设置：

```text
num_open_loop_steps = 1
```

Python 的有限长度 deque 在 `extend` 8 个动作时会丢弃左侧旧元素，最后只保留 `a7`。因此当前实现执行的是 chunk 的最后一步，而不是 `a0`。

同理：

```text
num_open_loop_steps = 2  → 保留并执行 a6、a7
num_open_loop_steps = 4  → 保留并执行 a4、a5、a6、a7
```

所以，不能把原脚本里 `--num_open_loop_steps 1` 直接解释成“每次执行 chunk 的第一步并重新推理”。如果要做真正的短闭环消融，应显式截取前 N 步，例如概念上改成：

```python
action_queue = deque()
action_queue.extend(actions[:cfg.num_open_loop_steps])
```

这是代码行为分析，不代表已经修改了你的 GPU 仓库。

### 13.4 `env.step`

每个动作完成 gripper 后处理后执行：

```python
obs, reward, done, info = env.step(action.tolist())
```

数据在这里完成闭环：

```text
模型动作
  → robosuite/LIBERO 控制器
  → MuJoCo 物理更新
  → 新的相机图像和机器人状态
  → 下一轮策略观测
```

## 14. 环节 10：终止、成功判定、日志与视频

### 14.1 终止逻辑

当前官方 `run_episode` 中：

```python
if done:
    success = True
    break
```

如果一直没有 `done=True`，Spatial 在执行满 220 个有效动作后退出，`success` 保持 `False`。

注意：当前 `main` 分支的评测循环没有显式调用 `env._get_success()`；它把环境返回的 `done=True` 当作成功。阅读其他版本代码时不能把两种实现混在一起。

### 14.2 视频内容

每个有效控制步在执行动作之前，把第三人称原始旋转图像加入 `replay_images`。视频：

- 不包含前 10 个稳定步；
- 只保存第三人称相机，不保存腕部相机；
- 使用 30 fps 编码；
- 30 fps 是视频播放帧率，不代表仿真按真实时间 30 Hz 运行。

一个失败 Spatial episode 如果正好有 220 帧，则说明它运行到了 220 个有效控制步上限；按 30 fps 播放时长度约为：

```text
220 / 30 = 7.33 秒
```

这与你当前5条失败视频的规格一致。

### 14.3 日志与成功率

每个 episode 结束记录：

```text
Success: True/False
# episodes completed so far
# successes
```

每个 task 结束记录：

```text
Current task success rate
Current total success rate
```

全部任务结束记录：

```text
Total episodes
Total successes
Overall success rate
```

## 15. 张量与数据形状总表

| 位置 | 名称 | 典型形状/类型 | 说明 |
|---|---|---|---|
| LIBERO obs | `agentview_image` | `(256,256,3) uint8` | 第三人称图像 |
| LIBERO obs | `robot0_eye_in_hand_image` | `(256,256,3) uint8` | 腕部图像 |
| 图像预处理 | `full_image` | `(224,224,3) uint8` | 旋转、JPEG round-trip、resize 后 |
| 图像预处理 | `wrist_image` | `(224,224,3) uint8` | 同上 |
| observation | `state` | `(8,)` | eef pos 3 + axis-angle 3 + gripper qpos 2 |
| proprio 归一化 | `proprio` | `(8,)`，范围 `[-1,1]` | q01/q99 缩放并 clip |
| processor | `input_ids` | `(1,L)` | prompt tokens |
| processor | `attention_mask` | `(1,L)` | prompt attention mask |
| 视觉骨干 | patch features | `(1,512,Dv)` | 双图像时每路256 patch |
| 视觉投影 | projected patches | `(1,512,D)` | 映射到 LLM 隐空间 |
| proprio 投影 | proprio feature | `(1,1,D)` | 一个额外 token |
| action 隐状态 | `actions_hidden_states` | `(1,56,D)` | 8×7 个 action 位置 |
| L1 head | `normalized_actions` | `(8,7)` | 归一化动作块 |
| 反归一化后 | `actions` | `(8,7)` | 真实动作尺度 |
| queue 取出 | `action` | `(7,)` | 单个环境动作 |

其中 `D` 是语言模型隐藏维度；具体数值应从实际 checkpoint 配置读取，不在本文中猜测。

## 16. 一次完整 Spatial episode 的时间结构

假设任务一直未成功：

```text
10 个稳定步
+ 220 个有效控制步
= 230 次 env.step
```

默认每 8 个有效步推理一次，因此最多需要：

```text
ceil(220 / 8) = 28 次 VLA 前向
```

第 28 个 action chunk 只会执行前 4 个动作，episode 就达到步数上限。

你的完整 Spatial 基线实测为 500 episodes、15560 秒，约等于：

```text
31.1 秒 / episode
4 小时 19 分 / suite
```

这段墙钟时间包含 VLA 前向、MuJoCo 步进、渲染、日志和 MP4 写入，不是纯模型推理延迟。

## 17. 五个失败案例如何映射到 Pipeline

当前5个失败都首先表现为 grasp 阶段失效。Pipeline 中最相关的边界是：

```text
观测/预处理
  → 模型一次预测8步
  → 8步开环执行
  → 没有显式“是否抓住”检查
```

### 17.1 空抓后仍移动到盘子

对应 episodes 233、378。可见事实是夹爪没有带走黑碗，但仍继续执行类似放置的轨迹。

Pipeline 解释：当前 action queue 不会在 chunk 中间检查目标物是否跟随夹爪；短时间内继续执行是代码允许的行为。仅凭视频仍不能断言模型内部“没看见失败”，因为模型在队列执行期间根本没有被重新查询。

### 17.2 推动而不是抓取

对应 episode 321。需要对照以下中间量：

- 抓取接触前后的 `a0...a7`；
- 第六/第七个动作附近的 z 方向与 gripper 命令；
- 当前 chunk 结束后新观测是否促成纠正。

### 17.3 抓起黑碗和 ramekin 整体

对应 episode 286。需要同时记录黑碗和 ramekin 的位姿；第三人称视频能证明两者共同移动，但不能区分是抓取高度过低、碰撞、摩擦耦合还是物体状态估计偏差。

### 17.4 抽屉内停滞

对应 episode 230。需要记录末端位姿、前六维动作和接触信息，判断模型仍在发动作但被碰撞约束，还是模型动作本身接近零。

## 18. 推荐诊断点

为定位失败，建议按数据边界记录，而不是只保存最终视频。

| 诊断点 | 记录内容 | 能回答的问题 |
|---|---|---|
| `prepare_observation` 后 | 两路图像、原始8维 state | 输入是否正确、图像是否翻转 |
| `normalize_proprio` 前后 | raw/norm proprio | q01/q99 是否匹配 |
| prompt 构造后 | 完整 prompt、token 长度 | 任务语言是否正确 |
| `predict_action` 内 | normalized `(8,7)` | 模型归一化输出是否异常 |
| `_unnormalize_actions` 后 | real `(8,7)`、q01/q99 | 动作尺度是否异常 |
| `process_action` 前后 | gripper 原值/二值/反号值 | 开合时机是否正确 |
| `env.step` 前后 | action、eef pose、object pose、done | 动作是否执行、物体是否跟随 |
| episode 结束 | task、init state index、success | 能否精确复现 |

最低限度的每步记录格式：

```text
task_id
episode_idx
t
chunk_id
action_index_in_chunk
prompt
raw_proprio[8]
normalized_proprio[8]
normalized_action[7]
unnormalized_action[7]
processed_action[7]
eef_pose
target_object_pose
support_object_pose
done
```

## 19. 常见误区与正确说法

| 误区 | 正确说法 |
|---|---|
| proprio 是7关节角+夹爪 | 当前 LIBERO eval 是 eef位置3 + eef轴角3 + gripper qpos2 |
| 每步都重新看图推理 | 默认每8步重新推理一次 |
| `num_open_loop_steps=1` 就是执行 action[0] | 当前有限 deque 实现会保留 action[7]，需修改截取逻辑 |
| 反归一化就是 gripper 反号 | 反归一化在模型内部；gripper 映射、二值化、反号在评测循环 |
| 视频30fps代表控制频率30Hz | 30fps只是写视频的播放帧率 |
| success 一定来自显式 `_get_success()` | 当前官方 main 直接把 `env.step` 返回的 `done=True` 视为成功 |
| 220帧视频包含10步稳定期 | 稳定期没有加入 replay；220帧是有效控制循环 |

## 20. 验证当前 GPU 仓库版本

在 GPU 实例执行：

```bash
cd /mnt/workspace/openvla-oft

git rev-parse HEAD
git status --short

grep -n "def prepare_observation\|def process_action\|def run_episode" \
  experiments/robot/libero/run_libero_eval.py

grep -n "prompt =\|def normalize_proprio\|def prepare_images_for_vla\|def get_vla_action" \
  experiments/robot/openvla_utils.py

grep -n "def _unnormalize_actions\|def predict_action" \
  prismatic/extern/hf/modeling_prismatic.py

grep -n "action_queue\|popleft\|env.step" \
  experiments/robot/libero/run_libero_eval.py
```

如果 GPU 上的代码与本文描述不一致，应以实际 commit 的源码为准，并在实验报告中记录差异。

## 21. 自测题

1. 为什么模型不是每个控制步都重新读取图像？
2. `normalized_actions` 与送入 `env.step` 的 action 中间经过了哪两类变化？
3. 8维 proprio 分别由什么组成？
4. 为什么 episode 233 抓空后仍可能继续朝盘子移动？
5. 为什么不能直接用当前代码的 `num_open_loop_steps=1` 做严格的一步闭环实验？
6. 一条失败视频有220帧、30fps，为什么播放时间只有约7.33秒，却不代表机器人只运行了7.33秒真实时间？

### 答案要点

1. 模型一次输出8步动作，默认 queue 清空后才重新推理。
2. 先根据 q01/q99 反归一化；再对 gripper 做 `[0,1]→[-1,1]`、二值化和符号翻转。
3. eef位置3维、eef四元数转轴角3维、gripper qpos2维。
4. 抓取失败可能发生在8步 action chunk 中间，剩余开环动作仍会继续执行，代码没有显式抓取成功检查。
5. 有限长度 deque 在一次 extend 8步时丢掉左侧动作，长度1时保留的是 action[7]。
6. MP4 的30fps是编码播放速度；仿真循环没有按真实时间睡眠，墙钟时间还包含推理、渲染和写盘。

## 22. 官方源码索引

- [评测入口、任务循环、episode循环](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py)
- [LIBERO环境、图像旋转、视频保存](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/libero_utils.py)
- [图像处理、prompt、proprio、VLA调用](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/openvla_utils.py)
- [gripper处理和动作分发](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/robot_utils.py)
- [多模态融合、action预测、反归一化](https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py)
- [L1 regression action head](https://github.com/moojink/openvla-oft/blob/main/prismatic/models/action_heads.py)
- [Proprio projector](https://github.com/moojink/openvla-oft/blob/main/prismatic/models/projectors.py)
- [LIBERO平台常量](https://github.com/moojink/openvla-oft/blob/main/prismatic/vla/constants.py)

