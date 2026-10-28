# 02 · 评测 Pipeline 逐环节详解（图像观测 → prompt 构造 → action 反归一化 → 步进）

> 本文所有代码引用均来自官方仓库 `moojink/openvla-oft`（main 分支），文件路径相对于仓库根目录。跑通冒烟测试后对照本文读一遍代码，是熟悉 pipeline 最快的方式。

## 总览

```
每个 episode（LIBERO 每任务 50 条，每条至多 N 步：spatial 220 / object 280 / goal 300 / long 520）
│
│  env.reset() → LIBERO 固定初始状态 → 前 10 步发空动作 (num_steps_wait=10, [0,0,0,0,0,0,-1])
│
├─ ① 图像观测    obs["agentview_image"] + obs["robot0_eye_in_hand_image"] (256×256×3 uint8)
│                 翻转180°对齐训练数据 → 中心90%裁剪 → resize 224×224（第三人称 + 腕部两路）
│
├─ ② 输入构造    prompt = "In: What action should the robot take to {任务描述}?\nOut:"
│                 proprio(8维) 用 q01/q99 归一化到 [-1,1] → MLP 投影 → 拼入序列
│
├─ ③ 一次前向    action token 位置的隐状态 → L1RegressionActionHead(MLP)
│                 → 归一化 action chunk (8×7)，整个 chunk 一次前向并行读出（无需自回归解码）
│
├─ ④ 反归一化    a_real = 0.5·(a+1)·(q99-q01+1e-8) + q01    （BOUNDS_Q99，逐维）
│                 统计量来自 checkpoint 里的 dataset_statistics.json[unnorm_key]
│
├─ ⑤ 后处理+步进 gripper: [0,1]→[-1,1] 且 binarize(取符号) → 取反（RLDS 与 robosuite 约定相反）
│                 action queue (deque, maxlen=8)：一次推理出 8 步，开环依次执行
│                 env.step(action) —— robosuite 控制频率 20Hz（非实时，尽快跑）
│
└─ ⑥ 终止        达到步数上限或任务 checker 判成功 → env._get_success()
                  存视频 ./rollouts/<日期>/...success=True|False...task=....mp4
                  写日志 ./experiments/logs/EVAL-<suite>-openvla-<时间>.txt
```

## 环节 0：常量与平台自动检测（一个隐蔽的坑）

`prismatic/vla/constants.py` 会扫描 `sys.argv`：命令行里含 `libero` → 用 LIBERO 常量；含 `aloha`/`bridge` 同理；都识别不出来**默认 LIBERO**。

```python
LIBERO_CONSTANTS = {"NUM_ACTIONS_CHUNK": 8, "ACTION_DIM": 7, "PROPRIO_DIM": 8,
                    "ACTION_PROPRIO_NORMALIZATION_TYPE": BOUNDS_Q99}
```

- **chunk=8**：模型一次输出 8 步动作；**ACTION_DIM=7**：6 DoF 末端增量（Δ位置×3 + Δ姿态×3）+ 1 维 gripper；**PROPRIO_DIM=8**：7 关节 + gripper。
- 坑：如果你自己写脚本调用模型、命令行里不含这些关键字，会用默认值（碰巧也是 LIBERO）。但换 ALOHA/Bridge 数据时务必确认。

## 环节 1：环境创建与 episode 初始化

`experiments/robot/libero/libero_utils.py :: get_libero_env`

```python
env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
env = OffScreenRenderEnv(**env_args)   # LIBERO 对 robosuite 的离屏渲染封装
env.seed(0)   # 注：官方注释指出即使使用固定初始状态，seed 也会影响物体位置
```

- 任务以 bddl 文件定义；观测相机分辨率 256×256（`env_img_res=256`）。
- 初始状态：LIBERO benchmark 为每个任务提供固定初始状态集合，官方评测按序取用保证可复现、跨方法可比。
- episode 开头 10 步发空动作 `get_libero_dummy_action()` = `[0,0,0,0,0,0,-1]`（让物理稳定下来，机器人不动）。

## 环节 2：图像观测与预处理（对齐训练是成功率的生命线）

`libero_utils.py :: get_libero_image` —— 先做 180° 旋转：

```python
img = obs["agentview_image"]
img = img[::-1, ::-1]   # IMPORTANT: rotate 180 degrees to match train preprocessing
```

（Bridge/RLDS 训练数据的图像朝向与 robosuite 相反，不翻转会系统性掉点。腕部相机 `robot0_eye_in_hand_image` 同样处理。）

`experiments/robot/openvla_utils.py :: get_vla_action` —— 收集两路图像并预处理：

```python
all_images = [obs["full_image"]]                                  # 第三人称
all_images.extend([obs[k] for k in obs if "wrist" in k])          # 腕部（num_images_in_input=2）
```

`prepare_images_for_vla` 对每路做：校验 H×W×3 uint8 → resize 到 224（内部走 JPEG 编解码 round-trip + lanczos3，对齐 RLDS 训练管线）→ **`center_crop=True` 时取中心 90% 区域再缩放到 224×224**。

- 为什么 center_crop 至关重要：训练时用了 `image_aug=True`（随机 90% 面积裁剪），测试时用中心 90% 裁剪与之对齐。**忘了开这个 flag 是"复现失败、成功率暴跌"的第一大原因。**
- 两路图像用同一个 prompt 分别过 processor，然后 `pixel_values` 沿 **patch 维** 拼接：`torch.cat([primary] + wrists, dim=1)` → 2×256 个 patch token。

## 环节 3：prompt 构造与本体感知（proprio）

`openvla_utils.py :: get_vla_action`：

```python
prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
```

- 任务描述直接来自环境（LIBERO 任务的自然语言指令，如 "put the black bowl on the stove"），模板就是 OpenVLA 的对话式格式。
- proprio：`obs["robot0_proprio-state"]`（8 维）先按该数据集的 q01/q99 归一化：

```python
# normalize_proprio（BOUNDS_Q99 分支）
proprio = 2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1   # → [-1,1]，再 clip
```

- 归一化后的 proprio 由 `ProprioProjector`（小 MLP）投影到 LLM 隐层维度，在 `predict_action` 内部拼入输入序列参与前向（`use_proprio=True` 是这些 checkpoint 的训练配置，eval 必须一致）。

## 环节 4：一次前向输出整个 action chunk（OFT 的核心提速点）

`prismatic/extern/hf/modeling_prismatic.py :: predict_action`（checkpoint 自带的 modeling 代码，`trust_remote_code=True` 加载的就是它）：

1. 输入序列 = 图像 patch tokens + prompt tokens + proprio + **56 个（8×7）action token 槽位**；
2. action 槽位的输入 embedding 被置零（`input_embeddings = input_embeddings * ~all_actions_mask`）；
3. **一次前向**后取最后一层隐状态中 action 槽位对应的 56 个位置：

```python
last_hidden_states = language_model_output.hidden_states[-1]           # (B, seq_len, D)
# 切出 action 区间: [NUM_PATCHES + NUM_PROMPT_TOKENS, + ACTION_DIM*NUM_ACTIONS_CHUNK)
normalized_actions = action_head.predict_action(actions_hidden_states) # L1RegressionActionHead (MLP)
normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)   # (8, 7)，取值 [-1,1]
```

与原版 OpenVLA 的区别（这就是 OFT 论文的两个改动）：
- **连续头替代离散 token**：不再把每维动作离散成 256 个 token 自回归解码，而是用 L1 回归头直接读出连续动作；
- **一次前向出整个 chunk**：不需要逐 token 生成，推理延迟大幅下降。

## 环节 5：action 反归一化（你的关注点之一）

仍在 `modeling_prismatic.py :: _unnormalize_actions`：

```python
action_norm_stats = self.norm_stats[unnorm_key]["action"]     # dataset_statistics.json
action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])  # BOUNDS_Q99
actions = 0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low
```

- `norm_stats` 挂在模型上：`_load_dataset_stats` 从 checkpoint 目录读 `dataset_statistics.json` 赋给 `vla.norm_stats`；
- `unnorm_key` 由评测脚本自动推导：先试 `libero_spatial`，不存在则回退 `libero_spatial_no_noops`（`_no_noops` 指 RLDS 版数据集剔除了 no-op 帧——训练数据版本，统计量必须与之配套）；
- 公式含义：模型输出 [-1,1] → 线性映射回 [q01, q99] 的真实动作区间，**逐维独立**；
- 输出 shape `(8, 7)`：8 步 × 7 维（6 DoF 末端增量 + gripper）。

## 环节 6：gripper 后处理与环境步进

`run_libero_eval.py` 主循环 + `experiments/robot/robot_utils.py`：

```python
# 每 8 步才推理一次（开环执行整个 chunk）
action_queue = deque(maxlen=cfg.num_open_loop_steps)   # 默认 8 = NUM_ACTIONS_CHUNK
if len(action_queue) == 0:
    action_queue.extend(get_action(...))               # 返回 8×7 的 list
action = action_queue.pop()

# gripper 两步后处理
action = normalize_gripper_action(action, binarize=True)
#   最后一维 [0,1] → [-1,1]，binarize 时再取符号（sgn）→ 只有 ±1
action = invert_gripper_action(action)
#   最后一维 ×(-1)：RLDS 惯例 0=close/1=open，robosuite -1=open/+1=close，两边约定相反

obs, reward, done, info = env.step(action.tolist())    # robosuite 旧 gym API（4 元组）
```

要点：
- **开环 chunk**：推理一次执行 8 步，是速度与控制平滑性的 tradeoff。官方警告：`num_open_loop_steps` 与 `NUM_ACTIONS_CHUNK` 不一致时会打 warning，推荐执行完整 chunk；
- gripper 被 binarize 成全开/全关两档——这些 checkpoint 训练时 gripper 就是二值的；
- 20Hz 控制频率是 robosuite 仿真设定，评测不睡实时（wall-clock 远快于真实时间）。

## 环节 7：成功判定、日志与视频（后续分析的原料）

- 终止：步数达上限（见总览）或 done；成功 = `env._get_success()`（LIBERO 每个 bddl 任务自带 checker，通常基于目标物体位姿）；
- 日志 `./experiments/logs/EVAL-<suite>-openvla-<DATE_TIME>.txt`，关键行（04 脚本就解析这些）：

```
Total episodes: 500
Total successes: 486
Overall success rate: 0.9720 (97.2%)
```

- 视频 `./rollouts/<日期>/<时间>--openvla_oft--episode=<i>--success=<True|False>--task=<任务名>.mp4`（30fps，原始 256×256 第三人称视角，**每个 episode 都存**）——文件名自带成败与任务名，05 脚本直接扫文件名做失败清单，不用改官方代码。

## 建议动手做的小实验（吃透 pipeline 的最快路径）

跑通 spatial 冒烟测试后，每个实验改动一行、跑 10 trials/task 对比：

1. **预处理对齐**：`--center_crop False` → 观察成功率暴跌，体会"训练-测试预处理一致"的分量；
2. **语言敏感性**：把 `libero_utils.py` 里取到的 task_description 换成同义句（如 "put the black bowl on the stove" → "place the black bowl onto the stove"）→ 观察变化，理解 prompt 的作用方式；
3. **开环长度**：`--num_open_loop_steps 1`（每步都推理，近似闭环）→ 对比成功率与总耗时；注意脚本会 warning（与训练 chunk 不一致）；
4. **proprio 消融**：`--use_proprio False` → 观察掉点幅度，量化本体感知的贡献；
5. **看中间量**：在 `get_vla_action` 返回前 `np.save("dbg_actions.npy", action)`，画出 8×7 chunk 的分布，看 gripper 维是否总是 ±1、末端增量是否平滑——为失败分析积累直觉。

## 常用入口文件索引

| 文件 | 作用 |
|---|---|
| `experiments/robot/libero/run_libero_eval.py` | 评测主循环、CLI 参数（draccus GenerateConfig）、日志 |
| `experiments/robot/libero/libero_utils.py` | 环境创建、图像 180° 翻转、视频保存 |
| `experiments/robot/openvla_utils.py` | get_vla（模型加载）、get_vla_action（图像/prompt/proprio/推理入口） |
| `experiments/robot/robot_utils.py` | get_action 分发、gripper 后处理 |
| `prismatic/extern/hf/modeling_prismatic.py` | predict_action + 反归一化（随 checkpoint 走） |
| `prismatic/vla/constants.py` | 平台常量自动检测 |
