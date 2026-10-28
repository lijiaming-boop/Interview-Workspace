# OpenVLA-OFT 环境搭建与训练方案文档

> [!NOTE]
> **全局权威口径导读（2026-09-15）：**  
> 1. 本文档是项目早期至中期的**云端工程搭建、依赖锁定、排坑全实录与 DAgger 微调方案原始记录**。  
> 2. 关于项目全链路的**最新综合大总结与权威指标看板**，请首选查阅根目录 [总结文档.md](总结文档.md) 及 [README.md](README.md)。  
> 3. **结果口径更正提醒**：本文第十节记录的是 2026-09-04 阶段判断。后续严格评估确认：基座为 `48/50`，head-only 自定义统计为 `41/50`，head-only 基座统计为 `37/50`；此前“head-only 48/50”的归属已纠正为基座模型。最新权威汇总见 [总结文档.md §7](总结文档.md#七训练侧微调实验与物理接触层分析)。

---

### 快速导航目录

- [一、实例选择结论](#一实例选择结论)
- [二、最终验证通过的版本清单](#二最终验证通过的版本清单)
- [三、踩坑记录（症状 → 根因 → 解法）](#三踩坑记录症状--根因--解法)
- [四、环境搭建完整执行步骤](#四环境搭建完整执行步骤)
- [五、验证环境是否正常](#五验证环境是否正常)
- [六、LIBERO 评测（500 轮基准）](#六libero-评测)
- [七、HDF5 到 RLDS 数据集转换](#七hdf5-到-rlds-数据集转换)
- [八、微调训练方案与执行](#八微调训练方案与执行)
- [九、新实例迁移与踩坑全集（15个问题清单）](#九新实例迁移与踩坑全集)
- [十、阶段性实验结论与决策](#十阶段性实验结论与决策)

---

> 起始日期：2026-09-02  
> 测试实例：魔搭免费 GPU 实例（A10 24G，ModelScope 官方镜像 Python 3.10）  
> 核心目标：LIBERO 仿真评估 → 失败分析 → state 29 微实验（DAgger 式增训验证）与环境迁移避坑  

***

## 一、实例选择结论

| 项目 | 选择                                 | 理由                                            |
| -- | ---------------------------------- | --------------------------------------------- |
| 入口 | 魔搭「我的Notebook」→ 免费 GPU 实例（PAI-DSW） | 免费额度按计算时抵扣                                    |
| 规格 | A10 24G（勾选"支持资源包抵扣"）               | 7B bf16 推理需 ≥14G 显存（实测 15.13G）                |
| 镜像 | ModelScope 官方镜像 **Python 3.10** 版  | openvla-oft 要求 py310；flash-attn 预编译轮子只有 cp310 |
| 存储 | 一切放 `/mnt/workspace`               | 唯一持久化目录，关机/重启不丢                               |

**持久化内容**：`venv_oft`（虚拟环境）、`openvla-oft`（代码）、`LIBERO`（代码）、`openvla-oft-ckpts`（16G 权重）、`activate_oft.sh`（激活脚本）。
**重启会丢的**：apt 装的系统库（重装一次即可）、家目录配置。

***

## 二、最终验证通过的版本清单

| 包                          | 版本                                    | 备注                          |
| -------------------------- | ------------------------------------- | --------------------------- |
| Python                     | 3.10                                  | 镜像自带                        |
| torch                      | 2.2.0+cu121                           | `cuda.is_available()=True`  |
| numpy                      | **1.26.4**                            | 必须锁死，2.x 会炸 torch/tf        |
| opencv-python-headless     | 4.9.0.80                              | 5.0 要求 numpy>=2，禁用          |
| protobuf                   | **4.25.3**                            | TF2.15 舒适区，勿动               |
| tensorflow-cpu             | 2.15.0                                | 只用于 RLDS 数据加载               |
| tensorflow\_datasets       | 4.9.3                                 | --no-deps 装                 |
| tensorflow-metadata        | **1.15.0**                            | 新版 gencode 6.x 与运行时冲突       |
| tensorflow\_graphics       | 2021.12.3                             | dlimp 需要，--no-deps 装        |
| wandb                      | 0.16.6                                | 0.29 要求 protobuf>=5，冲突      |
| transformers               | moojink/transformers-openvla-oft fork | --no-deps 装                 |
| tokenizers / sentencepiece | 0.19.1 / 0.19.99                      | 版本必须对齐                      |
| peft                       | 0.11.1                                | checkpoint 是 LoRA，版本必须匹配    |
| diffusers                  | 0.30.3                                | action\_heads 导入时硬需求        |
| draccus                    | 0.8.0                                 | <br />                      |
| dlimp                      | moojink/dlimp\_openvla                | --no-deps 装                 |
| robosuite                  | 1.4.1                                 | <br />                      |
| mujoco                     | **3.3.2**                             | 3.4+ 与 robosuite 断言冲突       |
| gym                        | 0.25.2                                | <br />                      |
| LIBERO                     | git 源码                                | editable 安装不可靠，用 PYTHONPATH |
| flash-attn                 | 2.5.8（可选）                             | 失败则用 SDPA 补丁，结果等效           |

***

## 三、踩坑记录（症状 → 根因 → 解法）

1. **`numpy.core.umath failed to import`**
   根因：某包把 numpy 升到 2.2.6，torch2.2/tf2.15 按 1.x 编译。
   解法：`pip install --force-reinstall "numpy==1.26.4"`。

2. **opencv 5.0 要求 numpy>=2**
   解法：卸掉 `opencv-python` 和 `opencv-python-headless`，只装 `opencv-python-headless==4.9.0.80`（服务器无显示器，GUI 版无用）。

3. **`cannot import name 'runtime_version' from 'google.protobuf'`**
   根因：`runtime_version` 是 protobuf **5.27（v27.0）才引入**的；报错来自某个包的 pb2 文件用新版 protoc 生成。肇事者是新版 `tensorflow-metadata`（gencode 6.31.1）。
   解法：`protobuf==4.25.3` + `tensorflow-metadata==1.15.0`（均 --no-deps）。
   教训：凡是会拉 protobuf 的包（tfds、tf-metadata、tf-graphics）一律 `--no-deps` 装。

4. **`VersionError: gencode 6.31.1 runtime 5.29.5`**
   根因：protobuf 规则"生成代码版本不能比运行时新"，升 protobuf 是死路（TF2.15 要求 <5）。
   解法：反向降级 tensorflow-metadata 到旧 gencode 版本。

5. **flash-attn v2.5.5 wheel 404**
   根因：v2.5.5 只编译到 torch2.1；torch2.2 轮子从 v2.5.8 起。
   解法：`flash_attn-2.5.8+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`（pip 版 torch 是 FALSE ABI，必须匹配）。
   兜底：不装，改 `attn_implementation="sdpa"`（一行 sed，数值等效，慢 10\~20%）。

6. **`cannot import name 'get_vla' from 'prismatic'`**
   根因：`get_vla` 不在 prismatic，在 `experiments/robot/openvla_utils.py`；且参数是 `GenerateConfig` 对象不是字符串路径。
   正确用法：

   ```python
   from experiments.robot.libero.run_libero_eval import GenerateConfig
   from experiments.robot.openvla_utils import get_vla
   cfg = GenerateConfig(pretrained_checkpoint=CKPT, use_l1_regression=True,
                        use_diffusion=False, use_film=False)
   vla = get_vla(cfg)   # 没有 load_for_training 参数
   ```

7. **`No module named 'libero'`（换目录就失败）**
   根因：`pip install -e .` 未生效；在 LIBERO 目录下能 import 是因为 cwd 在搜索路径里。
   解法：放弃 editable，统一 `PYTHONPATH=/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO`。

8. **LIBERO 依赖链缺包**：`easydict`、`matplotlib` 逐个补（--no-deps 安装的代价）。

9. **`binding_utils.py AssertionError`（robosuite）**
   根因：pip 默认装了 mujoco 3.4+，枚举实现变更，robosuite 1.4.1 不兼容。
   解法：`pip install "mujoco==3.3.2"`（LIBERO 社区 PR #140 验证版本，渲染与训练数据一致）。

10. **`EGL_NOT_INITIALIZED`（实例重启后）**
    根因：重启换了容器，GPU EGL 驱动/库未就绪。
    解法：切 OSMesa 软渲染 `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`（CPU 渲染，模型推理仍走 GPU，无头环境最稳）。

11. **pip 无限回溯**：tensorflow 与 torch 依赖解析互相打架。
    解法：关键包全部 `--no-deps` 分步装，手动补依赖。

12. **一堆** **`openvla-oft requires xxx which is not installed`** **警告**
    其中 `diffusers`、`json-numpy`、`tensorflow_graphics` 是真需求（导入时触发），已补；
    `fastapi`、`uvicorn`、`OpenEXR`、`tensorflow-addons` 是起服务/渲染用的，评估与微调不需要，**故意跳过**。

***

## 四、冒烟测试结果（2026-09-02）

| 测试项                                   | 结果                                           |
| ------------------------------------- | -------------------------------------------- |
| 依赖总验证（torch/tf/dlimp/libero/get\_vla） | 全部通过                                         |
| 模型加载（4 个 safetensors 分片）              | 成功，耗时 \~3s，显存 **15.13 GB**                   |
| 2-episode 仿真冒烟（top drawer, state 0/1） | **2/2 成功（100%）**                             |
| 视频自动保存                                | `./rollouts/<日期>/`，文件名含 `success=True/False` |
| 渲染后端                                  | OSMesa（EGL 在重启后不可用）                          |

评估脚本关键参数（`run_libero_eval.py`）：

- `--target_episodes "4:0,4:1,..."`：零基 task:state 对，定向跑指定初始状态

- `--num_open_loop_steps 8`：action chunk 大小

- `--num_steps_wait 10`：等待物体稳定

- `--local_log_dir`：日志目录；结果逐条记录在 `EVAL-*.txt`

***

## 五、实验结果（2026-09-02）

### 5.1 top drawer 定向 50 trials（--seed 7）

- **49/50 = 98%**，唯一失败：episode=30 ↔ `initial_state_index=29`（已用日志确认映射）

- 视频自动保存于 `rollouts/2026_09_02/`，文件名含 `success=True/False`

### 5.2 旧全量评估（08\_28，500 episodes）各任务成功率

| 任务                            | 成功率     | 失败 state |
| ----------------------------- | ------- | -------- |
| top drawer（in the top drawer） | **96%** | 29, 32   |
| next to cookie box            | 98%     | 20       |
| on ramekin                    | 98%     | 35       |
| on stove                      | 98%     | 27       |
| 其余 6 个任务                      | 100%    | —        |

整体 **495/500 = 99%**。最低任务 96%，**不存在 <90% 的薄弱任务**。

### 5.3 state 29 失败根因（Part A 结论）

1. **确定性失败**：8/28（EGL 渲染）与 9/02（OSMesa 渲染）两次失败视频的 gripper 曲线逐帧一致、文件大小仅差 1 字节 → 与渲染后端/运行时无关，纯由初始状态决定。
2. **失败形态**（gripper 开合曲线解读）：

   - chunk 3~~5（t=34~~50）：张开接近碗；

   - chunk 5~~23（t=50~~194，约 6 秒）：**全程保持张开**，长时间游移/对齐；

   - chunk 24 边界（t=202）：才闭合抓取（过晚）；

   - chunk 26 边界（t=218）：重新张开，episode 结束。
3. **chunk 边界机制再次验证**：两次动作切换点均落在 t = 10 + 8k，与 233/378 结论一致。
4. **失败类型**：抓取对齐失败/超时，非"抓起后搬运掉落"。

### 5.4 决策

- 无薄弱任务 + 仅 1 个孤立确定性难例 → **大规模 DAgger 增训不划算**（收益 +1\~2 episode，风险是 98% 部分灾难性遗忘）。

- 但 state 29 是极佳的**最小实验样本**：在其上做微实验，验证 DAgger 管线能否修复确定性难例且不退化。

***

## 六、state 29 微实验方案（M1\~M5）

> 目标：在 state 29 上采集少量专家示教 → LoRA 微调 → 复测验证。
> 原则：最小改动、混合原数据防遗忘、同 seed 复测保证可比。

### M1 位姿提取（理解 state 29 为何难）

脚本 `/mnt/workspace/extract_poses.py`：重建 state 28/29/30 环境，等待 10 步稳定后打印所有物体位姿。
**注意**：robosuite 1.4.1 对象层没有 `get_geom_center`/`get_quat`/`eef_pos` 这些 API，必须用 MuJoCo 底层 `sim.data.xpos/xquat`（已踩坑验证）。

```python
import numpy as np
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

def obj_pose(env, obj):
    sim = env.env.sim
    bid = sim.model.body_name2id(obj.root_body)
    return np.array(sim.data.xpos[bid]), np.array(sim.data.xquat[bid])

def eef_pos(env):
    try:
        return np.array(env.env.robots[0].robot_model.eef_pos)
    except Exception:
        sim = env.env.sim
        for name in ['grip_site', 'right_grip_site', 'eef_site']:
            try:
                return np.array(sim.data.site_xpos[sim.model.site_name2id(name)])
            except Exception:
                continue
        return np.array([0, 0, 0])

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(4)
init_states = task_suite.get_task_init_states(4)

env, task_desc = get_libero_env(task, "openvla", resolution=256)
for sid in [28, 29, 30]:
    env.reset()
    obs = env.set_init_state(init_states[sid])
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(7))
    print(f"\n=== state {sid} ===")
    for obj in env.env.objects:
        pos, quat = obj_pose(env, obj)
        print(f"  obj={obj.name:24s} pos={np.round(pos,3)} quat={np.round(quat,3)}")
    print(f"  eef={np.round(eef_pos(env),3)}")
```

运行：

```bash
cd /mnt/workspace/openvla-oft
PYTHONPATH=/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python /mnt/workspace/extract_poses.py
```

**实测结果**（目标碗 = `akita_black_bowl_1`，位于 top drawer 内；盘子 `plate_1` 在桌面 z≈0.90）：

| state  | 碗位置 (x, y, z)              | 旋转      | 结果     |
| ------ | -------------------------- | ------- | ------ |
| 28     | (0.081, −0.152, 1.069)     | 旋转约 15° | 成功     |
| **29** | **(0.097, −0.136, 1.063)** | 无旋转     | **失败** |
| 30     | (0.083, −0.154, 1.063)     | 无旋转     | 成功     |

**M1 结论**：

1. **无极端几何异常**：state 29 的碗相对邻域仅偏移约 2cm（+x 1.5cm、+y 1.7cm，略偏右、略靠抽屉口），高度相当；state 28 有旋转反而成功，旋转不是因素。
2. **失败机制佐证**：模型对该特定构型存在轻微接近偏差，开环 chunk 执行无法纠正 → 长时间游移对齐后超时（与 5.3 的 gripper 曲线一致）。2cm 偏移恰好落在策略的"坏盆"里。
3. **对 M3 的启示**：专家示教无需处理特殊几何，只需对该构型及其 ±2cm 扰动演示正确的"正上方接近→抓取→抬升→搬到盘 (≈0.07, 0.20, 0.90)→释放"轨迹。

### M2 微调数据管线调研（决定 M3 存储格式）

**已定位**（2026-09-02 调研）：

- 微调入口脚本：`vla-scripts/finetune.py`（根目录没有 finetune.py，注意路径）

- 数据管线：RLDS/OXE 体系，位于 `prismatic/vla/datasets/rlds/`

  - `oxe/configs.py`：各数据集的 TFDS 加载配置

  - `oxe/mixtures.py`：数据混合定义（含 libero 相关 mixture）

  - `oxe/transforms.py`：观测/动作变换

- 结论方向：微调走 **TFDS/RLDS 格式**，非 HDF5。因此 M3 采集的示教需要先存 HDF5（robosuite 原生），再转成 TFDS 并注册自定义 mixture。

**调研结论**（2026-09-02 已读代码确认）：

1. **微调入口** `vla-scripts/finetune.py`，`FinetuneConfig` 关键项：

   - `data_root_dir` 默认 `datasets/rlds`；`dataset_name` 指定 TFDS 数据集名

   - `use_lora=True`、`lora_rank=32`（与 checkpoint 一致）

   - `use_proprio` 默认 **False** → 微调 OFT 时必须设 True（checkpoint 用 proprio，PROPRIO\_DIM=8）

   - `num_images_in_input=1`；`image_aug=True`（官方强烈推荐）

   - 官方默认 batch\_size=8 / lr=5e-4 / max\_steps=200k；微实验需调小 batch（2~~4）、max\_steps（1~~2k）

   - `merge_lora_during_training=True`：训练完自动合并 LoRA 保存，复测可直接用
2. **数据格式为 TFDS/RLDS**：`dataset.py` 用 `tfds.builder(name, data_dir=data_dir)` 加载，
   即数据集必须以 TFDS 格式放在 `data_root_dir/<dataset_name>` 下。
3. **新示教必须遵循的 schema**（照抄 `libero_spatial_no_noops` 配置，configs.py L645-651）：

   - image\_obs\_keys：primary=`image`，wrist=`wrist_image`

   - state\_obs\_keys：`["EEF_state", "gripper_state"]`

   - state\_encoding=`POS_EULER`，action\_encoding=`EEF_POS`
4. **现成 mixtures**（mixtures.py L197-213）：`libero_spatial_no_noops`、`libero_4_task_suites_no_noops` 等。

**因此 M3/M4 管线**：示教存 HDF5 → 转 TFDS（按上述 schema）→ 在 configs.py 注册自定义数据集 +
mixtures.py 注册混合（新示教 + 原 `libero_spatial_no_noops`）→ `finetune.py` 设
`dataset_name=<自定义 mixture>`、`use_proprio=True`。

**剩余问题已解决**（LIBERO.md 确认）：

1. **原版 RLDS 数据**：HF 数据集 `openvla/modified_libero_rlds`（约 10G，含 4 个 suite 的 `_no_noops` 版）。
   官方下载方式 `git clone git@hf.co:datasets/openvla/modified_libero_rlds`；
   我们用 hf-mirror 部分下载（只要 spatial）：

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   huggingface-cli download openvla/modified_libero_rlds --repo-type dataset \
     --include "libero_spatial_no_noops/*" \
     --local-dir /mnt/workspace/datasets/rlds/libero_spatial_no_noops
   ```

   当前 `datasets/rlds` 不存在，必须先下载。
2. **官方微调命令关键差异**（LIBERO.md L99-120，修正此前假设）：

   - `--num_images_in_input 2`（第三人称 + wrist，不是 1！必须与 checkpoint 一致）

   - `--use_proprio True`

   - 显存：batch8 ≈ 62G，**batch1 ≈ 25G** → A10 24G 紧张，用 batch1 + `grad_accumulation_steps 4`

   - 官方超参：lr 5e-4、max\_steps 150005、100K 步 10x 衰减、L1 loss 目标 <0.01

   - `merge_lora_during_training` 峰值显存高且慢 → 微实验设 False，训后用
     `vla-scripts/merge_lora_weights_and_save.py` 离线合并

   - **务必在与训练相同的 GPU 上测试**（官方警告跨设备性能下降）

### M3 专家示教采集（state 29 及邻域 28\~32）

- **对象**：state 29 + 邻域 28/30/31/32（天然 ±2cm 扰动，M1 已测位姿），每状态 2 条

- **方法**：脚本策略 7 阶段（上方接近→下降→抓取→抬升→搬运→释放→撤离），OSC 位置控制，
  仅保存成功示教（碗落在盘上），失败自动丢弃

- **记录**：`image`（第三人称）+ `wrist_image` + `eef_state`（6 维 POS\_EULER）+ `gripper_state` + 7 维 action，
  与 `libero_spatial_no_noops` schema 对齐

- **结果**：**7/10 成功**（state 29×2、30×1、31×2、32×2），存 `/mnt/workspace/demos_state29.hdf5`

**调试踩坑（3 个，均已修复）**：

1. **body 名带** **`_main`** **后缀**：MuJoCo body 名是 `akita_black_bowl_1_main` 不是 `akita_black_bowl_1`；
   EEF body 是 `gripper0_eef`（`grip_site` 不存在）。
2. **观测键名**：robosuite 1.4.1 是 `agentview_image` / `robot0_eye_in_hand_image`，不是 `_rgb` 后缀。
3. **动作缩放（关键）**：OSC 控制器输入是归一化 \[-1,1]（映射 ±0.05m/步）。
   直接喂原始 delta 会导致每步只走 2.5mm，永远到不了碗。
   修复：`cmd = clip(delta / 0.05, -1, 1)`。

**最终采集脚本** `/mnt/workspace/collect_demos.py`：

```python
"""M3: 脚本策略在 state 29 及邻域(28~32)采集专家示教（OSC 归一化动作版）"""
import numpy as np, h5py
from transforms3d.euler import quat2euler
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

STATES = [28, 29, 30, 31, 32]
DEMOS_PER_STATE = 2
OUT = "/mnt/workspace/demos_state29.hdf5"
EEF_BODY = "gripper0_eef"
ACT_SCALE = 0.05   # OSC: 输入[-1,1] -> 每步 ±0.05m

def obj_pos(env, name):
    sim = env.env.sim
    for cand in (name + "_main", name):
        try:
            return np.array(sim.data.xpos[sim.model.body_name2id(cand)])
        except ValueError:
            continue
    raise ValueError(f"body {name} not found")

def eef_pos(env):
    return np.array(env.env.sim.data.xpos[env.env.sim.model.body_name2id(EEF_BODY)])

def eef_state(env):
    sim = env.env.sim
    pos = np.array(sim.data.xpos[sim.model.body_name2id(EEF_BODY)])
    quat = np.array(sim.data.xquat[sim.model.body_name2id(EEF_BODY)])
    return np.concatenate([pos, np.array(quat2euler(quat))])

def gripper_state(env):
    sim = env.env.sim
    robot = env.env.robots[0]
    idx = [sim.model.joint_name2id(j) for j in robot.gripper.joints]
    return np.array([sim.data.qpos[i] for i in idx])

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(4)
init_states = task_suite.get_task_init_states(4)
env, task_desc = get_libero_env(task, "openvla", resolution=256)

KEYS = {}
rec = None
def step(action):
    obs, r, done, _ = env.step(action)
    if not KEYS:
        KEYS["third"] = next(k for k in ["agentview_image", "agentview_rgb"] if k in obs)
        KEYS["wrist"] = next(k for k in ["robot0_eye_in_hand_image", "robot0_eye_in_hand_rgb"] if k in obs)
    rec["image"].append(obs[KEYS["third"]][::-1])
    rec["wrist_image"].append(obs[KEYS["wrist"]][::-1])
    rec["eef_state"].append(eef_state(env))
    rec["gripper_state"].append(gripper_state(env))
    rec["action"].append(action)

def move_to(target, n_steps, grip):
    for _ in range(n_steps):
        delta = np.clip(target - eef_pos(env), -0.05, 0.05)
        cmd = np.clip(delta / ACT_SCALE, -1, 1)          # 归一化到 OSC 输入范围
        step(np.concatenate([cmd, [0, 0, 0], [grip]]))

f = h5py.File(OUT, "w")
data_grp = f.create_group("data")
n_saved = 0

for sid in STATES:
    for trial in range(DEMOS_PER_STATE):
        env.reset()
        obs = env.set_init_state(init_states[sid])
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(7))

        bowl_p = obj_pos(env, "akita_black_bowl_1")
        plate_p = obj_pos(env, "plate_1")
        rec = {"image": [], "wrist_image": [], "eef_state": [], "gripper_state": [], "action": []}

        move_to(bowl_p + [0, 0, 0.15], 40, 1)     # 上方接近
        move_to(bowl_p + [0, 0, 0.03], 30, 1)     # 下降
        for _ in range(15):                       # 抓取
            step(np.concatenate([np.zeros(3), [0, 0, 0], [-1]]))
        move_to(bowl_p + [0, 0, 0.25], 40, -1)    # 抬升
        move_to(plate_p + [0, 0, 0.15], 60, -1)   # 搬运
        move_to(plate_p + [0, 0, 0.06], 30, -1)   # 下降
        for _ in range(15):                       # 释放
            step(np.concatenate([np.zeros(3), [0, 0, 0], [1]]))
        move_to(plate_p + [0, 0, 0.20], 20, 1)    # 撤离

        bp, pp = obj_pos(env, "akita_black_bowl_1"), obj_pos(env, "plate_1")
        ok = (np.linalg.norm(bp[:2] - pp[:2]) < 0.06) and (pp[2] < bp[2] < pp[2] + 0.15)
        print(f"state {sid} trial {trial}: success={ok} bowl={np.round(bp,3)}")
        if not ok:
            continue
        g = data_grp.create_group(f"demo_{n_saved}")
        g.attrs["init_state_id"] = sid
        for k, v in rec.items():
            g.create_dataset(k, data=np.array(v))
        n_saved += 1

f.attrs["num_demos"] = n_saved
f.close()
print(f"saved {n_saved} demos -> {OUT}")
```

运行：

```bash
cd /mnt/workspace/openvla-oft
PYTHONPATH=/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python /mnt/workspace/collect_demos.py
```

**采集结果明细**：

| state  | trial 0     | trial 1 |
| ------ | ----------- | ------- |
| 28     | 失败（抓取微偏，丢弃） | 失败（丢弃）  |
| **29** | **成功**      | **成功**  |
| 30     | 失败（丢弃）      | 成功      |
| 31     | 成功          | 成功      |
| 32     | 成功          | 成功      |

### M4 LoRA 微调（完整流程）

**微调流程总览**：

```
M3 示教 HDF5
   │  M4-1: 写 TFDS builder（RLDS schema）
   ▼
state29_builder.py
   │  M4-2: 注册 builder + 注入 configs/mixtures 条目
   ▼
oxe 数据管线可识别 "state29_demos"
   │  M4-3: 构建 TFDS 数据集
   ▼
/mnt/workspace/datasets/rlds/state29_demos/1.0.0/
   │  M4-4: LoRA 微调（在 OFT checkpoint 基础上继续训）
   ▼
/mnt/workspace/runs/<run_id>/checkpoints/
   │  M4-5: 合并 LoRA 权重（merge_lora_weights_and_save.py）
   ▼
完整微调模型 → M5 复测
```

#### M4-1：写 TFDS 转换 builder（RLDS schema，与 `libero_spatial_no_noops` 对齐）

创建 `prismatic/vla/datasets/rlds/oxe/state29_builder.py`：

```python
"""自定义 TFDS builder：把 /mnt/workspace/demos_state29.hdf5 转成 RLDS 格式"""
import h5py
import numpy as np
import tensorflow_datasets as tfds

TASK = "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
HDF5 = "/mnt/workspace/demos_state29.hdf5"

class State29Demos(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "state29 expert demos for DAgger micro-experiment."}

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "discount": tfds.features.Scalar(np.float32),
                    "is_first": tfds.features.Scalar(np.bool_),
                    "is_last": tfds.features.Scalar(np.bool_),
                    "is_terminal": tfds.features.Scalar(np.bool_),
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8),
                        "wrist_image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8),
                        "EEF_state": tfds.features.Tensor(shape=(6,), dtype=np.float32),
                        "gripper_state": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                    }),
                    "reward": tfds.features.Scalar(np.float32),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "language_instruction": tfds.features.Text(),
                }),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        with h5py.File(HDF5, "r") as f:
            for i in range(int(f.attrs["num_demos"])):
                d = f["data"][f"demo_{i}"]
                n = len(d["action"])
                steps = []
                for t in range(n):
                    steps.append({
                        "action": d["action"][t].astype(np.float32),
                        "discount": 1.0,
                        "is_first": t == 0,
                        "is_last": t == n - 1,
                        "is_terminal": t == n - 1,
                        "observation": {
                            "image": d["image"][t].astype(np.uint8),
                            "wrist_image": d["wrist_image"][t].astype(np.uint8),
                            "EEF_state": d["eef_state"][t].astype(np.float32),
                            "gripper_state": d["gripper_state"][t].astype(np.float32),
                        },
                        "reward": 1.0 if t == n - 1 else 0.0,
                    })
                yield i, {"steps": steps,
                          "episode_metadata": {"language_instruction": TASK}}
```

#### M4-2：注册 builder + 注入 config/mixture 条目（自动补丁，幂等）

```bash
cd /mnt/workspace/openvla-oft

# 1) 在 oxe/__init__.py 注册 import
grep -q state29_builder prismatic/vla/datasets/rlds/oxe/__init__.py || \
  echo "from . import state29_builder  # noqa" >> prismatic/vla/datasets/rlds/oxe/__init__.py

# 2) 在 configs.py / mixtures.py 注入条目（插到 libero_spatial_no_noops 条目前）
python - <<'PYEOF'
CONFIG_BLOCK = '''"state29_demos": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    '''
MIX_BLOCK = '''"state29_demos": [
        ("state29_demos", 1.0),
    ],
    '''
anchor = '"libero_spatial_no_noops"'
for path, block, marker in [
    ("prismatic/vla/datasets/rlds/oxe/configs.py", CONFIG_BLOCK, '"state29_demos": {'),
    ("prismatic/vla/datasets/rlds/oxe/mixtures.py", MIX_BLOCK, '"state29_demos": ['),
]:
    src = open(path).read()
    if marker in src:
        print(f"{path}: already patched")
        continue
    idx = src.index(anchor)
    src = src[:idx] + block + src[idx:]
    open(path, "w").write(src)
    print(f"{path}: patched")
PYEOF
```

#### M4-3：构建 TFDS 数据集

```bash
cd /mnt/workspace/openvla-oft
PYTHONPATH=/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO python -c "
import tensorflow_datasets as tfds
from prismatic.vla.datasets.rlds.oxe import state29_builder
b = tfds.builder('state29_demos', data_dir='/mnt/workspace/datasets/rlds')
b.download_and_prepare()
print(b.info)
"
```

预期：7 examples，生成 `/mnt/workspace/datasets/rlds/state29_demos/1.0.0/`。

#### M4-4：LoRA 微调（先只用新示教，验证管线）

```bash
cd /mnt/workspace/openvla-oft
PYTHONPATH=/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO \
WANDB_MODE=offline \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial \
  --data_root_dir /mnt/workspace/datasets/rlds \
  --dataset_name state29_demos \
  --run_root_dir /mnt/workspace/runs \
  --use_l1_regression True --use_diffusion False --use_film False \
  --num_images_in_input 2 --use_proprio True \
  --batch_size 1 --grad_accumulation_steps 4 \
  --learning_rate 2e-5 --max_steps 1000 --save_freq 500 \
  --save_latest_checkpoint_only False \
  --image_aug True --lora_rank 32 \
  --merge_lora_during_training False \
  --run_id_note dagger-state29-micro
```

**关键参数说明**：

| 参数                                           | 值               | 说明                                       |
| -------------------------------------------- | --------------- | ---------------------------------------- |
| `--vla_path`                                 | OFT checkpoint  | 在现有策略基础上继续训（DAgger 精神：修正当前策略）            |
| `--dataset_name`                             | `state29_demos` | 仅用新示教（7 条），验证管线                          |
| `--batch_size 1 + grad_accumulation_steps 4` | 有效 batch 4      | A10 24G 显存紧，官方默认 25.1G 会顶格               |
| `--learning_rate 2e-5`                       | 保守              | 7 条数据，防过拟合                               |
| `--max_steps 1000`                           | ≈2\~3 epoch     | 每条约 350 步，7 条 ≈ 2450 步/epoch，1000 步为部分遍历 |
| `--lora_rank 32`                             | 与 checkpoint 一致 | **必须匹配**，否则加载失败                          |
| `--image_aug True`                           | 开启              | 小数据增广，提升泛化                               |
| `--merge_lora_during_training False`         | 训练时不合并          | 省峰值显存，训后离线合并                             |

**显存预算**：模型 15.13G + LoRA 训练增量（梯度/优化器仅作用于 LoRA 参数）≈ 20\~22G，
A10 24G 可容纳但接近上限。若 OOM，加环境变量
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 或降 `--max_steps`。

**后续扩展（验证管线通过后）**：混合原版 `libero_spatial_no_noops` 数据（下载自
`gs://gresearch/open-x-embodiment`）一起训，防灾难性遗忘。

#### M4-5：合并 LoRA 权重（训后）

```bash
cd /mnt/workspace/openvla-oft
PYTHONPATH=/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO python scripts/merge_lora_weights_and_save.py \
  --model_path /mnt/workspace/runs/<run_id>/checkpoints/<step>/ \
  --save_path /mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-dagger-state29
```

合并后的完整模型即可用标准评估命令加载（`--pretrained_checkpoint` 指向合并路径）。

#### M4-4 最终可用的微调启动命令（已含全部修复）

```bash
cd /mnt/workspace/openvla-oft
nohup env PYTHONPATH=/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO \
  WANDB_MODE=offline OMP_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path /mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial \
    --data_root_dir /mnt/workspace/datasets/rlds \
    --dataset_name state29_demos \
    --run_root_dir /mnt/workspace/runs \
    --use_l1_regression True --use_diffusion False --use_film False \
    --num_images_in_input 2 --use_proprio True \
    --batch_size 1 --grad_accumulation_steps 4 \
    --learning_rate 2e-5 --max_steps 1000 --save_freq 500 \
    --save_latest_checkpoint_only False \
    --image_aug True --lora_rank 32 \
    --merge_lora_during_training False \
    --shuffle_buffer_size 1000 \
    --run_id_note dagger-state29-micro \
    > /mnt/workspace/finetune_state29.log 2>&1 &
echo "PID: $!"
```

监控：`tail -f /mnt/workspace/finetune_state29.log` + `watch -n 5 nvidia-smi`

### M4 踩坑排查全记录（12 个问题，按出现顺序）

> 排查总思路：**先看真实异常，不看外壳**。`torchrun` 只报 `exitcode`，
> 真正的 Python 异常在日志里，用 `grep -n -B5 -A30 "^Traceback" 日志` 抓。
> `exitcode` 含义：`1`=代码异常；`2`=参数解析错；`-9`(SIGKILL)=被系统杀（多为内存）。

| #  | 症状                                       | 根因                                                                              | 解决方案                                                          |
| -- | ---------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1  | `No module named 'tensorflow_datasets'`  | 虚拟环境没激活（提示符无 `(venv_oft)`），用了系统 Python                                          | `source /mnt/workspace/activate_oft.sh`                       |
| 2  | `No module named '...features_dict'` 等   | TFDS 4.9.3 用懒加载包整个包，**子模块路径不是真实可 import 文件**                                    | 改用顶层访问器 `tfds.features.FeaturesDict` 等                        |
| 3  | `No module named '...scalar_feature'`    | `Scalar` 实际在 `scalar` 模块，且子模块导入本就不可用                                            | 同上，统一用 `tfds.features.Scalar`                                 |
| 4  | 卡 61 秒 + GCS/Google auth 报错              | TFDS 检查 Google Cloud 有无同名托管数据集，阿里云访问不了                                          | `gcs_utils._is_gcs_disabled = True`（超时后其实也会自动继续）              |
| 5  | `KeyError: 'state29_demos'`（mixtures）    | 数据注册表查不到                                                                        | M4-2 补丁注入 `mixtures.py` + `configs.py`                        |
| 6  | `KeyError: 'state29_demos'`（transforms）  | 漏了**第三个注册表** `OXE_STANDARDIZATION_TRANSFORMS`                                   | 复用 `libero_dataset_transform` 注册 `state29_demos`              |
| 7  | `KeyError: 'state'`                      | `libero_dataset_transform` 期望合并的 8 维 `state` 键，我们存了拆开的 EEF/gripper              | builder 改为 `concat(eef_state(6), gripper_state(2))` 存 `state` |
| 8  | `missing keys: {'language_instruction'}` | 该键必须在 **step 级**（原版 LIBERO 每步都有），我们只存在 `episode_metadata`                       | builder 在每个 step 里加 `language_instruction`                    |
| 9  | （隐患）夹爪动作符号相反                             | 原版 LIBERO 约定 `-1=开 +1=关`，我们录的是 robosuite 运行时 `+1=开 -1=关`                        | builder 存 `-gripper` 翻转，否则模型学到反向夹爪                            |
| 10 | `exitcode -9`（SIGKILL）                   | **主机内存 OOM**：`shuffle_buffer_size=100000`，每帧 2×256×256×3≈390KB，10 万帧≈39GB > 28G | `--shuffle_buffer_size 1000`（数据总共才 \~2450 帧）                  |
| 11 | GPU 显存 OOM（跑出第一步 loss 后）                 | 轨迹长度不一，激活值峰值超 24G + 碎片化                                                         | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`            |
| 12 | `exitcode 2`                             | `--load_in_8bit` 不是 `finetune.py` 的有效参数（是评估脚本的）                                 | 去掉该参数                                                         |

**关键教训**：

- **自定义数据集接入 OXE 管线要动 3 个注册表**：`configs.py`（观测/动作编码）、
  `mixtures.py`（采样权重）、`transforms.py`（标准化变换），缺一不可。

- **数据格式必须严格对齐** **`libero_spatial_no_noops`**：8 维合并 `state`、step 级
  `language_instruction`、夹爪符号原始约定。最稳的做法是先读 `transforms.py` 里
  `libero_dataset_transform` 的期望，反推 builder 该存什么。

- **内存类崩溃先看** **`exitcode`**：`-9` 查主机内存（`shuffle_buffer`/并行 worker），
  GPU OOM 查激活值（`expandable_segments`/关 `image_aug`/降 batch）。

- 容器环境限制：`swapon` 不允许、`fallocate` 不支持（overlay FS），别在 swap 上浪费时间。

### M5 复测与判定

- 与阶段 1 **完全相同命令**（`--seed 7`）复测：

  - state 29（目标）→ 期望转成功

  - 邻域 state 28/30/31/32 → 期望不退化

  - 抽测 2 个其他任务各 10 trials → 期望不降级

- 成功标准：state 29 成功 且 抽测无退化；否则回滚 LoRA 权重，结论记为"微实验未通过"

### 微实验产出

无论成败，最终更新本文档：state 29 几何特征、示教条数、微调超参、复测对比表，作为"DAgger 修复确定性难例"的最小可行案例记录。

***

## 七、全新实例一键复现脚本

将以下内容保存为 `/mnt/workspace/setup_fresh.sh`，新实例上执行 `bash /mnt/workspace/setup_fresh.sh` 即可。
所有版本钉死 + 关键包 `--no-deps`，规避了本文档第三节的全部坑。

```bash
#!/bin/bash
# =============================================================
# setup_fresh.sh — OpenVLA-OFT LIBERO 环境一键复现
# 适用：魔搭免费 A10 实例，ModelScope 官方镜像 Python 3.10
# 用法：bash /mnt/workspace/setup_fresh.sh
# =============================================================
set -e
MIRROR="https://mirrors.cloud.aliyuncs.com/pypi/simple"
WS=/mnt/workspace

echo "=== [0/9] 虚拟环境 ==="
python -m venv $WS/venv_oft
source $WS/venv_oft/bin/activate
pip install -U pip -i $MIRROR

echo "=== [1/9] PyTorch + 锁 numpy（防被后续包升到 2.x）==="
pip install -i $MIRROR torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install -i $MIRROR "numpy==1.26.4"

echo "=== [2/9] flash-attn 预编译轮子（失败自动降级 SDPA）==="
pip install packaging ninja
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  || echo "[warn] flash-attn 安装失败，稍后打 SDPA 补丁"

echo "=== [3/9] 克隆代码 ==="
cd $WS
[ -d openvla-oft ] || git clone https://github.com/openvla/openvla-oft.git
[ -d LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

echo "=== [4/9] 主仓库 + forks（全部 --no-deps 防依赖漂移）==="
cd $WS/openvla-oft && pip install -e . --no-deps -i $MIRROR
pip install --no-deps git+https://github.com/moojink/transformers-openvla-oft.git
pip install --no-deps git+https://github.com/moojink/dlimp_openvla
# flash-attn 没装上则打 SDPA 补丁（幂等）
python -c "import flash_attn" 2>/dev/null || \
  sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/' \
    $WS/openvla-oft/experiments/robot/openvla_utils.py

echo "=== [5/9] 手动补依赖（全部为今日验证过的版本）==="
pip install -i $MIRROR \
  einops "timm==0.9.10" "tokenizers==0.19.1" "huggingface_hub>=0.23" safetensors regex \
  accelerate "sentencepiece==0.1.99" "draccus==0.8.0" h5py pillow tqdm pyyaml rich \
  "peft==0.11.1" imageio imageio-ffmpeg "opencv-python-headless==4.9.0.80" \
  "matplotlib==3.8.4" "wandb==0.16.6" json-numpy jsonlines filelock requests \
  trimesh termcolor "diffusers==0.30.3"

echo "=== [6/9] TensorFlow 栈（protobuf 4.25.3 + 旧 gencode tf-metadata）==="
pip install -i $MIRROR tensorflow-cpu==2.15.0
pip install --no-deps -i $MIRROR \
  "tensorflow_datasets==4.9.3" "tensorflow-metadata==1.15.0" "tensorflow_graphics==2021.12.3"
pip install --no-deps --no-cache-dir -i $MIRROR "protobuf==4.25.3"

echo "=== [7/9] LIBERO + 仿真栈（mujoco 必须 3.3.2）==="
cd $WS/LIBERO && pip install -e . --no-deps -i $MIRROR
pip install -i $MIRROR "robosuite==1.4.1" bddl "gym==0.25.2" hydra-core \
  "mujoco==3.3.2" numba scipy easydict
sudo apt-get install -y libosmesa6-dev libgl1-mesa-dev libegl1-mesa-dev || true

echo "=== [8/9] 模型权重（hf-mirror 加速，16G）==="
pip install -U huggingface_hub -i $MIRROR
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-spatial \
  --local-dir $WS/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial
du -sh $WS/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial   # 应 >= 14G

echo "=== [9/9] 激活脚本 + 总验证 ==="
cat > $WS/activate_oft.sh <<'EOF'
source /mnt/workspace/venv_oft/bin/activate
export PYTHONPATH=/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO:$PYTHONPATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=offline
EOF
source $WS/activate_oft.sh

cd $WS/openvla-oft
# LIBERO 首次导入会询问数据集目录：回答 Y + /mnt/workspace/libero_datasets
printf 'Y\n/mnt/workspace/libero_datasets\n' | python -c "
import numpy, cv2, torch, google.protobuf as pb
import tensorflow as tf, dlimp, robosuite, mujoco
from libero.libero import benchmark
from experiments.robot.openvla_utils import get_vla
print('numpy', numpy.__version__, '| cv2', cv2.__version__)
print('torch', torch.__version__, torch.cuda.is_available())
print('tf', tf.__version__, '| protobuf', pb.__version__, '| mujoco', mujoco.__version__)
print('ALL OK')
"
echo "完成！日常使用：source /mnt/workspace/activate_oft.sh"
```

### 使用说明

1. **新实例创建后**：上传本脚本到 `/mnt/workspace`，`bash setup_fresh.sh`，全程约 30\~60 分钟（主要是权重下载）。
2. **日常激活**：`source /mnt/workspace/activate_oft.sh`（PYTHONPATH、OSMesa、镜像源全部自动设好）。
3. **若权重已存在于旧实例磁盘**（同磁盘复用时）：跳过 \[8/9]。
4. **LIBERO 数据集提问**：脚本已自动应答；若手动遇到，回答 `Y` + `/mnt/workspace/libero_datasets`。
5. **pip 警告忽略清单**：`fastapi / uvicorn / OpenEXR / tensorflow-addons` 缺失警告不影响评估与微调。

***

## 八、常用命令速查

```bash
# 激活
source /mnt/workspace/activate_oft.sh

# 定向评估（示例：top drawer 50 个初始状态）
cd /mnt/workspace/openvla-oft
TARGETS=$(python -c "print(','.join(f'4:{i}' for i in range(50)))")
nohup env PYTHONPATH=$PYTHONPATH MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --target_episodes "$TARGETS" \
    --num_steps_wait 10 --num_open_loop_steps 8 \
    --local_log_dir /mnt/workspace/eval_logs \
    --run_id_note topdrawer50 --seed 7 \
    > /mnt/workspace/eval_logs/topdrawer50.log 2>&1 &
tail -f /mnt/workspace/eval_logs/topdrawer50.log

# 结果与视频位置
ls /mnt/workspace/eval_logs/EVAL-*.txt        # 逐条结果
ls /mnt/workspace/openvla-oft/rollouts/*/     # 每 episode 视频（文件名含 success 标记）

# 各任务成功率统计（解析视频文件名）
python /mnt/workspace/analyze_rollouts.py
```

***

## 九、新实例迁移与复测问题记录（2026-09-03，dsw-2150895）

> 背景：魔搭免费额度用尽，数据迁移到新实例继续。新实例镜像自带 torch 2.13.0+cu130、
> numpy 2.2.6 等**高版本包**，与旧环境（torch 2.2.0+cu121 / numpy 1.26.4）完全不同，
> 导致合并权重（M4-5）阶段连续踩坑。以下按出现顺序记录。

### 为什么新实例会重新踩坑

- 旧实例的 `venv_oft` 是**自己创建的虚拟环境**，所有包版本钉死，所以 50 trials 评估一次通过。

- 新实例是**全新容器 + 全新磁盘**（魔搭免费实例与付费实例的 NAS 不共享），只搬来了
  checkpoint 和代码，**没有搬环境的** **`site-packages`**（20G 太大，不值得搬）。

- 新镜像预装的包版本很新（torch 2.13 / numpy 2.2），和 openvla-oft 要求的组合
  （torch 2.2 / numpy 1.26）冲突 → 需要手动逐一对齐。

### 问题清单（15 个，按出现顺序）

| #  | 症状                                                                                                         | 根因                                                                                                    | 为什么必须解决                                             | 解决方案                                                                                                                                                |
| -- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | `bash: setup_fresh.sh: 没有那个文件或目录` + `tar: lora_bundle.tar.gz 无法打开`                                         | 新实例是全新磁盘，脚本和备份包都没上传                                                                                   | 没有脚本无法重建环境，没有 tar 包无法恢复 checkpoint                  | JupyterLab 上传 `lora_bundle.tar.gz`（1.5G）到 `/mnt/workspace`，再 `tar xzf`                                                                              |
| 2  | `pip install` 报 `SSLError: hostname 'mirrors.cloud.aliyuncs.com' doesn't match 'mirrors-ssl.aliyuncs.com'` | 阿里云内网域名的 SSL 证书只覆盖 `mirrors-ssl.aliyuncs.com` 和 `mirrors.aliyun.com`，不覆盖 `mirrors.cloud.aliyuncs.com` | 内网源不可用，装不了任何包                                       | 所有源改成 `pip install -i https://mirrors-ssl.aliyuncs.com/pypi/simple`（文档第七节 `setup_fresh.sh` 已同步更新）                                                   |
| 3  | torch 报 `NVIDIA driver too old (found version 12040)`                                                      | 新镜像自带 torch 2.13.0+cu130 要求 CUDA 13 驱动，实例只有 **12.4**                                                  | 模型无法上 GPU，合并/评估都跑不了                                 | 装 torch **2.5.1+cu124**（兼容 CUDA 12.4 驱动）：`pip install -i https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1` |
| 4  | 装完 torch 后 `numpy 2.2.6` 又被带上                                                                              | torch 2.5.1 默认依赖 numpy>=2                                                                             | tensorflow 2.15 和 wandb 0.16 都要求 numpy 1.x，2.x 会直接崩 | 强制降回 `pip install --force-reinstall "numpy==1.26.4"`                                                                                                |
| 5  | merge 报 `unrecognized arguments: --vla_path`                                                               | 合并脚本实际参数名是 `--base_checkpoint` / `--lora_finetuned_checkpoint_dir`，不是 `--vla_path` / `--lora_path`    | 参数名不对脚本直接退出                                         | 先用 `python vla-scripts/merge_lora_weights_and_save.py --help` 查真实参数名，再运行                                                                            |
| 6  | merge 输出目录报 `FileNotFoundError`                                                                            | `--config_path` 指向的路径不存在                                                                              | 脚本要写输出但目录不存在                                        | 先 `mkdir -p` 输出目录                                                                                                                                   |
| 7  | merge 强制调用 `.to("cuda")`，但之前的 CPU sed 改动没生效                                                                | 脚本内部写死 `.to("cuda")`                                                                                  | 合并操作本身不需要 GPU，但脚本强制走 GPU                            | 临时 `sed -i 's/.to("cuda")/.to("cpu")/g'` 改 cpu 版合并（有一定风险，见 #8）；**最终不需要**，因为装了 torch 2.5.1+cu124 后 CUDA 可用                                           |
| 8  | flash\_attn 报 `undefined symbol: _ZN3c104cuda9SetDeviceEi`                                                 | flash\_attn 2.5.8 是为 torch 2.2.0+cu121 编译的，与 torch 2.5.1+cu124 **二进制不兼容**                             | 加载模型时 flash\_attn 直接崩                               | 卸载 flash\_attn、打 SDPA 补丁（`sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/g'`）。**SDPA 与 flash-attn 数值等效，评估结果不受影响**     |
| 9  | `Saved merged model checkpoint at: <lora dir>`                                                             | 合并脚本把合并后的完整模型写**回原 LoRA 目录**                                                                          | 评估脚本 `--pretrained_checkpoint` 指向该目录即可加载合并模型        | 直接用合并后的目录作为 `--pretrained_checkpoint`                                                                                                               |
| 10 | 评估报 `No module named 'imageio'`                                                                            | 新实例没装评估用的视频录制库                                                                                        | 评估脚本 `import imageio` 录视频                           | `pip install imageio imageio-ffmpeg`                                                                                                                |
| 11 | 评估报 `No module named 'robosuite'`                                                                          | 新实例没装仿真栈                                                                                              | LIBERO 环境需要 robosuite                               | `pip install "robosuite==1.4.1" bddl "gym==0.25.2" "mujoco==3.3.2"`（mujoco 必须 3.3.2）                                                                |
| 12 | 评估报 `np.float_ was removed in NumPy 2.0`                                                                   | wandb 0.16 调用 numpy 1.x 才有的 `np.float_`                                                               | wandb 导入即崩                                          | **再次**降级 numpy 1.26.4（torch 2.5.1 安装时把 numpy 带上来了）                                                                                                  |
| 13 | 评估报 `OpenGL AttributeError: 'NoneType' object has no attribute 'glGetError'`                               | PyOpenGL 找不到系统 GL 库（新容器没装）                                                                            | 渲染后端无法初始化                                           | `sudo apt-get install -y libgl1-mesa-dev libosmesa6-dev libglfw3`                                                                                   |
| 14 | 评估报 `ModuleNotFoundError: No module named 'diffusers'`                                                     | 新实例没装 diffusers                                                                                       | `action_heads.py` 导入时就要求 diffusers                  | `pip install "diffusers==0.30.3"`                                                                                                                   |
| 15 | 评估报 `unrecognized arguments: --target_episodes`                                                            | **新实例的 openvla-oft 是官方原版**，旧实例打了补丁才有 `--target_episodes` 参数                                           | 无法像旧实例那样只测指定 state                                  | 方案 A：直接用 `--num_trials_per_task 50` 跑全量（新实例评估脚本更旧，无该参数）；或重新打补丁                                                                                      |

### 关键经验总结

1. **跨实例迁移时，环境（venv/site-packages）不值得搬，但要带一份"依赖清单"**：
   在旧实例执行 `pip freeze > requirements.txt` 保存，新实例用 `pip install -r requirements.txt` 还原
   （版本钉死可避免 #3/#4/#12 这类版本不匹配问题）。
2. **新镜像的"预装包"是最大的坑**：torch/numpy 等被镜像锁在高版本，必须主动
   `--force-reinstall` 降级到项目要求的版本，否则会出现各种 ABI/API 不兼容（#3/#4/#8/#12）。
3. **先** **`--help`** **再传参**：任何脚本参数名以 `--help` 输出为准，不要凭记忆或旧文档传参（#5）。
4. **flash-attn 是版本敏感组件**：它按 (torch 版本, CUDA 版本) 精确编译，升级 torch 后
   必须同步换 flash-attn 轮子或直接禁用改 SDPA（#8）。
5. **官方仓库 vs 打过补丁的仓库是两份代码**：迁移时若只搬 checkpoint 不搬代码补丁
   （如 `--target_episodes`），新实例会退回官方行为。要么把旧实例 `openvla-oft` 整个目录
   打包带走，要么重新打补丁（#15）。

### M5 复测连环踩坑（norm\_stats 注入 + 全量评估，2026-09-03 下午）

> 本节记录合并权重（M4-5 成功）后、复测评估（M5）阶段遇到的 3 个问题，
> **其中第 1 个问题直接导致 31/31 全部失败**，是本次实验中最隐蔽的坑。

#### 踩坑 16：`AssertionError: Action un-norm key libero_spatial not found in VLA norm_stats`

**现象**：合并权重完成后，运行评估脚本 `run_libero_eval.py`，模型加载成功但立即报：
`assert unnorm_key in model.norm_stats` → libero\_spatial 不在 norm\_stats 里。

**排查思路（3 步）**：

1. 读合并后 checkpoint 的 `config.json`，发现 `norm_stats` 只有 25 个 OXE 数据集键（austin/berkeley/...），
   **没有任何 libero 相关键**。
2. 检查发现：评估脚本加载模型时输出 `Updated config.json`——openvla-oft 的加载流程
   **首次加载会自动重写 config.json**（补齐 AutoConfig/AutoModel 映射），
   且**不会保留我们手动注入的 norm\_stats 键** → 注入无效。
3. 尝试"先注入 → 再加载验证"，发现模型加载后 `runtime norm_stats` 仍然缺 libero 键，
   说明 check\_unnorm\_key 检查的是**运行时模型内存里的 norm\_stats**（来自模型内部的
   `norm_stats` 属性，不是独立读 config.json）→ 需要在 `initialize_model` 里动态注入。

**最终解决**：给评估脚本 `run_libero_eval.py` 打补丁，在 `check_unnorm_key(cfg, model)` 调用前，
从 checkpoint 目录的 `dataset_statistics.json` 读取统计，动态注入 `model.norm_stats`：

```python
# [PATCH] 若 norm_stats 缺 libero_spatial，从 dataset_statistics.json 注入
if cfg.task_suite_name not in model.norm_stats:
    # 优先用基座 checkpoint 的原版 libero_spatial_no_noops 统计
    base_ds = "/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial/dataset_statistics.json"
    if os.path.exists(base_ds):
        bd = json.load(open(base_ds))
        for bk, bs in bd.items():
            if "libero" in bk.lower():
                model.norm_stats[cfg.task_suite_name] = bs
                break
```

**注意**：这段必须插在 `check_unnorm_key` 调用之前、`initialize_model` 返回之后。

#### 踩坑 17（致命）：91 个 episode 全失败（31 个验证 + 后续）—— norm\_stats 用了错误统计源

**现象**：补丁生效、`libero_spatial` 键存在后，评估能跑但**每个 episode 全部失败**（0%）。
日志显示任务也变样了（跑的是 `between the plate and the ramekin` 等其它任务，不是 top drawer）。

**根因（双层）**：

1. **norm\_stats 注入源错误**：第一次补丁从**微调 checkpoint** 的 `dataset_statistics.json`
   （只有 `state29_demos` 键，来自 7 条示教）取统计。这 7 条示教是用**脚本策略**采集的，
   策略只做平移不做旋转 → **欧拉角维度（3/4/5）的统计全是 0**：
   `std=[0,0,0]`、`q01=q99=[0,0,0]`。而反归一化公式是 `(raw - mean) / (q99 - q01)`，
   **除以 0 产生 inf/nan** → 输出动作全是 nan/乱飞 → 全部失败。

2. **评估范围错误**：`--num_trials_per_task 50` 会让**所有 10 个任务**各跑 50 轮
   （共 500 轮），不是只跑 top drawer。失败日志显示的是第 0 个任务（between plate and ramekin）。

**最终解决（双层修复）**：

1. **改用基座 checkpoint 的统计**（`openvla-oft-ckpts/.../dataset_statistics.json` 里的
   `libero_spatial_no_noops` 键，欧拉角维 std 正常 `[0.037, 0.072, 0.058]`）注入 `libero_spatial`。
   关键对比：

   ```text
   维度             基座 std      示教 std
   x (0)          0.413         0.321
   y (1)          0.347         0.371
   z (2)          0.509         0.499
   roll  (3)      0.037         0.000  ← 元凶
   pitch (4)      0.072         0.000  ← 元凶
   yaw   (5)      0.058         0.000  ← 元凶
   gripper (6)    0.498         0.494
   ```

2. **限制只跑 top drawer（task\_id=4）**：给评估脚本任务循环打补丁：

   ```python
   for task_id in tqdm.tqdm(range(num_tasks)):
       if task_id != 4:   # [PATCH] 只评估 top drawer
           continue
   ```

**验证通过**：重新注入后，欧拉角维 std 正确显示 `[0.037, 0.072, 0.058]`，加载正常。

#### 踩坑 18：`--target_episodes` 参数不存在（官方原版评估脚本）

**现象**：新实例的 `run_libero_eval.py` 是官方原版，没有旧实例补丁加的
`--target_episodes` 参数，传入直接 `unrecognized arguments` 退出。

**解决**：改用 `--num_trials_per_task N` 配合"任务循环过滤补丁"（踩坑 17 第 2 层的补丁）
实现等价的"只评估 top drawer"效果。

**最终验证命令**（task 4 × 3 trials 快速验证）：

```bash
cd /mnt/workspace/openvla-oft
PYTHONPATH=/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
WANDB_MODE=offline \
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /mnt/workspace/runs/openvla-7b-oft-finetuned-libero-spatial+state29_demos+b4+lr-2e-05+lora-r32+dropout-0.0--image_aug--dagger-state29-micro--1000_chkpt \
  --task_suite_name libero_spatial \
  --num_trials_per_task 3 \
  --num_steps_wait 10 --num_open_loop_steps 8 \
  --local_log_dir /mnt/workspace/eval_logs \
  --run_id_note verify3 --seed 7
```

#### 踩坑 16-18 关键教训

1. **反归一化统计（norm\_stats）必须用原版训练数据的统计，绝不能用自定义示教的统计**
   ——示教覆盖不到的动作维度（如旋转）统计为 0，反归一化时除 0 产生 nan，全部动作乱飞。
2. **模型加载后 norm\_stats 是运行时内存属性**，改 config.json 不一定生效；
   要在 `check_unnorm_key` 前动态注入。
3. **`--num_trials_per_task N`** **是"每个任务都跑 N"，不是"只跑 N 个"**。
   要只测特定任务必须改任务循环代码。
4. **脚本策略采集的示教天然缺失旋转维度**：如果后续用 DAgger 数据真正微调，
   应在脚本策略中加入随机旋转扰动，否则模型学不到旋转、评估时该维度反归一化失真。

### M5 复测最终结论（2026-09-03 晚间，旧实例历史记录）

> 本节保留旧实例的微实验记录，但迁移后无法完整复现，不能单独作为当前实例的最终结论。
> 当前实例的基线、官方 50 条专家轨迹训练结果及后续决策，以文末“2026-09-04 当前实例复盘”为准。

#### 三组对照评估结果

| 模型                 | 测试范围                  | 结果         | 结论            |
| ------------------ | --------------------- | ---------- | ------------- |
| 基座（未微调）            | top drawer 3 trials   | **3/3 成功** | 评估链路正常        |
| 微调后（7 示教 + 1000 步） | top drawer 3 trials   | 0/3        | 泛化区域崩溃        |
| 微调后（7 示教 + 1000 步） | **示教覆盖区 state 28-32** | **0/5**    | **连示教内容都没学会** |

#### 最终结论：DAgger 微实验**未通过**

1. **微调彻底破坏模型**：不仅灾难性遗忘（基座 98% → 微调 0%），
   连示教覆盖的 state 28-32 都失败 → **不是"局部过拟合"，而是微调本身失败**。
2. **正确决定：回滚到基座模型**。基座 98% 是可靠的，微调模型弃用。
3. **实验价值已达成**：

   - 完整跑通"采集示教 → TFDS → LoRA 微调 → 合并 → 复测"的 DAgger 全流程（管线验证成功）；

   - 用最小代价证明了**微实验级（7 条示教）直接 LoRA 微调会失败**——为后续真正增训提供明确反例。

#### 失败根因分析（三条叠加）

1. **示教数据与模型行为分布严重不匹配**：脚本策略的"直线段逼近"轨迹 vs 模型学过的
   "平滑分布"轨迹。7 条示教把 LoRA 权重拉向模型从未见过的分布 → 输出动作全乱。
2. **数据量太少 + 步数过多**：7 条 ≈ 2450 frame，1000 步循环十几轮 → LoRA 权重被拉爆，
   远超合理过拟合程度。
3. **示教质量未严格校验**：M3 的"成功"判定较宽松（碗落盘上即可），动作质量未经
   （平滑度/成功率）双重校验；夹爪翻转、欧拉角缺失等隐患叠加。

#### 后续真实增训的正确做法（如果要做）

- **必须混合原版数据**：`libero_spatial_no_noops`（10G）+ 新示教一起训，防灾难性遗忘；

- **数据量至少 50\~200 条**（每个难例状态 5-10 条），不能只有 7 条；

- **示教用真实专家数据**（LIBERO 自带 expert policy 或人工遥操作），
  不要用脚本直线段轨迹；

- **微调超参更保守**：lr 1e-5 以下、max\_steps 200\~500（对应数据量）；

- **评估必须混合验证**：既测示教覆盖状态，也测未覆盖状态，双重指标判断是否真正提升。

***

## 十、2026-09-04 当前实例复盘：官方 50 条专家轨迹 LoRA 微调

### 10.1 当前可复现结果

本节只记录迁移后当前实例上已经重新执行并得到日志支撑的结果。旧实例中的 49/50、
确定性 state 29 失败等结论不再直接沿用。

| 项目 | 当前实例结果 | 说明 |
| --- | --- | --- |
| 基座 top drawer 50 trials | **48/50（96%）** | 当前可靠基线 |
| 基座 state 29，5 个 seed | **5/5 成功** | 当前实例中不是稳定失败状态 |
| 基座 state 32，5 个 seed | **5/5 成功** | 当前实例中不是稳定失败状态 |
| 官方专家 HDF5 | **50 条、7479 个 transition** | 单条长度 127～190，文件可完整读取 |
| 观测 | 双路 128×128 RGB + 8 维 proprio | `ee_states(6) + gripper_states(2)` |
| 动作 | 7 维，位置和旋转均有变化 | 旋转统计的 q01/q99 非零 |
| 夹爪语义 | HDF5 中 `-1=开，+1=关` | 由动作与 gripper state 的对应关系确认 |
| LoRA 训练 | rank 32，lr=1e-5，有效 batch=4，500 steps | 最终 loss 约 **0.187** |
| Step 250 / Step 500 | state 29 均失败 | 未观察到中间 checkpoint 优势 |
| 自定义 norm stats | 正确加载后仍失败 | 不能把失败只归因于反归一化键 |
| 推理消融 | 去 proprio、open-loop=1 均失败 | 不是单一 proprio 或 chunk 长度问题 |
| gripper-flip 试验 | 失败，且该试验标签语义不成立 | 翻转规则只适用于此前自采 robosuite 动作，不适用于官方 HDF5 |

### 10.2 对 LoRA 效果的判断

**本轮 LoRA 微调没有产生可用模型，但现有证据不足以证明 LoRA 不适合 OpenVLA-OFT。**

更准确的结论是：当前“单任务 50 条数据 + rank 32 + 500 steps”的训练配方没有收敛，
并且破坏了原本成功率较高的基座行为。需要先替换训练配方和验证方法，再决定是否替换 LoRA。

判断依据如下：

1. **训练损失仍然偏高。** 文档中的官方长期训练目标为 L1 loss 小于 0.01，本轮结束时约为
   0.187，差距较大。0% rollout 更像是模型尚未学成或初始化/归一化链路不一致，不能仅解释为过拟合。
2. **500 steps 对 7479 个 transition 并不充分。** 按每个时刻形成一个训练样本粗算，
   有效 batch 4 对应约 2000 个样本曝光，不到一轮完整遍历；这与旧实验“7 条数据训练 1000 steps”
   的过度重复不是同一种情况。
3. **这是同任务继续训练，不需要优先改写视觉语言主干。** 当前目标只是修正少量控制行为，
   大容量 LoRA 可能带来不必要的策略漂移；先训练连续 action head / proprio projector 更符合问题规模。
4. **必须核验增量训练是否真正加载了基座的 action head 和 proprio projector。** 如果这两个模块
   被随机初始化，500 steps 和 0.187 loss 很可能不足以恢复原策略，rollout 全失败是预期结果。
5. **自定义数据统计虽非零，但仍可能改变基座 action head 的目标坐标系。** 后续应对比
   `libero_spatial_no_noops` 与自定义数据的 7 维 q01/q99，并进行“训练和评估统一使用基座统计”的对照。

因此，当前决策不是立即改成全参数微调，而是按以下优先级处理：

1. **首选：冻结 VLA 主干，先只训练 action head + proprio projector。** 这是成本最低、最能隔离
   数据链路问题的方案，也最不容易破坏已经达到 96% 的视觉语言能力。
2. **次选：保留 LoRA，但加入基座数据 replay。** 推荐从
   `libero_spatial_no_noops : 新专家数据 = 80～90% : 10～20%` 开始，学习率降至
   `1e-6～5e-6`，保存密集 checkpoint，并同时评估目标状态和全任务回归。
3. **若两者仍失败：采用 DAgger/残差策略。** 只学习基座失败附近的动作修正，而不是重写完整策略。
4. **暂不推荐全参数微调或 QLoRA。** 全参数微调在当前 A10 和小数据条件下更容易遗忘；
   QLoRA主要解决显存，不会自动解决数据对齐、初始化或控制标签问题。
5. **暂不优先更换 diffusion head。** 当前还没有证据证明失败来自多模态动作分布；过早更换动作头
   会同时改变更多变量，使问题更难定位。

### 10.3 专家数据质量判断

**50 条官方专家轨迹本身不能判定为“质量差”。** 从文件完整性、动作维度、旋转覆盖和夹爪状态看，
它明显优于此前 7 条脚本直线轨迹，足以用于管线验证和小规模适配。但它仍未达到“可直接用于稳定增训”
的证据标准，主要缺口不是文件数量，而是目标覆盖和闭环验收。

当前数据的优点：

- 来源是官方 LIBERO expert demonstrations，而非手写直线控制器；
- 50 条轨迹全部结构完整，共 7479 个 transition；
- 双相机、8 维 proprio、7 维 action 均齐全；
- 旋转动作有真实变化，避免了旧脚本策略旋转维统计为零的问题；
- 夹爪动作与实际开合状态能够对应。

仍需补做的质量验证：

1. **逐条 replay 成功率。** 在当前 LIBERO 环境中回放 HDF5 action，至少要求 50 条中 48 条以上
   成功；若 replay 自身失败，说明环境版本、控制频率或 action scale 不一致。
2. **时间对齐。** 确认 RLDS 中 `observation[t]` 对应 `action[t]`，而不是错一帧；同时检查
   action chunk 的未来 8 步构造是否跨越 episode 末尾。
3. **图像方向。** 抽样检查第三人称和腕部图像是否需要垂直翻转，避免训练图像与评估图像上下颠倒。
4. **初始状态映射。** 不能仅因文件有 50 条轨迹，就假定它们分别覆盖 benchmark 的 50 个
   init state；需要从保存的 simulator state 或 replay 结果建立 demo 与 init state 的对应表。
5. **训练/验证拆分。** 至少保留 10 条轨迹作离线验证，记录 held-out action L1；只看训练 loss
   无法判断数据是否可泛化。
6. **动作统计对照。** 对比原版 `libero_spatial_no_noops` 与当前数据的每维 min/max、q01/q99、
   零动作比例、夹爪切换次数和轨迹长度，识别采样频率或控制尺度差异。

因此，当前数据量的判断是：

- 对“验证 HDF5 → RLDS → 训练”链路：**数量足够**；
- 对“单独覆盖整个 top drawer 分布并保持基座能力”：**偏少且过于单任务化**；
- 对“修复特定失败状态”：只有在确认包含该状态或其邻域后才算有效，目前尚未证明；
- 下一轮不应只盲目增加数量，应优先补齐 replay、时间对齐、图像方向和初始状态覆盖证据。

### 10.4 下一轮实验顺序与通过标准

后续只允许一次改变一个关键变量，按以下顺序执行：

1. **数据闭环验收**：完成官方 HDF5 action replay、图像方向检查、动作/观测时间对齐检查。
2. **初始化验收**：确认 action head 和 proprio projector 从基座 checkpoint 加载，且训练前同一观测
   的动作输出与基座一致或仅有浮点误差。
3. **小样本过拟合测试**：取 1～5 条轨迹，关闭图像增强，只训练 head/projector；要求训练 L1
   明显降至 0.02 以下，并能在对应初始状态复现成功。该测试失败时禁止启动更大训练。
4. **稳定增训**：通过小样本测试后，使用基座 replay + 目标专家数据，尝试 head-only 与低学习率 LoRA
   两组对照；建议保存 100/250/500/1000 steps，而不是只看最终 checkpoint。
5. **统一评估**：每个候选模型至少评估目标状态 5 次、top drawer 50 states，并抽测其他 spatial
   任务；不得用单次成功或单次失败下结论。

候选模型的最低通过标准：

- top drawer 不低于当前基座 **48/50**；
- 目标失败状态成功率有可重复提升，而非单 seed 偶然成功；
- 其他 spatial 任务无明显退化；
- held-out action L1 与 rollout 成功率方向一致；
- 训练、合并、norm stats 和评估所用 checkpoint 均可追溯。

### 10.5 当前最终决策

1. 当前官方 50 条数据训练得到的 Step 250、Step 500 和 gripper-flip 模型均标记为**未通过**，
   不用于后续结论或部署。
2. 保留基座模型作为当前有效方案，其可复现 top drawer 基线为 **48/50（96%）**。
3. **暂不放弃 LoRA，但停止当前单任务直接 LoRA 配方。** 下一步优先完成数据 replay 和模块初始化核验，
   然后比较 head-only 与“低学习率 LoRA + 基座 replay”。
4. 若 head-only 能学会而 LoRA 仍退化，说明任务不需要改动 VLA 主干，应固定采用 head-only；
   若两者都学不会，应回到 RLDS/动作尺度/时序链路，而不是继续增加训练步数。
