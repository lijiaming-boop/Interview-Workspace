# Isaac Lab × OpenVLA × Franka：Bridge 风格抓取放置

本目录把现有 OpenVLA-OFT/Bridge 代码接到 Isaac Lab 官方 Franka lift 场景，选取一个最小可验证任务：

> 从桌面抓起一个立方体，移动到绿色目标区上方，松开并让立方体稳定落在目标区内。

实现链路是：

```text
Isaac Lab Camera RGB
        │
        ├─ Bridge 风格裁剪/颜色对齐/可控随机化
        ▼
OpenVLA-OFT (Bridge action head)
        │  7D: Δx, Δy, Δz, Δrx, Δry, Δrz, gripper
        ▼
BridgeActionMapper
        │  Isaac Lab arm IK: 6D + binary gripper: 1D
        ▼
ManagerBasedRLEnv.step()
        │
        ├─ success: 抓起过 + 物体落入目标区 + 夹爪打开
        └─ MP4 / JSONL / 并行成功率
```

## 文件

- `bridge_action_mapper.py`：Bridge 7D 到 Isaac Lab IK + 夹爪的动作对齐，含单位、坐标系、夹爪符号和限幅。
- `domain_randomization.py`：Bridge 风格 RGB 预处理、亮度/对比度/饱和度/噪声随机化以及 Isaac stage 的材质/灯光 hook。
- `observation.py`：从 Isaac Lab 的 camera/FrameTransformer/Franka 关节状态构造 OpenVLA 所需的 observation。
- `openvla_policy.py`：复用现有 `02-实验代码/patch-check3` 的 OpenVLA-OFT 加载逻辑；另带无模型 heuristic smoke policy。
- `bridge_pick_place_env_cfg.py`：继承 Isaac Lab 官方 `FrankaCubeLiftEnvCfg`，补上固定 Bridge 桌面相机和绿色放置目标。
- `run_closed_loop.py`：单个或并行环境的闭环 demo、视频录制、失败阶段记录。
- `analyze_results.py`：迁移前后掉点、置信区间和失败阶段汇总。
- `record_video.py` / `camera_sweep.py`：raw vs aligned 并排视频、相机位姿对齐扫描。
- `check_env.py`：环境自检（环境变量 / 路径资产 / 依赖 / 硬件门槛 / Isaac 运行态），只依赖标准库，可用 `--json` 供 CI 消费。
- `tests/`：不依赖 Isaac Sim 的动作映射、RGB 对齐单测，以及 `check_env.py` 自身的回归测试。

## 文档

- [`docs/A10-IsaacLab-仿真与Franka抓取实验全流程.md`](docs/A10-IsaacLab-仿真与Franka抓取实验全流程.md)：**全流程主文档**。环境配置（A10 可行 / A100 不可行的硬件门槛、版本矩阵、隔离铁律）、启动运行时序、关键参数逐项说明、14 条症状级排障表、抓取实验四阶段流程与结果评估口径（含 `n ≠ --episodes`、raw→aligned ≠ sim-to-real 掉点等三个统计坑）。
- [`docs/A10-IsaacLab-安装与入门.md`](docs/A10-IsaacLab-安装与入门.md)：安装步骤、官方示例、Franka 基础练习、远程可视化。
- [`docs/Bridge-迁移掉点分析报告.md`](docs/Bridge-迁移掉点分析报告.md)：成对评测协议、结果表与失败归因规则。

## 运行前提

当前工作区本身不是 Isaac Lab 运行环境；本机检查不到 `isaaclab` 和 `torch`。因此这里提交的是可直接放进 Isaac Lab Python 环境执行的实验包，当前机器只做了纯 Python 语法和无 Isaac 依赖单测。Isaac Lab 需要 Linux/Windows + NVIDIA Isaac Sim 对应版本、CUDA、PyTorch 和官方 Isaac Lab 安装。

**先跑一次环境自检**，确认环境变量、路径、依赖、硬件门槛是否齐全（只依赖标准库，本机和云实例都能跑）：

```bash
python 12-IsaacLab-Bridge-Franka/check_env.py                    # 人读版报告 + 可粘贴的配置命令
python 12-IsaacLab-Bridge-Franka/check_env.py --json             # 机器可读
python 12-IsaacLab-Bridge-Franka/check_env.py --ignore-hardware  # 本机自检时跳过硬件门槛
python 12-IsaacLab-Bridge-Franka/check_env.py --checkpoint /path/to/ckpt   # 顺带校验 unnorm_key 配对
python 12-IsaacLab-Bridge-Franka/check_env.py \
  --simulate-argv "./isaaclab.sh -p 12-IsaacLab-Bridge-Franka/run_closed_loop.py --policy openvla --unnorm_key bridge_orig"
```

退出码：`0` 无 FAIL，`1` 有 FAIL，`2` 加了 `--strict` 且存在 WARN。加了 `--simulate-argv` 还能先验证真实命令会被上游 `prismatic/vla/constants.py` 判成哪个平台（判不成 BRIDGE 会静默退回 LIBERO，PROPRIO_DIM 变 8）。

然后在 Isaac Lab 环境中设置：

```powershell
$env:OFT_ROOT = 'C:\Users\27013\Desktop\oft'
$env:OPENVLA_OFT_ROOT = "$env:OFT_ROOT\02-实验代码\patch-check3"
$env:PYTHONPATH = "$env:OFT_ROOT\12-IsaacLab-Bridge-Franka;$env:OPENVLA_OFT_ROOT;$env:PYTHONPATH"
```

### 1. 先验证 Franka/相机/动作接口

不加载 VLA，只验证环境能渲染并走 IK：

```bash
./isaaclab.sh -p 12-IsaacLab-Bridge-Franka/run_closed_loop.py \
  --policy heuristic --num_envs 4 --episodes 4 --domain_mode bridge_aligned \
  --video --headless
```

### 2. OpenVLA 闭环

`--checkpoint` 可以是本地 Bridge 微调 checkpoint；`--unnorm_key` 必须与该 checkpoint 的 `dataset_statistics.json` 一致，默认 `bridge_orig`。

```bash
./isaaclab.sh -p 12-IsaacLab-Bridge-Franka/run_closed_loop.py \
  --policy openvla \
  --checkpoint /path/to/openvla-bridge-checkpoint \
  --openvla_root "$OPENVLA_OFT_ROOT" \
  --unnorm_key bridge_orig \
  --num_envs 8 --episodes 50 --domain_mode raw --video --headless

./isaaclab.sh -p 12-IsaacLab-Bridge-Franka/run_closed_loop.py \
  --policy openvla \
  --checkpoint /path/to/openvla-bridge-checkpoint \
  --openvla_root "$OPENVLA_OFT_ROOT" \
  --unnorm_key bridge_orig \
  --num_envs 8 --episodes 50 --domain_mode bridge_aligned --video --headless
```

默认 `--open_loop_steps 1`，即每个控制周期重新渲染和推理；不要把 `5` 当成闭环结果，`5` 只用于可选的 action chunk 消融。

### 3. 并行评测与掉点报告

每个 batch 的物理仿真是并行的，模型接口保持单帧 action 语义；结果会写入 `outputs/<run>/episodes.jsonl` 和 `summary.json`。

Linux 用 `run_domain_gap_eval.sh`，Windows Isaac Lab 安装用 `run_domain_gap_eval.ps1`；二者都会用同一 seed 跑 raw 与 bridge_aligned，再生成掉点报告。

```powershell
.\run_domain_gap_eval.ps1 -Checkpoint C:\ckpts\openvla-bridge -Episodes 50 -NumEnvs 8
```

```bash
python 12-IsaacLab-Bridge-Franka/analyze_results.py \
  --before outputs/raw/summary.json \
  --after outputs/bridge_aligned/summary.json \
  --report outputs/domain_gap_report.md
```

## 动作空间契约

| Bridge 维度 | 含义 | 单位/符号 | Isaac Lab 下游 |
|---|---|---|---|
| 0:3 | 末端位置增量 | m，Bridge 世界/基座坐标 | `DifferentialInverseKinematicsActionCfg` 位置 3 维 |
| 3:6 | 末端轴角增量 | rad，小角度 exponential coordinates | 同一 IK action 的旋转 3 维 |
| 6 | 夹爪 | Bridge 默认绝对值 `0=close, 1=open`；adapter 统一成 `-1/+1` | `BinaryJointPositionActionCfg` |

Bridge 数据转换在现有项目中先用相邻 EEF state 差分构造前 6 维，再保留绝对夹爪维（Bridge 原始 transform 是 `0=close, 1=open`）；因此 observation 的第 7 个 proprio 维也保持 `[0,1]`，不能误写成 `[-1,1]`。OpenVLA 的 `get_vla_action()` 返回已经按 checkpoint 的统计量反归一化的 action。映射器只做坐标系、限幅和夹爪二值化，不再二次反归一化。

如果机器人 base 不是单位姿态，把 `--bridge_frame world --ik_frame body` 打开；映射器会用当前 EEF 四元数把小位移/小旋转从世界坐标转到 body 坐标。默认场景的 Panda base 与 Bridge 桌面坐标轴一致，因此 `root` 模式下无需额外旋转。

## 域差距实验设计

固定相同 episode seed、目标物体初始状态、checkpoint、推理频率和并行度，至少跑三组：

1. `raw`：只把 Isaac RGB resize 到 OpenVLA 输入，不做 Bridge 视角/颜色对齐。
2. `camera_aligned`：Bridge 风格 4:3 中心裁剪、固定相机位姿、曝光/色温对齐。
3. `bridge_aligned`：`camera_aligned` + 亮度/对比度/饱和度/轻微 JPEG/传感器噪声随机化 + stage 材质和灯光随机化。

报告同时记录：总体成功率、抓取成功率、放置成功率、首次接触步、推理延迟、碰撞/掉落失败数。掉点定义为相同 seed 的 `raw - bridge_aligned` 成功率差；补救收益定义为 `bridge_aligned - raw`。没有完成 Isaac Lab 实跑前，报告不会填入虚构的百分比。

## 官方接口依据

- Isaac Lab 官方环境列表包含 `Isaac-Lift-Cube-Franka-IK-Rel-v0` 与 `Isaac-Stack-Cube-Franka-IK-Rel-v0`；本实验从前者继承，降低场景复杂度。
- 官方 Franka IK 配置使用 `DifferentialInverseKinematicsActionCfg`、`panda_joint.*`、`panda_hand`、相对 pose + DLS。
- 官方 OpenVLA-OFT Bridge 配置定义 `ACTION_DIM=7`、`PROPRIO_DIM=7`，EEF action 是 `Δxyz + Δrpy + gripper`。

参考：[Isaac Lab Franka IK 配置](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/ik_rel_env_cfg.py)、[Isaac Lab task-space controller](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/05_controllers/run_diff_ik.html)、[OpenVLA-OFT action encoding](https://github.com/moojink/openvla-oft/blob/main/prismatic/vla/datasets/rlds/oxe/configs.py)。
