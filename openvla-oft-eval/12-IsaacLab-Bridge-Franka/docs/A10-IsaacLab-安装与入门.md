# A10 上安装 Isaac Lab 与基础操作

本文先完成 Isaac Lab 的安装、官方示例运行和 Franka 基础练习，为后续 OpenVLA 闭环实验准备环境。命令按 Linux A10 云实例编写。

## 0. 版本和环境约定

- Isaac Sim：5.1.0。
- Isaac Lab：稳定标签 `v2.3.2`。
- Isaac Sim 5.x 使用 Python 3.11；现有 OpenVLA-OFT 环境为 Python 3.10，必须分开安装，不能在原 OFT 环境中覆盖 PyTorch 或 NumPy。
- 初始只跑一个仿真环境和一台 Franka。先验证渲染、物理、控制，再评估与 7B 模型共享 A10 显存是否可行。
- 默认工作区沿用 A10 实例的 `/mnt/workspace` 持久目录。若平台的持久目录不同，替换下文 `WS`。

Isaac Lab v2.3.2 与 Isaac Sim 5.1 是稳定配套版本。新版 Isaac Lab 3.0 开发线已切换到 Isaac Sim 6.x / Python 3.12，且含 API 变化；当前实验代码基于 2.x 接口，因此先不跟开发线。

## 1. 安装前检查

在 A10 Notebook 的终端执行：

```bash
nvidia-smi
uname -a
ldd --version | head -n 1
free -h
df -h /mnt/workspace
python3.11 --version
```

核对项目：

1. GPU 确实是 NVIDIA A10，空闲显存足以单独启动 Isaac Sim。
2. Linux x86_64，建议 Ubuntu 22.04；主机内存至少 32 GB，工作盘给安装包、代码和资产留出空间。
3. Isaac Sim pip 安装要求 GLIBC 2.35 或以上。若系统低于此版本，不要升级系统 GLIBC；改用 Isaac Sim 官方预编译包或平台允许的容器方案。
4. A10 是否能正常启动 RTX 渲染，以 Compatibility Checker 和下节渲染冒烟为准。云平台可能限制图形 API、驱动或远程显示。

官方安装要求列出至少 16 GB GPU 显存，并提醒相机渲染会额外占显存；A10 的 24 GB 还要给 OpenVLA 留出空间，所以必须先单独测 Isaac，再测两者并存。

## 2. 建立独立 Python 3.11 环境

不要复用 `/mnt/workspace/venv_oft`。在 Linux 终端执行：

```bash
export WS=/mnt/workspace
mkdir -p "$WS/isaaclab"
python3.11 -m venv "$WS/venv_isaaclab"
source "$WS/venv_isaaclab/bin/activate"
python -m pip install --upgrade pip
```

先安装 Isaac Sim 5.1 和匹配的 CUDA PyTorch，再安装 Isaac Lab：

```bash
python -m pip install 'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com

python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

git clone --branch v2.3.2 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git "$WS/isaaclab/IsaacLab"
cd "$WS/isaaclab/IsaacLab"
./isaaclab.sh --install none
```

`--install none` 安装 Isaac Lab 本体扩展，不额外安装当前入门阶段用不到的 RL 算法库。后续要跑某种训练框架时再按需添加。

首次启动会下载或编译扩展、填充着色器缓存，并出现 NVIDIA Omniverse EULA 提示；启动时间可能超过 10 分钟。按实际使用条款在终端交互确认。不要在无人值守脚本里擅自设置 EULA 接受变量。

## 3. 安装验收

先检查包和 CUDA：

```bash
python - <<'PY'
import torch
import isaaclab
import isaacsim
print('torch:', torch.__version__)
print('torch CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')
print('isaaclab:', isaaclab.__file__)
print('isaacsim:', isaacsim.__file__)
PY
```

再执行 Isaac Lab 官方空场景示例：

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

验收标准：进程启动并能正常退出；日志中没有 `ModuleNotFoundError`、CUDA 初始化错误、RTX 渲染器初始化失败或 PhysX 崩溃。记录 Isaac Sim/Lab、PyTorch、驱动版本和启动日志。若无头模式脚本不接受 `--headless`，先运行 `--help`，根据该标签下脚本的参数启动，避免混用开发分支教程。

## 4. 入门练习顺序

### 练习 A：查看 Stage 和 USD 场景

有图形界面时启动 Isaac Sim，熟悉 Stage 树、Viewport、属性面板、Play/Stop，以及 USD 中的物体层级。添加桌面和 Cube，观察位置、旋转、缩放以及物理属性。云端没有桌面显示时，可先在本地安装 Isaac Sim 做 GUI 熟悉；A10 负责 headless 执行并保存图像/视频。

### 练习 B：创建场景并步进

阅读 `scripts/tutorials/00_sim/create_empty.py`：找到 `AppLauncher`、`SimulationContext`、场景初始化和 `sim.step()`。理解应用启动、场景创建、物理步进的顺序。后续所有独立脚本都要先启动 `AppLauncher`，再导入或创建 Isaac 场景。

### 练习 C：与刚体交互

运行官方刚体/物体教程，练习创建 Cube、读取其位置、修改初始状态、重置仿真并观察重力和碰撞。重点理解世界坐标、米制单位、刚体碰撞属性和 `reset()`。

### 练习 D：Franka 和末端执行器 IK

```bash
./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py \
  --robot franka_panda --num_envs 1
```

观察 Franka 末端目标框与真实末端运动。阅读 `DifferentialIKController` 示例，确认关节名称、末端 body 名称、末端位姿和控制命令。第一次学习只用 1 个环境；不要直接照抄 128 环境的性能示例。

### 练习 E：运行官方方块任务

先确认环境注册名称：

```bash
./isaaclab.sh -p scripts/environments/list_envs.py | grep 'Lift-Cube-Franka-IK-Rel'
```

再用该标签内的官方策略播放脚本，以 `--help` 查找 `--task`、`--num_envs`、`--checkpoint`、`--headless` 参数。官方环境名是 `Isaac-Lift-Cube-Franka-IK-Rel-v0`。先以 1 个环境运行，熟悉 reset、observation、action、terminated/truncated 和成功条件。

### 练习 F：读懂 Manager-Based 环境

打开官方 Franka 方块任务配置，依次找到 Scene、Actions、Observations、Rewards、Terminations 和 Curriculum。理解 `env.step(action)` 输入什么、返回什么；相对 IK 和绝对 IK 的差别；episode 终止与超时如何区分。

## 5. A10 上的远程可视化

熟悉 GUI 有两条路径：

1. **本地工作站 GUI**：在有受支持 NVIDIA RTX GPU 的工作站安装同版本 Isaac Sim/Lab，适合练 Stage、Viewport 和属性面板。
2. **云端串流**：若 A10 平台允许开放 Isaac Sim WebRTC 所需网络端口，可运行官方 headless livestream 并使用 WebRTC Streaming Client。先确认平台网络策略和驱动支持，再配置端口；普通 Notebook 的浏览器代理通常不会自动转发这些端口。

若两者都不可用，A10 上仍可完成基础 API 学习：运行 headless 官方脚本，保存 RGB、MP4、状态日志，在本地查看输出。

## 6. 接入当前项目之前的门槛

完成以上练习后，再运行本项目里的环境 smoke。先确认实际 Isaac Lab checkout 是 `v2.3.2`，然后检查本项目配置能否导入官方 `FrankaCubeLiftEnvCfg`，再进行单环境相机渲染和零动作步进。项目代码路径见 [README](../README.md)。

当前 OpenVLA-OFT 环境使用 Python 3.10 / PyTorch 2.2；Isaac Sim 5.1 需要 Python 3.11 / PyTorch 2.7。不要把两边的 pip 包混装。后续可在 A10 评估单进程兼容方案，或由 Isaac Lab 与 OFT 推理分别运行并通过本地 HTTP/队列通信；无论哪种方式，都先记录合并运行的显存峰值和单步延迟。

## 官方参考

- [Isaac Lab v2.3.1 Python 包安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html)：Isaac Sim 5.x 对应 Python 3.11、安装及 `isaaclab.sh` 验收流程。
- [Isaac Lab v2.3.2 发布页](https://github.com/isaac-sim/IsaacLab/releases)：稳定版标签与版本说明。
- [Isaac Lab 空场景教程](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/tutorials/00_sim/create_empty.html)：应用启动、仿真上下文和基础步进。
- [Franka 差分 IK 教程](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/tutorials/05_controllers/run_diff_ik.html)：Franka 末端执行器控制。
- [Isaac Sim 5.1 容器与兼容性检查](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html)：远程 headless 部署及 Compatibility Checker。
