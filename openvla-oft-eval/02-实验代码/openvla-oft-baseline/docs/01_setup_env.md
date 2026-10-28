# 01 · 环境搭建手册（魔搭 A10 实例 · 全国内源）

`scripts/01_setup_env.sh` 是本文的自动化版本。建议先读本文理解每一步在做什么、为什么这样做，出问题时才能定位。

## 0. 环境适配：镜像 `ubuntu22.04-cuda12.1.0-py310-torch2.3.0` vs 官方依赖栈

这个镜像选得很好——python 和 CUDA 都与官方栈完全对齐，只差 torch 版本和几个未预装的包：

| 依赖 | 官方版本 | 镜像自带 | 结论 |
|---|---|---|---|
| python | 3.10（论文用 3.10.14） | 3.10 | ✅ 一致 |
| CUDA | torch 轮子自带 cu121 | 12.1.0 | ✅ 一致 |
| torch | 2.2.0（硬 pin） | 2.3.0 | ⚠️ pip 会自动替换为 2.2.0（官方 pin，勿手动升级） |
| tensorflow | 2.15.0 | 无 | 需安装（仅支持 py≤3.11，镜像满足；numpy<2） |
| transformers | moojink 的 4.40.1 fork（git 安装） | 无 | 需安装（双向注意力支持，不能换新版） |
| timm / tokenizers | 0.9.10 / 0.19.1 | 无 | 需安装 |
| flash-attn | 2.5.5（**仅训练需要**） | 无 | 评测不需要：`get_vla` 里 `attn_implementation="flash_attention_2"` 是注释掉的，默认走 PyTorch SDPA |

两种装法（后续 02/03 脚本通过 `~/.oft_use_base` 标记文件自动识别，命令完全相同）：

- **默认（推荐）**：新建独立 conda 环境，不动镜像自带环境；
- **USE_BASE=1**：直接装进镜像自带 py3.10——省时省盘，但镜像的 torch 2.3.0 会被降为 2.2.0、setuptools 降为 65.5.0（实例还要跑其他任务就别用）。

显存 24GB > 推理所需 ~16GB；bf16 需要 Ampere 及以上（A10/3090/4090 等 24G 卡均可）。8核32GB 内存对 robosuite 仿真 + 推理绰绰有余。

## 1. 磁盘预算

| 项目 | 大小 |
|---|---|
| conda 环境（torch + TF 2.15 + 全家桶） | ~15GB |
| 每个 checkpoint（4 个 suite + 可选四合一） | ~15GB × N |
| 评测视频（每 episode ~1-3MB × 500/suite） | ~1GB/suite |

四个 suite 全量建议预留 **≥ 90GB**。磁盘紧张就只下四合一 checkpoint（`openvla-7b-oft-finetuned-libero-spatial-object-goal-10`，15GB 跑全部 suite，论文均值 96.8% vs 97.1%）。

> ⏱ **额度提示（剩余约 30h）**：全量 4 个 suite（2000 episodes）大概率跑不完——推荐"Spatial 全量 + 其余三个 suite 10 trials 快速基线"，计划表见 README。**实例释放后磁盘数据可能不保留，每跑完一个 suite 就把 `results/`、日志、失败视频下载到本地**；环境随时可用本套脚本重建。

## 2. 国内镜像清单（common.sh 可覆盖）

| 用途 | 地址 | 备选 |
|---|---|---|
| pip | `https://pypi.tuna.tsinghua.edu.cn/simple` | `https://mirrors.aliyun.com/pypi/simple/` |
| conda / Miniconda | `https://mirrors.tuna.tsinghua.edu.cn/anaconda/` | 中科大 mirrors.ustc.edu.cn/anaconda/ |
| GitHub 加速 | `https://ghfast.top/https://github.com/...` | `https://gh-proxy.com/`、`https://ghproxy.net/` |
| HuggingFace | `HF_ENDPOINT=https://hf-mirror.com` | （魔搭无官方 ckpt 镜像，只能走这里） |

GitHub 代理失效时改环境变量即可：`GH_PROXY=https://gh-proxy.com bash scripts/01_setup_env.sh`。

## 3. 分步详解

### 3.1 git 加速改写（关键技巧）

```bash
git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"
```

之后所有 `https://github.com/...` 的 clone/fetch 都会被 git 自动改写走代理。**这一招同时解决了 `pip install -e .` 的两个 git 依赖**（`transformers @ git+https://github.com/moojink/transformers-openvla-oft.git` 和 dlimp fork）——pip 对 git+ 地址是交给 git 处理的，改写规则照样生效。
撤销：`git config --global --unset url."https://ghfast.top/https://github.com/".insteadOf`

### 3.2 torch 2.2.0 不需要去 pytorch 官网

PyPI 上的 `torch==2.2.0` Linux 轮子本身就是 CUDA 12.1 构建（通过 `nvidia-*-cu12` pip 包带运行时），清华 PyPI 镜像直接可下，速度拉满。镜像自带 torch 2.3.0（同为 cu121）：USE_BASE=1 模式下 pip 会把它降级到官方 pin 2.2.0，属预期行为；独立 conda 环境模式则环境里装的就是 2.2.0。

### 3.3 gym==0.21.0 的老坑（脚本已处理）

LIBERO/robosuite 1.4.1 是 2023 年的旧栈，`env.step()` 返回 4 元组、`reset()` 不带 info——gym 0.26+ 的 API 变更会把它打崩。gym 0.21.0 的 sdist 又有元数据 bug（`opencv-python>=3.` 不合法），新 setuptools 直接拒装。解决顺序：

```bash
pip install "setuptools==65.5.0" "wheel==0.38.4"
pip install gym==0.21.0 --no-build-isolation
```

**必须先于 openvla-oft / LIBERO 安装**，这样后续依赖解析时 `gym` 已被满足，不会被升到 0.26+。

### 3.4 numpy 必须 <2.0

`pip install -e .` 不 pin numpy，会装出 numpy 2.x，然后 TF 2.15、robosuite、LIBERO 全部报错。脚本用 constraints 文件（`numpy==1.24.4`）+ 安装完成后再强制装一次 `numpy==1.24.4` 双保险。**任何时候 `pip list | grep numpy` 看到 2.x 都是错的**。

### 3.5 LIBERO 安装

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git   # 会被 3.1 的改写走代理
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt   # 在 openvla-oft 目录下
```

`libero_requirements.txt` 内容：`imageio[ffmpeg]`、`robosuite==1.4.1`、`bddl`、`easydict`、`cloudpickle`、`gym`（robosuite 1.4.1 这个 pin 是硬性的）。

### 3.6 首次运行的交互式询问（nohup 杀手）

LIBERO 第一次 import 时若 `~/.libero/config.yaml` 不存在，会 `input()` 询问数据集保存路径（评测其实用不到数据集，但要走过场）。**后台运行（nohup/tmux detach 后无 stdin）会直接 EOFError**。所以流程永远是：前台 SMOKE 一次 → 之后随便后台跑。

### 3.7 无头渲染（EGL）

脚本导出 `MUJOCO_GL=egl`（robosuite 的 MuJoCo offscreen 渲染走 EGL，NVIDIA 驱动自带 `libEGL_nvidia`，无需 X server）。验证：

```bash
MUJOCO_GL=egl python -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); print('egl ok')"
```

## 4. 故障排查表（按报错关键字）

| 报错/现象 | 原因 | 解法 |
|---|---|---|
| `gym==0.21.0` 安装报 `extras_require` / `opencv-python>=3.` | setuptools 太新 | 见 3.3（setuptools 65.5.0 + `--no-build-isolation`） |
| `numpy.dtype size changed` / TF import 崩 | numpy 被升到 2.x | `pip install numpy==1.24.4` 后重跑 |
| pip 提示将降级/卸载 torch 2.3.0（USE_BASE=1 时） | 官方 pin 就是 2.2.0 | 预期行为，放心继续 |
| `git clone` / pip 拉 git 依赖超时 | 加速代理失效 | 换 `GH_PROXY=https://gh-proxy.com`；或手动 clone 两个 fork 后 `pip install -e` |
| `EGL_NOT_INITIALIZED`（进程退出阶段） | MuJoCo 销毁 GL context 的已知噪声 | 只要退出码为 0 就忽略 |
| 渲染真失败 `Failed to create GLFW/EGL context` | 缺 EGL 库 | `apt install -y libegl1 libgl1`；仍不行试 `MUJOCO_GL=osmesa`（慢但能跑） |
| nohup 跑挂，`EOFError` | 见 3.6 | 前台 SMOKE 一次 |
| `KeyError: 'libero_spatial_no_noops'` | unnorm_key 没对上 / dataset_statistics.json 缺失 | 检查 checkpoint 目录完整性（02 脚本会校验） |
| 成功率远低于预期（<80%） | 多半是预处理不一致 | 确认 `--center_crop True`（脚本默认）、`use_proprio=True`、`num_images_in_input=2`；再看 numpy 是否 2.x |
| `assert len(checkpoint_files) == 1` | checkpoint 目录里 action_head/proprio_projector 的 .pt 缺失或重复 | 重下 checkpoint（02 脚本断点续传） |
| CUDA OOM | 有其他进程占卡 | `nvidia-smi` 查看；A10 24GB 跑单个评测进程足够 |
| pip 报 tensorflow_graphics 依赖冲突警告 | 该包只被训练数据管线用到 | `pip check` 只涉及它时可忽略，评测不 import |

## 5. 安装完成自检清单

```bash
conda activate oft
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
# 期望: 2.2.0+cu121  1.24.4
python -c "import gym, robosuite, transformers; print(gym.__version__, robosuite.__version__, transformers.__version__)"
# 期望: 0.21.0  1.4.1  4.40.x
python -c "from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM; print(NUM_ACTIONS_CHUNK, ACTION_DIM)"
# 期望: 8 7
```
