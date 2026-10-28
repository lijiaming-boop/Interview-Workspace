# OpenVLA-OFT × LIBERO 基线评测套件（魔搭 A10 实例 · 全国内源）

目标：在魔搭社区 GPU 实例（24GB 显存 / 镜像 `ubuntu22.04-cuda12.1.0-py310-torch2.3.0` / 8核32GB）上跑通 **OpenVLA-OFT 官方 checkpoint 在 LIBERO 仿真基准上的评测 pipeline**（图像观测 → prompt 构造 → action 反归一化 → 环境步进），产出 **baseline 成功率表格** 和 **失败 case 分析**。

## 先读这五条结论

1. **官方支持的仿真评测是 LIBERO**（4 个 suite，各 10 任务 × 50 trials，另有 1 个四合一 checkpoint）。SimplerEnv 主仓库**不内置** OpenVLA-OFT 的 agent（官方只支持 RT-1/Octo；OpenVLA 系要走社区 fork，如 [DelinQu/SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA)）。本套件以 LIBERO 为主线，SimplerEnv 见文档末尾"扩展路线"。
2. **评测不需要 flash-attn**。模型加载代码里 `attn_implementation="flash_attention_2"` 是被注释掉的（`experiments/robot/openvla_utils.py` 的 `get_vla`），默认走 PyTorch SDPA——省去国内编译 flash-attn 的最大麻烦。A10 24GB 跑 bf16 推理（约 16GB）完全够。
3. **新镜像（py3.10 + CUDA 12.1）与官方栈基本对齐**，只差三样：torch 官方 pin 是 2.2.0（镜像自带 2.3.0，安装时自动替换）、TF 2.15、transformers fork + timm 0.9.10 等依赖。`01_setup_env.sh` 默认新建独立 conda 环境（最稳）；镜像已是 py3.10，也可 `USE_BASE=1 bash scripts/01_setup_env.sh` 直接装进自带环境（更快，但会把镜像的 torch 2.3.0 降为 2.2.0）。
4. **模型只能走 hf-mirror.com**。魔搭（ModelScope）上没有官方 checkpoint 镜像（只有社区用户 `Sylvest` 的变体），pip/conda/GitHub 分别用清华源和 ghfast.top 加速。
5. **首次运行 LIBERO 会交互式询问"数据集保存路径"**（`~/.libero/config.yaml` 不存在时）。务必先在前台跑一次冒烟测试再上 nohup，否则后台任务会因 `input()` 读到 EOF 直接报错。

## 快速开始

```bash
# 0. 自检：GPU / 磁盘 / 国内源连通性
bash scripts/00_preflight.sh

# 1. 搭环境（约 30-60 min，磁盘 ~15GB；更快：加 USE_BASE=1 直接用镜像自带 py3.10）
bash scripts/01_setup_env.sh

# 2. 下载 checkpoint（每个 ~15GB；先下 spatial 打通全链路）
bash scripts/02_download_ckpts.sh spatial

# 3. 冒烟测试：每任务 2 条共 20 episodes，验证 pipeline 通了（务必前台跑一次！）
SMOKE=1 bash scripts/03_run_libero_eval.sh spatial

# 4. 正式 baseline：先 10 trials/task 快速版，再上官方 50 trials/task 全量
bash scripts/03_run_libero_eval.sh spatial 10
bash scripts/03_run_libero_eval.sh spatial 50

# 5. 四个 suite 全量（后台 + 断点说明见 docs/01）
bash scripts/02_download_ckpts.sh all
nohup bash scripts/03_run_libero_eval.sh all 50 > run_all.log 2>&1 &

# 6. 生成 baseline 成功率表格（results/baseline_success_rates.md）
python scripts/04_aggregate_results.py

# 7. 生成失败 case 清单 + 人工标注模板（results/failure_cases/）
python scripts/05_collect_failure_cases.py
```

## 期望 baseline（对照用）

| Suite | 官方单 suite checkpoint | 四合一 checkpoint |
|---|---|---|
| LIBERO-Spatial | ~97.2% | ~96.8% |
| LIBERO-Object | ~97.4% | ~97.0% |
| LIBERO-Goal | ~95.8% | ~95.4% |
| LIBERO-10 (Long) | ~98.0% | ~98.0% |
| **平均** | **97.1%**（官方 README 背书） | **96.8%**（官方 README 背书） |

说明：均值 97.1% / 96.8% 来自官方仓库 LIBERO.md；per-suite 数值来自论文表格的社区复现记录（seed=7、50 trials/task、A100、py3.10.14 + torch 2.2.0 + transformers fork）。官方明确提示"不同 GPU 上结果可能有轻微波动"。**你的 A10 baseline 与该表对比，偏差在 ±1-2% 内即视为复现成功**；更大的偏差先查 `--center_crop True` 是否生效、numpy 是否被升到 2.x、以及图像预处理 flag 是否与训练一致。

## 时间预算（剩余额度 ~30h 的推荐计划）

| 阶段 | 预估 | 说明 |
|---|---|---|
| 环境搭建 | 0.5-1h | 01 脚本 |
| 下载 checkpoint | 0.5-1h / 个 | 先下 spatial，其余按计划补 |
| 冒烟测试（必做，前台） | ~0.5h | `SMOKE=1` |
| LIBERO-Spatial 全量（50 trials） | 4-12h | 用冒烟耗时 ×25 外推后再决定 |
| 其余 3 个 suite 快速基线（10 trials） | 1-3h / 个 | 出表格足够，注明 10 trials |
| 汇总 + 失败视频标注 | 1-2h | 04/05 脚本 + docs/03 |

- 全量跑满 4 个 suite（2000 episodes）大概率超出 30h 额度——先 spatial 全量 + 其余 10 trials，富余再补全量。
- **额度用尽/实例释放后磁盘数据可能不保留**：每跑完一个 suite，立即把 `results/`、`$OFT_ROOT/experiments/logs/` 和失败视频下载到本地。环境随时可用脚本一键重建——这正是全套脚本化的意义。
- 长跑一律 `nohup ... &` 或 tmux，防 SSH 断线。

## 耗时预估（按冒烟测试实测外推）

- 冒烟（每任务 2 条 = 20 episodes）：约 10-30 分钟。
- 全量单 suite（500 episodes）：约 4-12 小时（每个 episode 需 220-520 个控制步，每 8 步一次 VLA 前向）。
- 外推公式：`全量耗时 ≈ 冒烟耗时 × (500 / 冒烟episodes数)`。
- 建议顺序：spatial 冒烟 → spatial 全量 → 其余三个 suite（可 nohup 串行过夜跑）。

## 目录结构

```
openvla-oft-baseline/
├── README.md                    # 本文件
├── docs/
│   ├── 01_setup_env.md          # 环境搭建手册（每步原理 + 故障排查表）
│   ├── 02_pipeline_walkthrough.md  # 评测 pipeline 逐环节详解（带官方代码引用）
│   └── 03_failure_analysis.md   # 失败 case 分析方法论 + 分类体系 + 报告模板
├── scripts/
│   ├── common.sh                # 全局配置（路径/镜像源，可用环境变量覆盖）
│   ├── 00_preflight.sh          # 运行前自检
│   ├── 01_setup_env.sh          # 一键搭环境（国内源）
│   ├── 02_download_ckpts.sh     # 从 hf-mirror 下载官方 checkpoint
│   ├── 03_run_libero_eval.sh    # 跑官方评测（SMOKE/全量/指定seed）
│   ├── 04_aggregate_results.py  # 解析日志 → 成功率表格 (md + csv)
│   └── 05_collect_failure_cases.py  # 扫描失败视频 → 清单 + 标注模板
└── results/                     # 产出目录（表格、清单、标注文件）
```

## 把本套件放到 GPU 实例上

```bash
# 方式一：scp 直传（在本机执行）
scp -r openvla-oft-baseline/ <user>@<实例IP>:~/

# 方式二：推到 gitee 私有仓库后实例上 clone（国内快）
```

## 常见坑 Top 5（详见 docs/01 排查表）

1. **`gym==0.21.0` 装不上**（sdist 元数据 bug）→ 先装 `setuptools==65.5.0` 再 `--no-build-isolation`（脚本已处理）。
2. **numpy 被依赖悄悄升级到 2.x** → TF 2.15 / robosuite / LIBERO 全崩；脚本用 constraints + 最后强制锁 `numpy==1.24.4`。
3. **MUJOCO EGL 报错** → `MUJOCO_GL=egl` 已在脚本导出；`EGL_NOT_INITIALIZED` 出现在进程退出阶段时是噪声可忽略；真渲染失败看 docs/01。
4. **忘记 `--center_crop True`**（脚本默认已带）→ 成功率大幅下降，这是与训练时 random 90% crop 对齐的关键。
5. **nohup 后台跑挂掉** → 首次 LIBERO 交互询问数据路径；先前台 SMOKE 一次生成 `~/.libero/config.yaml`。

## 扩展路线：SimplerEnv（可选，非官方支持）

如需 Bridge/Google Robot real2sim 评测：SimplerEnv 主仓库没有 OpenVLA-OFT agent，需用社区 fork（如 [DelinQu/SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA)，另需 ManiSkill2 real2sim + SAPIEN 2.2 + Vulkan 渲染，安装复杂度显著高于 LIBERO）。建议先用 LIBERO 拿到 baseline、吃透 pipeline，再评估是否投入。

## 参考链接

- 官方仓库：https://github.com/moojink/openvla-oft （SETUP.md / LIBERO.md）
- 官方 checkpoint（hf-mirror 可下）：https://hf-mirror.com/moojink/openvla-7b-oft-finetuned-libero-spatial 等
- 论文：Fine-Tuning Vision-Language-Action Models (arXiv:2502.19645)
- LIBERO 基准：https://github.com/Lifelong-Robot-Learning/LIBERO
