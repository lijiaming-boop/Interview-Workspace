# OpenVLA-OFT 环境搭建与训练方案文档

> 日期：2026-09-02
> 实例：魔搭免费 GPU 实例（A10 24G，ModelScope 官方镜像 Python 3.10）
> 目标：LIBERO 仿真评估 + top drawer 失败分析 + DAgger 式增训微调

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
| tokenizers / sentencepiece | 0.19.1 / 0.1.99                       | 版本必须对齐                      |
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

| 测试项                                   | 结果                                                      |
| ------------------------------------- | ------------------------------------------------------- |
| 依赖总验证（torch/tf/dlimp/libero/get\_vla） | 全部通过                                                    |
| 模型加载（4 个 safetensors 分片）              | 成功，耗时 \~3s，显存 **15.13 GB**                              |
| 2-episode 仿真冒烟（top drawer, state 0/1） | **2/2 成功（100%）**                                        |
| 视频自动保存                                | `./rollouts/2026_09_02/*.mp4`，文件名含 `success=True/False` |
| 渲染后端                                  | OSMesa（EGL 在重启后不可用）                                     |

评估脚本关键参数（`run_libero_eval.py`）：

- `--target_episodes "4:0,4:1,..."`：零基 task:state 对，定向跑指定初始状态

- `--num_open_loop_steps 8`：action chunk 大小

- `--num_steps_wait 10`：等待物体稳定

- `--local_log_dir`：日志目录；结果逐条记录在 `EVAL-*.txt`

***

## 五、训练方案（针对性增训，5 阶段）

**目标**：top drawer（libero\_spatial task\_id=4）失败率定位 → 失败状态补专家数据 → LoRA 微调 → 复测。

### 阶段 1：定向评估（进行中）

- 50 个初始状态 × 1 trial：`--target_episodes "4:0,4:1,...,4:49"`，`--seed 7` 固定

- 产出：逐条成功/失败记录（`EVAL-*.txt`）+ 每个 episode 的视频（自动保存）

- 预计 2~~3.5 小时（OSMesa 每 trial 2~~4 分钟）

### 阶段 2：失败模式分析

- 离线脚本按 state id 重建环境，读出每个初始状态的碗位姿，与成败对齐聚类

- 判定：某桶失败率 ≥60% 且其他桶 ≤30% → 失败集中于特定初始状态，进阶段 3

- 对失败视频跑 `video_chunk_analysis.py`，验证 grasp→transfer 是否发生在 chunk 边界（t = 10 + 8k）

### 阶段 3：专家修正数据收集（DAgger 核心）

- 只对失败初始状态采集，每状态 5~~10 条，共 50~~200 条

- 首选：运动学脚本策略（IK 规划：碗上方→抓取→搬到盘→松开），全自动可批量

- 备选：键盘遥操作 / 复用 LIBERO 官方 demo

- 存 LIBERO 标准 HDF5 格式（obs/image、action、reward、done）

### 阶段 4：LoRA 微调

- 官方 `finetune.py`；数据 = 原 LIBERO spatial 示教 + 新增示教混合（防灾难性遗忘）

- A10 24G 必须 LoRA（checkpoint 本身即 LoRA，peft==0.11.1）

- 超参：batch 2~~4 + gradient checkpointing，lr 2e-5，1~~2 epoch

- 此阶段真正用到 TensorFlow/dlimp（RLDS 数据加载）

### 阶段 5：复测对比

- 阶段 1 完全相同命令重跑 50 trials（同 seed，可比）

- 成功标准：失败桶成功率提升 ≥20 个百分点，其他任务抽测不退化

***

## 六、全新实例一键复现脚本

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

## 七、常用命令速查

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
```

