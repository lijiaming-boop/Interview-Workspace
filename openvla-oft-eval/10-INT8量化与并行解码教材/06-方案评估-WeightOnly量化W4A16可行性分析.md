# 方案评估：转向 Weight-Only 量化（AWQ / Marlin W8A16 / W4A16）是否可行

> 评估对象：`06-文档/模型量化误差策略与OpenVLA-OFT时延分析.md` §7 方案二
> 评估基准：`04-评测结果/baseline/openvla_oft_fp16_baseline.json` + 500 episodes 配对实测
> 硬件：NVIDIA A10（sm_86，24 GB，HBM 600 GB/s）

---

## 0. 结论（先给答案）

**机制成立，收益算错了。**

| 判据 | 结论 |
|---|---|
| 技术路线对不对 | ✅ 对。weight-only 确实能消除 `LLM.int8()` 的离群值分解与动态反量化开销 |
| 在本负载上能拿到多少 | ❌ **≤ 1.10×**（W4A16 投影 217.2 → 197.9 ms） |
| 能压到 150 ms 吗 | ❌ **算术不可能**。任何 weight-only 位宽的物理下限是 **191.5 ms** |
| 工程代价 | ⚠️ 数天工作量 + 破坏冻结 torch 环境的风险 |
| 该不该做 | ❌ 在 A10 上**不该做**。见 §6 |

---

## 1. 为什么「1.3~1.7×」这个经验值在这里不成立

1.3~1.7× 是 **LLM 自回归 decode** 的经验值：`batch=1`、每步只处理 **1 个 token**，此时接近 100% 的时间都在「把 15 GB 权重从 HBM 搬进 SM」。砍掉 4 倍权重搬运，就接近砍掉 4 倍时间——所以那个数字是真的。

**但 OpenVLA-OFT 不是这个形态。** 它是**并行解码**（`use_l1_regression=True`，action head 一次吐出整个 `8×7` chunk），每次推理是**一次 prefill**：

| 输入构成 | token 数 |
|---|---|
| 主视角图像（224px / patch14 → 16×16） | 256 |
| 腕部相机图像 | 256 |
| prompt + BOS + `29871` 补位 token（`modeling_prismatic.py:981`） | ~23 |
| **合计** | **≈ 535** |

一次前向 = 535 个 token 全部穿过 32 层。**计算项与访存项同量级**，而 weight-only 只砍后者，所以加速上限被结构性卡死在个位数百分比。

---

## 2. 硬算术：权重的时延预算只有 25.8 ms

数据来源：`openvla_oft_fp16_baseline.json`（A10 / torch 2.3.1+cu121 / 2 图 / chunk=8 / warmup 5 / iters 30）。

| 项 | 数值 | 依据 |
|---|---|---|
| 实测权重常驻 | **14.39 GiB = 15.45 GB** | `load_allocated_gib = 14.3908` |
| A10 HBM 带宽 | **600 GB/s** | 官方规格 |
| **权重搬运下限** | 15.45 / 600 = **25.8 ms** | 纯算术 |
| **实测端到端时延** | **217.24 ms** | 500 episodes 评测均值 |
| 交叉验证 | 219.52 ms（bf16/fp16 bench 两次独立测量差 <1%） | `latency_ms/mean_ms` |
| **权重搬运占比** | **11.9%** | 25.8 / 217.24 |

> **权重搬运只占 11.9%。** 这是本次评估最关键的一个数，也是整份判决的地基。

### 2.1 各方案投影

| 方案 | 权重字节 | 搬运时间 | 省下 | 投影时延 | 加速比 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| BF16（现状） | 15.45 GB | 25.8 ms | — | 217.2 ms | 1.00× |
| W8A16 | 7.73 GB | 12.9 ms | 12.9 ms | 204.4 ms | **1.06×** |
| W4A16 | 3.86 GB | 6.4 ms | 19.3 ms | 197.9 ms | **1.10×** |
| W2A16（假想） | 1.93 GB | 3.2 ms | 22.6 ms | 194.6 ms | 1.12× |
| **权重归零（物理极限）** | 0 | 0 ms | **25.8 ms** | **191.5 ms** | **1.13×** |

**最后一行就是判决书。** 把权重**全部删掉**（物理上不可能），时延也只能从 217.2 降到 **191.5 ms**。要命中 150 ms 需要砍掉 **67 ms**，是全部权重预算（25.8 ms）的 **2.6 倍**。

→ 「1.3~1.7×、压到 150 ms 以内」在这个负载上是**算术不可能**，不是工程困难。

---

## 3. 剩下的 88% 是什么？weight-only 碰得到吗

粗估分解：

| 分项 | 估算 | 占比 | W4A16 能减少吗 |
| :--- | ---: | ---: | :--- |
| 权重搬运（HBM → SM） | **25.8 ms** | ~12% | ✅ **只此一项** |
| Tensor Core GEMM 计算 | ~64 ms @100% MFU（~143 ms @45% MFU） | 30~60% | ❌ **完全不能** |
| Kernel launch / CPU 发射 / 显存分配 / dtype cast / 未融合算子 | 剩余 | 30~58% | ❌ 完全不能 |

**关键点**：Marlin 的「片上解量化」意味着 **GEMM 依然跑在 FP16 Tensor Core 上**。它省掉的是「每读一次权重就要从 HBM 搬一遍」，**不是省计算**。所以只要计算项与访存项同量级（本负载正是如此），加速上限必然被卡在个位数。

> ⚠️ 上表「计算 / 开销」两行的切分是**估算**，需要 profile 才能定案（见 §7）。但**第一行 25.8 ms 是硬数**，判决不依赖估算。

---

## 4. 就算要做，工程上也不是「换个 flag」

| 环节 | A10（sm_86）可行性 |
| :--- | :--- |
| Marlin kernel 架构支持 | ✅ sm_80 / 86 / 89 在支持列表内（但调优目标是 sm_80 / sm_89，A10 非最优目标） |
| 拿到 Marlin / AWQ 格式的 checkpoint | ❌ **主要拦路虎** |
| 校准集 | ⚠️ 需要，且不能用通用文本校准集 |

### 4.1 拦路虎的具体位置

`experiments/robot/openvla_utils.py:282-290`：

```python
vla = AutoModelForVision2Seq.from_pretrained(
    cfg.pretrained_checkpoint,
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    load_in_8bit=cfg.load_in_8bit,
    load_in_4bit=cfg.load_in_4bit,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
```

它**只透传 `load_in_8bit` / `load_in_4bit`，没有任何 `quantization_config` 通道**。而模型走的是 ckpt 自带的 `transformers_modules.<repo>.modeling_prismatic`（`trust_remote_code=True`）。要让 Marlin / GPTQ / AWQ 生效，只能二选一：

1. 在 ckpt 的 modeling 代码里打补丁；或
2. 加载后**手工遍历 `llm_backbone` 把 `nn.Linear` 替换成 Marlin 模块**——但前提是你得先有 GPTQ/AWQ **量化后的权重**。

### 4.2 校准问题

AWQ / GPTQ 的激活统计对 VLA 而言**依赖图像 token**，不能用通用文本校准集糊弄。需要把 LIBERO 观测流水线接进校准循环（数百张真实 obs + prompt，图片部分必须走同一套 `prepare_images_for_vla` 预处理）。

### 4.3 环境风险

我们的 venv 是**冻结**的（torch 2.2.0+cu121 / 项目基线）。装 `autoawq` / `auto-gptq` 必须 `--no-deps`，且它们要**现场编译 CUDA kernel**——有把 torch 环境弄坏的历史前科。

> **综合代价：数天工作量 + 环境风险，换 1.10×。**

---

## 5. 替代路径：更该做的三个杠杆（按 ROI 排序）

### ① 砍掉腕部相机：`num_images_in_input: 2 → 1`

- **原理**：直接删掉 256 个图像 token（序列 535 → ~279）和一次完整的 SigLIP 前向。
- **预期**：计算项近乎减半，**−35 ~ −40 ms**（217 → ~180 ms）。
- **代价**：⚠️ OFT 是按 2 图微调的，这是**分布外输入**，必须回归验证成功率。
- **改法**：`run_libero_eval.py --num_images_in_input 1`（`run_libero_eval.py:102`）。

### ② 打掉那 30~58% 的调度开销：CUDA Graph / `torch.compile`

- **原理**：本负载序列长度固定（静态 shape），是 CUDA Graph 的**理想场景**，理论上能把这部分开销压到接近零。
- **预期**：**−20 ~ −45 ms**。
- **改法**：试 `torch.compile(vla.llm_backbone, mode="reduce-overhead")`。
- **注意**：与 bnb 8bit **不兼容**，必须在 BF16 基线上做。

### ③ 打开 flash-attention-2（一行改动）

- **现状**：`openvla_utils.py:284` 的 `attn_implementation="flash_attention_2"` 是**注释掉的** —— 装了 flash-attn 2.5.8 却没用上。
- **预期**：−5 ~ −10 ms（535 token 下注意力 FLOPs 占比不大，主要是省 softmax / mask 的显存往返）。
- **改法**：取消注释。

> **三者叠加：217 → ~135~150 ms。这才是进 150 ms 区间的实际路径，而且每一项都比 AWQ/Marlin 便宜得多。**

---

## 6. 战略层面：这个负载也许根本不该量化

| 事实 | 数值 |
| :--- | :--- |
| A10 物理显存 | 24 GB（23028 MiB） |
| BF16 峰值占用 | 15.8 GiB = **66%** |
| BF16 时延 vs 400 ms 预算 | 217 ms → **余量 1.84×** |

**BF16 在 A10 上本来就装得下，本来就满足实时预算。** INT8 省下的 6.5 GiB 显存**没有任何人需要**，代价是 2.56× 时延；W4A16 会买来更多没人需要的显存。

所以 INT8 实验的真正价值是**方法论演示**：

> 对 OpenVLA-OFT 做 **99.8% 覆盖**的 INT8 量化，显存 **−41.2%**、成功率 **+0.4 pp（无损）**，但时延 **0.391×** —— 因为 `LLM.int8()` 的动态离群值分解与 `batch=1` 的访存特征根本不匹配。

**这个结论本身就是教材的核心论点**，它解释了「为什么量化不一定提速」。而「转向 W4A16」如果只换来 1.10×，会把这条论点稀释成一个平庸的工程细节。

**什么时候 W4A16 才值得做**：
- 目标机型是 **12 GB 卡**（BF16 装不下）；
- 需要在一张卡上**并发跑多个策略实例**；
- 要上 Jetson / Orin 等边缘设备。

**在 A10 上，答案是不做。**

---

## 7. 建议的下一步（一锤定音，约 20 分钟）

在做任何量化改造之前，先用**一次 profile** 把 §3 的估算变成实测：

```bash
cd /mnt/workspace/openvla-oft; source /mnt/workspace/activate_oft.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python /mnt/workspace/profile_stages.py --checkpoint <ckpt> --iters 30
```

脚本需分别计时三段 + 统计 kernel 数：

| 计时对象 | 说明 |
| :--- | :--- |
| `vision_backbone` | 2 次 SigLIP 前向 |
| `llm_backbone` | 单次 prefill（~535 token × 32 层） |
| `action_head` + `proprio_projector` | L1 回归头 |
| `torch.profiler` | kernel 总数、CPU 发射时间占比、HBM 读写字节数 |

**判据**：

| 实测结果 | 应采取的行动 |
| :--- | :--- |
| `llm_backbone` 的**权重搬运占比 > 40%** | 本评估的判决错了 → W4A16 值得做 |
| **< 20%**（按当前数据强烈预期如此） | 放弃 W4A16，转 §5 的三个杠杆 |
| 计算项占比 > 50% 且 MFU < 40% | 瓶颈是 kernel 效率 → 优先 CUDA Graph / flash-attn |
| 三段之和 << 217 ms | 大块时间在 Python / 预处理 / dtype cast → 优先消灭 host 侧开销 |

---

## 附：本文用到的全部硬数据出处

| 数值 | 出处 |
| :--- | :--- |
| 权重 14.39 GiB | `openvla_oft_fp16_baseline.json` → `load_allocated_gib` |
| 时延 219.52 ms（bench） | 同上 → `latency_ms/mean_ms`（30 次，warmup 5） |
| 时延 217.24 ms（评测） | 500 episodes 配对实测（`metrics.csv`） |
| token 构成 256 / 256 / ~23 | `modeling_prismatic.py:1026` + `openvla_utils.py:792-817` + `:981` |
| 量化覆盖 436 / 99.8% | `audit_int8_coverage.txt` |
| 显存 15839 → 9315 MiB | 500 episodes 配对实测 |
| A10 HBM 600 GB/s、BF16 125 TFLOPS | NVIDIA 官方规格 |

**尚未实测、标注为估算的项**：计算项与调度开销的切分（§3）、各杠杆的收益区间（§5）。这些必须由 §7 的 profile 定案，**不要直接写进对外报告**。
