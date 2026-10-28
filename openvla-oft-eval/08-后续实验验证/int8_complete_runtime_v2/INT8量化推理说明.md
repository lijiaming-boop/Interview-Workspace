# OpenVLA-OFT 并行解码与 INT8 混合精度评测

## 1. OFT 并行解码是什么

标准自回归策略逐 token / 逐动作生成，后一个输出依赖前一个输出，8 个动作需要连续解码多次。
OpenVLA-OFT 的连续动作头一次接收一组动作位置对应的 LLM hidden states，并行输出
`8 × 7` 的 action chunk。一次模型查询的延迟因此可以摊到 8 个动作上：

```text
policy query FPS = 1000 / 平均单次 forward 延迟(ms)
predicted action FPS = 8 × policy query FPS
```

`num_open_loop_steps=8` 表示执行完整 8 步后重新观察；设为 4 时模型仍预测 8 步，但只执行前 4 步，
所以闭环更紧、模型查询更频繁。视频编码的 30 FPS 与推理 FPS、控制频率都不是同一个指标。

## 2. 当前项目中的 INT8 实际覆盖范围

`load_in_8bit=True` 传给 Hugging Face `AutoModelForVision2Seq.from_pretrained`，bitsandbytes 会替换
VLA 主模型中符合条件的线性层。具体覆盖可能同时涉及 LLM 和视觉编码器，取决于模型模块及
Transformers/bitsandbytes 版本，因此不能未经运行时审计就写成“只量化 LLM”。

OFT 的 L1 action head 与 proprio projector 不在这个 Hugging Face 主模型里面：它们在主模型加载后
单独构造并载入 checkpoint。当前代码明确把 action head 保持为 `torch.bfloat16`，因此本实验是：

```text
BF16 baseline
vs
INT8 VLA 主模型 + BF16 外置 action head/proprio projector
```

这里用 BF16 而不是字面 FP16，是因为当前 checkpoint 和推理代码原生使用 BF16。强行把动作头改成
FP16 还需要同步转换输入 hidden states，否则会发生 dtype 不匹配；也会引入第二个实验变量。

## 3. 为什么动作头保留高精度

视觉编码器的误差会改变目标几何特征，LLM 主干的误差会改变语言—视觉融合后的 hidden states；
动作头则把这些 hidden states 直接映射到末端位置、旋转和夹爪命令。动作输出的微小数值偏差会在
开环执行中累积，并可能跨过夹爪开合阈值，所以动作头通常比分类 logits 更敏感。

因此首轮工程方案先量化主模型、保留动作头 BF16，再通过同状态成功率判断能否接受。若成功率下降，
下一步应分别跳过视觉编码器或部分 LLM 层量化，而不是继续量化动作头。

## 4. 运行与指标

快速对照运行 5 个已知困难状态：

```bash
RUN_TAG=int8-quick-$(date +%Y%m%d-%H%M%S) bash tools/09_int8_benchmark.sh all
```

完整 LIBERO-Spatial 500 episodes：

```bash
TARGETS="" RUN_TAG=int8-full-$(date +%Y%m%d-%H%M%S) bash tools/09_int8_benchmark.sh all
```

自动产物：

```text
<RESULT_BASE>/summary/metrics.csv
<RESULT_BASE>/summary/report.md
```

报告包含去掉前三个 warmup chunk 后的平均/P50/P95 延迟、policy query FPS、8-action chunk
吞吐、`nvidia-smi` 采样峰值显存、成功率及成功率损失。快速 5 状态结果只用于冒烟检查；量化成功率
损失应以完整 500 episodes 的同状态 BF16/INT8 对照为准。
