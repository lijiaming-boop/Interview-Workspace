# 性能工程

## 1. 30 秒回答

性能优化从可复现基准开始：固定提交、模型、素材、机器、线程设置和预热，分别测内容 FPS、墙钟吞吐、首帧、端到端 p95/p99、阶段耗时、队列驻留、A/V 偏差、CPU/GPU 和内存。先用分段计时确定瓶颈，再用 profiler 和火焰图解释 CPU 去向，单变量 A/B 验证优化，同时用口型质量和同步指标防止以降质换速度。

## 2. 基准契约

没有以下信息的性能数字不可比较：

- Git commit 和构建类型；
- 编译器、依赖版本和优化选项；
- 模型文件 SHA、精度和后端；
- 头像、音频、帧数和分辨率；
- CPU/GPU、驱动、功耗模式；
- OpenCV、OpenMP、ncnn 线程设置；
- warmup 次数、测量次数和统计方法；
- 是否包含检测、编码、网络和文件 IO。

建议基准使用 `RelWithDebInfo`，既保留优化也保留可解析符号。

## 3. 指标定义

### 吞吐与实时性

```text
content_fps = 输出帧数 / 媒体内容时长
throughput_fps = 输出帧数 / 墙钟运行时长
speed_x = 媒体内容时长 / 墙钟运行时长
```

`content_fps = 25` 只说明内容按 25 FPS 编号，不代表程序实时生成。只有 `speed_x ≥ 1` 才说明不慢于实时。

### 延迟

```text
e2e_latency(seq) = emit_wall_time(seq) - ingress_wall_time(seq)
queue_residence = dequeue_wall_time - enqueue_wall_time
```

必须用同一 seq 关联入口和出口。用“当前墙钟减媒体 PTS”得到的是调度迟到量，不是严格的逐包端到端延迟。

### 建议指标

| 类别 | 指标 |
|---|---|
| 用户体验 | first token、first PCM、first frame、首包 |
| 延迟 | 各阶段 p50/p95/p99、E2E p50/p95/p99 |
| 流水线 | 队列深度/驻留、任务/输出帧比、DROP/SKIP/DUPLICATE |
| 同步 | AV skew p50/p95/p99、PTS 单调性 |
| 资源 | CPU、线程、上下文切换、RSS、GPU/显存 |
| 发布 | 编码耗时、码率、重连次数/耗时、写包错误 |
| 质量 | 嘴部运动、滞后相关性、黑帧、人工审片 |

平均值会掩盖少量严重卡顿，实时系统至少报告 median、p95、p99、最大值和样本量。

## 4. 定位顺序

1. **验证正确性**：错误的任务数、PTS 或重试会放大负载。
2. **阶段计时**：Audio、Video、Matcher、Infer、Render、Encode。
3. **看队列**：持续增长的上游队列指向下游瓶颈。
4. **采样 profiler**：判断计算、内存复制、锁还是忙等。
5. **展开符号和火焰图**：定位具体库和函数。
6. **单变量 A/B**：每次只改一个线程、缓存或后端选项。
7. **复测质量**：速度、同步和视觉质量一起验收。

## 5. 案例一：OpenMP Busy Wait

历史采样中，libgomp barrier 自旋一度占约 60.3% self CPU。设置 `OMP_WAIT_POLICY=passive` 和 `GOMP_SPINCOUNT=0` 后，历史对比下降到约 0.6%。“模型加载前设置线程数”是随后提出的独立优化建议，不应与这次实测因果混在一起。

这个案例的重点不是背数字，而是方法：

```text
CPU 很高但吞吐不高
  → DSO 聚合发现 libgomp
  → symbol 聚合发现 barrier 自旋
  → 检查外部 worker 与内部 OpenMP 过度订阅
  → 先调整 OpenMP 等待策略
  → 再把线程预算作为独立变量测试
  → 固定输入 A/B
```

数字来自特定 WSL/CPU/提交，只能称为历史案例，不能作为当前基线。见 [火焰图分析](../perf/perf_flamegraph_analysis.md)。

## 6. 案例二：任务放大

Mel hop 为 12.5 ms，理论产生约 80 个 Mel 时间步每秒；目标视频约 25 FPS。如果每个 Mel 都驱动推理，任务数会显著高于输出帧数。

face-driven Matcher 让每个视频帧最多产生一个任务。关键指标不是只看推理耗时，而是：

```text
inference_tasks / output_frames
retries / inference_tasks
stale_drops / inference_tasks
```

如果比值远大于 1，先修任务生成和重试策略，再优化单次模型。

## 7. 案例三：ROI 与 Amdahl 定律

将融合限制在嘴部/人脸 ROI 能减少 OpenCV 运算，但端到端收益受其他阶段限制。假设推理占总耗时 60%，即使 GPU 让推理快 10 倍，理论总加速也约为：

```text
1 / (0.4 + 0.6 / 10) ≈ 2.17×
```

模型 kernel 的 10× 不等于端到端 10×，因为检测、转换、复制、融合、编码和排队仍然存在。

## 8. 当前值得优化的热点

先测量再决定是否实施：

- Pipeline 输入、Render 和 Publisher 存在多次全图 clone；
- AudioProcessor 热路径有 `shrink_to_fit` 和临时 vector 分配；
- Mel 提取重复构造 Hann 和 OpenCV 临时矩阵；
- 文本 SSE 客户端保留完整响应并频繁从 vector/string 前端 erase；
- Publisher 等待音频和视频同时就绪，一路停顿可能饿死另一路；
- 每会话约 11 线程，多会话会与 ncnn/OpenMP 内部线程过度订阅；
- 静态头像缓存的判断成本低，但签名可靠性不足；
- 指标采集以平均值和普通字段为主，尚不足以支持稳定 p95/p99。

## 9. 推荐优化顺序

### 第一层：正确性

- 单调 PTS、正确 EOS、正确任务数；
- 修复无限重试、旧 turn 数据和错误队列策略；
- 先保证输出质量和完成语义。

### 第二层：分配与复制

- 不可变 FrameBuffer/AudioBuffer；
- buffer pool 和 scratch 复用；
- 缓存 Hann、变换矩阵和静态头像特征；
- 增量解析，不保留无用完整响应。

### 第三层：调度

- 每设备共享推理调度器；
- 明确 CPU 核和 OpenMP 线程预算；
- 按 deadline 丢弃过期实时任务；
- 多会话容量准入和公平性。

### 第四层：模型与硬件

- CPU/Vulkan 同一数据集比较；
- FP16/INT8 质量门禁；
- 真实目标硬件上评估 micro-batching；
- 保留 FP32/CPU 回退。

## 10. 性能优化不能牺牲质量

每次优化同时检查：

- 口型对音频的响应；
- AV skew 和长时间漂移；
- 黑帧、边缘接缝和颜色；
- DROP/SKIP/DUPLICATE；
- 首帧与长尾；
- 模型输出数值差异和人工审片。

项目已有 QualityGate 默认阈值，但尚未按正式数据集校准，不能把默认值宣称为通用 SLA。模型量化方法见 [模型精度与量化](../models/model_precision_quantization_guide.md)。

## 11. 当前证据边界

- 原始 `perf.data` 和 folded 文件已经清理；保留的 SVG 和报告是历史证据，不能重新聚合得到新结论。
- 历史报告中的 25 content FPS 不等于墙钟实时；5 秒内容若生成 9 秒，实际约为 `0.56×` 实时。
- 旧报告可能基于不同人脸检测器、Mel 预处理和代码版本。
- 当前没有重新构建和运行基准，因此不能声明当前 CPU/GPU FPS。

## 12. 高频追问

### CPU 100% 但吞吐低怎么办？

先区分有效计算、忙等、锁竞争和过度订阅。看 DSO/symbol 采样、运行线程数、上下文切换和队列驻留，再做线程数单变量测试。

### 为什么平均推理 30 ms 仍可能不实时？

端到端还包含排队、预处理、渲染和编码；p99 抖动也会使队列增长。需要同时看 throughput、E2E 和各队列驻留。

### 怎样证明优化不是降质？

固定数据集比较数值、口型运动、同步和人工审片；性能结果和质量结果使用同一提交、模型和素材共同发布。
