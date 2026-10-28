# 核心实现导读

本文按一条真实数据链路阅读源码：初始化 SDK，输入 PCM 和头像，生成口型帧，再编码为直播流。

## 1. 入口与初始化

公共 C++ 入口是 [`include/digital_human_sdk.h`](../../include/digital_human_sdk.h)，实现位于 [`src/core/sdk_entry.cpp`](../../src/core/sdk_entry.cpp)。初始化阶段主要完成：

1. 校验音频、视频、队列和模型配置；
2. 构造 Pipeline 及七级队列；
3. 加载人脸检测、关键点和 Wav2Lip 模型；
4. 配置 CPU/OpenMP 或 Vulkan 后端；
5. 创建 AudioProcessor、VideoProcessor、AVMatcher、InferenceWorker 和 RenderThread；
6. 注册 worker，等待 Start。

模型和队列应在启动线程前验证。否则工作线程启动后才发现权重缺失，错误传播和回收会更复杂。

当前 GPU 配置还有一致性问题：上层 SDK 默认 `enable_gpu=false`，但底层 ModelInferencer 默认 `use_gpu=true`，上层关闭并不可靠等于底层禁用。目标实现应使用 `Auto/CPU/Vulkan` 后端枚举，在模型加载前一次性决定，并对外报告 selected backend 与 fallback reason。

当前模型资源已从工作区清理，阅读代码不受影响，但运行前需要重新准备权重。

## 2. 启动顺序

WorkerRegistry 按注册顺序启动：

```text
Render → Inference → Matcher → Video → Audio
```

这是下游到上游的顺序。Audio 一旦产生 Mel，下游已经具备消费能力。停止请求按相反方向发出，使上游先停止产生新数据。

源码：[`src/core/worker_registry.cpp`](../../src/core/worker_registry.cpp)、[`src/core/pipeline.cpp`](../../src/core/pipeline.cpp)。

## 3. 音频输入到 Mel

调用方通过 `PushAudio(pcm, pts_ms)` 提交 float PCM：

```text
PushAudio
  → AudioRawQueue
  → AudioProcessor
  → pending audio
  → 降噪 / RMS / 预加重 / 分帧 / VAD
  → STFT / Mel / dB / 对称归一化
  → MelFeatureQueue
```

AudioProcessor 需要处理任意 chunk 边界。一个 HTTP/TTS chunk 不一定等于一个模型 frame，因此要把残留样本保存在内部缓冲区，达到 frame size 后再处理，并按 hop 前进。

关键不变量：

- chunk 分片方式不能改变最终 Mel 序列；
- 所有 Mel 值应为有限数；
- 序号和 PTS 单调；
- EOS 时残留数据的补齐或丢弃策略必须确定；
- sample 与 sample frame 不能混用。

当前队列模式会根据内部输出计数重新计算 Mel PTS，没有完整保留输入 PTS。面试时应将其说明为待收敛问题。

## 4. 视频输入到标准人脸

调用方通过 `PushVideo(frame, pts_ms)` 提交 BGR：

```text
PushVideo
  → 入口 clone
  → VideoRawQueue
  → SCRFD 检测
  → 2D106 关键点
  → 仿射对齐 96×96
  → 口部 mask / 逆变换信息
  → ProcessedFaceQueue
```

VideoProcessor 会尝试缓存静态头像的人脸检测、对齐矩阵和 mask，避免相同底图每帧重复计算。当前缓存使用轻量像素签名，适合固定头像演示；更严格的生产方案应由调用方传入 immutable avatar ID 或 revision。

关键源码：[`src/core/video_processor.cpp`](../../src/core/video_processor.cpp)。

## 5. AVMatcher

Matcher 位于 Pipeline 内部，以 face packet 为驱动：

1. 读取一帧标准人脸；
2. 根据该帧 PTS 计算目标 Mel 起点；
3. 从缓存和 Mel 队列获得连续 16 个时间步；
4. 检查 Mel 与视频 PTS 的误差；
5. 生成 InferenceTask。

为什么不能只找“当前最近的一个 Mel”：Wav2Lip 需要时间上下文，不是单点特征。为什么不能每来一个 Mel 就推理：Mel 速率高于目标视频帧率，会重复计算并放大队列压力。

## 6. 推理任务与张量

InferenceWorker 将 OpenCV 数据转换成 ncnn 张量：

```text
Mel: 16×80 → ncnn w=16, h=80, c=1
Face: 96×96 BGR
      → RGB
      → 下半脸遮罩 RGB + 完整 RGB
      → ncnn w=96, h=96, c=6
```

然后创建 Extractor，输入 `audio_sequences` 和 `face_sequences`，读取 `output`。推理结果与原始 face metadata 一起放入 InferenceOutputQueue。

需要分开测量：

- 任务排队时间；
- OpenCV/ncnn 转换；
- Extractor 推理；
- 输出转换；
- 失败、重试和过期任务数量。

源码：[`src/core/inference_worker.cpp`](../../src/core/inference_worker.cpp)、[`src/model/model_inferencer.cpp`](../../src/model/model_inferencer.cpp)。

## 7. Render 与 OutputProcessor

RenderThread 消费推理输出，在同一线程内调用 OutputProcessor：

```text
模型 RGB 输出
  → 转成图像
  → 色彩/边缘处理
  → 逆仿射回原图坐标
  → mask 羽化融合
  → ROI 写回 BGR
  → OutputFrameQueue
```

OutputProcessor 不是独立 worker。ROI 优化减少融合区域，但当前路径仍存在整图 clone，后续可通过不可变 FrameBuffer 和 buffer pool 减少复制。

RenderThread 还负责帧调度。DISPLAY、DROP、DUPLICATE 和 EOS drain 最终应该走统一的 `EmitFrame` 语义，保证回调和 `GetOutputFrame` 得到一致结果。当前 DUPLICATE 和尾帧路径仍需修正。

源码：[`src/core/render_thread.cpp`](../../src/core/render_thread.cpp)、[`src/model/output_processor.cpp`](../../src/model/output_processor.cpp)。

## 8. ConversationSession

实时文本会话使用四条线程：

```text
用户文本
  → Generation：调用 LLM，增量分句
  → TTS：按句生成 PCM
  → Audio Feeder：提交 PCM
  → Video Feeder：根据音频水位提交头像帧
```

这样做让网络、TTS 和核心反压互相隔离，但也增加了共享状态和关闭复杂度。每轮回复和整个会话不是同一生命周期：每轮结束不能向 one-shot Pipeline 发送最终 EOS，否则下一轮无法继续。

源码：[`src/dialog/conversation_session.cpp`](../../src/dialog/conversation_session.cpp)。

## 9. Bridge 与 Publisher

ConversationStreamBridge 将 TTS PCM 分成两路：

- 一路进入 DigitalHumanSDK，驱动口型；
- 一路进入 StreamPublisher，作为最终音轨。

Bridge 独立线程读取渲染 BGR，再送入 Publisher。Publisher 完成：

```text
BGR → swscale → YUV420P → H.264
PCM → swresample → Audio FIFO → AAC
H.264 + AAC → interleaved mux → RTMP / RTSP / File
```

视频队列过载时旧帧价值较低，可以丢旧；音频更重视连续性，通常限时等待。两者都必须有最大媒体时长和关闭策略，不能无限积压。

源码：[`src/media/conversation_stream_bridge.cpp`](../../src/media/conversation_stream_bridge.cpp)、[`src/media/stream_publisher.cpp`](../../src/media/stream_publisher.cpp)。

## 10. EOS、Stop 与打断

三个概念不能混用：

| 事件 | 含义 | 期望行为 |
|---|---|---|
| EOS | 正常输入结束 | 下游排空、flush、输出尾帧 |
| Stop | 会话或 SDK 终止 | 不再接受输入，按 deadline 回收线程 |
| Barge-in | 取消旧 turn | 丢弃旧 epoch，保持会话和全局 PTS 继续 |

当前打断会重置部分上层游标，但核心和发布器保持运行，可能破坏单调时间线。目标设计应让每个 packet 携带 `session_id + turn_epoch + pts + deadline`，打断只提升 epoch，并让各阶段丢弃旧任务。

## 11. C++ API 与 C API

C++ API 提供 Init、Start、PushAudio、PushVideo、GetOutputFrame、EOS、Stop 和 Metrics。C API 使用 opaque handle 和 POD 配置，但目前缺少完整的输出帧接口，部分 width/height 和 metrics 映射也未落实。

推荐演进方向：

- 稳定 C ABI 作为跨语言、跨编译器边界；
- C++ wrapper 提供 RAII 和类型安全；
- `struct_size` 使用前缀兼容读取；
- 输出使用 caller-owned buffer 或明确生命周期的回调；
- API version、库版本和能力查询由构建系统统一生成。

## 12. 源码阅读顺序

1. [`include/digital_human_sdk.h`](../../include/digital_human_sdk.h)
2. [`src/core/sdk_entry.cpp`](../../src/core/sdk_entry.cpp)
3. [`src/core/pipeline.cpp`](../../src/core/pipeline.cpp)
4. [`src/core/audio_processor.cpp`](../../src/core/audio_processor.cpp)
5. [`src/core/video_processor.cpp`](../../src/core/video_processor.cpp)
6. [`src/core/inference_worker.cpp`](../../src/core/inference_worker.cpp)
7. [`src/core/render_thread.cpp`](../../src/core/render_thread.cpp)
8. [`src/dialog/conversation_session.cpp`](../../src/dialog/conversation_session.cpp)
9. [`src/media/stream_publisher.cpp`](../../src/media/stream_publisher.cpp)

## 13. 高频追问

### 为什么 OutputProcessor 不独立成线程？

它与 Render 的一帧结果强关联，同步执行简化所有权和输出顺序。只有后处理成为独立瓶颈且能明确队列、时序和内存收益时，才值得再拆线程。

### 为什么每次推理创建 Extractor？

Net 保存长期模型，Extractor 保存单次执行状态。按任务创建可以隔离输入输出和中间状态；优化时应通过基准确认创建成本，而不是直接跨线程复用可变 Extractor。

### 为什么音频与视频使用不同过载策略？

旧视频帧错过显示时间后价值很低；音频丢样本会产生爆音并改变时间轴。但音频也不能无限等待，因此需要 deadline 和失败策略。

### 一帧最终没有输出，应该看哪些阶段？

按同一 seq/PTS 检查 VideoRaw、ProcessedFace、InferenceTask、InferenceOutput 和 OutputFrame，确定在哪个阶段消失，再检查对应的 SKIP/FATAL/EOS 和队列指标。
