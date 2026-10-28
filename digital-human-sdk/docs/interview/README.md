# Digital Human SDK 面试文档学习路线

本目录是项目面试准备的统一入口。内容按“底层基础 → 音视频与模型 → 系统架构 → 核心实现 → 调试 → 测试 → 性能 → 面试表达”组织。

面试回答以当前源码为准。`docs/reference/`、`docs/review/`、部分早期架构文档记录的是历史设计与演进过程，可以作为案例，但不能代替当前实现口径。

## 1. 推荐阅读顺序

| 顺序 | 文档 | 解决的问题 |
|---|---|---|
| 1 | [C++、内存与并发基础](01_cpp_concurrency.md) | 为什么使用 RAII、Pimpl、有界队列、条件变量和协作式停止 |
| 2 | [音视频与 Wav2Lip 基础](02_audio_video_ai.md) | PCM、Mel、PTS、人脸对齐、张量和 ncnn 推理是什么 |
| 3 | [系统架构设计](03_system_architecture.md) | 模块如何分层，数据、线程和依赖如何流动 |
| 4 | [核心实现导读](04_core_implementation.md) | 一段音频和一张头像如何变成带口型的视频流 |
| 5 | [调试方法与案例](05_debugging_playbook.md) | 怎样从现象定位到 PCM、Mel、模型、渲染或发布阶段 |
| 6 | [测试与质量体系](06_testing_strategy.md) | 怎样验证功能、并发、故障、ABI 和端到端链路 |
| 7 | [性能工程](07_performance_engineering.md) | 怎样定义指标、采样、定位热点并证明优化有效 |
| 8 | [面试表达手册](08_interview_playbook.md) | 怎样用 30 秒、90 秒和 3 分钟讲清项目 |

按岗位分类的扩展问题见 [C++ 岗位面试准备指南](../career/cpp_interview_guide.md)。

## 2. 当前项目统一口径

面试中应统一使用以下事实：

- 项目是 C++17 数字人 SDK。文本生成和 TTS 是外部服务；核心 SDK 负责音频特征、人脸处理、Wav2Lip 推理、图像融合和媒体发布。
- 核心 Pipeline 是 **5 个 worker、7 级有界队列**，不是 7 个 worker。五个 worker 是 AudioProcessor、VideoProcessor、AVMatcher、InferenceWorker 和 RenderThread。
- OutputProcessor 在 RenderThread 中同步执行，不是独立线程。
- 完整实时会话通常包含 4 个会话线程、5 个核心 worker、1 个 Bridge 输出线程和 1 个 Publisher 线程，共约 11 条后台线程。
- 默认音频契约是 16 kHz 单声道 float PCM。音频 frame 为 800 个采样点，hop 为 200 个采样点，即 50 ms 帧长、12.5 ms 步长。
- Matcher 由视频帧驱动，每帧组装 `16 × 80` Mel 窗，覆盖约 200 ms 音频上下文；Mel 归一化滚动上下文默认 300 帧 ≈ 3 s。
- 停止协议是：给音频/视频上游打 EOS → 停止全部 7 个队列唤醒等待者 → 按注册逆序请求 worker 停止 → 在共享 deadline 内等待退出（默认 2 s）。`Stop()` 之后不可再 `Start()`，超时也只可再次 `Stop()`。
- 实时链路的数据源是外部 TTS 的流式 PCM。`RingBuffer` 已具备，但**没有麦克风采集回调**（PortAudio 目前只用于播放）。
- 最终音轨与驱动口型的是同一份 TTS PCM 的两路拷贝：`ConversationStreamBridge` 把渲染帧抽给 `StreamPublisher`，后者编码 H.264/AAC 发布到 RTMP / RTSP / 文件。
- 当前人脸链路使用 SCRFD 检测、2D106 关键点、`96 × 96` 对齐和口部 mask，不再使用 dlib 作为主链路。
- Wav2Lip 人脸输入是 `96 × 96 × 6`：前三通道为下半脸遮罩后的 RGB，后三通道为完整 RGB。
- 模型输入 blob 为 `audio_sequences`、`face_sequences`，输出 blob 为 `output`。
- 构建结构是多个内部 STATIC 职责模块合入一个 SHARED facade，不是早期报告中的 OBJECT 库。

主要源码入口：

- [`src/core/pipeline.cpp`](../../src/core/pipeline.cpp)
- [`src/dialog/conversation_session.cpp`](../../src/dialog/conversation_session.cpp)
- [`src/core/audio_processor.cpp`](../../src/core/audio_processor.cpp)
- [`src/core/video_processor.cpp`](../../src/core/video_processor.cpp)
- [`src/core/inference_worker.cpp`](../../src/core/inference_worker.cpp)
- [`src/core/render_thread.cpp`](../../src/core/render_thread.cpp)
- [`src/media/stream_publisher.cpp`](../../src/media/stream_publisher.cpp)

## 3. 回答问题的统一结构

技术问题建议使用五段式回答：

1. 先给结论，用一句话说明设计或根因。
2. 说明约束，例如实时性、线程安全、模型输入或 ABI。
3. 结合当前实现说明数据怎样流动。
4. 说明取舍和已知不足，不把规划说成已实现。
5. 给出验证方法或指标，证明结论不是猜测。

调试和优化案例建议使用：

> 现象 → 最小复现 → 关键证据 → 根因链 → 修复 → 回归测试 → 剩余边界

## 4. 当前能力边界

以下内容应主动说明，不应包装成已完成能力：

- Pause 当前只改变状态，还没有真正阻止 worker 消费数据。
- 音频 PTS、帧调度和发布端时间线还没有收敛到一个统一 MediaClock。
- 用户打断的完整 CancelTurn/Flush/epoch 语义仍需完善。
- ASR 接口存在，但完整的识别结果提交和打断闭环尚未完成。
- C API 缺少完整的输出帧读取接口，部分配置和指标仍需校正。
- GPU 配置存在默认值不一致：SDK 的 `enable_gpu` 默认关闭，但底层 inferencer 默认启用，当前不能仅凭上层配置断言实际后端。
- 构建缓存、模型和测试资源已清理，当前不能宣称现版本已经重新构建或测试通过。
- 没有麦克风采集：PortAudio 只用于播放，`RingBuffer` 尚未接到任何采集设备。
- `face_size` 只校验大于 0，非 96 会静默产生错误结果，缺少显式校验与报错。

## 5. 根目录面试稿与本目录的关系

仓库根目录另有两份可直接背诵的稿子，内容已按本目录口径同步：

- `数字人sdk面试文档_优化版.md`：Q1–Q34 表达稿，含会话编排、媒体发布、C API 与模型工程章节。
- `数字人SDK项目面试题库_整合补充版.md`：六部分题库，开头“七个统一口径”与本目录一致。

两者与本目录冲突时以本目录为准；发现不一致应回来修本目录，而不是改稿子去迁就旧文档。

## 6. 历史材料的使用方式

- [当前架构总览](../architecture/overview.md)是最接近现状的系统说明。
- [Mel 预处理修复](../bugfix/mel_preprocess_fix.md)适合作为数值与模型调试案例。
- [Pimpl Move 崩溃修复](../bugfix/pimpl_move_semantics_fix.md)适合作为 C++ 生命周期案例。
- [性能火焰图分析](../perf/perf_flamegraph_analysis.md)适合作为性能方法论案例，其中数字只代表当时环境。
- `reference/` 和 `review/` 统一视为历史材料。出现线程数、模型、人脸检测器或性能数字冲突时，以当前源码和本目录为准。
