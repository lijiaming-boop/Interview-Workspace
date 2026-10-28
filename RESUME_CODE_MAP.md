# 🗺️ `RESUME_CODE_MAP.md` —— 简历描述与工业级代码物理坐标全景映射表

> **唯一基准**：严格对齐 `RESUME_FINAL.md` 中的每一条实习与项目技术描述，将全部 **11 条简历要点 / 10 个核心技术节点（`Node-01` ~ `Node-10`）** 映射到工作区真实源码的物理路径、核心类/函数签名、起止行号、底层运行逻辑与三项防伪代码细节。

---

## 一、全局节点与简历条目速查矩阵

| 节点编号 | 所属模块 | 对应 `RESUME_FINAL.md` 原文核心机制 | 主控源码物理坐标 |
| :--- | :--- | :--- | :--- |
| **`Node-01`** | **项目一：C++ 数字人 SDK**（第 1 条） | C++17 流量反压有界阻塞队列 + SPSC 无锁环形缓冲区 + 四阶段流水线解耦 | `digital-human-sdk/include/core/thread_safe_queue.h` (`L21-L584`)<br>`digital-human-sdk/include/core/spsc_ring_buffer.h` (`L56-L260`)<br>`digital-human-sdk/src/audio/audio_ring_buffer.cpp` (`L13-L142`)<br>`digital-human-sdk/src/core/pipeline.cpp` (`L50-L520`) |
| **`Node-02`** | **项目一：C++ 数字人 SDK**（第 2 条） | SCRFD + 106→68 点人脸对齐 + Wav2Lip NCNN 图引擎 + OpenMP 被动等待 + 静态人脸缓存 + 嘴部 ROI 局部融合（`113ms -> 14.3ms`） | `digital-human-sdk/src/model/model_inferencer.cpp` (`L26-L773`)<br>`digital-human-sdk/src/core/inference_worker.cpp` (`L147-L292`)<br>`digital-human-sdk/src/core/face_detector.cpp` (`L24-L376`)<br>`digital-human-sdk/src/core/video_processor.cpp` (`L51-L211`)<br>`digital-human-sdk/src/model/output_processor.cpp` (`L514-L617`) |
| **`Node-03`** | **项目一：C++ 数字人 SDK**（第 3 条） | 以音频播放时钟为主时钟（Audio Master Clock）动态 PTS 校准 + 帧丢弃/补偿 + FFmpeg H.264/AAC RTMP/RTSP 推流（音画同步误差 `< 30ms`） | `digital-human-sdk/src/core/av_sync.cpp` (`L12-L185`)<br>`digital-human-sdk/src/core/render_thread.cpp` (`L131-L212`)<br>`digital-human-sdk/src/core/audio_sync_scheduler.cpp` (`L20-L157`)<br>`digital-human-sdk/src/media/stream_publisher.cpp` (`L28-L300`) |
| **`Node-04`** | **项目一：C++ 数字人 SDK**（第 4 条） | 显式线程与会话生命周期状态机（FSM）+ 协作式取消（Cooperative Cancellation）+ 彻底消除 `std::thread::detach()` 悬挂引用 + 30+ 项单测与 Valgrind 零泄漏 | `digital-human-sdk/include/core/thread_base.h` (`L20-L270`)<br>`digital-human-sdk/src/core/worker_registry.cpp` (`L9-L95`)<br>`digital-human-sdk/include/dialog/conversation_session.h` (`L21-L115`)<br>`digital-human-sdk/src/dialog/conversation_session.cpp` (`L42-L300`)<br>`digital-human-sdk/tests/core_infrastructure_coverage_test.cpp` (`L82-L330`) |
| **`Node-05`** | **实习：蘑菇物联 Agent 后端**（第 1 条） | 基于 Pydantic 标准 JSON Schema 工具注册表 + 有限状态机（FSM）动态白名单裁剪（10 步预算、连续 3 次异常熔断、`fill_ratio > 25%` 挂起落盘与断点恢复） | `moguwulian-agent-backend/.../agent/orchestrator.py` (`L41-L357`)<br>`moguwulian-agent-backend/.../agent/tools.py` (`L35-L595`)<br>`moguwulian-agent-backend/.../agent/brain.py` (`L25-L227`) |
| **`Node-06`** | **实习：蘑菇物联 Agent 后端**（第 2 条） | FastAPI + SQLAlchemy 2.0 异步会话持久化 + Redis 租约与后台协程心跳续租跨进程分布式锁 + SSE 断线重连（`Last-Event-ID`）增量回放 | `moguwulian-agent-backend/.../apps/api/lock_lease.py` (`L20-L246`)<br>`moguwulian-agent-backend/.../apps/api/storage.py` (`L68-L636`)<br>`moguwulian-agent-backend/.../apps/api/events.py` (`L16-L189`)<br>`moguwulian-agent-backend/.../apps/api/agent_service.py` (`L31-L482`) |
| **`Node-07`** | **实习：蘑菇物联 Agent 后端**（第 3 条） | Go 反向代理网关（`httputil.ReverseProxy` + `FlushInterval=-1` 禁用缓冲透传 SSE）+ 滑动窗口限流 + JWT 鉴权与 `X-Tenant-ID` 头部重写 + 36 个回归用例（工具调用准确率 `96.4%`） | `moguwulian-agent-backend/.../proxy/main.go` (`L36-L469`)<br>`moguwulian-agent-backend/.../apps/api/deps.py` (`L18-L186`)<br>`moguwulian-agent-backend/.../evaluation/run_eval.py` (`L18-L231`)<br>`moguwulian-agent-backend/.../evaluation/report.md` (`L1-L46`) |
| **`Node-08`** | **实习：蘑菇物联 Agent 后端**（第 4 条） | 10.5 万条真实工况数据与机理模型 -> 7.2 万条结构化样本与 6 步 CoT 数据合成 + 14 类物理约束规则过滤幻觉样本 + Qwen3-8B SFT 与设备安全护栏拦截（越界率 `0%`） | `Cooling Tower Modeling/全系统寻优/07_cot_data_factory.py` (`L41-L487`)<br>`Cooling Tower Modeling/全系统寻优/09_cot_dataset_builder.py` (`L19-L231`)<br>`Cooling Tower Modeling/全系统寻优/10_quality_sample_selector.py` (`L25-L388`)<br>`Cooling Tower Modeling/全系统寻优/12_validate_student_physics.py` (`L24-L422`)<br>`Cooling Tower Modeling/全系统寻优/inference/safety_guard.py` (`L11-L84`) |
| **`Node-09`** | **项目二：OpenVLA-OFT 具身智能**（第 1 条 + 第 3 条） | HDF5→RLDS 转换管线 + LIBERO-Spatial 500 Episodes 基座评测（`99.0%`）与 MuJoCo 接触动力学诊断 + BridgeData V2 LoRA（`rank=32`）20k 轨迹拐点（SIMPLER `41.3% -> 60.5%`）+ Isaac Lab 7D 动作映射与三阶域对齐（`+17.5 pp`） | `openvla-oft-eval/tools/convert_libero_hdf5_to_rlds.py` (`L20-L155`)<br>`openvla-oft-eval/tools/bridge_lora_scaling_eval.py` (`L26-L209`)<br>`openvla-oft-eval/09-行为蒸馏实验/scripts/finetune_teacher_distill.py` (`L65-L305`)<br>`openvla-oft-eval/12-IsaacLab-Bridge-Franka/bridge_action_mapper.py` (`L16-L120`)<br>`openvla-oft-eval/12-IsaacLab-Bridge-Franka/domain_randomization.py` (`L11-L150`)<br>`openvla-oft-eval/12-IsaacLab-Bridge-Franka/run_closed_loop.py` (`L84-L254`) |
| **`Node-10`** | **项目二：OpenVLA-OFT 具身智能**（第 2 条） | OFT 并行解码（`NUM_ACTIONS_CHUNK=8`，400ms 控制时间预算，Open-loop `8 vs 4` 消融）+ `bitsandbytes` INT8 混合精度量化审计（436 层 `Linear8bitLt` 覆盖 99.8%，显存 `-41.2%`，`Batch=1` 时延 `217ms -> 555ms` 劣化根因） | `openvla-oft-eval/02-实验代码/patch-check/prismatic/models/action_heads.py` (`L84-L108`)<br>`openvla-oft-eval/10-INT8量化与并行解码教材/audit_int8_coverage.py` (`L49-L220`)<br>`openvla-oft-eval/08-后续实验验证/scripts/09_int8_benchmark.sh` (`L1-L145`)<br>`openvla-oft-eval/08-后续实验验证/scripts/summarize_int8_benchmark.py` (`L38-L185`)<br>`openvla-oft-eval/10-INT8量化与并行解码教材/analyze_regressions.py` (`L188-L320`) |

---

## 二、逐节点深度源码坐标与防伪细节拆解

### 📌 `Node-01`：C++ 数字人 SDK —— 并发流水线、流量反压有界阻塞队列与 SPSC 无锁环形缓冲区

- **对应简历原文**：
  > **高并发流水线与无锁缓冲架构**：采用 C++17 实现支持流量反压（Backpressure）的线程安全有界阻塞队列与单生产者单消费者（SPSC）无锁环形缓冲区，解耦音频解码、特征提取、模型推理与实时推流四阶段流水线，消除高频音频采样与视频帧合成间的锁竞争与内存拷贝开销。

- **核心物理坐标与符号映射**：
  1. **支持流量反压的线程安全有界阻塞队列**：
     - 文件：[`digital-human-sdk/include/core/thread_safe_queue.h`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/include/core/thread_safe_queue.h#L21-L584)
     - 核心结构体与类：`struct QueueMetrics`（`L21-L48`）、`template <typename T> class ThreadSafeQueue`（`L68-L584`）
     - 关键函数：
       - `bool Push(const T& item)` / `bool Push(T&& item)` / `bool Emplace(Args&&... args)`（`L98-L188`）：当 `queue_.size() >= max_capacity_ && max_capacity_ > 0` 时，生产者线程在条件变量 `not_full_cv_.wait(lock, [this]() { return stopped_.load(std::memory_order_acquire) || queue_.size() < max_capacity_; })` 上挂起，天然向上游传递**流量反压（Backpressure）**。
       - `bool TryPush(T&& item)`（`L198-L250`）：非阻塞尝试入队，满载时递增 `total_overflows_.fetch_add(1, std::memory_order_relaxed)` 并返回 `false`。
       - `bool WaitAndPop(T& item, int timeout_ms = -1)`（`L262-L315`）：支持无限等待与基于 `std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms)` 的 `not_empty_cv_.wait_until` 超时出队，出队后调用 `not_full_cv_.notify_one()` 唤醒被反压挂起的上游生产者。
       - `size_t TryPopBatch(std::vector<T>& batch, size_t max_items = 0)`（`L342-L378`）：批量移动出队并执行 `not_full_cv_.notify_all()`。
       - `void Stop()`（`L424-L432`）：置位 `stopped_.store(true, std::memory_order_release)` 并同时广播 `not_empty_cv_.notify_all()` 与 `not_full_cv_.notify_all()`，确保优雅停机时无死锁。
  2. **通用编译期容量 SPSC 无锁环形缓冲区**：
     - 文件：[`digital-human-sdk/include/core/spsc_ring_buffer.h`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/include/core/spsc_ring_buffer.h#L56-L260)
     - 核心模板类：`template <typename T, std::size_t Capacity> class SpscRingBuffer`
     - 关键机制：
       - 静态断言 `static_assert((Capacity & (Capacity - 1)) == 0)` 强制容量为 2 的幂次，使用位掩码 `idx & kMask`（`kMask = Capacity - 1`）替代取模除法指令。
       - 生产者写指针 `alignas(kCacheLineSize) std::atomic<size_type> tail_{0}` 与消费者写指针 `alignas(kCacheLineSize) std::atomic<size_type> head_{0}` 各自独占 64 字节缓存行（`kCacheLineSize = 64`），并分别维护对端下标本地缓存 `cached_head_` 与 `cached_tail_`，仅当本地快照判满/判空时才通过 `std::memory_order_acquire` 刷新对端原子变量，消除多核高频伪共享（False Sharing）。
       - 采用 `std::aligned_storage_t<sizeof(T), alignof(T)>` 原始槽位 + `::new (RawSlot(tail)) T(...)` 原位构造 + `std::launder` 访问 + 显式析构 `slot->~T()`，支持非平凡类型（如含 `cv::Mat` / `std::vector` 的帧结构）零额外堆分配流转。
  3. **音频 PCM 专用运行期容量 SPSC 无锁环形缓冲区**：
     - 文件：[`digital-human-sdk/include/audio/audio_ring_buffer.h`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/include/audio/audio_ring_buffer.h#L10-L38) 与 [`digital-human-sdk/src/audio/audio_ring_buffer.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/audio/audio_ring_buffer.cpp#L13-L142)
     - 核心实现：`struct RingBuffer::Impl`（`L13-L94`），使用单调递增的 64 位无符号索引 `alignas(64) std::atomic<uint64_t> writeIdx{0}` 与 `readIdx{0}`，通过两段式回绕 `std::memcpy`（`firstPart = std::min(toWrite, capacity - writePos)`）在 PortAudio 实时回调线程与 `AudioProcessor` 特征线程间零锁传递 float PCM 采样。
  4. **四阶段异步流水线解耦与音画窗口配对**：
     - 文件：[`digital-human-sdk/include/core/pipeline.h`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/include/core/pipeline.h#L23-L307) 与 [`digital-human-sdk/src/core/pipeline.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/core/pipeline.cpp#L50-L520)
     - 核心配置与线程编排：`PipelineConfig` 定义六条有界队列容量（`audio_raw_queue_size = 30`, `mel_queue_size = 60`, `video_raw_queue_size = 30`, `face_queue_size = 30`, `infer_queue_size = 30`, `output_queue_size = 10`）；`Pipeline::Impl::MatcherThread`（`L277-L497`）以视频人脸帧 `face_pts_ms` 为基准（`mel_start = static_cast<int64_t>(std::llround(face_pts_ms / hop_ms_))`），从滑动缓冲区 `mel_history`（上限 `mel_context_frames = 300`）切出 `80 × 16` Mel 窗口送入 `infer_queue_`。

- **3 个防伪代码细节**：
  1. `ThreadSafeQueue::UpdatePeakSize(size_t current_size)`（`thread_safe_queue.h:L553-L561`）使用无锁 CAS 循环 `peak_size_.compare_exchange_weak(prev_peak, current_size, std::memory_order_relaxed)` 记录队列历史水位峰值，并内置 `CheckDeadlock(int timeout_seconds = 30)` 监测长时间满队列无消费死锁。
  2. `Pipeline::Impl::MatcherThread`（`pipeline.cpp:L292-L370`）专门修复了“旧实现按每收到一帧 100Hz Mel 就拼一次 16 帧窗口导致 100FPS 推理风暴”的缺陷，改为**以 25FPS 视频帧驱动（Face-driven）**拼装 `[mel_start, mel_start + 16)` 窗口，将无效推理削减 75%。
  3. `Pipeline::Impl::Start()`（`pipeline.cpp:L122-L162`）严格按照**从下游到上游**顺序启动线程（`RenderThread -> InferenceWorker -> AVMatcher -> VideoProcessor -> AudioProcessor`），而 `Stop()`（`L172-L246`）严格按照**从上游到下游**顺序关停队列并汇合线程，防止上游生产者在下游退出后因反压满队列而永久阻塞。

---

### 📌 `Node-02`：C++ 数字人 SDK —— NCNN 推理引擎、OpenMP 调优、人脸缓存与嘴部 ROI 局部融合

- **对应简历原文**：
  > **深度学习推理与图像后处理优化**：封装基于 SCRFD 与 68 点关键点的人脸检测对齐模块，基于 NCNN 构建 Wav2Lip 跨平台推理引擎（支持 CPU / Vulkan GPU 加速）；通过设置 OpenMP 被动等待策略消除线程空转，引入静态人脸特征缓存与嘴部 ROI 局部仿射融合（替代全图 Warp），将视频合成前/后处理耗时从 **113ms 降至 14.3ms（提速约 8 倍）**。

- **核心物理坐标与符号映射**：
  1. **SCRFD 人脸检测 + 106→68 点关键点对齐**：
     - 文件：[`digital-human-sdk/src/core/face_detector.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/core/face_detector.cpp#L24-L376) 与 [`digital-human-sdk/src/core/face_aligner.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/core/face_aligner.cpp#L1-L220)
     - 核心实现：`FaceDetector::Impl` 加载 NCNN 模型 `scrfd_2.5g_kps-opt2.param/.bin`（输入 `kDetectorInputSize = 640`，步长 `strides = {8, 16, 32}`，均值 `127.5f`，缩放 `1/128.0f`）与 OpenCV DNN 关键点模型 `2d106det.onnx`（输入 `192×192`），通过静态映射表 `kLandmark106To68[68]`（`L34-L47`）将 InsightFace 106 点精确转换为标准 iBUG 68 点关键点。
  2. **NCNN Wav2Lip 推理引擎与 OpenMP 被动等待策略**：
     - 文件：[`digital-human-sdk/src/model/model_inferencer.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/model/model_inferencer.cpp#L26-L773) 与 [`digital-human-sdk/src/core/inference_worker.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/core/inference_worker.cpp#L147-L292)
     - 核心实现：
       - `ConfigureOpenMPPassiveWait()`（`model_inferencer.cpp:L38-L55`）：通过 `std::call_once` 在首次 NCNN/OpenMP 调用前注入环境变量 `OMP_WAIT_POLICY=PASSIVE` 与 `GOMP_SPINCOUNT=0`，将 `libgomp` 屏障自旋占用从 `60.3%` self CPU 压降至 `< 3%`。
       - `ModelInferencer::Impl::doLoadModel()`（`L172-L248`）：将 `net.opt.num_threads` 设置严格置于 `net.load_param()` / `net.load_model()` **之前**，修复 NCNN Winograd 卷积在 `load_param` 阶段固化线程数导致的 `convolution winograd will use load-time value` 失效问题；支持 Vulkan GPU（`AcquireVulkanInstance()` 引用计数管理 `ncnn::create_gpu_instance()` / `destroy_gpu_instance()`）。
       - `InferenceWorker::Impl::FaceToNCNN`（`inference_worker.cpp:L175-L229`）：构建 `96×96×6` 输入张量 `face_sequences`，其中前 3 通道对下半张脸 `y >= h / 2` 执行 `std::memset(dst_row, 0, w * sizeof(float))` 置零遮罩（防止模型直接拷贝输入帧嘴型而忽略音频），后 3 通道保留完整参考人脸，并通过 `cached_face_data_ptr_` 缓存静态参考帧张量。
  3. **静态人脸特征缓存与嘴部 ROI 局部仿射融合**：
     - 文件：[`digital-human-sdk/src/core/video_processor.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/core/video_processor.cpp#L51-L211)、[`digital-human-sdk/src/model/output_processor.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/model/output_processor.cpp#L514-L617) 与 [`digital-human-sdk/examples/perf_baseline_compare.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/examples/perf_baseline_compare.cpp#L58-L212)
     - 核心实现：
       - `VideoProcessor::Impl::ComputeSignature`（`video_processor.cpp:L72-L97`）：按 `kSampleRows = 16, kSampleCols = 32` 网格采样帧像素生成轻量级签名 `std::vector<uchar>`，命中 `FaceCache` 时直接复用 `aligned_face`、逆仿射矩阵 `M_inv`、嘴部羽化掩码 `face_mask`（`generatePreciseMouthAlphaMask96`）并采用浅拷贝传递原图，省去每帧重复检测与深拷贝。
       - `OutputProcessor::ProcessROI`（`output_processor.cpp:L514-L617`）：将 `96×96` 四个角点经 `M_inv` 投影至原图坐标系求最小外接矩形 `roi`（外扩 2px 边界并裁剪至图像尺寸），构造局部平移矩阵 `M_roi`（`M_roi(0, 2) -= roi.x; M_roi(1, 2) -= roi.y`），仅在嘴部小区域 `roi`（约 `144×144`）内执行 `doSharpen`、`cv::warpAffine`、`cv::blendLinear`（`doFaceFusion`）与 LAB 色彩融合，避免对 `1280×720` 全图做三次重采样与逐像素浮点混合。

- **3 个防伪代码细节**：
  1. `InferenceWorker::Impl::MelToNCNN`（`inference_worker.cpp:L147-L161`）修复了 `Eigen::MatrixXf(80, 16)` 列主序转换至 `ncnn::Mat(w=16, h=80, c=1)` 时宽高转置的隐蔽 Bug，显式按 `out.row(b)[t] = mel(b, t)` 逐元素写入。
  2. `FaceDetector::Impl::GenerateProposals`（`face_detector.cpp:L103-L140`）兼容了 NCNN 导出 SCRFD 模型时 `scores.c == 1`（将 `num_anchors=2` 展平到 `h` 维）与常规布局两种张量形态。
  3. `OutputProcessor::Impl`（`output_processor.cpp:L43-L61`）预分配了可复用工作缓冲区 `warped_face_buf_`、`warped_mask_buf_`、`inv_mask_1ch_buf_`、`fusion_out_buf_`，并使用 `cv::blendLinear(warped_face, original, mask_1ch, inv_mask_1ch_, out)` 单算子替代先 `convertTo(CV_32FC3)` 再乘加融合的慢路径。

---

### 📌 `Node-03`：C++ 数字人 SDK —— 音频主时钟动态 PTS 校准与 FFmpeg 实时推流

- **对应简历原文**：
  > **音视频主从同步与实时推流**：构建以音频播放时钟为主时钟（Audio Master Clock）的动态 PTS 校准机制，设计帧丢弃与重复帧补偿策略，将端到端音画同步误差控制在 **30ms 以内**；集成 FFmpeg 实现 H.264 视频编码与 AAC 音频封装，支持 RTMP/RTSP 低延迟直播推流。

- **核心物理坐标与符号映射**：
  1. **Audio Master Clock 与动态 PTS 漂移计算**：
     - 文件：[`digital-human-sdk/src/core/av_sync.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/core/av_sync.cpp#L12-L185)、[`digital-human-sdk/src/core/audio_sync_scheduler.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/core/audio_sync_scheduler.cpp#L20-L157) 与 [`digital-human-sdk/src/core/media_clock.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/core/media_clock.cpp#L66-L140)
     - 核心实现：
       - `AVSync::Impl`（`av_sync.cpp:L12-L97`）：维护原子音频主时钟 `std::atomic<double> audio_clock_{0.0}` 与视频时钟 `video_clock_{0.0}`，默认同步阈值 `sync_threshold_ms = 30.0`（即简历中的 **30ms 以内**）、最大漂移阈值 `max_drift_ms = 100.0`。
       - `doGetSyncStatus()`（`av_sync.cpp:L59-L83`）：计算 `double drift = v_pts - a_clk`（正值代表视频超前音频，负值代表视频落后音频），当 `std::abs(drift) <= sync_threshold_ms`（`30.0ms`）时判定 `in_sync = true`。
       - `MediaClock::ObserveVideoPts`（`media_clock.cpp:L94-L125`）：基于音频累计采样帧推进主时钟 `audio_pts_ms_ = total_audio_frames_ * 1000.0 / audio_sample_rate`，当音画漂移超出 `drift_correction_threshold_ms` 时按单步上限 `max_correction_step_ms` 渐进修正视频 PTS，避免画面跳变。
  2. **帧丢弃（Drop）/ 重复帧补偿（Duplicate）与微秒级帧率步进**：
     - 文件：[`digital-human-sdk/src/core/render_thread.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/core/render_thread.cpp#L131-L212)
     - 核心实现：
       - `RenderThread::Impl::GetSyncAction(double video_pts_ms)`（`render_thread.cpp:L131-L164`）：查询音频主时钟计算 `drift = video_pts_ms - status.audio_clock_ms`：
         - 若 `drift < -status.max_drift_ms`（视频严重滞后超过 `100ms`）：返回 `FrameAction::DROP` 丢弃过期帧追赶音频；
         - 若 `drift > status.max_drift_ms`（视频严重超前超过 `100ms`）：返回 `FrameAction::DUPLICATE` 复写上一帧 `last_displayed_frame_` 等待音频；
         - 其余情况返回 `FrameAction::DISPLAY`。
       - `RenderThread::Impl::PaceFrame()`（`render_thread.cpp:L175-L212`）：计算微秒级目标时间点 `target_time = last_frame_time_ + microseconds(int64_t(target_interval_ms_ * 1000.0))`，先调用 `std::this_thread::sleep_until(target_time - milliseconds(1))` 休眠至目标前 1ms，最后 1ms 通过 `while (steady_clock::now() < target_time) { std::this_thread::yield(); }` 精确对齐，将 Windows 定时器整毫秒截断抖动降低 90% 以上。
  3. **FFmpeg H.264 + AAC 编码与 RTMP / RTSP 实时推流**：
     - 文件：[`digital-human-sdk/src/media/stream_publisher.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/media/stream_publisher.cpp#L28-L300)
     - 核心实现：
       - 根据 URL 前缀自动选择封装格式：`rtmp://` 使用 `"flv"`，`rtsp://` 使用 `"rtsp"`（默认 TCP 传输 `av_dict_set(&opts, "rtsp_transport", "tcp", 0)`），文件或其它流使用 `"mpegts"` / `"mp4"`（`L39-L47`）。
       - 视频编码器优先探测硬件加速 `avcodec_find_encoder_by_name("h264_nvenc")`，回退至软件编码 `libx264`（配置 `preset=ultrafast`, `tune=zerolatency`, `max_b_frames=0`, `AV_PIX_FMT_YUV420P`，`sws_scale` 完成 BGR24→YUV420P 转换）。
       - 音频编码器采用 `AV_CODEC_ID_AAC`（`AV_SAMPLE_FMT_FLTP`），通过 `SwrContext` 重采样与 `AVAudioFifo` 缓冲凑齐固定帧大小 `audio_ctx_->frame_size`（1024 采样点）后送入 `avcodec_send_frame` 与 `av_interleaved_write_frame`。

- **3 个防伪代码细节**：
  1. `StreamPublisher::Impl::InterruptIo`（`stream_publisher.cpp:L31-L37`）向 `AVFormatContext::interrupt_callback` 注册了基于 `steady_clock::now().time_since_epoch().count() > deadline` 的超时中断回调，防止远端 RTMP/RTSP 服务器卡死导致推流线程永久阻塞在 `avio_open2` 或 `av_interleaved_write_frame`。
  2. `RenderThread::Impl::PaceFrame`（`render_thread.cpp:L207-L211`）在正常步进时采用 `last_frame_time_ = target_time`（增量累加基准时间）而非 `steady_clock::now()`，彻底消除了单帧微小的正负唤醒误差在长周期运行下的累积帧率漂移。
  3. `AudioSyncScheduler::Impl::doWaitForSyncPoint`（`audio_sync_scheduler.cpp:L125-L157`）在视频超前（`drift > threshold_ms`）时，仅对超出阈值的部分做 `sleep_ms = std::min(drift - threshold_ms, threshold_ms)` 渐进式微调等待，避免单次大跨度休眠引发卡顿。

---

### 📌 `Node-04`：C++ 数字人 SDK —— 状态机生命周期、协作式取消与工程质量门禁

- **对应简历原文**：
  > **工程质量与稳定性保障**：基于显式状态机（FSM）与协作式取消（Cooperative Cancellation）重构线程与会话生命周期管理，彻底消除 `std::thread::detach()` 引发的悬挂引用风险；编写 30+ 项核心模块单元测试，通过 Valgrind 内存检测实现长稳运行零泄漏。

- **核心物理坐标与符号映射**：
  1. **线程生命周期状态机（`ThreadState`）与协作式取消基类（`ThreadBase`）**：
     - 文件：[`digital-human-sdk/include/core/thread_base.h`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/include/core/thread_base.h#L20-L270) 与 [`digital-human-sdk/src/core/worker_registry.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/core/worker_registry.cpp#L9-L95)
     - 核心实现：
       - `enum class ThreadState { INIT, RUNNING, STOPPING, STOPPED, ERROR }`（`thread_base.h:L20-L26`）。
       - `RequestStop()`（`L85-L94`）：设置 `stop_requested_.store(true, std::memory_order_release)`，将状态从 `RUNNING` 切为 `STOPPING`，调用子类钩子 `OnStopRequested()`（唤醒阻塞队列或条件变量），并触发 `exit_cv_.notify_all()`。
       - `Wait(int timeout_ms = 5000)` / `WaitUntil(...)`（`L102-L162`）：在 `exit_mutex_` 上通过 `exit_cv_.wait_until` 等待工作线程将 `exited_` 置为 `true` 后再调用 `thread_.join()`。**核心安全铁律**：当等待超时（`!exited_`）时，**绝不调用 `thread_.detach()`**（注释明确说明：`Never detach on timeout: detaching while Run() still accesses 'this' causes use-after-free when the worker object is destroyed`），而是保持 `thread_` 归属权并返回 `false` 供上层重试或上报。
  2. **多轮对话会话状态机（`SessionState`）与 Barge-in 打断取消**：
     - 文件：[`digital-human-sdk/include/dialog/conversation_session.h`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/include/dialog/conversation_session.h#L21-L115) 与 [`digital-human-sdk/src/dialog/conversation_session.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/dialog/conversation_session.cpp#L42-L300)
     - 核心实现：
       - `enum class SessionState { IDLE, GENERATING, SYNTHESIZING, PLAYING, INTERRUPTING, FAILED, STOPPING, STOPPED }`（`conversation_session.h:L21-L30`）。
       - `OnBargeIn()` / `InterruptCurrentTurn()`：递增 `turn_generation_`，置位当前轮次的 `TurnCancelToken`，中断正在流式输出的 LLM / TTS 请求并清空未播放的音频缓冲，安全切回 `IDLE`。
       - `FailTurn(TurnContext& turn, const Error& err)`（`conversation_session.cpp:L82-L108`）：通过 `turn.turn_failed.exchange(true)` 原子去重，确保同一轮对话中 LLM 与 TTS 级联报错时只向调用方触发一次 `FAILED` 状态转换与错误回调。
  3. **核心模块单元测试体系与质量门禁**：
     - 文件：[`digital-human-sdk/tests/core_infrastructure_coverage_test.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/tests/core_infrastructure_coverage_test.cpp#L82-L330)、[`digital-human-sdk/tests/streaming_ resilience_test.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/tests/streaming_resilience_test.cpp#L1-L300)、[`digital-human-sdk/tests/pipeline_integration_test.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/tests/pipeline_integration_test.cpp#L1-L260) 与 [`digital-human-sdk/src/core/quality_gate.cpp`](file:///c:/Users/27013/Desktop/Interview-Workspace/digital-human-sdk/src/core/quality_gate.cpp#L24-L95)
     - 覆盖 `ThreadSafeQueue`、`SpscRingBuffer`、`ThreadBase`、`WorkerRegistry`、`MediaClock`、`QualityGate`、`ConversationSession`、`StreamPublisher` 等 30+ 项核心单测断言（仅 `core_infrastructure_coverage_test` 单套件即包含 77 项检查点全部通过）。

- **3 个防伪代码细节**：
  1. `ThreadBase::~ThreadBase()`（`thread_base.h:L52-L60`）析构函数采用 `RequestStop()` + 无限期阻塞 `Wait(-1)`，确保子类成员析构前后台线程 100% 已经退出 `Run()`，从根源上杜绝了虚表指针（vptr）已销毁或成员变量 Use-After-Free。
  2. `WorkerRegistry::WaitAllUntil`（`worker_registry.cpp:L68-L94`）使用**逆序等待**（`for (auto it = entries_.rbegin(); it != entries_.rend(); ++it)`）共享同一个全局 `deadline` 时间点，避免 N 个线程串行累加 `N * timeout_ms` 造成停机雪崩。
  3. `ThreadBase::ThreadEntry`（`thread_base.h:L228-L256`）捕获 `std::exception` 与 `...` 未知异常，在异常发生时将状态设为 `ThreadState::ERROR`，并在 `lock_guard<mutex> lock(exit_mutex_)` 保护下置位 `exited_ = true` 后广播 `exit_cv_.notify_all()`。

---

### 📌 `Node-05`：蘑菇物联 Agent 后端 —— 状态感知型 ReAct 编排引擎与工具注册表

- **对应简历原文**：
  > **Agent 编排引擎与状态机**：基于 Pydantic 设计标准化的 JSON Schema 工具注册表，实现状态感知的 ReAct 推理循环；通过有限状态机（FSM）按会话阶段动态裁剪可用工具白名单，引入最大步数（10 步）、连续异常（3 次）熔断及人工审批门禁（缺失参数填充率 `>25%` 自动挂起并落盘上下文，支持跨进程断点恢复），消除越序调用与死循环。

- **核心物理坐标与符号映射**：
  1. **基于 Pydantic 的标准化 JSON Schema 工具注册表**：
     - 文件：[`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/agent/tools.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/agent/tools.py#L35-L176)
     - 核心模型与类：
       - Pydantic 入参模型：`ExtractSurveyArgs`、`EmptyToolArgs`、`RequestUserInputArgs`、`RequestApprovalArgs`、`FinishArgs`（`L36-L70`），通过 `pydantic_to_json_schema(model_cls)`（`L73-L93`）调用 `model_cls.model_json_schema()` 自动导出精简 JSON Schema。
       - `Tool` 数据类（`L110-L143`）与 `ToolRegistry`（`L146-L176`）：在 `ToolRegistry.execute(name, ctx, args)`（`L167-L176`）中先执行 `tool.validate_args(args)`（内部调用 `self.args_model.model_validate(payload)`），校验失败直接抛出 `AgentToolError` 转化为结构化观察反馈给决策层。
       - `build_registry_for_module(module_type)`（`L482-L600`）：支持 `"air_conditioner"`（中央空调线）与 `"air_compressor"`（空压机线）双业务线统一注册 4 个确定性计算工具（`extract_survey`, `suggest_filling`, `apply_filling`, `calculate_saving_rate`）与 3 个控制类工具（`request_user_input`, `request_approval`, `finish`）。
  2. **状态感知的 ReAct 推理循环与 FSM 动态工具白名单裁剪**：
     - 文件：[`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/agent/orchestrator.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/agent/orchestrator.py#L41-L357) 与 [`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/agent/brain.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/agent/brain.py#L25-L227)
     - 核心实现：
       - 常量与阈值：`APPROVAL_FILL_RATIO = 0.25`（`orchestrator.py:L41`，即 **25% 人工审批门禁**）、`max_steps: int = 10`（`L56`，即 **最大 10 步预算**）、`consecutive_errors >= 3`（`L205`，即 **连续 3 次异常熔断**）。
       - `AgentOrchestrator._allowed_tools(ctx)`（`orchestrator.py:L230-L271`）：基于会话阶段产物实现严格的单向有限状态机（FSM）白名单裁剪：
         1. `extracted_data is None` $\rightarrow$ 仅开放 `["extract_survey", "request_user_input"]`；
         2. 质检拒止（`rejected`）或制冷站类型不支持（非水冷螺杆/离心且置信度 `>= 0.6`）$\rightarrow$ 仅开放 `["request_user_input"]`；
         3. `suggestions is None` $\rightarrow$ 仅开放 `["suggest_filling", "request_user_input"]`；
         4. `!can_proceed` 或 `unfillable_fields` 非空 $\rightarrow$ 仅开放 `["request_user_input"]`；
         5. `filled_data is None` 且 `fill_ratio > APPROVAL_FILL_RATIO (0.25)` 且未获批（`not approved`）$\rightarrow$ 仅开放 `["request_approval", "request_user_input"]`（**硬性移除 `apply_filling`，从集合层面阻断模型跳过审批**）；
         6. `filled_data is None`（占比 `<= 25%` 或已获批）$\rightarrow$ 仅开放 `["apply_filling", "request_user_input"]`；
         7. `result is None` $\rightarrow$ 仅开放 `["calculate_saving_rate", "request_user_input"]`；
         8. 已有 `result` $\rightarrow$ 仅开放 `["finish"]`。
  3. **人工审批挂起落盘与跨进程断点恢复**：
     - `hydrate_from_session(ctx)`（`orchestrator.py:L84-L101`）：从 `session_dir` 磁盘扫描并恢复 `extracted_data.json`、`quality_report.json`、`filling_suggestions.json`、`filled_data.json`、`result_data.json` 及 `approval.json`（含 `approved: True` 标记），使审批通过后的新 Worker 进程无需重跑前序阶段即可从 `apply_filling` 断点无缝续跑。

- **3 个防伪代码细节**：
  1. `fill_ratio()`（`tools.py:L312-L334`）计算分母时使用 `max(1, len(available_set) * 12)`（模板实际可用字段数 × 12 个月，默认 `12 * 7 = 84`），将模板天生不包含、仅靠行业默认值补充的字段排除在分子之外，避免误触发 25% 审批门禁。
  2. `AgentOrchestrator._state_view(ctx)`（`orchestrator.py:L273-L357`）对观测状态做了紧凑投影（例如将完整的 `fillable_suggestions` 数组压缩为字段名列表 `fillable_fields`），防止 12 个月完整 JSON 在多轮 ReAct 循环中撑爆 LLM Context Window。
  3. 当 LLM 幻觉尝试调用不在 `_allowed_tools(ctx)` 白名单内的工具时，`orchestrator.py:L155-L169` 会直接在编排层拦截（不执行 handler），将 `"当前阶段不允许调用 <tool>，允许的工具: [...]"` 写入 `StepRecord.error` 并令 `consecutive_errors += 1`。

---

### 📌 `Node-06`：蘑菇物联 Agent 后端 —— 异步会话持久化、Redis 租约锁与 SSE 断点回放

- **对应简历原文**：
  > **分布式并发控制与实时事件流**：采用 FastAPI 与 SQLAlchemy 2.0 构建会话状态持久化层；引入基于 Redis 租约机制与后台协程心跳续租的跨进程分布式锁，彻底解决多 Worker 并发消费下的状态竞态与重复执行问题；基于 Redis 与 Server-Sent Events（SSE）实现实时推理进度推送，支持基于 `Last-Event-ID` 的断线重连与增量事件回放。

- **核心物理坐标与符号映射**：
  1. **基于 Redis 租约机制与后台协程心跳续租的跨进程分布式锁**：
     - 文件：[`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/lock_lease.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/lock_lease.py#L20-L246) 与 [`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/storage.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/storage.py#L482-L636)
     - 核心实现：
       - `RedisLeaseStore`（`lock_lease.py:L45-L84`）：
         - 抢锁 `claim_task`：生成 `token = uuid.uuid4().hex`，调用 Redis 原子指令 `SET cs:lock:{task_type}:{session_id} token NX EX ttl_seconds`。
         - 续租 `renew_task`：执行 Lua 脚本 `_LUA_RENEW_SCRIPT`（先校验 `redis.call("GET", KEYS[1]) == ARGV[1]`，匹配才执行 `PEXPIRE`），保證只有锁持有者本人才能续期。
         - 释放 `release_task`：执行 Lua 脚本 `_LUA_RELEASE_SCRIPT`（校验 `GET == token` 后才执行 `DEL`），彻底杜绝因任务超时导致旧 Worker 误删新 Worker 锁的经典分布式竞态。
       - `AsyncRedisLeaseLock`（`lock_lease.py:L87-L172`）与 `LockLease`（`L175-L246`）：
         - 通过 `asyncio.create_task(self._heartbeat_loop())` 启动后台续租协程，每隔 `max(5.0, ttl_seconds / 3.0)` 秒自动续租；
         - 若 `renew` 返回 `False`（说明锁已超时被其它 Worker 接管），立即置位 `self.lock_lost.set()`；业务主循环在每个推理步骤边界调用 `lease.check()`（参见 `agent_service.py:L157` `step_hook`），检测到 `lock_lost` 时立即抛出 `LockLostError` 中止当前旧任务，防止双实例并发写脏会话状态。
  2. **SQLAlchemy 2.0 会话状态持久化层与 CAS 状态流转**：
     - 文件：[`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/storage.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/storage.py#L68-L480)
     - 核心模型：`SessionRow`、`EventRow`、`TaskLock`（`L68-L120`），定义合法状态集合 `SESSION_STATUSES = ("uploaded", "extracted", "suggested", "needs_input", "needs_approval", "approved", "filled", "completed", "rejected", "error")`；提供 `cas_status(session_id, expected_statuses, new_status)` 条件更新与数据库级租约降级兜底。
  3. **Redis + SSE 实时事件流与 `Last-Event-ID` 断点增量回放**：
     - 文件：[`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/events.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/events.py#L16-L189) 与 [`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/routers.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/routers.py#L660-L745)
     - 核心实现：
       - `EventBus.publish(session_id, event)`（`events.py:L73-L116`）：为每条事件分配单调递增序号 `seq`，先持久化到 SQLite/PostgreSQL（若注入 `persist`），再通过 Redis `RPUSH cs:events:{session_id}` 追加并设置 `EXPIRE 3600`。
       - `EventBus.history(session_id, after_seq=0)`（`events.py:L118-L148`）：优先读取 Redis `LRANGE 0 -1` 缓存列表，若 Redis 键已过期或不可用则自动回源查询持久化数据库，过滤出 `e["seq"] > after_seq` 的增量事件并按 `seq` 升序去重。
       - `stream_session_events`（`routers.py:L668-L742`）：同时解析 HTTP Header `Last-Event-ID` 与 Query 参数 `after_seq`（取较大值 `cursor = max(after_seq, header_seq)`），建连后先回放 `bus.history(session_id, after_seq=cursor)` 中的历史漏发事件，再进入实时队列消费，每条 SSE 帧均携带标准 `id: {seq}\ndata: {...}\n\n`，空闲 15 秒发送 `: keepalive\n\n` 注释帧保活。

- **3 个防伪代码细节**：
  1. 心跳续租循环（`lock_lease.py:L122-L132` 与 `L206-L212`）严格区分了**“网络/存储瞬时异常（Exception）”**与**“续租条件未命中（`renewed == False`）”**：遇到瞬时异常时 `continue` 等下一轮重试，只有明确返回 `False`（Token 不匹配或锁被删）才置位 `lock_lost`，防止网络闪断误杀正常长任务。
  2. `agent_service.py:L150-L162` 将 `lease.check()` 注入到 `AgentOrchestrator` 的 `step_hook` 回调中，确保在每一步工具执行前都验证租约有效性，一旦锁易主立即抛出 `LockLostError` 并跳过最终文件写入。
  3. `EventBus.subscribe`（`events.py:L150-L176`）在跨进程模式下以 `0.3s` 间隔轮询 `self.history(session_id, after_seq=cursor)` 并推进游标 `cursor = item["seq"]`，收到终止事件 `type in ("done", "error", "paused")` 时自动退出生成器关闭 SSE 流。

---

### 📌 `Node-07`：蘑菇物联 Agent 后端 —— Go 反向代理网关、多租户隔离与 36 例回归评测集

- **对应简历原文**：
  > **高并发网关与自动化评测体系**：使用 Go（`httputil.ReverseProxy`）构建反向代理网关，接管连接池复用、滑动窗口限流、JWT 鉴权、多租户上下文注入及 SSE 流式长连接透传；构建覆盖正常工况、缺项追问、越权拦截与异常恢复的 **36 个端到端回归评测集**，工具调用准确率达 **96.4%**，流程完成率提升至 **94.4%**。

- **核心物理坐标与符号映射**：
  1. **Go 反向代理网关（`httputil.ReverseProxy` + SSE 零缓冲透传 + 限流鉴权）**：
     - 文件：[`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/proxy/main.go`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/proxy/main.go#L36-L469)
     - 核心实现：
       - `newUpstreamProxy`（`main.go:L101-L154`）：创建 `httputil.ReverseProxy`，配置 `FlushInterval: -1`（**关键：禁用代理缓冲，每次后端 `Write` 立即 `Flush` 到客户端，确保 SSE 实时流式透传**），连接池参数 `MaxIdleConns: 256`, `MaxIdleConnsPerHost: 128`, `IdleConnTimeout: 90 * time.Second`, `ResponseHeaderTimeout: 300 * time.Second`。
       - `Director` 钩子（`L115-L133`）：注入 `X-Request-ID`，并调用 `req.Header.Del("X-Tenant-ID")` **强制剥离客户端伪造的租户头**，仅当 JWT 验签通过后由网关重写可信的 `req.Header.Set("X-Tenant-ID", tenantID)`，配合 `ModifyResponse` 在 `/events` 路径强制设置 `X-Accel-Buffering: no` 与 `Cache-Control: no-cache`。
       - ` verifyJWT_HS256`（`L214-L280`）：无第三方依赖纯标准库实现 HMAC-SHA256 签名校验（`hmac.Equal` 常量时间比对防时序攻击）、`exp` 过期检查及 `tenant_id` / `sub` 声明提取。
       - `SlidingWindowLimiter`（`L296-L355`）：按客户端/租户维度维护时间戳切片 `hits map[string][]time.Time`，在 `Allow(key)` 中原地清理滑出窗口 `[now - window, now]` 的旧请求，超限返回 `429 Too Many Requests` 与 `Retry-After: 1`，并由后台协程 `cleanupLoop` 定期回收闲置桶防内存泄漏。
  2. **Python 后端多租户数据隔离闭环**：
     - 文件：[`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/deps.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/apps/api/deps.py#L18-L186)
     - 核心实现：`get_current_tenant`（`L78-L115`）与 `require_session_Access`（`L136-L186`），校验会话所属 `owner_tenant` 与当前租户一致，跨租户访问直接拦截并抛出 `403 Forbidden`（对应回归集中的 `tenant_isolation` 越权拦截用例）。
  3. **36 个端到端回归评测集与黄金轨迹对拍引擎**：
     - 文件：[`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/evaluation/run_eval.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/evaluation/run_eval.py#L18-L231) 与 [`moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/evaluation/report.md`](file:///c:/Users/27013/Desktop/Interview-Workspace/moguwulian-agent-backend/conditioner_saving/conditioner_saving/conditioner_saving/evaluation/report.md#L1-L46)
     - 核心指标计算（`run_eval.py:L122-L186`）：
       - 构建 7 大类共 **36 个评测用例**（`normal_complete: 6`, `needs_approval: 6`, `approval_resume: 5`, `missing_input: 6`, `unsupported_station: 5`, `tool_fault_recovery: 4`, `tenant_isolation: 4`）；
       - 对比 Agent 实际工具调用序列与黄金标准序列 `expected_tools`，实测达成：**工具调用准确率 `96.4%`（`108 / 112`）**、**端到端流程完成率 `94.4%`（`34 / 36`）**、**越权与危险操作拦截率 `100.0%`（`21 / 21`）**。

- **3 个防伪代码细节**：
  1. `proxy/main.go:L124-L129` 实施了“网关剥离客户端 `X-Tenant-ID` + 仅注入 JWT 载荷中的 `tenant_id`”零信任策略，防止攻击者绕过 JWT 声明通过手动构造 HTTP Header 越权读取其它租户的节能测算报告。
  2. `proxy/main.go:L64-L82` 的 `statusRecorder` 结构体显式实现了 `http.Flusher` 接口转发（`if f, ok := r.ResponseWriter.(http.Flusher); ok { f.Flush() }`），解决了 Go 标准库包装 `http.ResponseWriter` 后丢失 `http.Flusher` 类型断言导致 `ReverseProxy` 流式刷新失效的经典坑点。
  3. `evaluation/run_eval.py:L88-L118` 在评测 `tool_fault_recovery` 用例时，通过注入带预设故障次数的模拟器验证 `AgentOrchestrator` 在连续报错 `< 3` 次时自动重试恢复、达到 `3` 次时精确触发熔断降级。

---

### 📌 `Node-08`：蘑菇物联 Agent 后端 & 冷却塔寻优 —— 机理数据合成、14 类物理约束过滤、Qwen3-8B 微调与安全护栏

- **对应简历原文**：
  > **工业垂直大模型微调与安全护栏**：基于 10.5 万条真实工况数据与机理模型，构建 7.2 万条结构化样本及 6 步思维链（CoT）推理数据，设计 14 类物理约束规则过滤异常幻觉样本；完成 Qwen3-8B 监督微调（SFT）与设备频率硬边界安全护栏（Safety Guard）部署，物理红线越界率降至 **0%**。

- **核心物理坐标与符号映射**：
  1. **机理模型驱动的 6 步 CoT 数据工厂（10.5 万原始工况 $\rightarrow$ 72,503 条结构化样本）**：
     - 文件：[`Cooling Tower Modeling/全系统寻优/07_cot_data_factory.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/Cooling%20Tower%20Modeling/%E5%85%A8%E7%B3%BB%E7%BB%9F%E5%AF%BB%E4%BC%98/07_cot_data_factory.py#L41-L487)、[`Cooling Tower Modeling/全系统寻优/08_llm_cot_synthesizer.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/Cooling%20Tower%20Modeling/%E5%85%A8%E7%B3%BB%E7%BB%9F%E5%AF%BB%E4%BC%98/08_llm_cot_synthesizer.py#L1-L260) 与 [`Cooling Tower Modeling/全系统寻优/cot_dataset/quality_sample_summary.json`](file:///c:/Users/27013/Desktop/Interview-Workspace/Cooling%20Tower%20Modeling/%E5%85%A8%E7%B3%BB%E7%BB%9F%E5%AF%BB%E4%BC%98/cot_dataset/quality_sample_summary.json#L1-L65)
     - 核心实现：
       - 从 `B_train`（27,206 条）、`B_test`（25,536 条）、`C_test`（19,761 条）共 **72,503 条（7.2 万条）** 结构化工况样本出发，调用冷却塔-水泵-冷机耦合机理模型生成严格对齐的 **6 步思维链（CoT）** 结构（`step1_load_assessment` 热负荷评估 $\rightarrow$ `step2_topology_screening` 塔台数拓扑初筛 $\rightarrow$ `step3_boundary_envelope` 安全包络边界确定 $\rightarrow$ `step4_closed_loop_iterations` 闭环不动点迭代寻优轨迹 $\rightarrow$ `step5_power_tradeoff` 冷机/水泵/风机功耗博弈核算 $\rightarrow$ `step6_final_output` 推荐频率与安全断言）。
  2. **14 类物理约束与机理一致性过滤规则（`PHYSICS_CONSTRAINT_RULES_14`）**：
     - 文件：[`Cooling Tower Modeling/全系统寻优/10_quality_sample_selector.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/Cooling%20Tower%20Modeling/%E5%85%A8%E7%B3%BB%E7%BB%9F%E5%AF%BB%E4%BC%98/10_quality_sample_selector.py#L25-L256)
     - 核心实现：`PHYSICS_CONSTRAINT_RULES_14`（`L42-L57`）与 `evaluate_full_record(record, text)`（`L147-L256`），对合成样本实施 14 类物理与机理硬约束过滤（统计结果完整记录于 `quality_sample_summary.json:L24-L39`）：
       1. `chiller_cop_out_of_range`：冷机 COP 超出物理合理区间 `[2.0, 8.0]`（拦截 372 条）；
       2. `airflow_too_close_to_envelope`：冷却塔总风量贴近训练包络边界，余量 `< 5%`（拦截 634 条）；
       3. `worse_than_model_baseline`：寻优总功率劣于机理模型基线超过 `1%` 容差（拦截 401 条）；
       4. `factory_quality_not_high`：07 数据工厂未标记为 `high_quality` 或闭环寻优未收敛（拦截 309 条）；
       5. `optimizer_not_feasible`：SLSQP/网格优化器未找到满足热力学约束的可行解（拦截 309 条）；
       6. `tower_frequency_out_of_range`：运行冷却塔风机频率超出 `[1.01, 50.0] Hz`（拦截 27 条）；
       7. `approximate_capacity_screen_failed`：选定塔拓扑排热容量 `capacity_est_kW < Qc_estimated_kW`（拦截 192 条）；
       8. `safety_assertion_failed`：Step 6 四项终端安全断言未全量通过（拦截 295 条）；
       9. `pump_frequency_out_of_range`：冷却水泵频率超出变频安全边界 `[30.01, 49.99] Hz`（拦截 118 条）；
       10. `dt_too_close_to_limit`：冷却水温差 `dt_cw` 距安全上下限不足 `0.10 °C`（拦截 139 条）；
       11. `flow_too_close_to_envelope`：冷却水流量 `M_opt` 贴近水力包络边界，余量 `< 5%`（拦截 204 条）；
       12. `temperature_too_close_to_limit`：出水温度 `Tout_pred` 距凝汽器上限或湿球逼近度不足 `0.20 / 0.10 °C`（拦截 225 条）；
       13. `nonphysical_power`：总功率或分项功率非正，或能量守恒残差 `|Ptotal - (Pch + Ppump + Ptower)| > max(2.0 kW, 2%)`（拦截 14 条）；
       14. `no_active_tower`：全部冷却塔风机均处于停机状态（拦截 64 条）。
  3. **Qwen3-8B SFT 推理评测与设备频率硬边界安全护栏（Safety Guard）**：
     - 文件：[`Cooling Tower Modeling/全系统寻优/09_cot_dataset_builder.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/Cooling%20Tower%20Modeling/%E5%85%A8%E7%B3%BB%E7%BB%9F%E5%AF%BB%E4%BC%98/09_cot_dataset_builder.py#L19-L231)、[`Cooling Tower Modeling/全系统寻优/12_validate_student_physics.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/Cooling%20Tower%20Modeling/%E5%85%A8%E7%B3%BB%E7%BB%9F%E5%AF%BB%E4%BC%98/12_validate_student_physics.py#L24-L422) 与 [`Cooling Tower Modeling/全系统寻优/inference/safety_guard.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/Cooling%20Tower%20Modeling/%E5%85%A8%E7%B3%BB%E7%BB%9F%E5%AF%BB%E4%BC%98/inference/safety_guard.py#L11-L84)
     - 核心实现：`SafetyGuard.validate_and_clamp(raw_action, context)`（`safety_guard.py:L26-L84`）在模型输出送入 PLC 控制器前执行硬边界截断与联锁校验：水泵频率强制钳位至 `[30.0, 50.0] Hz`、冷却塔风机频率钳位至 `{0.0} ∪ [10.0, 50.0] Hz`（避开 `[0.1, 10.0) Hz` 机械共振死区）、单周期频率变化率限制 `max_delta_hz = 5.0 Hz`、凝汽器进水温度上限 `35.0 °C` 联锁升频，将物理红线越界率严格压降至 **0%**。

- **3 个防伪代码细节**：
  1. `09_cot_dataset_builder.py` 采用**按时间块（Temporal Block Split）**而非随机打乱划分训练/验证集，防止工业时序传感器数据相邻采样点高度自相关引发数据泄漏（Data Leakage）。
  2. `10_quality_sample_selector.py:L164-L176` 对塔组合标签执行规范化重算 `canonical_combo = [1 if x >= TOWER_MIN_ON_HZ else 0 for x in ft_opt]`，并与 Step 6 输出的 `tower_combo_opt` 交叉比对，抓出“文本声称开启 2 台塔但实际输出频率有 3 台”的模态内自相矛盾幻觉。
  3. `12_validate_student_physics.py` 不信任学生模型自报的预测水温和省电率，而是提取学生模型输出的 `(Fp_opt_Hz, Ft_opt_Hz)` 反向代入 `06_system_optimization` 物理机理仿真器重算真实热平衡出水温度 `Tout` 与冷机 COP，实现闭环物理归因验证。

---

### 📌 `Node-09`：OpenVLA-OFT 具身智能 —— 多模态微调、20k 数据缩放拐点与 Isaac Lab 仿真迁移

- **对应简历原文**：
  > - **多模态微调与数据缩放规律探究**：构建 HDF5 至 RLDS 的标准化数据转换管线；在 LIBERO-Spatial 基准上完成 500 Episodes 全量评测（成功率 99.0%）与 MuJoCo 接触动力学诊断。基于 BridgeData V2 开展 LoRA（`rank=32`）数据缩放实验（2k~38k 轨迹），定位 **20k 轨迹数据收益拐点**，将 SIMPLER 跨场景抓取成功率从 41.3% 提升至 60.5%。
  > - **Isaac Lab 仿真迁移与域差距消融**：搭建 OpenVLA-OFT 至 Isaac Lab（Franka 机械臂）的闭环评估管线，实现 7D 末端动作至相对逆运动学（DLS IK）与二值夹爪的坐标映射及安全限幅；设计三阶视觉域对齐策略（4:3 中心裁剪、色彩抖动与 USD 场景光照/材质随机化），将跨仿真器迁移抓取成功率提升 **17.5 个百分点**。

- **核心物理坐标与符号映射**：
  1. **HDF5 至 RLDS/TFDS 标准化转换管线与归一化生命周期**：
     - 文件：[`openvla-oft-eval/tools/convert_libero_hdf5_to_rlds.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/tools/convert_libero_hdf5_to_rlds.py#L20-L155) 与 [`openvla-oft-eval/09-行为蒸馏实验/scripts/finetune_teacher_distill.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/09-行为蒸馏实验/scripts/finetune_teacher_distill.py#L65-L305)
     - 核心实现：
       - `LiberoSpatialTopDrawer(tfds.core.GeneratorBasedBuilder)`（`convert_libero_hdf5_to_rlds.py:L31-L142`）：校验 `actions`, `rewards`, `dones`, `ee_states`, `gripper_states`, `agentview_rgb`, `eye_in_hand_rgb` 七组序列长度严格一致，将双路 `128×128×3` RGB 双线性重采样为 `256×256×3`，拼接 8D Proprio 状态并保持官方夹爪编码（`-1=open, +1=close`）导出为 8 个 TFRecord 分片。
       - `_rebase_bounds_q99` 与 `RebasedRLDSBatchTransform`（`finetune_teacher_distill.py:L65-L119`）：解决单任务微调统计量与基座全局统计量 `libero_spatial_no_noops` 的 `BOUNDS_Q99`（`[q01, q99] -> [-1, 1]`）坐标系漂移问题；并对动作头末层 `student.model.fc2.weight` 注册梯度掩码钩子 `register_hook(lambda grad: torch.cat((grad[:6], torch.zeros_like(grad[6:])), dim=0))`（`L207-L212`），仅微调前 6 维位姿而冻结第 7 维夹爪。
  2. **BridgeData V2 LoRA（`rank=32`）数据缩放实验与 20k 轨迹拐点定位**：
     - 文件：[`openvla-oft-eval/02-实验代码/patch-check/vla-scripts/finetune.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/02-实验代码/patch-check/vla-scripts/finetune.py#L68-L120) 与 [`openvla-oft-eval/tools/bridge_lora_scaling_eval.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/tools/bridge_lora_scaling_eval.py#L26-L209)
     - 核心实现：
       - `BRIDGE_V2_SCALING_BENCHMARK`（`bridge_lora_scaling_eval.py:L37-L115`）：记录 BridgeData V2（`NUM_ACTIONS_CHUNK=5, ACTION_DIM=7, PROPRIO_DIM=7, BOUNDS_Q99`）在 `2k, 5k, 10k, 20k, 30k, 38k` 六个轨迹规模下的 LoRA（`lora_rank=32`）微调验证 L1 Loss 与 SIMPLER 跨场景抓取成功率（`41.3% -> 47.9% -> 54.6% -> 60.5% -> 61.4% -> 61.9%`）。
       - `locate_scaling_knee_point()`（`L118-L163`）：采用归一化 Kneedle 曲率距离算法（最大化曲线与 `[2k, 38k]` 割线的正交垂直偏差 $D(x) = y_{\text{norm}}(x) - x_{\text{norm}}(x)$）结合每千条轨迹边际收益（`marginal_gain_pp_per_1k` 从 `1.34 pp/1k` 断崖跌至 `20k` 之后的 `0.09 pp/1k`），精准定位 **20k 轨迹数据收益拐点（`kneedle_distance = 0.432`，成功率 `41.3% -> 60.5%`）**。
  3. **Isaac Lab 闭环迁移、7D 动作映射与三阶视觉域对齐**：
     - 文件：[`openvla-oft-eval/12-IsaacLab-Bridge-Franka/bridge_action_mapper.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/12-IsaacLab-Bridge-Franka/bridge_action_mapper.py#L16-L120)、[`openvla-oft-eval/12-IsaacLab-Bridge-Franka/domain_randomization.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/12-IsaacLab-Bridge-Franka/domain_randomization.py#L11-L150)、[`openvla-oft-eval/12-IsaacLab-Bridge-Franka/observation.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/12-IsaacLab-Bridge-Franka/observation.py#L15-L92) 与 [`openvla-oft-eval/12-IsaacLab-Bridge-Franka/run_closed_loop.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/12-IsaacLab-Bridge-Franka/run_closed_loop.py#L84-L254)
     - 核心实现：
       - `BridgeActionMapper.to_isaac_action`（`bridge_action_mapper.py:L76-L113`）：对反归一化后的 7D 动作前 3 维位置增量执行 `np.clip(raw[:3], -0.06, 0.06)`（`max_position_delta_m = 0.06m`），对 3:6 轴角旋转增量执行 `np.clip(raw[3:6], -0.35, 0.35)`（`max_rotation_delta_rad = 0.35rad`），必要时通过 `quat_wxyz_to_matrix(current_eef_quat_wxyz).T` 完成 `world/root -> body` 坐标系旋转，并将 Bridge 绝对夹爪 `[0, 1]`（`open_threshold = 0.5`）二值化为 Isaac Lab `BinaryJointPositionActionCfg` 所需的 `+1.0 / -1.0`。
       - `apply_rgb_alignment` 与 `apply_stage_randomization`（`domain_randomization.py:L66-L150`）：实现 `raw -> camera_aligned -> bridge_aligned` 三阶视觉域对齐（`center_crop_to_aspect(4, 3)` 中心裁剪、亮度 `[0.92, 1.08]` / 对比度 `[0.90, 1.10]` / 饱和度 `[0.92, 1.08]` / 高斯传感器噪声 `std=1.5` 调整，以及 USD Stage `/World/light` 强度 `[2200, 3400]` 与桌面/物体漫反射材质随机化）。

- **3 个防伪代码细节**：
  1. `OpenVLAAdapter.__init__`（`openvla_policy.py:L45-L46`）显式检查 `if "bridge" not in " ".join(sys.argv).lower(): sys.argv.append("--bridge")`，防止上游 `prismatic/vla/constants.py` 的 `detect_robot_platform()` 因启动脚本名不含 `"bridge"` 而静默退回 `LIBERO` 常量（导致 `PROPRIO_DIM` 从 7 变成 8、`NUM_ACTIONS_CHUNK` 从 5 变成 8 引发张量维度崩溃）。
  2. `observation.py:L46-L69` 在构造 Bridge 7D Proprio 状态时，不仅减去了并行环境的局部原点偏移 `pos = pos - env.scene.env_origins[env_id]`，还将 Panda 双指关节宽度归一化为 `np.clip(width / 0.04, 0.0, 1.0)`（保持 `[0, 1]` 绝对开度，区别于动作指令的 `[-1, +1]`），因为 Bridge 数据集的 `BOUNDS_Q99` 掩码对第 7 维不作分位数缩放。
  3. `run_closed_loop.py:L192-L197` 使用 `success_seen |= success_now` 在每个物理子步累积成功事件，防止 Isaac Lab 的 `ManagerBasedRLEnv` 在 episode 终止瞬间自动重置（Auto-Reset）环境槽位导致在循环末尾读取物体坐标时被误判为失败。

---

### 📌 `Node-10`：OpenVLA-OFT 具身智能 —— OFT 并行解码机制与 INT8 混合精度量化性能剖析

- **对应简历原文**：
  > **并行解码机制分析与混合精度量化**：剖析 OFT 连续动作块（Action Chunking, `H=8`）并行解码机制，通过消融实验量化开环步长与 400ms 控制周期预算的耦合边界；采用 `bitsandbytes` 对视觉塔与 LLM 主干共 **436 个线性层**实施 INT8 混合精度量化（保留 FP16/BF16 连续动作头），在成功率几乎无损（98.8%）的前提下将峰值显存从 **15.8GB 压缩至 9.3GB（下降 41.2%）**，并深入定位小 Batch（`Batch=1`）场景下动态反量化算子拆分导致推理时延劣化（217ms 增至 555ms）的底层瓶颈。

- **核心物理坐标与符号映射**：
  1. **OFT 并行动作头（`L1RegressionActionHead`）与 400ms 控制时间预算**：
     - 文件：[`openvla-oft-eval/02-实验代码/patch-check/prismatic/models/action_heads.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/02-实验代码/patch-check/prismatic/models/action_heads.py#L38-L108) 与 [`openvla-oft-eval/02-实验代码/patch-check/prismatic/vla/constants.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/02-实验代码/patch-check/prismatic/vla/constants.py#L26-L31)
     - 核心实现：
       - `L1RegressionActionHead.predict_action(actions_hidden_states)`（`action_heads.py:L98-L108`）：将 Llama-2 7B 最后一层对应动作占位符的隐状态 `(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)`（在 LIBERO 下为 `(B, 8 * 7 = 56, 4096)`）通过 `.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)` 重排为 `(B, 8, 28672)`，经包含 2 个 `MLPResNetBlock`（Pre-LayerNorm + Linear + ReLU + 残差连接）的 `MLPResNet` 一次性并行回归出 `(B, 8, 7)` 连续动作块，彻底取代传统 VLA 逐 Token 自回归生成（56 次串行前向）。
       - 控制周期时间预算公式：在 `control_freq = 20 Hz`（单步 `50 ms`）与 `open_loop = 8` 下，$T_{\text{budget}} = \frac{8}{20\text{ Hz}} \times 1000 = 400\text{ ms}$。当消融实验将 `open_loop` 从 `8` 缩短至 `4` 时，时间预算腰斩为 $200\text{ ms}$，而 BF16 单次推理均值为 `217.24 ms`，导致 `chunk_deadline_met` 达标率从 `97.8%` 直接跌至 **`0.0%`**。
  2. **G1 量化覆盖率只读审计工具（436 层 `Linear8bitLt` 精确解剖）**：
     - 文件：[`openvla-oft-eval/10-INT8量化与并行解码教材/audit_int8_coverage.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/10-INT8量化与并行解码教材/audit_int8_coverage.py#L49-L220)
     - 核心实现：
       - `count_linear(model, bnb)`（`L49-L76`）：递归遍历 `model.named_modules()`，统计 `bnb.nn.Linear8bitLt` 与普通 `torch.nn.Linear` 数量，并按顶层模块名前缀 `n.split(".")[0]` 聚合归组。
       - 实测审计结果：全模型 437 个线性层中有 **436 个被替换为 `Linear8bitLt`（覆盖率 `99.8%`）**，其中 `language_model` 占 **224 层**（32 层 Transformer × 7 个线性投影）、`vision_backbone`（SigLIP + DINOv2 双视觉塔）占 **209 层**、`projector` 占 **3 层**；唯一保留的 1 个普通 `nn.Linear` 为末端输出层，且外置 `action_head`（16 个参数/缓冲张量）与 `proprio_projector`（4 个张量）经 `dtype_counts()` 验证全部保持 `torch.bfloat16`。
  3. **500-Episode 配对基准评测与 `Batch=1` 时延劣化根因分析**：
     - 文件：[`openvla-oft-eval/08-后续实验验证/scripts/09_int8_benchmark.sh`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/08-后续实验验证/scripts/09_int8_benchmark.sh#L1-L145)、[`openvla-oft-eval/08-后续实验验证/scripts/summarize_int8_benchmark.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/08-后续实验验证/scripts/summarize_int8_benchmark.py#L38-L185)、[`openvla-oft-eval/10-INT8量化与并行解码教材/analyze_regressions.py`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/10-INT8量化与并行解码教材/analyze_regressions.py#L188-L320) 与 [`openvla-oft-eval/06-文档/模型量化误差策略与OpenVLA-OFT时延分析.md`](file:///c:/Users/27013/Desktop/Interview-Workspace/openvla-oft-eval/06-文档/模型量化误差策略与OpenVLA-OFT时延分析.md#L192-L312)
     - 核心实测账本与根因：
       - 峰值显存从 **`15,839 MiB`（`15.8 GB`）降至 `9,315 MiB`（`9.3 GB`），净降 `6,524 MiB`（`-41.2%`）**；500 Episodes 成功率从 `496/500 (99.2%)` 微变为 `494/500 (98.8%)`（损失仅 `0.4 pp`，处于统计噪声内）。
       - 平均推理延迟从 **`217.24 ms` 增至 `555.49 ms`（慢 `2.56×`，`Speedup = 0.391×`）**，动作吞吐从 `36.82 FPS` 跌至 `14.40 FPS < 20 Hz`，`400 ms` Deadline 达标率从 `98.4%` 跌至 `0.0%`。
       - 四大底层物理根因：① `bitsandbytes` `LLM.int8()` 在前向传播中按阈值 `llm_int8_threshold = 6.0` 动态扫描激活离群列（Outliers），将单次融合 GEMM 拆碎为“动态量化 + 离群列提取 + FP16 GEMM + INT8 GEMM + 动态反量化 + 累加拼接”6 个细碎算子；② 在线闭环控制强制 `Batch=1`，矩阵乘退化为访存受限（Memory-bound）的 GEMV，436 层每步触发 `1500~2000` 次 CUDA Kernel Launch 调度风暴；③ 视觉塔 209 个小线性层被全量量化仅省约 `1.2 GiB` 显存却贡献近半数算子发射开销；④ `BF16 <-> FP16 <-> INT8` 频繁跨精度类型转换阻断算子融合。

- **3 个防伪代码细节**：
  1. `09_int8_benchmark.sh:L41-L45` 在启动评测前执行静态代码门禁正则检查 `grep -Eq 'action_head[[:space:]]*=[[:space:]]*action_head\.to\(torch\.bfloat16\)'`，若无法确认外置 `action_head` 保持 BF16 则立即以退出码 2 终止脚本，严防连续动作头被误量化毁掉回归精度。
  2. `summarize_int8_benchmark.py:L38-L48` 在统计推理延迟时强制剔除前 `warmup_chunks = 3` 个 Chunk（消除首次 CUDA Context 初始化与 cuBLAS workspace 分配毛刺），并在 `L122-L134` 对两组实验的 `(task_id, initial_state_index)` 集合做严格相等校验。
  3. `analyze_regressions.py:L35-L37` 专门针对上游 `run_libero_eval.py` 的真实字段结构，指出开启 `verify_grasp` 空抓重规划干预时会中途清空动作队列，因此真实控制步数必须读取 `executed_control_steps`（绝不等于 `replan_count * num_open_loop_steps`）。
