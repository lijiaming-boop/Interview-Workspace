# 音视频与 Wav2Lip 基础

## 1. 30 秒回答

项目把 16 kHz 单声道 float PCM 转成 Mel 频谱窗口，同时对头像做人脸检测、关键点定位和 `96 × 96` 对齐。Matcher 以视频 PTS 为基准选取 `16 × 80` Mel 上下文，构造音频张量和 `96 × 96 × 6` 人脸张量交给 ncnn Wav2Lip 推理。输出人脸经过逆仿射和 mask 融合回 BGR 帧，最终与原始 PCM 分别编码为 H.264 和 AAC。

## 2. PCM 的基本单位

必须区分 sample、sample frame 和字节：

- sample：单个声道的一个采样值；
- sample frame：同一时刻所有声道的一组 sample；
- interleaved stereo：`L0, R0, L1, R1, ...`；
- float PCM：本项目通常使用 `[-1, 1]` 范围的 32 位浮点数。

音频时钟公式必须按 sample frame 计算：

```text
audio_pts_ms = frames_per_channel / sample_rate × 1000
```

如果把双声道 float 元素数量直接除以采样率，时钟会快两倍。

## 3. 从波形到 Mel

当前默认参数：

| 参数 | 数值 | 时间含义 |
|---|---:|---:|
| 采样率 | 16000 Hz | 每秒 16000 个单声道采样 |
| frame size | 800 | 50 ms |
| hop size | 200 | 12.5 ms |
| Mel bins | 80 | 每个时间步 80 维 |
| 上下文长度 | 16 | 约 200 ms |

当前音频链路可以概括为：

```text
PCM → 降噪 → 流式 RMS → 预加重 → 分帧/VAD
    → Hann 窗 → STFT 幅度谱 → Slaney Mel → dB → 对称归一化
```

关键理解：

- 分帧将连续波形转换为重叠短时窗口；
- 窗函数减少截断造成的频谱泄漏；
- STFT 给出时间局部的频率能量；
- Mel 滤波器组压缩频率轴，更接近语音感知尺度；
- 模型要求的归一化、维度方向和值域必须与训练一致。

实现见 [`src/core/audio_processor.cpp`](../../src/core/audio_processor.cpp) 和 [`src/audio/audio_mel_feature_extract.cpp`](../../src/audio/audio_mel_feature_extract.cpp)。

## 4. PTS、DTS 与 Time Base

- PTS：帧应该呈现的时间；
- DTS：帧应该解码的时间，存在 B 帧时可能与 PTS 不同；
- time base：时间戳的单位，例如毫秒、`1/25` 或 `1/48000`；
- wall clock：程序实际运行时间，不等于媒体时间。

视频理想 PTS：

```text
video_pts_ms = frame_index × 1000 / fps
```

进入 FFmpeg 后必须把业务时间基准转换成 codec/stream time base。持续会话的时间戳应单调递增；用户打断应该取消某个 turn，而不是把全局媒体时间重置为 0。

## 5. 为什么 Matcher 由视频驱动

目标视频通常是固定 25 FPS，而 Mel 每 12.5 ms 产生一个时间步。如果由 Mel 驱动推理，可能远高于目标视频帧率，造成任务放大。

face-driven Matcher 的思路是：

1. 取一个带 PTS 的 face frame；
2. 根据 PTS 计算对应 Mel 起点；
3. 缓存并组装连续 16 个 Mel 时间步；
4. 检查 A/V 误差阈值；
5. 每个目标视频帧只产生一个推理任务。

实现位于 [`src/core/pipeline.cpp`](../../src/core/pipeline.cpp)。

## 6. 人脸检测、对齐与 Mask

视觉链路：

```text
BGR → SCRFD 人脸检测 → 2D106 关键点
    → 仿射对齐到 96×96 → 口部 mask → 模型输入
```

为什么需要对齐：检测框只能给出粗略位置；旋转、尺度和姿态变化会让嘴部坐标不稳定。关键点与仿射变换把人脸映射到训练时的标准坐标，输出后再通过逆变换贴回原图。

为什么需要 mask：直接覆盖模型输出会形成矩形边界。mask 让主要替换区域集中在嘴部，并通过羽化减少颜色和纹理突变。

实现见 [`src/core/video_processor.cpp`](../../src/core/video_processor.cpp)、[`src/core/face_aligner.cpp`](../../src/core/face_aligner.cpp) 和 [`src/core/face_mask_generator.cpp`](../../src/core/face_mask_generator.cpp)。

## 7. Wav2Lip 张量契约

当前模型约定：

| Blob | 逻辑形状 | 含义 |
|---|---|---|
| `audio_sequences` | `1 × 80 × 16` | 约 200 ms Mel 上下文 |
| `face_sequences` | `6 × 96 × 96` | 遮罩脸 RGB + 完整脸 RGB |
| `output` | `3 × 96 × 96` | 合成后的 RGB 人脸 |

六通道人脸的前三通道不是另一份完整脸，而是下半脸置零后的 RGB；后三通道是完整 RGB。通道顺序、NCHW/NHWC、RGB/BGR、缩放和值域任何一个错位，都可能让模型输出“能跑但不受音频控制”。

张量构造见 [`src/core/inference_worker.cpp`](../../src/core/inference_worker.cpp)，blob 名称见 [`src/model/model_inferencer.cpp`](../../src/model/model_inferencer.cpp)。

## 8. ncnn 加载与推理

- `.param` 描述网络结构、算子和 blob 连接；
- `.bin` 保存权重；
- Net 生命周期长，加载和 warmup 在初始化阶段完成；
- Extractor 通常按单次推理创建，保存该次输入输出和执行状态；
- CPU 线程数、OpenMP 和 Vulkan 选项应在模型加载前确定；
- warmup 只证明形状和执行路径可用，不证明模型质量正确。

模型替换应采用“临时 Net 加载 + warmup + 成功后 swap”，避免加载失败先破坏当前可用模型。

## 9. BGR、RGB 与 YUV

- OpenCV 默认 BGR；
- 模型通常使用 RGB；
- H.264 编码器通常接收 YUV420P 或 NV12；
- YUV420 的色度在宽高方向二倍下采样，常要求偶数分辨率。

颜色异常时应逐段检查通道顺序、stride、色彩矩阵、full/limited range 和像素格式，不能只尝试交换 R/B。

## 10. 当前实现边界

- 对外配置允许多声道，但流式主链实际只可靠支持 16 kHz mono；短期应在入口强制契约。
- AudioProcessor 会重新生成部分 PTS，尚未完整保留输入时间线。
- Pipeline 没有向 RenderThread 注入真实 AudioPlayer，当前不能过度声称已经实现完整音频主时钟同步。
- 静态头像缓存使用较弱的像素签名；生产设计更适合显式 avatar/version ID。

## 11. 高频追问

### 为什么 16 个 Mel 时间步约为 200 ms？

hop 为 200 个采样，在 16 kHz 下是 12.5 ms；16 个时间步约覆盖 200 ms。面试回答要使用当前参数，不能沿用旧资料中的 10 ms/160 ms。

如果面试官追问原始波形支持范围，需要把 50 ms STFT 窗考虑进去：首窗到末窗结束约为 `50 + 15 × 12.5 = 237.5 ms`；“约 200 ms”描述的是 16 个 Mel 时间步的步长范围。

### 如何证明问题在预处理而不是模型？

逐层验证 PCM 能量、Mel 有限性和值域、时间维变化、张量方向，再固定 face 输入对两段不同音频做 A/B。如果模型输出几乎不变，再检查 blob、权重和模型本身。

### 为什么不能只看模型推理耗时？

端到端还包含人脸处理、张量转换、内存复制、融合、编码和队列等待。需要按同一 seq 测量各阶段耗时和驻留时间。

### 量化模型上线前验证什么？

代表性校准集、预处理一致性、输出误差、口型质量、目标硬件速度、内存和回退策略。不能凭模型文件更小就断言质量可接受。
