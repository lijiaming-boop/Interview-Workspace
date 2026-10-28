# Digital Human SDK

A digital-human lip-sync SDK based on Wav2Lip-SD-GAN. It provides a complete audio/video processing pipeline, including audio feature extraction, face detection and alignment, model inference, and lip fusion rendering.

---

## Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Installing Dependencies](#installing-dependencies)
- [Building](#building)
- [Module Architecture](#module-architecture)
- [Running Tests](#running-tests)
- [API Overview](#api-overview)
- [Configuration](#configuration)
- [FAQ](#faq)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/lijiaming-boop/digital-human-sdk.git
cd digital-human-sdk

# 2. Install dependencies (Ubuntu 22.04)
sudo apt install -y build-essential cmake pkg-config \
    libopencv-dev libavformat-dev libavcodec-dev \
    libavutil-dev libswresample-dev libswscale-dev \
    libportaudio-dev libcurl4-openssl-dev

# 3. Install ncnn (manually)
git clone https://github.com/Tencent/ncnn.git
cd ncnn && mkdir build && cd build
cmake .. -DNCNN_VULKAN=OFF && make -j$(nproc) && sudo make install
cd ../..

# 4. Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 5. Run the tests
./bin/full_pipeline_test
```

---

## System Requirements

| Item | Minimum requirement |
|------|---------------------|
| Operating system | Ubuntu 22.04 / Windows WSL2 |
| Compiler | GCC 11+ / Clang 14+ |
| CMake | 3.16+ |
| CPU | 4-core x86_64 (AVX2 supported) |
| Memory | 8 GB |
| Audio device | ALSA-compatible (PortAudio) |
| GPU (optional) | Vulkan 1.1+ |

---

## Installing Dependencies

### Ubuntu 22.04

```bash
# Basic tools
sudo apt install -y build-essential cmake pkg-config git

# OpenCV 4
sudo apt install -y libopencv-dev

# FFmpeg
sudo apt install -y libavformat-dev libavcodec-dev \
    libavutil-dev libswresample-dev libswscale-dev

# PortAudio
sudo apt install -y portaudio19-dev

# Optional: HTTP text-generation and TTS clients
sudo apt install -y libcurl4-openssl-dev

# ncnn (build from source)
git clone https://github.com/Tencent/ncnn.git
cd ncnn
git checkout 20240820
mkdir build && cd build
cmake .. -DNCNN_VULKAN=OFF \
         -DNCNN_BUILD_EXAMPLES=OFF \
         -DNCNN_BUILD_TOOLS=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../..
```

---

## Building

### Full build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build only the core library

```bash
make -j$(nproc) digital_human_core
```

### Build selected tests

```bash
make -j$(nproc) full_pipeline_test
make -j$(nproc) pipeline_test
make -j$(nproc) inference_worker_test
```

### Output directories

```
build/
├── lib/libdigital_human_core.so   # Core shared library
└── bin/                            # Test executables
    ├── full_pipeline_test
    ├── pipeline_test
    ├── inference_worker_test
    ├── render_thread_test
    ├── audio_processor_test
    ├── frame_scheduler_test
    └── audio_sync_test
```

---

## Module Architecture

### Real-time session and streaming loop

```text
User text/ASR result → llama.cpp/OpenAI-compatible service → Incremental sentence splitting → HTTP TTS service
                                                ↓ 16 kHz mono PCM
Static digital-human image ─────────────────→ ConversationSession
                                                ↓
                                  PushAudio + PushVideo
                                                ↓
                                    Existing Wav2Lip Pipeline
                                                ↓
                                      BGR output frames + PTS
                                                ↓
                                      H.264 + AAC encoding
                                                ↓
                                      RTMP / RTSP publishing
```

Text generation and TTS are both integrated through abstract clients. A llama.cpp OpenAI Chat Completions adapter is currently provided. The surrounding media layer encodes TTS PCM as AAC, encodes SDK-rendered BGR frames as H.264, and supports publishing to RTMP/RTSP servers. See the [documentation index](docs/README.md) for the detailed architecture, llama.cpp configuration, and streaming integration instructions.

See the [real-time avatar conversation guide](docs/guides/realtime_avatar_conversation.md) for the complete commands for user image upload, hot avatar updates, real local TTS, and persistent multi-turn sessions.

### End-to-end data flow

```
Audio: PCM float → NoiseReduction → AudioFramer → VAD
      → PreEmphasis → RMSNormalize → MelFeatureExtract
      → CMVN → ncnn::Mat

Video: cv::Mat BGR → FaceDetect → FaceAlign → FaceMaskGen
       → ProcessedFaceData

Inference: Mel + AlignedFace → ModelInferencer(Wav2Lip) → ncnn::Mat

Rendering: ncnn::Mat → OutputToMat → InverseTransform
           → FaceFusion → Sharpen → ColorBlend → cv::Mat
```

### Thread model

```
AudioProducer ──► AudioProcessor ──► ┌─ SyncPoint ─┐
                                     │ (A/V match) │
VideoProducer ──► VideoProcessor ──►└─────────────┘
                                              │
                                              ▼
                                        InferenceWorker
                                              │
                                              ▼
                                        OutputProcessor
                                              │
                                              ▼
                                          RenderThread
                                              │
                                              ▼
                                     FrameCallback/OutputQueue
```

| Thread | Responsibility | Priority |
|--------|----------------|----------|
| AudioProducer | Reads audio data | High |
| AudioProcessor | PCM → Mel feature extraction (7-stage pipeline) | Medium |
| VideoProducer | Reads video frames | High |
| VideoProcessor | Face detection → alignment → masking | Medium |
| InferenceWorker | Wav2Lip model inference + retry on failure | Medium |
| OutputProcessor | Post-processes model output | Medium |
| RenderThread | Fusion + audio synchronization + frame pacing | High |
| AudioPlayback | PortAudio audio output (system callback) | Highest (real-time) |

### Core library structure

```
include/
├── dialog/                         # Dialog orchestration, text services, incremental sentence splitting
├── tts/                            # TTS client and PCM contract
├── network/                        # Optional libcurl HTTP transport
├── audio/                          # Audio processing modules
│   ├── audio_loader.h              # Audio file loading
│   ├── audio_framer.h              # Framing (frame/hop)
│   ├── audio_noise_reduction.h     # Spectral-subtraction noise reduction
│   ├── audio_vad.h                 # Voice activity detection
│   ├── audio_preemphasis.h         # Pre-emphasis
│   ├── audio_rms_normalize.h       # RMS normalization
│   ├── audio_mel_feature_extract.h # Mel spectrum (FFT → filter → log)
│   ├── audio_cmvn.h                # Cepstral mean and variance normalization
│   ├── audio_ring_buffer.h          # Lock-free SPSC ring buffer
│   └── audio_player.h               # PortAudio player
│
├── core/                           # Core modules
│   ├── image_loader.h               # Image loading
│   ├── face_detector.h              # Face detection
│   ├── face_aligner.h               # Face alignment (affine transform)
│   ├── face_mask_generator.h        # Lip mask generation
│   ├── frame_scheduler.h            # Frame scheduling (DROP/DUPLICATE/DISPLAY)
│   ├── av_sync.h                    # Audio/video synchronization (Audio Master Clock)
│   ├── audio_sync_scheduler.h       # Audio synchronization scheduler
│   ├── audio_processor.h            # Audio processing thread
│   ├── inference_worker.h           # Inference thread
│   ├── render_thread.h              # Rendering thread
│   ├── pipeline.h                  # Pipeline orchestration
│   ├── packet.h                     # Packet definitions
│   ├── thread_safe_queue.h          # Thread-safe queue
│   └── thread_base.h                # Thread base class (lifecycle management)
│
└── model/                          # Model modules
    ├── model_loader.h               # Asynchronous model loading (callback notification)
    ├── model_inferencer.h           # Wav2Lip inferencer (auto-tuning)
    └── output_processor.h            # Output post-processing (fusion/sharpening/color)
```

---

## Running Tests

### Full test suite

```bash
cd build

# End-to-end integration tests (162 tests, no Wav2Lip model required)
./bin/full_pipeline_test

# Inference and rendering integration (21 tests)
./bin/inference_render_pipeline_test

# Threading infrastructure (76 tests)
./bin/pipeline_test

# Independent module tests
./bin/audio_processor_test      # 23 tests
./bin/inference_worker_test     # 45 tests
./bin/render_thread_test        # 28 tests
./bin/frame_scheduler_test      # 62 tests
./bin/audio_sync_test           # 37 tests

# Phase-one real-time session tests
./bin/dialog_module_test
./bin/conversation_sdk_integration_test

# AAC/H.264 encoding and streaming
./bin/stream_publisher_test
./bin/conversation_stream_integration_test
./bin/stream_network_publish_test rtsp rtsp://127.0.0.1:8554/live/test

# HTTP client smoke tests (start tools/mock_dialog_service.py first)
./bin/http_service_client_test \
  http://127.0.0.1:18080/text \
  http://127.0.0.1:18080/tts

# llama.cpp SSE/JSON and ConversationSession loop
./bin/llama_cpp_client_test \
  http://127.0.0.1:8090/v1/chat/completions \
  Qwen3-4B-Q4_K_M.gguf

# Full llama.cpp → HTTP TTS → Wav2Lip → H.264/AAC pipeline
./bin/full_conversation_chain_test \
  http://127.0.0.1:8090/v1/chat/completions \
  http://127.0.0.1:18080/tts \
  full_conversation_chain.flv file Qwen3-4B-Q4_K_M.gguf
```

### Quick validation script

```bash
# run_all_tests.sh
cd build
tests=(
    full_pipeline_test
    inference_render_pipeline_test
    pipeline_test
    audio_processor_test
    inference_worker_test
    render_thread_test
    frame_scheduler_test
    audio_sync_test
)
for t in "${tests[@]}"; do
    echo "=== $t ==="
    ./bin/$t 2>/dev/null
done
```

### Test levels

| Level | Coverage | Command |
|-------|----------|---------|
| Level 0 | Independent module tests (no model required) | Individual `*_test.cpp` files |
| Level 1 | Threading infrastructure (queues/thread base class) | `pipeline_test` |
| Level 2 | Audio/inference/rendering threads | `*_worker_test` |
| Level 3 | Inference and rendering integration | `inference_render_pipeline_test` |
| Level 4 | Full end-to-end pipeline | `full_pipeline_test` |

---

## API Overview

### Audio processing

```cpp
#include "audio/audio_loader.h"
#include "audio/audio_framer.h"
#include "audio/audio_mel_feature_extract.h"

using namespace digital_human::audio;

// Load audio
AudioLoader loader;
AudioData data = loader.load("input.wav");

// Denoise → frame
NoiseReduction nr;
auto denoised = nr.process(data.samples, data.sampleRate);

AudioFramer framer;
auto frames = framer.frame(denoised, {400, 160});

// Mel features
MelFeatureExtract mel;
MelConfig mc{512, 80, 16000, 0, 8000};
auto spec = mel.extract(frames, mc);  // cv::Mat (T×80)
```

### Video processing

```cpp
#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/face_mask_generator.h"

using namespace digital_human::core;

// Face detection
FaceDetector detector;
auto faces = detector.detect(image);

// Alignment
FaceAligner aligner;
auto result = aligner.alignByRect(image, landmarks, 96, faces[0]);

// Masking
FaceMaskGenerator mask_gen;
auto mask = mask_gen.generateMouthMask(image.size(), landmarks);
```

### Model inference

```cpp
#include "model/model_inferencer.h"

digital_human::model::ModelInferencer model;
model.Init("models/Wav2Lip-SD-GAN-opt");

ncnn::Mat output = model.Infer(audio_feat_ncnn, face_input_ncnn);
```

### Pipeline

```cpp
#include "core/pipeline.h"

using namespace digital_human::core;

PipelineConfig config;
config.audio_sample_rate = 16000;
config.target_fps        = 25.0;

Pipeline pipeline;
pipeline.Init(config);
pipeline.Start();

// Input data
pipeline.PushAudio(pcm_data, pts_ms);
pipeline.PushVideo(frame, pts_ms);

// Read output
OutputFramePacket result;
while (pipeline.GetOutputFrame(result)) {
    cv::imshow("Output", result.payload);
}

pipeline.Stop();
```

### Audio/video synchronization

```cpp
#include "core/audio_sync_scheduler.h"

using namespace digital_human::core;

AudioSyncScheduler sched;
AudioSyncConfig cfg;
cfg.audio_sample_rate = 48000;
cfg.target_fps        = 25.0;
sched.Init(cfg);
sched.LoadAudio(pcm_data, channels);
sched.Play();

// Frame scheduling
auto decision = sched.ScheduleFrame(frame_id, pts_ms);
switch (decision.action) {
    case FrameAction::DISPLAY:   break;  // Display normally
    case FrameAction::DROP:      break;  // Drop a late frame
    case FrameAction::DUPLICATE: break;  // Duplicate an early frame
    case FrameAction::WAIT:      break;  // Wait for audio to catch up
}
```

---

## Configuration

### PipelineConfig

| Field | Default | Description |
|-------|---------|-------------|
| `audio_sample_rate` | 16000 | Audio sample rate (Hz) |
| `audio_frame_size` | 400 | Frame size (samples @ 16 kHz = 25 ms) |
| `audio_hop_size` | 160 | Hop size (samples @ 16 kHz = 10 ms) |
| `target_fps` | 25.0 | Target frame rate |
| `sync_threshold_ms` | 30.0 | Synchronization threshold (ms) |
| `max_drift_ms` | 100.0 | Maximum allowed drift (ms) |

### SDKConfig (file processing)

| Field | Default | Description |
|-------|---------|-------------|
| `file_audio_lead_ms` | 300 | Maximum audio lead over submitted video allowed by `ProcessFile`, preventing insufficient input or excessive prefetching |
| `file_stall_timeout_ms` | 30000 | Timeout for continuous lack of output progress during file processing; the pipeline stops and wakes blocked queues after the timeout |

`ProcessFile` uses independent audio and video producer threads. Audio remains the synchronization master clock; the lead watermark is used only to control input backpressure.

### InferenceWorkerConfig

| Field | Default | Description |
|-------|---------|-------------|
| `max_retries` | 3 | Maximum number of retries after inference failure |
| `pop_timeout_ms` | 100 | Queue pop timeout (ms) |
| `input_queue_warn_threshold` | 10 | Queue depth warning threshold |
| `input_queue_error_threshold` | 30 | Queue depth error threshold |
| `backlog_check_interval` | 1000 | Backlog check interval (ms) |
| `latency_warn_threshold_ms` | 100.0 | Inference latency warning threshold (ms) |

### RenderConfig

| Field | Default | Description |
|-------|---------|-------------|
| `target_fps` | 25.0 | Target frame rate |
| `sync_threshold_ms` | 30.0 | Synchronization threshold (ms) |
| `max_drift_ms` | 100.0 | Maximum allowed drift (ms) |
| `enable_audio_sync` | true | Enable audio synchronization |
| `enable_frame_pacing` | true | Enable frame pacing |
| `drain_max_frames` | 30 | Maximum number of frames to drain on exit |
| `pop_timeout_ms` | 100 | Queue pop timeout (ms) |

---

## FAQ

### Build issues

**Q: ncnn cannot be found**
```bash
export CMAKE_PREFIX_PATH=/usr/local/lib/cmake:$CMAKE_PREFIX_PATH
cmake ..
```

**Q: OpenCV version mismatch** — Ubuntu 22.04 installs OpenCV 4.5.4 by default.

**Face models** — Prepare SCRFD and 2D106 weights according to [Deep Learning Face Models](docs/models/face_models.md).

### Runtime issues

**Q: PortAudio reports "No audio devices"**
- WSL2 has no audio device by default; this does not affect functional tests.
- Production environments require an ALSA driver (`sudo apt install alsa-utils`).

**Q: The inference thread keeps retrying**
- Confirm that the model path is correct.
- Check that `.param` and `.bin` files are present under `models/`.

**Q: The rendering frame rate is unstable**
- Adjust `RenderConfig::target_fps`.
- Make sure `enable_frame_pacing = true`.
- Check whether the queues are backing up (`InferenceMetrics` logs).

### Debugging

```bash
# Check memory with Valgrind
valgrind --leak-check=full ./bin/full_pipeline_test 2>/dev/null

# Specify the model directory
export MODEL_DIR=/path/to/models
```

---

## Contributing

1. Create a feature branch from `master`: `feature/your-feature`.
2. Follow the PIMPL pattern when implementing new modules.
3. Add Chinese Doxygen comments for all public APIs.
4. Write independent tests for each module, covering normal, boundary, and failure paths.
5. Make sure the full test suite passes before submitting: `./run_all_tests.sh`.

---

## License

MIT License
