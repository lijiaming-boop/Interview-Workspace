# 文档索引

文档按用途分类。面试准备统一从 `interview/` 开始；架构总览是系统说明基线，面试事实最终以[面试学习路线](interview/README.md)和当前源码为准。早期线程设计、Code Wiki、项目分析及根目录旧版人脸面试稿已清理。`reference/`、`review/` 和 `bugfix/` 中保留的材料用于历史案例，不应替代当前架构说明。

## 架构

- [系统架构设计文档（当前基线，含 2026-08 P0 修复契约与全链路验证记录）](architecture/system_architecture.md)
- [架构总览](architecture/overview.md)
- [第一阶段稳定性改造：生命周期、线程回收与配置一致性](architecture/phase1_stability_refactor.md)
- [第二阶段改造：生命周期收敛、Worker Registry 与可观测性](architecture/phase2_lifecycle_observability_refactor.md)
- [用户图片驱动的实时数字人会话实现方案](architecture/realtime_avatar_conversation_implementation.md)
- [项目完善方案与实施路线](architecture/project_completion_roadmap.md)

## 使用与部署

- [会话服务接口协议](guides/dialog_service_protocol.md)
- [llama.cpp 接入指南](guides/llama_cpp_integration.md)
- [实时头像数字人会话运行指南](guides/realtime_avatar_conversation.md)
- [全链路闭环验收](guides/end_to_end_validation.md)
- [音视频编码与 RTMP/RTSP 推流](guides/stream_publishing.md)
- [Windows Vulkan GPU 验证](guides/windows_vulkan_gpu_validation.md)

## 求职与面试

- [面试文档学习路线：底层、实现、调试、测试与性能](interview/README.md)
  - [C++、内存与并发基础](interview/01_cpp_concurrency.md)
  - [音视频与 Wav2Lip 基础](interview/02_audio_video_ai.md)
  - [系统架构设计](interview/03_system_architecture.md)
  - [核心实现导读](interview/04_core_implementation.md)
  - [调试方法与案例](interview/05_debugging_playbook.md)
  - [测试与质量体系](interview/06_testing_strategy.md)
  - [性能工程](interview/07_performance_engineering.md)
  - [面试表达手册](interview/08_interview_playbook.md)
- [按 C++ 岗位分类的问题与参考答案](career/cpp_interview_guide.md)

## 模型

- [人脸模型](models/face_models.md)
- [模型精度与量化](models/model_precision_quantization_guide.md)

## 性能

- [流水线性能优化报告](perf/pipeline_optimization_report.md)
- [性能优化计划](perf/performance_optimization_plan.md)
- [火焰图分析](perf/perf_flamegraph_analysis.md)
- `perf/` 中保留的 SVG 和报告是历史分析证据；原始 `perf.data`、folded 文件和运行日志已清理。

## 维护记录

- `bugfix/`：历史缺陷修复说明。
- `review/`：阶段性代码审查报告。

## 历史参考

- [框架与推理面试参考](reference/interview_faq_framework_inference.md)
