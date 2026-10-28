# 测试与质量体系

## 1. 30 秒回答

测试按无资源单元契约、Pipeline 集成、真实模型、网络媒体、故障韧性和性能基准分层。并发测试不用长 sleep 猜调度，而用 promise/future、条件变量和测试专用屏障建立确定时序；EOS、Stop、timeout、重复 Start、队列满和异常传播都要成为一等测试。CTest 应是唯一门禁来源，缺目标、缺二进制和依赖未满足必须明确失败或 SKIP，不能假绿。

## 2. 测试分层

| 层级 | 目标 | 外部依赖 | 日常门禁 |
|---|---|---|---|
| unit | 算法、状态机、队列、解析器 | 无 | 是 |
| coverage/contract | 模块公共契约和错误路径 | 无 | 是 |
| integration | 多 worker、文件、编码闭环 | 可选本地依赖 | 是，按能力分组 |
| model | 真实权重、张量和质量 | 模型 | 独立 |
| network | HTTP、RTMP、RTSP | 可控服务 | 独立 |
| fault/resilience | 超时、取消、断线、关闭竞争 | 可控故障注入 | 是 |
| perf | 性能和质量基线 | 固定机器与素材 | 独立，不能混成功能测试 |

现有标签和覆盖矩阵见 [`tests/CMakeLists.txt`](../../tests/CMakeLists.txt)、[`examples/CMakeLists.txt`](../../examples/CMakeLists.txt) 和 [模块覆盖测试](../testing/module_coverage.md)。

## 3. 模块测试矩阵

每个模块至少从四个方向设计：正常、边界、失败、生命周期/并发。

| 模块 | 正常 | 边界 | 失败 | 生命周期/并发 |
|---|---|---|---|---|
| ThreadSafeQueue | Push/Pop 顺序 | 容量 1、空/满 | Stop 唤醒 | 多生产者、阻塞者退出 |
| ThreadBase | Start/Stop/Wait | timeout=0 | Run 抛异常 | 重复 Start、并发 Wait |
| AudioProcessor | 不规则 chunk → 稳定 Mel | 半帧、静音、EOS 残留 | NaN/格式错误 | Stop 时满下游队列 |
| VideoProcessor | 检测、对齐、mask | 小脸、奇数尺寸 | 无脸/模型缺失 | 缓存切换、并发输入 |
| Matcher | 正确 `16×80` 窗 | 短音频、PTS 临界 | Mel 缺失/超阈值 | EOS 与在途任务 |
| InferenceWorker | tensor → output | warmup/首帧 | blob 错、推理失败 | backlog、Stop、重载 |
| Render | DISPLAY 输出 | DROP/DUPLICATE/EOS drain | 非法 ROI | 回调重入、输出所有权 |
| Conversation | 多句顺序 | 短回复、空回复 | LLM/TTS timeout | barge-in、Stop deadline |
| Publisher | BGR/PCM → packet | FIFO 尾帧 | 写包失败、断线 | 幂等 Close、并发 Open/Close |
| C API | 完整输入输出 | 空句柄、小 struct | 版本/能力不匹配 | 多线程调用、销毁竞争 |

## 4. 并发测试怎样做成确定性

不推荐：

```text
启动线程 → sleep 2 秒 → 猜它已经运行到某行
```

推荐：

1. 在测试替身中暴露“已进入阻塞点”的 promise；
2. 主线程等待 future，确认目标状态；
3. 主线程触发 Stop、Close 或第二个操作；
4. 释放测试屏障；
5. 验证完成顺序、错误和最终状态。

本项目使用 C++17，因此优先使用 promise/future、条件变量或测试专用屏障；标准 `std::latch`/`std::barrier` 属于 C++20。

重点并发场景：

- 队列满时生产者被 Stop 唤醒；
- worker Wait 超时后再次 Wait 成功；
- Push 与 EOS 被线性化，不出现 EOS 后丢包；
- 外部回调同步重入时不持内部锁；
- Close 能打断网络 backoff、IO 和队列等待；
- 析构前不存在 joinable thread；
- GetMetrics 与 worker 更新不存在数据竞争。

同一测试重复数百次只能提高触发概率，不能替代明确时序。确定性测试与 TSan 应同时使用。

## 5. 数值和模型测试

模型测试需要分层：

1. 结构门禁：文件存在、blob 名称、输入输出 shape；
2. 数值门禁：无 NaN/Inf，范围合理；
3. 响应门禁：固定 face 下，不同 audio 应产生可测差异；
4. 质量门禁：口型运动、同步、黑帧和人工审片；
5. 性能门禁：同一硬件和同一数据集比较。

测试数据应版本化描述来源、授权、哈希和预处理配置。模型权重不需要进入 Git，但必须有 manifest 和受控下载方式。

## 6. 流式 HTTP 与媒体测试

HTTP/SSE 测试不能假设一次回调等于一行：

- 把一行拆成任意分片；
- 多行合并在一个分片；
- 最后一行没有换行；
- 非 2xx 响应包含看似合法业务数据；
- 慢响应、超大响应、取消和重定向；
- 用户回调抛异常。

媒体发布测试至少验证：

- 输出 stream、codec、time base；
- packet PTS/DTS 单调性；
- 音频时长与视频时长；
- flush 后尾部数据完整；
- 断线时队列不会无限增长；
- Close 能在 deadline 内完成。

Mock 服务证明客户端协议和状态机，不证明真实网络、防火墙、服务端背压和硬件编码器兼容性。

## 7. C/C++ API 与包消费测试

C API 需要覆盖：

- 空 handle、`destroy(NULL)` 和一次性销毁契约；当前裸 handle 在销毁后再次传入属于调用方使用失效指针，若要安全支持重复 destroy，应改成 pointer-to-handle 并在销毁时置空；
- 旧版较小 `struct_size`；
- 未知尾部字段；
- 错误码稳定性；
- caller-owned 输出 buffer 容量不足；
- 回调线程、重入和参数生命周期；
- 不同编译器或语言调用。

安装包测试应在全新目录中创建独立消费者项目，通过 `find_package` 编译、链接和运行。不能只证明仓库内部 target 能链接。

## 8. CI 门禁设计

推荐流水线：

```text
format/static analysis
  → Linux CPU minimal unit/contract
  → Linux full integration
  → Windows CPU install-consumer
  → ASan + UBSan
  → TSan
  → model/network/nightly
  → fixed-machine performance baseline
```

要求：

- 每个 preset 使用独立 `out/build/<preset>`；
- 创建 target 和注册测试使用同一个 CMake helper；
- 必需 target 缺失时配置失败；
- SKIP 使用明确返回码，不伪装 PASS；
- CTest 是唯一测试清单和退出状态来源；
- 性能回退与功能失败分开报告。

## 9. 当前测试状态必须如实说明

- 构建目录、模型和媒体资源已经清理，本轮没有重新构建或运行测试，不能声称当前版本通过。
- [模块覆盖测试](../testing/module_coverage.md)记录的 2026-08-12 结果是历史记录，不是当前验证。
- 文档中的内部断言数和 CTest 可执行项数是不同口径，不能相加成一个“总测试数”。
- [`run_tests.sh`](../../run_tests.sh) 遇到全部二进制缺失时仍可能成功退出，不能作为可靠门禁。
- `model_loader_test`、`model_inferencer_test`、`output_processor_test` 有源码或条件注册，但 target 创建不完整，CMake helper 会静默忽略不存在的 target。
- 现有覆盖率文档描述了覆盖测试，却没有可引用的实际 line/branch 百分比。

## 10. 面试回答：怎样证明不是 Demo

可以给出四类证据：

1. 正常路径：输入到输出闭环；
2. 生命周期：重复启动、EOS、Stop、timeout、析构；
3. 故障路径：缺模型、错误服务、断线、满队列、回调异常；
4. 交付路径：最小/完整构建、安装包消费者、ABI 和跨平台矩阵。

只有成功截图不能证明可靠性；只有覆盖率也不能证明并发和外部系统边界正确。

## 11. 高频追问

### 覆盖率高为什么仍会漏生命周期 Bug？

覆盖率只说明代码被执行，不说明并发顺序、对象状态组合和断言是否正确。析构、timeout 和重入需要专门构造时序。

### 单元测试与集成测试怎样划分？

单元测试验证一个契约并隔离外部依赖；集成测试验证模块之间的数据、时间和完成协议。若失败时无法判断是哪一层，测试范围通常太大。

### 如何测试音视频同步？

使用已知节奏的音频和视觉标记，检查 packet PTS、AV skew 分位数和口型运动相关性；同时验证打断、短输入和长时间漂移。
