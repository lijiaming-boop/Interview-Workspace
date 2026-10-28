# C++、内存与并发基础

## 1. 面试官考察什么

这部分不是考语法，而是考候选人能否建立清晰的所有权、生命周期和并发不变量。数字人 SDK 同时涉及线程、回调、第三方 C API、图像共享内存和超时关闭，是解释现代 C++ 工程能力的好材料。

### 30 秒回答

项目用 RAII 管理线程和第三方资源，用 Pimpl 隔离实现并缩小 ABI 变化；worker 通过有界队列通信，mutex 保护复合不变量，原子变量只表示简单状态，条件变量负责无忙等唤醒。停止采用协作式取消和 join，不能 detach。跨线程传递的图像和音频必须明确只读、复制或共享所有权，不能把引用计数误当成线程安全。

## 2. RAII 与所有权

RAII 的核心不是“用了智能指针”，而是资源状态与对象生命周期绑定：

- 构造或 Init 获得资源；
- 析构或 Stop 释放资源；
- 任意失败路径都不能泄漏线程、FFmpeg 上下文、PortAudio 流或模型对象；
- 部分初始化和 moved-from 对象也必须能够安全析构。

常见所有权选择：

| 类型 | 适用场景 | 项目中的关注点 |
|---|---|---|
| `std::unique_ptr` | 单一所有者、Pimpl | SDK、Pipeline 和各模块实现对象 |
| `std::shared_ptr` | 生命周期确实跨线程共享 | 异步回调依赖的会话状态、不可变媒体块 |
| `std::weak_ptr` | 回调不能延长宿主生命周期 | ASR/VAD/HTTP 回调捕获会话状态 |
| 值类型 | 数据较小、复制成本可接受 | 配置、错误码、指标快照 |
| 裸指针/引用 | 明确的非拥有、同步调用 | 不应跨异步边界长期保存 |

面试时要强调：`shared_ptr` 只保证引用计数安全，不自动保证被指向对象的字段安全。

## 3. Pimpl、Move 与 ABI

Pimpl 将实现成员放入 `Impl`，公共头只保留 `std::unique_ptr<Impl>`。收益包括：

- 减少公共头依赖和编译传播；
- 隐藏第三方实现类型；
- 实现字段变化时尽量不改变 C++ 类对象布局；
- 让公共接口更容易保持稳定。

代价包括一次间接访问、一次堆分配，以及必须认真处理析构和 Move 语义。

默认 Move 会把源对象中的 `unique_ptr` 置空。因此析构、`Destroy()` 和允许在 moved-from 对象上调用的成员函数必须处理空 `impl_`。案例见 [Pimpl Move 语义修复](../bugfix/pimpl_move_semantics_fix.md)。

需要避免的错误表述：Pimpl 不能自动保证跨编译器 C++ ABI，也不能隐藏公共接口里直接出现的 OpenCV 或 ncnn 类型。跨语言稳定边界更适合使用 C ABI。

## 4. `cv::Mat` 与媒体内存

`cv::Mat` 拷贝通常只复制矩阵头并共享像素数据。它不会因为任意写操作自动为业务代码提供可靠的 copy-on-write 隔离。

跨线程传递时必须明确一种契约：

1. 唯一所有权：移动一个独立 buffer；
2. 共享只读：`shared_ptr<const FrameBuffer>`；
3. 边界复制：在 API 入口或异步保存前 `clone()`；
4. Buffer pool：用 lease 表示归还时机。

全链路无条件 clone 虽然安全，但会增加内存带宽和延迟。优化前先画出所有权图，再决定哪里复制。

## 5. 原子变量与 Mutex 的边界

原子变量适合独立标量状态，例如 stop_requested、计数器或一次状态转换。它不能自动维护多个字段的一致快照。

例如音频播放器同时读取 data、total_frames、channels 和 read_position。如果这些字段分别原子化，仍可能组合出“旧游标 + 新 buffer”的非法状态。正确设计通常是：

- 用 mutex 保护一次完整更新；或
- 原子发布一个不可变快照对象；或
- 把全部状态放在单线程 actor 中串行访问。

需要 mutex 的典型场景：

- SDK 生命周期状态与资源句柄必须同步变化；
- Open/Close 的完整过程必须串行；
- 一组指标需要同一时刻的一致快照；
- Push 与 EOS 必须建立明确的线性化顺序。

## 6. 条件变量与有界队列

条件变量必须和谓词一起使用：

```cpp
cv.wait(lock, [&] { return stopped || !queue.empty(); });
```

谓词用于处理虚假唤醒，也使 Stop 能够唤醒阻塞的生产者和消费者。有界队列解决的是“生产长期快于消费”问题：

- 无界队列会把性能问题变成内存耗尽和高延迟；
- 文件处理可以选择有截止时间的阻塞；
- 直播视频通常优先丢旧帧；
- 音频更关注连续性，但也必须限制最大等待和保留时长。

项目队列实现见 [`include/core/thread_safe_queue.h`](../../include/core/thread_safe_queue.h)。

## 7. Worker 生命周期

一个可靠 worker 至少需要以下状态：

```text
CREATED → RUNNING → STOP_REQUESTED → STOPPED
                   ↘ ERROR
```

核心原则：

- 启动下游再启动上游，防止刚生产的数据无人消费；
- 停止先关闭输入，再传播 EOS 或取消，唤醒所有阻塞点；
- `Wait(timeout)` 超时不能 detach；线程仍可能访问 `this`；
- 析构前必须完成 join，或者把线程所需状态完整移入独立共享对象；
- 自然 EOS 排空和用户取消是两种不同语义。

现有线程基础设施见 [`include/core/thread_base.h`](../../include/core/thread_base.h) 和 [`src/core/worker_registry.cpp`](../../src/core/worker_registry.cpp)。

## 8. 异常边界

异常不能穿过 C ABI、libcurl/PortAudio/FFmpeg 回调或线程入口。标准做法是：

```text
noexcept 边界 → catch std::exception / catch (...) → 转换错误 → 请求全链路停止
```

同样，不能持有内部 mutex 调用用户回调或外部组件；对方可能同步回调并重入 SDK，形成死锁。

## 9. 当前项目可讲与不可过度声称的内容

可以讲：

- 有界队列、条件变量谓词、WorkerRegistry 和共享停止 deadline；
- Pimpl、opaque C handle 和可选适配器；
- 实时视频与音频采用不同反压策略。

需要说明仍待完善：

- 部分运行时指标存在无锁读写；
- ConversationSession 超时析构、Bridge drain 和外部回调的关闭协议还需收敛；
- StreamPublisher 的并发 Open/Close 需要完整生命周期状态机；
- Pause 尚未成为所有 worker 共享的不变量。

## 10. 高频追问

### 原子变量能否替代 mutex？

不能。原子只保证单次原子操作，不保证多个字段的组合不变量，也不自动解决对象生命周期。

### 为什么不能 detach 超时线程？

detach 后宿主对象可能析构，而线程继续访问成员，形成 use-after-free。C++ 没有安全强杀线程，必须让每个阻塞点可取消并最终 join。

### 为什么队列必须有界？

有界队列把过载显式转化为阻塞、丢弃或失败策略，避免延迟和内存无限增长，同时让队列深度成为可观测的瓶颈信号。

### 怎样做并发代码的确定性测试？

用 future、promise 或受控条件变量建立明确 happens-before，而不是依赖长时间 sleep 猜调度。`std::latch` 和 `std::barrier` 属于 C++20；本项目是 C++17，可使用测试专用屏障或上述原语实现同样目的。

## 11. 建议实验

- 构造容量为 1 的队列，验证满队列生产者能被 Stop 唤醒。
- 构造 Wait 超时后再次 Wait 的 worker，证明线程所有权没有丢失。
- 用浅拷贝 `cv::Mat` 修改消费者帧，观察共享像素的影响，再比较边界 clone。
- 用 TSan 运行并发 Push/EOS、GetMetrics 和 Open/Close 场景。
