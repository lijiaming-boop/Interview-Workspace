#pragma once

// ============================================================================
// SpscRingBuffer —— 单生产者单消费者（SPSC）无锁定长环形缓冲区
// ============================================================================
//
// 设计要点（面试可直接复述）：
//
// 1. 内存布局
//    - Capacity 必须是 2 的幂：下标用 `idx & kMask` 取模，避免除法指令。
//    - head_/tail_ 为单调递增的 size_t 计数器（不回绕到 [0,Capacity)），
//      `tail - head` 即元素个数，满/空无需浪费一个槽位来区分；
//      由于 Capacity | 2^64，无符号溢出回绕后差值依旧正确。
//    - 生产者写的 tail_ 与消费者写的 head_ 各独占一条 64B Cache Line
//      （alignas(kCacheLineSize)），消除多核伪共享（false sharing）。
//    - 每侧额外缓存一份“对端下标”（cached_head_ / cached_tail_），
//      只有在本地快照显示满/空时才去读对端原子量，
//      把跨核 Cache Line 迁移从“每次操作”降到“每轮一次”。
//    - 槽位是未初始化的原始存储（placement new + 显式析构），
//      T 无需默认构造，弹出后立即析构，不会延长资源（如 cv::Mat 引用计数）寿命。
//
// 2. 内存序（happens-before 链）
//    生产者：构造元素  →  tail_.store(t+1, release)
//    消费者：tail_.load(acquire) 看到 t+1  →  读取元素  →  head_.store(h+1, release)
//    生产者：head_.load(acquire) 看到 h+1  →  才会复用该槽位
//    自身下标只被自己写，因此读自身下标用 relaxed 即可。
//
// 3. 异常安全
//    - push：T 的构造在发布 tail_ 之前完成，构造抛异常则 tail_ 不前移（强保证）。
//    - pop ：先移动出元素再析构、最后发布 head_；移动抛异常则元素原地保留。
//    - 要求 T 的析构不抛异常（static_assert）。
//
// 4. 线程约束
//    恰好一个生产者线程调用 TryPush/TryEmplace/PushBulk，
//    恰好一个消费者线程调用 TryPop/Front/PopFront/PopBulk。
//    SizeApprox/Empty 任意线程可调用，但只是瞬时近似值。
//
// 与 audio::RingBuffer（float PCM 专用、运行期容量）互补：本类为编译期容量的
// 通用模板，用于流水线阶段之间传递任意可移动类型。
// ============================================================================

#include <atomic>
#include <cstddef>
#include <cstring>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

namespace digital_human {
namespace core {

/// x86-64 / ARMv8 主流 L1 Cache Line 大小。
/// 不直接使用 std::hardware_destructive_interference_size：
/// GCC 在头文件中使用它会触发 -Winterference-size（ABI 不稳定告警）。
inline constexpr std::size_t kCacheLineSize = 64;

template <typename T, std::size_t Capacity>
class SpscRingBuffer {
    static_assert(Capacity >= 2, "SpscRingBuffer: Capacity must be >= 2");
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "SpscRingBuffer: Capacity must be a power of two");
    static_assert(std::is_nothrow_destructible<T>::value,
                  "SpscRingBuffer: T must be nothrow destructible");

public:
    using value_type = T;
    using size_type  = std::size_t;

    static constexpr size_type kCapacity = Capacity;
    static constexpr size_type kMask     = Capacity - 1;

    SpscRingBuffer() = default;

    /// 析构时仍在缓冲区内的元素会被逐一析构。
    /// 调用方必须保证此时生产者/消费者线程均已 join。
    ~SpscRingBuffer() {
        size_type head = head_.load(std::memory_order_relaxed);
        const size_type tail = tail_.load(std::memory_order_relaxed);
        while (head != tail) {
            Slot(head)->~T();
            ++head;
        }
    }

    // 原子量与原始存储都不可安全搬移：禁止拷贝/移动。
    SpscRingBuffer(const SpscRingBuffer&)            = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;
    SpscRingBuffer(SpscRingBuffer&&)                 = delete;
    SpscRingBuffer& operator=(SpscRingBuffer&&)      = delete;

    // ========================================================================
    // 生产者接口（仅生产者线程）
    // ========================================================================

    /// 非阻塞入队。满时返回 false（调用方据此实施反压：重试/降级/丢帧）。
    template <typename U,
              typename = std::enable_if_t<std::is_constructible<T, U&&>::value>>
    bool TryPush(U&& value) noexcept(std::is_nothrow_constructible<T, U&&>::value) {
        return TryEmplace(std::forward<U>(value));
    }

    /// 非阻塞原位构造入队。
    template <typename... Args>
    bool TryEmplace(Args&&... args) noexcept(
        std::is_nothrow_constructible<T, Args&&...>::value) {
        const size_type tail = tail_.load(std::memory_order_relaxed);
        if (tail - cached_head_ == Capacity) {
            // 本地快照显示已满：刷新一次对端下标（acquire 与消费者 release 配对）
            cached_head_ = head_.load(std::memory_order_acquire);
            if (tail - cached_head_ == Capacity) {
                return false;
            }
        }
        ::new (static_cast<void*>(RawSlot(tail))) T(std::forward<Args>(args)...);
        tail_.store(tail + 1, std::memory_order_release);  // 发布元素
        return true;
    }

    /// 批量写入（仅可平凡拷贝类型，如 PCM float / int16）。
    /// 最多写入 count 个，返回实际写入数；整批只做一次 release 发布。
    template <typename U = T,
              typename = std::enable_if_t<std::is_trivially_copyable<U>::value>>
    size_type PushBulk(const T* data, size_type count) noexcept {
        if (data == nullptr || count == 0) return 0;
        const size_type tail = tail_.load(std::memory_order_relaxed);
        size_type free_slots = Capacity - (tail - cached_head_);
        if (free_slots < count) {
            cached_head_ = head_.load(std::memory_order_acquire);
            free_slots   = Capacity - (tail - cached_head_);
        }
        const size_type n = count < free_slots ? count : free_slots;
        if (n == 0) return 0;

        const size_type pos   = tail & kMask;
        const size_type first = (Capacity - pos) < n ? (Capacity - pos) : n;
        std::memcpy(RawSlot(tail), data, first * sizeof(T));
        if (first < n) {
            std::memcpy(RawSlot(0), data + first, (n - first) * sizeof(T));
        }
        tail_.store(tail + n, std::memory_order_release);
        return n;
    }

    // ========================================================================
    // 消费者接口（仅消费者线程）
    // ========================================================================

    /// 非阻塞出队到 out。空时返回 false。
    bool TryPop(T& out) noexcept(std::is_nothrow_move_assignable<T>::value) {
        const size_type head = head_.load(std::memory_order_relaxed);
        if (!RefreshTailIfEmpty(head)) return false;
        T* slot = Slot(head);
        out = std::move(*slot);     // 若抛异常：元素原地保留，head_ 不前移
        slot->~T();
        head_.store(head + 1, std::memory_order_release);  // 归还槽位
        return true;
    }

    /// 非阻塞出队，返回 std::optional（空时 nullopt）。
    std::optional<T> TryPop() noexcept(std::is_nothrow_move_constructible<T>::value) {
        const size_type head = head_.load(std::memory_order_relaxed);
        if (!RefreshTailIfEmpty(head)) return std::nullopt;
        T* slot = Slot(head);
        std::optional<T> result(std::move(*slot));
        slot->~T();
        head_.store(head + 1, std::memory_order_release);
        return result;
    }

    /// 窥视队首（零拷贝）。空时返回 nullptr。
    /// 返回的指针在调用 PopFront()/TryPop() 之前有效。
    T* Front() noexcept {
        const size_type head = head_.load(std::memory_order_relaxed);
        if (!RefreshTailIfEmpty(head)) return nullptr;
        return Slot(head);
    }

    /// 丢弃队首元素（与 Front() 配合实现零拷贝消费）。空时返回 false。
    bool PopFront() noexcept {
        const size_type head = head_.load(std::memory_order_relaxed);
        if (!RefreshTailIfEmpty(head)) return false;
        Slot(head)->~T();
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    /// 批量读取（仅可平凡拷贝类型）。返回实际读取数。
    template <typename U = T,
              typename = std::enable_if_t<std::is_trivially_copyable<U>::value>>
    size_type PopBulk(T* out, size_type count) noexcept {
        if (out == nullptr || count == 0) return 0;
        const size_type head = head_.load(std::memory_order_relaxed);
        size_type avail = cached_tail_ - head;
        if (avail < count) {
            cached_tail_ = tail_.load(std::memory_order_acquire);
            avail        = cached_tail_ - head;
        }
        const size_type n = count < avail ? count : avail;
        if (n == 0) return 0;

        const size_type pos   = head & kMask;
        const size_type first = (Capacity - pos) < n ? (Capacity - pos) : n;
        std::memcpy(out, RawSlot(head), first * sizeof(T));
        if (first < n) {
            std::memcpy(out + first, RawSlot(0), (n - first) * sizeof(T));
        }
        head_.store(head + n, std::memory_order_release);
        return n;
    }

    // ========================================================================
    // 查询接口（任意线程，近似值）
    // ========================================================================

    /// 瞬时元素个数近似值。先读 head 再读 tail，保证结果不会“负溢出”。
    size_type SizeApprox() const noexcept {
        const size_type head = head_.load(std::memory_order_acquire);
        const size_type tail = tail_.load(std::memory_order_acquire);
        const size_type diff = tail - head;
        return diff > Capacity ? Capacity : diff;
    }

    bool Empty() const noexcept { return SizeApprox() == 0; }
    bool Full() const noexcept { return SizeApprox() == Capacity; }
    static constexpr size_type capacity() noexcept { return Capacity; }

private:
    using Storage = std::aligned_storage_t<sizeof(T), alignof(T)>;

    /// 消费者侧：本地快照为空时刷新一次 tail_，仍为空返回 false。
    bool RefreshTailIfEmpty(size_type head) noexcept {
        if (head == cached_tail_) {
            cached_tail_ = tail_.load(std::memory_order_acquire);
            if (head == cached_tail_) return false;
        }
        return true;
    }

    void* RawSlot(size_type idx) noexcept { return &slots_[idx & kMask]; }

    T* Slot(size_type idx) noexcept {
        // placement new 构造的对象需经 std::launder 才能合法访问
        return std::launder(reinterpret_cast<T*>(&slots_[idx & kMask]));
    }

    // ---- 生产者独占 Cache Line ----
    alignas(kCacheLineSize) std::atomic<size_type> tail_{0};
    size_type cached_head_ = 0;   ///< 生产者对 head_ 的本地快照

    // ---- 消费者独占 Cache Line ----
    alignas(kCacheLineSize) std::atomic<size_type> head_{0};
    size_type cached_tail_ = 0;   ///< 消费者对 tail_ 的本地快照

    // ---- 数据槽位（与控制字段隔离，避免写数据时污染下标所在行） ----
    alignas(kCacheLineSize) Storage slots_[Capacity];
};

static_assert(std::atomic<std::size_t>::is_always_lock_free,
              "SpscRingBuffer requires lock-free std::atomic<size_t>");

}  // namespace core
}  // namespace digital_human
