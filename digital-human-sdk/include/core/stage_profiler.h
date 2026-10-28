#pragma once

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace digital_human::core {

// Optional benchmark instrumentation. No samples or locks when not configured.
// Enable only for bounded offline runs: each stage retains one sample per call.
class StageProfiler {
public:
    using Clock = std::chrono::steady_clock;
    void Record(const std::string& stage, double milliseconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        samples_[stage].push_back(milliseconds);
    }
    std::map<std::string, std::vector<double>> Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return samples_;
    }
private:
    mutable std::mutex mutex_;
    std::map<std::string, std::vector<double>> samples_;
};

class ScopedStageTimer {
public:
    ScopedStageTimer(std::shared_ptr<StageProfiler> profiler, const char* stage)
        : profiler_(std::move(profiler)), stage_(stage) {
        if (profiler_) start_ = StageProfiler::Clock::now();
    }
    ~ScopedStageTimer() { Stop(); }
    void Stop() {
        if (!profiler_) return;
        const double ms = std::chrono::duration<double, std::milli>(
            StageProfiler::Clock::now() - start_).count();
        profiler_->Record(stage_, ms);
        profiler_.reset();
    }
private:
    std::shared_ptr<StageProfiler> profiler_;
    const char* stage_;
    StageProfiler::Clock::time_point start_;
};

}  // namespace digital_human::core
