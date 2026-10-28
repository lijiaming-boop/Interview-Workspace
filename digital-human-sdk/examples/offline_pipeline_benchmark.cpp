// Offline benchmark: image + audio -> optional H.264/AAC MP4, with stage samples.
// Usage: offline_pipeline_benchmark image audio out_dir cpu|gpu seconds threads encode [baseline|direct|outer-feather]
// seconds=0 uses the complete audio. encode=0 discards frames (core throughput).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include "audio/audio_loader.h"
#include "core/pipeline.h"
#include "digital_human/logger.h"

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
using namespace digital_human;

static double Ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static std::string Quote(const std::string& s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') out << '\\';
        if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    out << '"';
    return out.str();
}
static std::string ShellPath(const fs::path& path) {
    const auto s = path.generic_string();
    if (s.find_first_of("\"&|<>^%\r\n") != std::string::npos)
        throw std::runtime_error("Unsupported shell metacharacter in encoder path");
    return '"' + s + '"';
}

int main(int argc, char** argv) try {
    if (argc != 8 && argc != 9) {
        std::cerr << "Usage: offline_pipeline_benchmark image audio out_dir cpu|gpu seconds threads encode [baseline|direct|outer-feather]\n";
        return 1;
    }
    const auto job_start = Clock::now();
    set_global_logger(create_default_logger(LogLevel::Warn));
    const fs::path image_path(argv[1]), audio_path(argv[2]), out_dir(argv[3]);
    const bool gpu = std::string(argv[4]) == "gpu";
    if (!gpu && std::string(argv[4]) != "cpu") throw std::runtime_error("backend must be cpu or gpu");
    const double seconds = std::stod(argv[5]);
    const int threads = std::stoi(argv[6]);
    const bool encode = std::stoi(argv[7]) != 0;
    const std::string variant = argc == 9 ? argv[8] : "baseline";
    if (variant != "baseline" && variant != "direct" && variant != "outer-feather")
        throw std::runtime_error("Unknown render variant");
    const bool save_snapshots = argc == 9 && encode;
    fs::create_directories(out_dir);
    const auto image_start = Clock::now();
    std::ifstream image_file(image_path, std::ios::binary);
    const std::vector<uchar> bytes((std::istreambuf_iterator<char>(image_file)), {});
    const auto image = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("Could not decode image");
    const double load_image_ms = Ms(image_start);
    const auto audio_start = Clock::now();
    audio::AudioLoader loader;
    auto audio = loader.load(audio_path.string());
    if (audio.sampleRate != 16000 || audio.channels != 1)
        throw std::runtime_error("AudioLoader did not normalize to 16kHz mono");
    if (seconds > 0 && audio.samples.size() > static_cast<size_t>(seconds * 16000))
        audio.samples.resize(static_cast<size_t>(seconds * 16000));
    const double load_audio_ms = Ms(audio_start);
    const double content_s = audio.samples.size() / 16000.0;
    constexpr double fps = 25;
    const int64_t expected = static_cast<int64_t>(std::floor(content_s * fps));
    if (expected <= 0) throw std::runtime_error("Empty audio");
    auto profiler = std::make_shared<core::StageProfiler>();
    core::PipelineConfig config;
    config.stage_profiler = profiler;
    config.target_fps = fps;
    config.enable_frame_pacing = false;
    config.opencv_num_threads = 4;
    config.mouth_generated_blend_weight = variant == "baseline" ? 0.7f : 1.0f;
    config.mouth_outward_feather_px = variant == "outer-feather" ? 3 : 0;
    core::Pipeline pipeline;
    const auto init_start = Clock::now();
    if (!pipeline.Init(config)) throw std::runtime_error("Pipeline Init failed");
    const auto models = fs::path(PROJECT_SOURCE_DIR) / "models";
    if (!pipeline.InitModelInferencer(models.string())) throw std::runtime_error("Wav2Lip Init failed");
    if (!pipeline.EnableGPU(gpu) || pipeline.IsGPUEnabled() != gpu)
        throw std::runtime_error("Requested backend was not selected");
    pipeline.SetInferenceThreads(threads);
    const double model_init_ms = Ms(init_start);
    const auto face_load_start = Clock::now();
    if (!pipeline.LoadFaceModel((models / "face").string()))
        throw std::runtime_error("Face models failed to load");
    const double face_model_load_ms = Ms(face_load_start);
    std::cout << "START " << argv[4] << " " << image.cols << "x" << image.rows
              << " frames=" << expected << " encode=" << encode << std::endl;

    FILE* encoder = nullptr;
    if (encode) {
        const auto output_path = out_dir / "lipsync.mp4";
        std::ostringstream cmd;
        cmd << "ffmpeg -nostdin -hide_banner -loglevel error -y -f rawvideo -pix_fmt bgr24 -s "
            << image.cols << 'x' << image.rows << " -r 25 -i pipe:0 -i " << ShellPath(audio_path)
            << " -map 0:v:0 -map 1:a:0 -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\""
            << " -c:v libx264 -preset veryfast -crf 20 -threads 2 -c:a aac -b:a 192k"
            << " -shortest -movflags +faststart " << ShellPath(output_path)
            << " 2> " << ShellPath(out_dir / "encoder.log");
#ifdef _WIN32
        encoder = _popen(cmd.str().c_str(), "wb");
#else
        encoder = popen(cmd.str().c_str(), "w");
#endif
        if (!encoder) throw std::runtime_error("Could not launch ffmpeg encoder");
    }
    std::atomic<bool> producer_ok{true};
    std::vector<Clock::time_point> ingress(static_cast<size_t>(expected));
    std::vector<double> e2e_ms;
    std::vector<int64_t> output_pts;
    std::vector<double> output_at_ms;
    std::vector<std::pair<int64_t, cv::Mat>> snapshots;
    const auto pipeline_start = Clock::now();
    if (!pipeline.Start()) throw std::runtime_error("Pipeline Start failed");
    std::thread video([&] {
        for (int64_t i = 0; i < expected; ++i) {
            ingress[static_cast<size_t>(i)] = Clock::now();
            if (!pipeline.PushVideo(image, i * 40)) { producer_ok = false; break; }
        }
        pipeline.MarkVideoEOS();
    });
    std::thread sound([&] {
        constexpr size_t chunk = 1600;
        for (size_t offset = 0; offset < audio.samples.size(); offset += chunk) {
            const size_t end = std::min(offset + chunk, audio.samples.size());
            std::vector<float> samples(audio.samples.begin() + offset, audio.samples.begin() + end);
            if (!pipeline.PushAudio(samples, static_cast<int64_t>(offset * 1000 / 16000))) {
                producer_ok = false; break;
            }
        }
        pipeline.MarkAudioEOS();
    });
    int64_t frames = 0;
    bool eos = false, encode_ok = true;
    double first_output_ms = 0;
    while (Ms(pipeline_start) < std::max(120000.0, content_s * 20000)) {
        core::OutputFramePacket packet;
        if (!pipeline.GetOutputFrame(packet, 1000)) continue;
        if (packet.header.IsEOS()) { eos = true; break; }
        if (packet.header.IsFatal()) break;
        if (!packet.header.IsOK() || packet.payload.empty()) continue;
        if (frames == 0) first_output_ms = Ms(pipeline_start);
        const int64_t idx = packet.header.pts_ms / 40;
        if (idx >= 0 && idx < expected) e2e_ms.push_back(Ms(ingress[static_cast<size_t>(idx)]));
        output_pts.push_back(packet.header.pts_ms);
        output_at_ms.push_back(Ms(pipeline_start));
        if (save_snapshots && idx % 75 == 0) {
            // Identical sparse copies in all A/B groups; PNG encoding is deferred
            // until after measured processing and encoder shutdown.
            snapshots.emplace_back(packet.header.pts_ms, packet.payload.clone());
        }
        if (encoder) {
            core::ScopedStageTimer timer(profiler, "encode_pipe_write");
            const auto& frame = packet.payload;
            if (frame.isContinuous()) {
                const size_t n = frame.total() * frame.elemSize();
                if (std::fwrite(frame.data, 1, n, encoder) != n) encode_ok = false;
            } else {
                for (int row = 0; row < frame.rows; ++row) {
                    const size_t n = frame.cols * frame.elemSize();
                    if (std::fwrite(frame.ptr(row), 1, n, encoder) != n) encode_ok = false;
                }
            }
        }
        ++frames;
        if (frames % 500 == 0) std::cout << "PROGRESS " << frames << '/' << expected
                                      << " wall_s=" << Ms(pipeline_start)/1000 << std::endl;
        if (!encode_ok) break;
    }
    if (!eos) pipeline.Stop(); // Unblock producers on a failed run.
    video.join();
    sound.join();
    const bool stopped = pipeline.Stop();
    const double pipeline_s = Ms(pipeline_start) / 1000;
    double finalize_ms = 0;
    if (encoder) {
        const auto close_start = Clock::now();
#ifdef _WIN32
        encode_ok = (_pclose(encoder) == 0) && encode_ok;
#else
        encode_ok = (pclose(encoder) == 0) && encode_ok;
#endif
        finalize_ms = Ms(close_start);
    }
    const double job_s = Ms(job_start) / 1000;
    const auto metrics = pipeline.GetMetrics();
    bool pts_ok = true;
    for (size_t i = 0; i < output_pts.size(); ++i)
        if (output_pts[i] != static_cast<int64_t>(i) * 40) pts_ok = false;
    const bool complete = producer_ok && eos && stopped && encode_ok
        && frames == expected && pts_ok && metrics.inference_count == frames;
    auto samples = profiler->Snapshot();
    samples["offline_ingress_to_output"] = std::move(e2e_ms);
    std::ofstream raw(out_dir / "stage_samples.csv");
    raw << "stage,sample_index,milliseconds\n" << std::setprecision(10);
    std::ofstream json(out_dir / "report.json");
    json << std::setprecision(10) << "{\n"
         << "\"image\":" << Quote(image_path.generic_string()) << ",\n"
         << "\"audio\":" << Quote(audio_path.generic_string()) << ",\n"
         << "\"backend\":" << Quote(argv[4]) << ",\n"
         << "\"render_variant\":" << Quote(variant) << ",\n"
         << "\"mouth_generated_blend_weight\":" << config.mouth_generated_blend_weight << ",\n"
         << "\"mouth_outward_feather_px\":" << config.mouth_outward_feather_px << ",\n"
         << "\"gpu_enabled\":" << (pipeline.IsGPUEnabled() ? "true" : "false") << ",\n"
         << "\"encode\":" << (encode ? "true" : "false") << ",\n"
         << "\"cpu_threads\":" << threads << ",\n\"opencv_threads\":4,\n"
         << "\"width\":" << image.cols << ",\n\"height\":" << image.rows << ",\n"
         << "\"content_seconds\":" << content_s << ",\n\"expected_frames\":" << expected << ",\n"
         << "\"output_frames\":" << frames << ",\n\"inference_count\":" << metrics.inference_count << ",\n"
         << "\"dropped\":" << metrics.frames_dropped << ",\n\"skipped\":" << metrics.frames_skipped << ",\n"
         << "\"av_match_misses\":" << metrics.av_match_miss_count << ",\n"
         << "\"pipeline_seconds\":" << pipeline_s << ",\n\"pipeline_fps\":" << frames/pipeline_s << ",\n"
         << "\"job_seconds\":" << job_s << ",\n\"job_fps\":" << frames/job_s << ",\n"
         << "\"speed_x\":" << (frames/fps)/pipeline_s << ",\n"
         << "\"load_image_ms\":" << load_image_ms << ",\n\"load_audio_ms\":" << load_audio_ms << ",\n"
         << "\"model_init_ms\":" << model_init_ms << ",\n\"face_model_load_ms\":" << face_model_load_ms << ",\n"
         << "\"first_output_ms\":" << first_output_ms << ",\n\"encoder_finalize_ms\":" << finalize_ms << ",\n"
         << "\"eos\":" << (eos ? "true" : "false") << ",\n"
         << "\"complete\":" << (complete ? "true" : "false") << ",\n\"stages\":{\n";
    bool first = true;
    for (auto& [name, values] : samples) {
        for (size_t i = 0; i < values.size(); ++i) raw << name << ',' << i << ',' << values[i] << '\n';
        if (values.empty()) continue;
        const double sum = std::accumulate(values.begin(), values.end(), 0.0);
        std::sort(values.begin(), values.end());
        auto quantile = [&](double q) { return values[static_cast<size_t>(q * (values.size()-1))]; };
        if (!first) json << ",\n";
        first = false;
        json << Quote(name) << ":{\"count\":" << values.size() << ",\"total_ms\":" << sum
             << ",\"mean_ms\":" << sum/values.size() << ",\"p50_ms\":" << quantile(.50)
             << ",\"p95_ms\":" << quantile(.95) << ",\"p99_ms\":" << quantile(.99)
             << ",\"max_ms\":" << values.back() << '}';
    }
    json << "\n}}\n";
    std::ofstream trace(out_dir / "output_timeline.csv");
    trace << "frame_index,pts_ms,output_wall_ms\n";
    for (size_t i = 0; i < output_pts.size(); ++i)
        trace << i << ',' << output_pts[i] << ',' << output_at_ms[i] << '\n';
    if (save_snapshots) {
        fs::create_directories(out_dir / "snapshots");
        for (const auto& [pts, frame] : snapshots) {
            cv::imwrite((out_dir / "snapshots" / ("frame_" + std::to_string(pts) + ".png")).string(), frame);
        }
    }
    std::cout << "RESULT frames=" << frames << '/' << expected << " pipeline_fps=" << frames/pipeline_s
              << " job_fps=" << frames/job_s << " complete=" << complete << std::endl;
    return complete ? 0 : 2;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
}
