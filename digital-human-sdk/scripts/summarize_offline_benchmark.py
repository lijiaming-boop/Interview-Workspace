"""Build a readable report and CSV from run_current_offline_benchmark.py outputs."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_root", type=Path)
    args = parser.parse_args()
    root = args.report_root.resolve()
    status = json.loads((root / "suite_status.json").read_text(encoding="utf-8"))
    results = status["results"]
    by_case = {r["case"]: r for r in results}
    def fmt(value):
        return "—" if value is None else f"{value:.2f}"
    def get(case, key):
        return by_case.get(case, {}).get(key)
    rows = []
    for r in results:
        row = {k: r[k] for k in ["case", "backend", "encode", "width", "height", "content_seconds",
               "expected_frames", "output_frames", "inference_count", "av_match_misses", "pipeline_seconds",
               "pipeline_fps", "job_seconds", "job_fps", "model_init_ms", "first_output_ms", "complete"]}
        for name, stats in r["stages"].items():
            if "cache" in name:
                row[name + "_count"] = stats["count"]
            else:
                for key in ("count", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
                    row[name + "_" + key] = stats[key]
        rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 当前源码离线 CPU / Vulkan 吞吐测试", "", "测试日期：2026-10-27。", "",
        f"执行状态：**{status['state']}**；已完成 {len(results)}/12 个用例。", "",
        "输入：测试1/2/3.png + Melody from Afar.mp3。音频预先统一为 16 kHz 单声道 PCM；完整内容约 182 秒。该一次性转换不计入单次任务耗时，单次任务的 load_audio_ms 是读取归一化 WAV 的时间；视频使用该归一化音轨。",
        "硬件：Ryzen 7 7840HS（8 核/16 线程）、RTX 4050 Laptop GPU；Windows 原生 Release 构建。",
        "CPU 推理固定 2 线程，OpenCV 4 线程；Vulkan 为实际验证启用的 GPU 后端。各用例顺序运行，没有并行压测。", "",
        "## 吞吐和视频", "",
        "纯流水线：取前 30 秒，不编码、不写 JPEG。视频任务：完整音频，H.264 libx264 veryfast / CRF20 / 2线程 + AAC。每个条件运行一次，不能据此推断跨次运行方差。", "",
        "| 底图 | CPU 纯流水线 FPS | GPU 纯流水线 FPS | GPU/CPU | CPU 视频任务 FPS | GPU 视频任务 FPS | 视频 |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for i in range(1, 4):
        c, g = get(f"face{i}_cpu_core", "pipeline_fps"), get(f"face{i}_gpu_core", "pipeline_fps")
        links = []
        for backend in ("cpu", "gpu"):
            name = f"face{i}_{backend}_video"
            if name in by_case:
                links.append(f"[{backend.upper()}]({name}/lipsync.mp4)")
        lines.append(f"| 测试{i} | {fmt(c)} | {fmt(g)} | {fmt(g/c if g and c else None)}× | "
                     f"{fmt(get(f'face{i}_cpu_video', 'pipeline_fps'))} | "
                     f"{fmt(get(f'face{i}_gpu_video', 'pipeline_fps'))} | {' / '.join(links)} |")
    lines += ["", "FPS = 实际生成帧数 / Pipeline Start 到输出排空和 Stop 的墙钟时长。视频任务 FPS 包含编码管道写入/背压，但不包含编码器最终关闭等待；下表总任务时间包含加载、模型预热和编码关闭。", "",
        "## 完整视频任务的时间与完整性", "",
        "| 用例 | 生成/预期帧 | 核心及编码写入秒 | 总任务秒 | 总任务 FPS | 模型初始化 ms | 首帧 ms | 匹配丢帧 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        if not r["encode"]: continue
        lines.append(f"| {r['case']} | {r['output_frames']}/{r['expected_frames']} | {r['pipeline_seconds']:.2f} | "
                     f"{r['job_seconds']:.2f} | {r['job_fps']:.2f} | {r['model_init_ms']:.1f} | "
                     f"{r['first_output_ms']:.1f} | {r['av_match_misses']} |")
    stage_names = {"audio_features": "音频特征（每个 12.5ms hop）", "input_video_clone": "视频入口 clone",
        "video_preprocess": "人脸预处理（含首帧及缓存命中）", "matcher_assemble": "Mel 窗口装配",
        "mel_to_tensor": "音频张量转换", "face_to_tensor": "人脸张量转换", "model_inference": "Wav2Lip 推理",
        "render_fusion": "逆变换和融合渲染", "encode_pipe_write": "编码管道写入（含背压）"}
    lines += ["", "## 每阶段耗时", "",
        "以下以完整视频任务为准。平均值/p95/p99 单位均为毫秒。阶段并行且调用次数不同，不能把各阶段均值直接相加当作端到端延迟。编码写入是管道提交和等待时间，不是编码器内部 kernel 时间。"]
    for r in results:
        if not r["encode"]: continue
        lines += ["", f"### {r['case']}", "", "| 阶段 | 次数 | 平均 ms | p95 ms | p99 ms | 最大 ms |",
                  "|---|---:|---:|---:|---:|---:|"]
        for stage, label in stage_names.items():
            s = r["stages"].get(stage)
            if not s: continue
            lines.append(f"| {label} | {s['count']} | {s['mean_ms']:.4f} | {s['p95_ms']:.4f} | "
                         f"{s['p99_ms']:.4f} | {s['max_ms']:.4f} |")
        cold = ", ".join(f"{label} {r['stages'][stage]['mean_ms']:.3f}ms"
            for stage, label in [("video_detect", "检测"), ("video_landmarks", "关键点"),
                                 ("video_align", "对齐"), ("video_mask", "遮罩")]
            if stage in r["stages"])
        hit = r["stages"].get("video_cache_hit", {}).get("count", 0)
        miss = r["stages"].get("video_cache_miss", {}).get("count", 0)
        tensor_hit = r["stages"].get("face_tensor_cache_hit", {}).get("count", 0)
        tensor_miss = r["stages"].get("face_tensor_cache_miss", {}).get("count", 0)
        lines += ["", f"首帧冷计算：{cold}。", "",
                  f"人脸缓存命中/未命中：{hit}/{miss}；张量缓存命中/未命中：{tensor_hit}/{tensor_miss}。"]
    lines += ["", "## 发现的问题与测量边界", "",
        "1. 短测已经复现尾部 4 帧匹配失败。当前 Matcher 在音频 EOS 时仍把窗口起点钳到最后一个完整 Mel 窗口，末尾视频帧与该起点的误差超过 40ms 后被跳过。没有修改此算法、补帧或把重复帧算成吞吐；各用例的 complete=false 必须如实保留。最终 MP4 使用 -shortest，可能因此比输入短约 0.16 秒。",
        "2. 视频预处理原来没有写 cost_ms，导致旧平均指标为 0；本次补齐计时。其余源码改动为可选阶段采样和测试入口，没有调整核心预处理、缓存、推理或融合算法。",
        "3. 当前 InitModelInferencer 默认先初始化 Vulkan；CPU 用例随后显式切回 CPU。模型初始化单列，其中可能包含不必要的 GPU 初始化；纯流水线 FPS 排除这段初始化，不能把该启动耗时视为纯 CPU 部署的下限。",
        "4. offline_ingress_to_output 包含离线快速喂入时的反压和队列积压，不等于实时播放延迟。matcher_audio_fill_wait 含等待上游音频，不属于纯匹配计算。",
        "5. 这是静态底图缓存命中场景，不覆盖动态底板失效和长时间网络推流。音乐包含伴奏；嘴部产生变化不等于已验证音素级同步质量。",
        "6. 奇数宽高通过编码端补一列/一行黑边满足 YUV420P；SDK 图像仍按原始尺寸处理。没有逐帧 JPEG 落盘。",
        "7. 模型是本次从 Hugging Face 下载并通过 pnnx 转换的版本，未声称与历史性能权重字节相同。", "",
        "8. 对 1/60/120/180 秒做了抽帧解码验证，并目视检查 60 秒画面。音视频轨、尺寸、帧数和抽帧可读性正常；底图3嘴部放大后明显偏软，不能把吞吐达标等同于口型画质验收通过。CPU/GPU 的底图3同时间抽帧外观接近。", "",
        "## 原始证据", "",
        "- [完整机器与输入清单](manifest.json)",
        "- [硬件和驱动](hardware.json)",
        "- [运行状态及全部结构化结果](suite_status.json)",
        "- [视频与阶段计数验证](validation.json)",
        "- [汇总 CSV](summary.csv)",
        "- 每个用例目录的 report.json、stage_samples.csv 和 output_timeline.csv；根目录同名 .log。",
        "- [当前源代码差异](source_changes.patch)；新增文件及源码内容哈希见 manifest.json。", ""]
    lines += ["- [源码与复现脚本快照](source_snapshot.zip)", ""]
    lines += ["- [CPU/GPU 视频对照页](videos.html)", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    cards = []
    for i in range(1, 4):
        players = []
        for backend in ("cpu", "gpu"):
            name = f"face{i}_{backend}_video"
            if name not in by_case: continue
            r = by_case[name]
            players.append(f'<article><h3>{backend.upper()} · {r["pipeline_fps"]:.2f} FPS</h3>'
                f'<p>含加载及编码收尾：{r["job_seconds"]:.2f} 秒</p>'
                f'<video controls preload="none" poster="{name}/frame_060.png" src="{name}/lipsync.mp4"></video>'
                f'<p><a href="{name}/lipsync.mp4">打开视频</a> · <a href="{name}/report.json">原始报告</a></p></article>')
        cards.append(f'<section><h2>测试{i}</h2><button onclick="playPair(this)">从头同步播放这一组</button>'
                     '<div class="pair">' + ''.join(players) + '</div></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>离线口型视频 · CPU / GPU</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;background:#111827;color:#edf2f7}
a{color:#7dd3fc}h1{line-height:1.3}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{background:#1f2937;padding:18px;border-radius:12px}
video{width:100%;height:500px;background:#070b12;object-fit:contain}section{margin:36px 0}button{padding:8px 16px;border:0;border-radius:6px;cursor:pointer}
.note{padding:14px 20px;background:#422e15;border-radius:8px}@media(max-width:650px){.pair{grid-template-columns:1fr}video{height:430px}}</style>
<h1>静态底图 + Melody from Afar</h1><p>2026-10-27 · 当前工作区 Release · CPU 2线程 / RTX 4050 Vulkan</p>
<p>显示的 FPS 是离线生成吞吐（包含编码管道写入），播放帧率统一为 25 FPS。</p>
<p class="note">当前匹配器每段末尾少生成4帧，输出4544/4548帧；没有补帧，帧完整性检查未通过。底图3嘴部有放大模糊。</p>
<p><a href="REPORT.md">逐阶段报告</a> · <a href="summary.csv">CSV</a> · <a href="validation.json">验证记录</a></p>'''
    page += ''.join(cards)
    page += '''<script>function playPair(b){const vs=b.closest('section').querySelectorAll('video');vs.forEach(v=>{v.currentTime=0;v.play().catch(()=>{});});}</script></html>'''
    (root / "videos.html").write_text(page, encoding="utf-8")
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
