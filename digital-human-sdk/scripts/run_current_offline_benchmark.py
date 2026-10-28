"""Run the current offline benchmark sequentially to avoid CPU/GPU contention.

Inputs: REPORT_ROOT/inputs/{face1.png,face2.png,face3.png,audio16k.wav}.
Each case retains report.json, raw stage samples, stdout/stderr and optionally MP4.
Exit code 2 from the benchmark means a completed run failed frame-completeness checks;
it is retained as evidence rather than silently discarded or counted as a pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("report_root", type=Path)
    args = parser.parse_args()
    root = args.report_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PATH"] = (r"D:\MSYC64\ucrt64\bin;D:\digital-human-dev\ncnn\ucrt64-vulkan\bin;"
                   + env.get("PATH", ""))
    sources = list((repo / "src").rglob("*.cpp")) + list((repo / "include").rglob("*.h"))
    sources += [repo / "examples/offline_pipeline_benchmark.cpp", repo / "examples/CMakeLists.txt"]
    hashes = {str(p.relative_to(repo)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sources}
    manifest = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_root": str(repo), "source_hashes": hashes,
        "executable": str(args.executable.resolve()),
        "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        "input_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (root / "inputs").iterdir() if p.is_file()},
        "model_source": json.loads((repo / "models/wav2lip-source.json").read_text()),
        "cpu_threads": 2, "opencv_threads": 4,
        "core_seconds": 30, "video_seconds": "complete audio",
        "runs_per_case": 1,
        "notes": ["Offline, no frame pacing; model/face loading reported separately.",
                  "Core runs discard output; video runs pipe frames to FFmpeg libx264 veryfast CRF20 threads=2.",
                  "Stage durations overlap across threads; audio features are per 12.5ms hop, others per call.",
                  "No algorithm/cache/AV-match changes; instrumentation is opt-in except fixing missing video cost_ms.",
                  "Frame completeness, PTS continuity, and AV matcher misses are explicit checks."]
    }
    save(root / "manifest.json", manifest)
    (root / "source_changes.patch").write_bytes(subprocess.check_output(
        ["git", "diff", "--no-ext-diff", "--", "src", "include", "examples", "CMakeLists.txt"], cwd=repo))
    status = {"state": "running", "results": [], "started_unix": time.time()}
    cases = [(image, backend, mode) for image in range(1, 4)
             for backend in ("cpu", "gpu") for mode in ("core", "video")]
    for image, backend, mode in cases:
        name = f"face{image}_{backend}_{mode}"
        status["current"] = name
        save(root / "suite_status.json", status)
        out_dir = root / name
        cmd = [str(args.executable), str(root / f"inputs/face{image}.png"),
               str(root / "inputs/audio16k.wav"), str(out_dir), backend,
               "30" if mode == "core" else "0", "2", "1" if mode == "video" else "0"]
        print("RUN", name, flush=True)
        start = time.monotonic()
        with (root / f"{name}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=repo, timeout=2400)
        report_path = out_dir / "report.json"
        if proc.returncode not in (0, 2) or not report_path.exists():
            status.update(state="failed", error=f"{name}: exit {proc.returncode}")
            save(root / "suite_status.json", status)
            raise RuntimeError(status["error"])
        report = json.loads(report_path.read_text())
        result = {"case": name, "image_index": image, "returncode": proc.returncode,
                  "process_seconds": time.monotonic() - start, **report}
        status["results"].append(result)
        save(root / "suite_status.json", status)
        print(f"DONE {name}: {report['pipeline_fps']:.2f} FPS, "
              f"{report['output_frames']}/{report['expected_frames']} frames, "
              f"job={report['job_seconds']:.2f}s", flush=True)
    status.update(state="completed", current=None, finished_unix=time.time())
    save(root / "suite_status.json", status)
    print("SUITE COMPLETED", flush=True)


if __name__ == "__main__":
    main()
