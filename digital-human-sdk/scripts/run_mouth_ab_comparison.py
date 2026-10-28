"""Three controlled GPU mouth-fusion variants, balanced repeated timings and videos."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    exe, root = map(Path, sys.argv[1:3])
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PATH"] = r"D:\MSYC64\ucrt64\bin;D:\digital-human-dev\ncnn\ucrt64-vulkan\bin;" + env.get("PATH", "")
    def save(name, value):
        (root / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    runtime_sources = list((repo / "src").rglob("*.cpp")) + list((repo / "include").rglob("*.h"))
    runtime_sources += [repo / "examples/offline_pipeline_benchmark.cpp"]
    save("manifest.json", {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_hashes": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_sources},
        "executable": str(exe), "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "input_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "inputs").iterdir()},
        "source_audio_seconds": [60, 90], "target_fps": 25,
        "variants": {"baseline": {"blend_weight": 0.7, "feather": "11x11 Gaussian"},
                     "direct": {"blend_weight": 1.0, "feather": "11x11 Gaussian"},
                     "outer-feather": {"blend_weight": 1.0, "feather": "opaque dilated mouth core + 3px outward smoothstep"}},
        "unchanged": ["input face", "audio", "model weights", "alignment", "RGB layout", "sharpener", "image interpolation", "encoder settings", "AV matching"],
        "gpu": "RTX 4050 Laptop Vulkan", "opencv_threads": 4,
        "timing": "Three no-encode runs per variant in balanced order; one encoded run per variant. PNG writing happens after measured work."
    })
    state = {"state": "running", "results": []}
    variants = ["baseline", "direct", "outer-feather"]
    plan = [(variants[(j+r)%3], f"core_r{r+1}", False) for r in range(3) for j in range(3)]
    plan += [(v, "video", True) for v in variants]
    for variant, suffix, encode in plan:
        name = variant + "_" + suffix
        state["current"] = name
        save("suite_status.json", state)
        cmd = [str(exe), str(root / "inputs/face3.png"), str(root / "inputs/audio_60_90.wav"),
               str(root / name), "gpu", "0", "2", str(int(encode)), variant]
        print("RUN", name, flush=True)
        started = time.monotonic()
        with (root / (name + ".log")).open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        path = root / name / "report.json"
        if proc.returncode not in (0, 2) or not path.exists():
            state.update(state="failed", error=f"{name}: exit {proc.returncode}")
            save("suite_status.json", state)
            raise RuntimeError(state["error"])
        result = json.loads(path.read_text())
        assert result["gpu_enabled"] is True
        state["results"].append({"case": name, "process_seconds": time.monotonic()-started, **result})
        save("suite_status.json", state)
        print(f"DONE {name}: {result['pipeline_fps']:.2f} FPS, render {result['stages']['render_fusion']['mean_ms']:.3f}ms, "
              f"{result['output_frames']}/{result['expected_frames']} frames", flush=True)
    state.update(state="completed", current=None)
    save("suite_status.json", state)


if __name__ == "__main__":
    main()
