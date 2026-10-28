"""Check generated artifacts without treating known tail-frame loss as a pass."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_root", type=Path)
    args = parser.parse_args()
    root = args.report_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    status = json.loads((root / "suite_status.json").read_text())
    assert status["state"] == "completed", "Suite is still running"
    env = os.environ.copy()
    env["PATH"] = r"D:\MSYC64\ucrt64\bin;" + env.get("PATH", "")
    checks = []
    for r in status["results"]:
        out = root / r["case"]
        stages = r["stages"]
        for stage in ("model_inference", "render_fusion", "face_to_tensor", "mel_to_tensor", "matcher_assemble"):
            assert stages[stage]["count"] == r["output_frames"], (r["case"], stage)
        assert stages["video_preprocess"]["count"] == r["expected_frames"]
        assert stages["video_cache_miss"]["count"] == 1
        assert stages["face_tensor_cache_miss"]["count"] == 1
        assert r["gpu_enabled"] == (r["backend"] == "gpu")
        timeline = list(csv.DictReader((out / "output_timeline.csv").open()))
        pts_contiguous = all(int(p["pts_ms"]) == i * 40 for i, p in enumerate(timeline))
        assert len(timeline) == r["output_frames"] and pts_contiguous
        assert r["output_frames"] + r["av_match_misses"] == r["expected_frames"]
        check = {"case": r["case"], "stage_counts_valid": True, "backend_verified": True,
                 "generated_pts_contiguous": True, "runtime_eos": r["eos"],
                 "complete": r["complete"], "tail_frames_missing": r["expected_frames"] - r["output_frames"]}
        if r["encode"]:
            video = out / "lipsync.mp4"
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], env=env))
            v = next(s for s in probe["streams"] if s["codec_type"] == "video")
            a = next(s for s in probe["streams"] if s["codec_type"] == "audio")
            assert v["codec_name"] == "h264" and a["codec_name"] == "aac"
            assert int(v["nb_frames"]) == r["output_frames"]
            assert v["avg_frame_rate"] == "25/1"
            assert int(v["width"]) == r["width"] + r["width"] % 2
            assert int(v["height"]) == r["height"] + r["height"] % 2
            (out / "ffprobe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
            for seconds in (1, 60, 120, 180):
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-y", "-ss", str(seconds),
                    "-i", str(video), "-frames:v", "1", "-threads", "1", str(out / f"frame_{seconds:03d}.png")],
                    env=env, check=True, timeout=30)
                assert (out / f"frame_{seconds:03d}.png").exists()
            check.update(video_frames=int(v["nb_frames"]), video_duration_s=float(v["duration"]),
                         audio_duration_s=float(a["duration"]), video_bytes=video.stat().st_size,
                         sampled_frames=[1, 60, 120, 180], video_sha256=hashlib.sha256(video.read_bytes()).hexdigest())
        checks.append(check)
    report = {"runtime_checks_passed": True, "frame_completeness_passed": all(c["complete"] for c in checks),
              "cases": checks, "note": "Frame sampling and container checks, not full-stream decode or phonetic lip-sync acceptance."}
    (root / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with zipfile.ZipFile(root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in ("src", "include", "examples", "scripts", "cmake"):
            for path in (repo / directory).rglob("*"):
                if path.is_file() and path.suffix in (".cpp", ".h", ".py", ".cmake", ".in", ".txt"):
                    archive.write(path, path.relative_to(repo))
        for name in ("CMakeLists.txt", "CMakePresets.json", "CMakeUserPresets.json"):
            archive.write(repo / name, name)
    print(json.dumps({"cases": len(checks), "runtime_checks_passed": True,
                      "frame_completeness_passed": report["frame_completeness_passed"]}))


if __name__ == "__main__":
    main()
