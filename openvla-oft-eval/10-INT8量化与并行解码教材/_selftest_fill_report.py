#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线自检：造一份合成产物，跑 fill_report.build()，检查模板占位符是否全部被填充。

不依赖实例，可随时本地运行：  python _selftest_fill_report.py
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fill_report  # noqa: E402

PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_fixture(root: Path):
    cfg_tpl = (
        "label={label}\nprecision={label}\nload_in_8bit={l8}\n"
        "task_suite=libero_spatial\nseed=7\ntargets=ALL\nopen_loop=8\n"
        "control_freq=20\nchunk_size=8\ncheckpoint=/mnt/workspace/openvla-oft-ckpts/"
        "openvla-7b-oft-finetuned-libero-spatial\naction_head_dtype=bf16\n"
    )
    for label, l8, lat, succ in (
        ("bf16", "False", 219.52, [True, True, True, False, True]),
        ("int8-mixed", "True", 203.11, [True, True, False, False, True]),
    ):
        d = root / label
        write(d / "run_config.txt", cfg_tpl.format(label=label, l8=l8))
        write(d / "console.log",
              "09/14 [19:59:51] start\n09/14 [20:31:02] done\n")
        with (d / "gpu_memory.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ms", "memory_used_mib", "gpu_util_percent"])
            for i in range(6):
                w.writerow([i, 20000 - i * 10, 88 + i])
        for ep, s in enumerate(succ):
            ed = d / "actions" / f"task_4_ep{ep}"
            ed.mkdir(parents=True, exist_ok=True)
            with (ed / "chunks.jsonl").open("w", encoding="utf-8") as f:
                for c in range(6):
                    f.write(json.dumps({"inference_latency_ms": lat + c}) + "\n")
            write(ed / "episode_summary.json",
                  json.dumps({"task_id": 4, "initial_state_index": ep,
                              "success": s}))
    # summary
    rows = []
    for label, lat, mem, succ in (("bf16", 219.52, 20000, 4),
                                  ("int8-mixed", 203.11, 17432, 3)):
        rows.append({
            "root": f"/x/{label}", "episodes": 5, "success_count": succ,
            "success_rate": succ / 5, "latency_samples_raw": 30,
            "latency_samples_measured": 30, "latency_mean_ms": lat,
            "latency_p50_ms": lat, "latency_p95_ms": lat + 20,
            "policy_query_fps": 1000 / lat, "predicted_action_fps": 8000 / lat,
            "executed_action_capacity_fps": 8000 / lat,
            "memory_min_mib": mem - 50, "memory_peak_mib": mem,
            "memory_delta_mib": 50, "gpu_util_mean_percent": 90.0,
            "gpu_samples": 6, "precision": label,
        })
    sd = root / "summary"
    sd.mkdir(parents=True, exist_ok=True)
    with (sd / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write(sd / "report.md", "# fake summary\n- speedup 1.08x\n")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="fillreport_"))
    run = tmp / "int8-full-SELFTEST"
    try:
        build_fixture(run)
        tpl = HERE / "05-INT8实验结果报告模板.md"
        text = fill_report.build(run, tpl)
        left = sorted(set(PLACEHOLDER_RE.findall(text)))
        if left:
            print("[FAIL] 未被填充的占位符:", left)
            return 1
        # 抽查关键值确实来自 CSV 而非 "—"
        checks = {
            "speedup": "1.081",
            "bf16_lat_mean": "219.52",
            "int8_mem_peak": "17432",
            "bf16_gpu_util": "90.0",
            "bf16_success": "4/5",
            "int8_success": "3/5",
            "loss_pp": "+20.0",
            "wins": "0",
            "regressions": "1",
            "scope_note": "配对子集 N=5",
            "limitation_partial": "本次为部分评测",
        }
        bad = [f"{k}: 期望含 {val!r}" for k, val in checks.items()
               if val not in text]
        # 找一个未填充的 markdown 单元格
        if bad:
            print("[FAIL] 抽查不通过:")
            for b in bad:
                print("   -", b)
            return 1
        print(f"[OK] 占位符全部填充，{len(checks)} 项抽查通过")
        print(f"     生成字符数 {len(text)}，行数 {len(text.splitlines())}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
