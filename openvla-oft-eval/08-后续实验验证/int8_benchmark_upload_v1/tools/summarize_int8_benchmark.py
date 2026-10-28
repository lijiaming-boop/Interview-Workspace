#!/usr/bin/env python3
"""Summarize paired BF16 vs INT8-mixed OpenVLA-OFT evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_config(root: Path) -> dict[str, str]:
    path = root / "run_config.txt"
    if not path.exists():
        raise SystemExit(f"缺少 {path}")
    config = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config


def read_latencies(root: Path, warmup_chunks: int) -> tuple[list[float], int]:
    records = []
    for path in sorted((root / "actions").rglob("chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("inference_latency_ms") is not None:
                    records.append(float(record["inference_latency_ms"]))
    if len(records) <= warmup_chunks:
        raise SystemExit(f"{root}: latency 样本数 {len(records)} 不足以丢弃 {warmup_chunks} 个 warmup chunk")
    return records[warmup_chunks:], len(records)


def read_successes(root: Path) -> dict[tuple[int, int], bool]:
    results = {}
    for path in sorted((root / "actions").rglob("episode_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        key = (int(summary["task_id"]), int(summary["initial_state_index"]))
        if key in results:
            raise SystemExit(f"{root}: 重复 episode {key}")
        results[key] = bool(summary.get("success", False))
    if not results:
        raise SystemExit(f"{root}: 没有 episode_summary.json")
    return results


def read_memory(root: Path) -> dict[str, float]:
    path = root / "gpu_memory.csv"
    if not path.exists():
        raise SystemExit(f"缺少 {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = [float(row["memory_used_mib"]) for row in rows if row.get("memory_used_mib")]
    utils = [float(row["gpu_util_percent"]) for row in rows if row.get("gpu_util_percent")]
    if not values:
        raise SystemExit(f"{path}: 没有显存采样")
    return {
        "memory_min_mib": min(values),
        "memory_peak_mib": max(values),
        "memory_delta_mib": max(values) - min(values),
        "gpu_util_mean_percent": statistics.fmean(utils) if utils else math.nan,
        "gpu_samples": len(values),
    }


def load_run(root: Path, warmup_chunks: int) -> dict:
    config = read_config(root)
    latencies, raw_latency_count = read_latencies(root, warmup_chunks)
    successes = read_successes(root)
    memory = read_memory(root)
    chunk_size = int(config.get("chunk_size", "8"))
    open_loop = int(config.get("open_loop", "8"))
    latency_mean = statistics.fmean(latencies)
    success_count = sum(successes.values())
    result = {
        "root": str(root),
        "config": config,
        "successes": successes,
        "episodes": len(successes),
        "success_count": success_count,
        "success_rate": success_count / len(successes),
        "latency_samples_raw": raw_latency_count,
        "latency_samples_measured": len(latencies),
        "latency_mean_ms": latency_mean,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "policy_query_fps": 1000.0 / latency_mean,
        "predicted_action_fps": chunk_size * 1000.0 / latency_mean,
        "executed_action_capacity_fps": open_loop * 1000.0 / latency_mean,
        **memory,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-root", type=Path, required=True)
    parser.add_argument("--int8-root", type=Path, required=True)
    parser.add_argument("--warmup-chunks", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    bf16 = load_run(args.bf16_root, args.warmup_chunks)
    int8 = load_run(args.int8_root, args.warmup_chunks)
    pair_keys = ("task_suite", "seed", "targets", "open_loop", "control_freq", "chunk_size", "checkpoint")
    mismatches = [key for key in pair_keys if bf16["config"].get(key) != int8["config"].get(key)]
    if mismatches:
        raise SystemExit("两组配置不一致: " + ", ".join(mismatches))
    if set(bf16["successes"]) != set(int8["successes"]):
        raise SystemExit("两组 task/state 集合不一致")
    if bf16["config"].get("load_in_8bit") != "False":
        raise SystemExit("BF16 组未声明 load_in_8bit=False")
    if int8["config"].get("load_in_8bit") != "True":
        raise SystemExit("INT8 组未声明 load_in_8bit=True")
    if int8["config"].get("action_head_dtype") != "bf16":
        raise SystemExit("INT8 组动作头不是 BF16")

    speedup = bf16["latency_mean_ms"] / int8["latency_mean_ms"]
    memory_reduction = bf16["memory_peak_mib"] - int8["memory_peak_mib"]
    success_loss_pp = (bf16["success_rate"] - int8["success_rate"]) * 100.0
    int8_wins = sum(
        int(not bf16["successes"][key] and int8["successes"][key])
        for key in bf16["successes"]
    )
    int8_regressions = sum(
        int(bf16["successes"][key] and not int8["successes"][key])
        for key in bf16["successes"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, run in (("bf16", bf16), ("int8-mixed", int8)):
        rows.append({key: value for key, value in run.items() if key not in ("config", "successes") } | {"precision": label})
    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = f"""# OpenVLA-OFT BF16 vs INT8-mixed

| 指标 | BF16 | INT8 主干 + BF16 动作头 |
|---|---:|---:|
| Episodes | {bf16['episodes']} | {int8['episodes']} |
| Success | {bf16['success_count']}/{bf16['episodes']} ({bf16['success_rate']:.1%}) | {int8['success_count']}/{int8['episodes']} ({int8['success_rate']:.1%}) |
| Mean latency | {bf16['latency_mean_ms']:.2f} ms | {int8['latency_mean_ms']:.2f} ms |
| P50 latency | {bf16['latency_p50_ms']:.2f} ms | {int8['latency_p50_ms']:.2f} ms |
| P95 latency | {bf16['latency_p95_ms']:.2f} ms | {int8['latency_p95_ms']:.2f} ms |
| Policy query FPS | {bf16['policy_query_fps']:.2f} | {int8['policy_query_fps']:.2f} |
| Predicted action FPS (8/chunk) | {bf16['predicted_action_fps']:.2f} | {int8['predicted_action_fps']:.2f} |
| Peak GPU memory | {bf16['memory_peak_mib']:.0f} MiB | {int8['memory_peak_mib']:.0f} MiB |
| Sampled GPU utilization | {bf16['gpu_util_mean_percent']:.1f}% | {int8['gpu_util_mean_percent']:.1f}% |

- INT8 latency speedup: **{speedup:.3f}x**
- Peak memory reduction: **{memory_reduction:.0f} MiB**
- Success-rate loss: **{success_loss_pp:+.1f} percentage points**
- Paired INT8 wins/regressions: **{int8_wins}/{int8_regressions}**
- Latency excludes the first {args.warmup_chunks} chunks; GPU memory is sampled by `nvidia-smi` and assumes no unrelated GPU process.
"""
    report_path = args.output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
