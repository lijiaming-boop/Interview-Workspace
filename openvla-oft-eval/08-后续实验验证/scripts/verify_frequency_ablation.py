#!/usr/bin/env python3
"""Verify the paired 20 Hz vs 50 Hz LIBERO frequency ablation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TARGETS = {(4, 32), (6, 20), (7, 27)}
OPEN_LOOP = 4


def config_of(root: Path, expected_freq: int) -> dict:
    path = root / "run_config.txt"
    if not path.exists():
        raise SystemExit(f"缺少配置文件: {path}")
    config = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    expected = {
        "task_suite": "libero_spatial",
        "seed": "7",
        "targets": "4:32,6:20,7:27",
        "open_loop": str(OPEN_LOOP),
        "freq": str(expected_freq),
    }
    errors = [
        f"{key}={config.get(key)!r}（应为 {value!r}）"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if errors:
        raise SystemExit(f"{root}: run_config 校验失败: " + "; ".join(errors))
    return config


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def load_run(root: Path, expected_freq: int) -> dict:
    config = config_of(root, expected_freq)
    summary_paths = sorted((root / "actions").rglob("episode_summary.json"))
    rows = {}
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        key = (summary.get("task_id"), summary.get("initial_state_index"))
        if key in rows:
            raise SystemExit(f"{root}: 重复状态 {key}")
        if summary.get("control_freq") != expected_freq:
            raise SystemExit(f"{summary_path}: control_freq 不为 {expected_freq}")
        if summary.get("num_open_loop_steps") != OPEN_LOOP:
            raise SystemExit(f"{summary_path}: open-loop 不为 {OPEN_LOOP}")
        chunk_path = summary_path.parent / "chunks.jsonl"
        latencies = [
            float(json.loads(line)["inference_latency_ms"])
            for line in chunk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not latencies:
            raise SystemExit(f"{chunk_path}: 没有 inference_latency_ms 记录")
        budget_ms = OPEN_LOOP / expected_freq * 1000.0
        rows[key] = {
            "success": bool(summary.get("success", False)),
            "chunks": len(latencies),
            "deadline_rate": (
                sum(latency <= budget_ms for latency in latencies) / len(latencies)
                if latencies
                else 0.0
            ),
            "mean_ms": sum(latencies) / len(latencies) if latencies else None,
            "p95_ms": p95(latencies),
            "budget_ms": budget_ms,
        }
    if set(rows) != TARGETS:
        raise SystemExit(f"{root}: 目标状态为 {sorted(rows)}，应为 {sorted(TARGETS)}")
    return {"config": config, "rows": rows}


def print_run(label: str, run: dict) -> None:
    rows = run["rows"]
    chunks = sum(row["chunks"] for row in rows.values())
    success = sum(row["success"] for row in rows.values())
    deadline = sum(row["deadline_rate"] * row["chunks"] for row in rows.values()) / chunks
    mean = sum(row["mean_ms"] * row["chunks"] for row in rows.values()) / chunks
    p95_max = max(row["p95_ms"] for row in rows.values() if row["p95_ms"] is not None)
    print(f"{label}: success={success}/{len(rows)} ({success / len(rows):.1%}), "
          f"deadline={deadline:.1%}, mean={mean:.1f} ms, p95={p95_max:.1f} ms")
    for task_id, state in sorted(TARGETS):
        row = rows[(task_id, state)]
        print(f"  task {task_id}/state {state}: success={row['success']}, "
              f"deadline={row['deadline_rate']:.1%}, p95={row['p95_ms']:.1f} ms, "
              f"budget={row['budget_ms']:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq20-root", type=Path, required=True)
    parser.add_argument("--freq50-root", type=Path, required=True)
    parser.add_argument("--min-deadline-rate", type=float, default=0.95)
    args = parser.parse_args()

    freq20 = load_run(args.freq20_root, 20)
    freq50 = load_run(args.freq50_root, 50)
    pair_keys = ("task_suite", "seed", "targets", "open_loop", "wrist", "motion_threshold", "ckpt")
    mismatches = [
        key for key in pair_keys if freq20["config"].get(key) != freq50["config"].get(key)
    ]
    if mismatches:
        raise SystemExit(f"两组实验配置不一致: {', '.join(mismatches)}")
    print("频率配对验收（固定 task/state、seed=7、open-loop=4）")
    print_run("20 Hz", freq20)
    print_run("50 Hz", freq50)
    success20 = sum(row["success"] for row in freq20["rows"].values())
    success50 = sum(row["success"] for row in freq50["rows"].values())
    chunks50 = sum(row["chunks"] for row in freq50["rows"].values())
    deadline50 = sum(row["deadline_rate"] * row["chunks"] for row in freq50["rows"].values()) / chunks50
    p95_50 = max(row["p95_ms"] for row in freq50["rows"].values() if row["p95_ms"] is not None)
    print(f"success delta (50-20 Hz): {success50 - success20:+d}/3")
    print(f"50 Hz deadline threshold: {args.min_deadline_rate:.1%}; observed: {deadline50:.1%}")

    if deadline50 < args.min_deadline_rate or p95_50 > 80.0:
        raise SystemExit("验收失败：50 Hz 未满足 80 ms 实时性要求，不能宣称高频控制有效")
    if success50 < success20:
        raise SystemExit("验收失败：50 Hz 虽实时，但固定状态成功数低于 20 Hz，疑似控制分布偏移")
    print("结论：50 Hz 满足实时性且未低于 20 Hz 成功数，可进入扩大样本复验。")


if __name__ == "__main__":
    main()
