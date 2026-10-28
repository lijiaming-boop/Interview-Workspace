#!/usr/bin/env python3
"""验收 LIBERO-Spatial 321 的配对位移重定位实验。

新脚本布局为：
  <group-root>/trial-01/actions/task_06/state_20/...

每个 trial 的 control/treatment 使用相同 seed；两组唯一实验变量是
target_motion_replan_threshold_m: 0.0 vs 0.01 m。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_config(path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config


def discover_trials(root: Path) -> dict[str, Path]:
    trials = {
        path.name: path
        for path in sorted(root.glob("trial-*"))
        if path.is_dir()
    }
    if trials:
        return trials
    # 兼容旧版单次输出；在 --min-trials=10 下会被明确判定为数量不足。
    if (root / "actions" / "task_06" / "state_20").is_dir():
        return {"single": root}
    return {}


def load_run(root: Path, expected_threshold: float, trial_id: str) -> dict:
    config_path = root / "run_config.txt"
    episode_dir = root / "actions" / "task_06" / "state_20"
    summary_path = episode_dir / "episode_summary.json"
    events_path = episode_dir / "events.jsonl"
    for required in (config_path, summary_path, events_path):
        if not required.exists():
            raise SystemExit(f"{trial_id}: 缺少结果文件: {required}")

    config = read_config(config_path)
    config_errors = []
    for key, expected in (
        ("task_suite", "libero_spatial"),
        ("targets", "6:20"),
        ("open_loop", "4"),
        ("wrist", "1.0"),
        ("freq", "20"),
    ):
        if config.get(key) != expected:
            config_errors.append(f"{key}={config.get(key)!r}（应为 {expected!r}）")
    if config_errors:
        raise SystemExit(f"{trial_id}: run_config 校验失败: " + "; ".join(config_errors))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = []
    if summary.get("task_id") != 6:
        errors.append(f"task_id={summary.get('task_id')}（应为 6）")
    if summary.get("initial_state_index") != 20:
        errors.append(f"initial_state_index={summary.get('initial_state_index')}（应为 20）")
    if summary.get("num_open_loop_steps") != 4:
        errors.append(f"open_loop={summary.get('num_open_loop_steps')}（应为 4）")
    if summary.get("control_freq") != 20:
        errors.append(f"control_freq={summary.get('control_freq')}（应为 20）")
    actual_threshold = float(summary.get("target_motion_replan_threshold_m", -1.0))
    if abs(actual_threshold - expected_threshold) > 1e-9:
        errors.append(
            f"motion_threshold={actual_threshold:g}（应为 {expected_threshold:g}）"
        )
    if errors:
        raise SystemExit(f"{trial_id}: 结果配置校验失败: " + "; ".join(errors))

    motion_events = [
        event for event in events if event.get("event") == "target_motion_forced_replan"
    ]
    if int(summary.get("forced_replan_count", 0)) != len(motion_events):
        raise SystemExit(
            f"{trial_id}: summary forced_replan_count={summary.get('forced_replan_count')} "
            f"与 events 数量={len(motion_events)} 不一致"
        )

    analysis_path = root / "analysis" / "episode_summary.csv"
    max_chunk_motion = None
    if analysis_path.exists():
        with analysis_path.open(encoding="utf-8-sig", newline="") as analysis_file:
            rows = list(csv.DictReader(analysis_file))
        values = [
            float(row["max_target_motion_between_chunks_m"])
            for row in rows
            if row.get("max_target_motion_between_chunks_m") not in (None, "", "None")
        ]
        max_chunk_motion = max(values) if values else None

    invalid_events = [
        event
        for event in motion_events
        if float(event.get("gripper_command", 0.0)) >= 0
        or float(event.get("target_motion_m", 0.0)) < expected_threshold
        or int(event.get("discarded_queued_actions", 0)) <= 0
    ]
    return {
        "trial_id": trial_id,
        "root": str(root),
        "config": config,
        "success": bool(summary.get("success", False)),
        "forced_replans": int(summary.get("forced_replan_count", 0)),
        "motion_events": motion_events,
        "invalid_events": invalid_events,
        "max_motion_m": max(
            (float(event.get("target_motion_m", 0.0)) for event in motion_events),
            default=0.0,
        ),
        "discarded_actions": sum(
            int(event.get("discarded_queued_actions", 0)) for event in motion_events
        ),
        "max_chunk_motion_m": max_chunk_motion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--treatment-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--min-trials", type=int, default=10)
    parser.add_argument(
        "--require-effect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="要求处理组成功率高于对照组且至少出现 1 次 paired success flip",
    )
    args = parser.parse_args()
    if args.threshold <= 0:
        raise SystemExit("--threshold 必须大于 0")
    if args.min_trials < 10:
        raise SystemExit("--min-trials 必须大于等于 10")

    control_trials = discover_trials(args.control_root)
    treatment_trials = discover_trials(args.treatment_root)
    if len(control_trials) < args.min_trials or len(treatment_trials) < args.min_trials:
        raise SystemExit(
            f"重复次数不足：control={len(control_trials)}, treatment={len(treatment_trials)}，"
            f"要求两组都 >= {args.min_trials}"
        )
    if set(control_trials) != set(treatment_trials):
        raise SystemExit(
            "两组 trial 集合不一致："
            f"control-only={sorted(set(control_trials) - set(treatment_trials))}, "
            f"treatment-only={sorted(set(treatment_trials) - set(control_trials))}"
        )

    results = []
    for trial_id in sorted(control_trials):
        control = load_run(control_trials[trial_id], 0.0, f"control/{trial_id}")
        treatment = load_run(
            treatment_trials[trial_id], args.threshold, f"treatment/{trial_id}"
        )
        pair_keys = ("task_suite", "seed", "targets", "open_loop", "wrist", "freq", "ckpt")
        pair_mismatch = [
            key
            for key in pair_keys
            if control["config"].get(key) != treatment["config"].get(key)
        ]
        if pair_mismatch:
            raise SystemExit(
                f"{trial_id}: 两组 run_config 不一致: {', '.join(pair_mismatch)}"
            )
        results.append((control, treatment))

    control_successes = sum(int(control["success"]) for control, _ in results)
    treatment_successes = sum(int(treatment["success"]) for _, treatment in results)
    control_events = sum(control["forced_replans"] for control, _ in results)
    treatment_events = sum(treatment["forced_replans"] for _, treatment in results)
    invalid_events = sum(len(treatment["invalid_events"]) for _, treatment in results)
    paired_flips = sum(
        int(not control["success"] and treatment["success"])
        for control, treatment in results
    )
    paired_regressions = sum(
        int(control["success"] and not treatment["success"])
        for control, treatment in results
    )
    n = len(results)
    control_rate = control_successes / n
    treatment_rate = treatment_successes / n

    print(f"321 配对验收（task 6, state 20；open-loop=4；共 {n} 对）")
    print(f"control:   {control_successes}/{n} success ({control_rate:.1%}), forced_replans={control_events}")
    print(f"treatment: {treatment_successes}/{n} success ({treatment_rate:.1%}), forced_replans={treatment_events}")
    print(f"paired success flips: {paired_flips}; paired regressions: {paired_regressions}")

    failures = []
    if control_events:
        failures.append(f"对照组阈值为 0，但出现了 {control_events} 个位移重定位事件")
    if treatment_events == 0:
        failures.append(f"处理组没有达到 {args.threshold:g} m 的位移重定位事件")
    if invalid_events:
        failures.append(f"处理组有 {invalid_events} 个事件不满足‘开爪、超过阈值、丢弃队列’条件")
    if args.require_effect and not (treatment_rate > control_rate and paired_flips >= 1):
        failures.append("处理组成功率未高于对照组，或没有出现 control=False → treatment=True 的成功翻转")

    if failures:
        print("验收失败：")
        for failure in failures:
            print(f"- {failure}")
        if treatment_events and not invalid_events and not control_events:
            print("结论：重定位机制已通过触发校验，但未通过任务效果验收。")
        raise SystemExit(1)

    print("验收通过：两组重复数、配对配置、事件语义和任务效果均满足标准。")
    print("结论：位移重定位在 321 上恢复成功；处理组相对对照组出现成功翻转。")


if __name__ == "__main__":
    main()
