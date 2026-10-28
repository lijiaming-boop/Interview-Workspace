#!/usr/bin/env python3
"""Summarize chunk boundaries, gripper transitions, target motion, and inference deadlines."""

import argparse
import csv
import json
import math
from pathlib import Path


def read_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def target_position(chunk):
    objects = chunk.get("sim_snapshot_before_chunk", {}).get("objects", {})
    for name, state in objects.items():
        if "bowl" in name.lower():
            return name, state.get("position")
    return None, None


def distance(left, right):
    if left is None or right is None or len(left) != len(right):
        return None
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_episode(summary_path: Path, translation_threshold: float):
    episode_dir = summary_path.parent
    episode = json.loads(summary_path.read_text(encoding="utf-8"))
    chunks = read_jsonl(episode_dir / "chunks.jsonl")
    steps = read_jsonl(episode_dir / "steps.jsonl")
    events = read_jsonl(episode_dir / "events.jsonl")

    executed_pairs = {(step["chunk_index"], step["action_index_in_chunk"]) for step in steps}
    chunk_rows = []
    previous_target_position = None
    previous_executed_gripper = -1.0
    first_close = None
    first_transfer = None
    first_close_with_motion = None
    discarded_close_candidates = 0

    for chunk in chunks:
        chunk_idx = int(chunk["chunk_index"])
        processed_actions = chunk.get("processed_actions", [])
        open_loop_steps = int(chunk.get("open_loop_steps", len(processed_actions)))
        target_name, current_target_position = target_position(chunk)
        target_motion = distance(previous_target_position, current_target_position)
        previous_target_position = current_target_position

        local_previous_gripper = previous_executed_gripper
        first_close_idx = None
        first_transfer_idx = None
        for action_idx, action in enumerate(processed_actions):
            gripper = float(action[-1])
            is_executed = (chunk_idx, action_idx) in executed_pairs
            if local_previous_gripper < 0 <= gripper and first_close_idx is None:
                first_close_idx = action_idx
                if is_executed and first_close is None:
                    first_close = (chunk_idx, action_idx)
                elif not is_executed:
                    discarded_close_candidates += 1
            if is_executed:
                translation_norm = math.sqrt(sum(float(value) ** 2 for value in action[:3]))
                # A close command can itself contain the final approach.  Do not
                # call that transfer: only a later closed motion is a post-grasp
                # transfer candidate.
                is_close = local_previous_gripper < 0 <= gripper
                if is_close and translation_norm >= translation_threshold and first_close_with_motion is None:
                    first_close_with_motion = (chunk_idx, action_idx)
                elif first_close is not None and gripper >= 0 and translation_norm >= translation_threshold:
                    if first_transfer_idx is None:
                        first_transfer_idx = action_idx
                    if first_transfer is None:
                        first_transfer = (chunk_idx, action_idx)
                local_previous_gripper = gripper
        previous_executed_gripper = local_previous_gripper

        chunk_rows.append(
            {
                "episode_dir": str(episode_dir),
                "task_id": episode.get("task_id"),
                "initial_state_index": episode.get("initial_state_index"),
                "chunk_index": chunk_idx,
                "control_step": chunk.get("control_step"),
                "control_freq": chunk.get("control_freq"),
                "open_loop_steps": open_loop_steps,
                "predicted_actions": len(processed_actions),
                "actually_executed_actions": sum(1 for pair in executed_pairs if pair[0] == chunk_idx),
                "first_close_action_index": first_close_idx,
                "close_at_chunk_boundary": first_close_idx == 0,
                "close_inside_executed_prefix": first_close_idx is not None and first_close_idx < open_loop_steps,
                "first_post_close_transfer_action_index": first_transfer_idx,
                "post_close_transfer_at_chunk_boundary": first_transfer_idx == 0,
                "target_name": target_name,
                "target_motion_since_previous_chunk_m": target_motion,
                "contact_pairs_at_boundary": len(chunk.get("sim_snapshot_before_chunk", {}).get("contacts", [])),
                "inference_latency_ms": chunk.get("inference_latency_ms"),
                "chunk_deadline_ms": chunk.get("chunk_deadline_ms"),
                "chunk_deadline_met": chunk.get("chunk_deadline_met"),
            }
        )

    latencies = [float(row["inference_latency_ms"]) for row in chunk_rows if row["inference_latency_ms"] is not None]
    deadline_values = [bool(row["chunk_deadline_met"]) for row in chunk_rows if row["chunk_deadline_met"] is not None]
    target_motions = [
        float(row["target_motion_since_previous_chunk_m"])
        for row in chunk_rows
        if row["target_motion_since_previous_chunk_m"] is not None
    ]
    episode_row = {
        "episode_dir": str(episode_dir),
        "task_id": episode.get("task_id"),
        "initial_state_index": episode.get("initial_state_index"),
        "task_description": episode.get("task_description"),
        "success": episode.get("success"),
        "control_freq": episode.get("control_freq"),
        "open_loop_steps": episode.get("num_open_loop_steps"),
        "chunks": len(chunks),
        "executed_steps": len(steps),
        "forced_replans": episode.get("forced_replan_count", 0),
        "grasp_checks": episode.get("grasp_check_count", 0),
        "empty_grasp_replans": episode.get("empty_grasp_replan_count", 0),
        "grasp_retry_exhausted": episode.get("grasp_retry_exhausted", False),
        "first_close_chunk": first_close[0] if first_close else None,
        "first_close_action_index": first_close[1] if first_close else None,
        "first_close_at_boundary": bool(first_close and first_close[1] == 0),
        "first_close_with_motion_chunk": first_close_with_motion[0] if first_close_with_motion else None,
        "first_close_with_motion_action_index": first_close_with_motion[1] if first_close_with_motion else None,
        "first_transfer_chunk": first_transfer[0] if first_transfer else None,
        "first_transfer_action_index": first_transfer[1] if first_transfer else None,
        "first_transfer_at_boundary": bool(first_transfer and first_transfer[1] == 0),
        "grasp_to_transfer_same_chunk": bool(
            first_close and first_transfer and first_close[0] == first_transfer[0]
        ),
        "boundary_check_before_transfer": bool(
            first_close
            and first_transfer
            and first_transfer[0] > first_close[0]
            and first_transfer[1] == 0
        ),
        "discarded_close_candidates": discarded_close_candidates,
        "max_target_motion_between_chunks_m": max(target_motions) if target_motions else None,
        "mean_inference_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "deadline_met_rate": sum(deadline_values) / len(deadline_values) if deadline_values else None,
        "motion_replan_events": sum(1 for event in events if event.get("event") == "target_motion_forced_replan"),
    }
    return episode_row, chunk_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Experiment directory containing task_*/state_*")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--translation-threshold", type=float, default=0.02)
    args = parser.parse_args()

    summary_paths = sorted(args.root.rglob("episode_summary.json"))
    if not summary_paths:
        raise SystemExit(f"No episode_summary.json found under {args.root}")

    episode_rows, chunk_rows = [], []
    for summary_path in summary_paths:
        episode_row, per_chunk_rows = analyze_episode(summary_path, args.translation_threshold)
        episode_rows.append(episode_row)
        chunk_rows.extend(per_chunk_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episode_summary.csv", episode_rows)
    write_csv(args.output_dir / "chunk_summary.csv", chunk_rows)

    successful = sum(str(row["success"]).lower() == "true" for row in episode_rows)
    report_lines = [
        "# Action-chunk diagnostic summary",
        "",
        f"- Episodes: {len(episode_rows)}",
        f"- Successes: {successful}",
        f"- Failures: {len(episode_rows) - successful}",
        "",
        "| Task | State | Hz | Open-loop | Success | First close | First post-close transfer | Boundary check before transfer | Deadline met | Grasp checks | Empty-grasp replans | Retry exhausted | Forced replans |",
        "|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---|---:|",
    ]
    for row in episode_rows:
        close = f"c{row['first_close_chunk']}/a{row['first_close_action_index']}"
        transfer = f"c{row['first_transfer_chunk']}/a{row['first_transfer_action_index']}"
        deadline_rate = row["deadline_met_rate"]
        report_lines.append(
            f"| {row['task_id']} | {row['initial_state_index']} | {row['control_freq']} | "
            f"{row['open_loop_steps']} | {row['success']} | {close} | {transfer} | "
            f"{row['boundary_check_before_transfer']} | {deadline_rate:.1%} | "
            f"{row['grasp_checks']} | {row['empty_grasp_replans']} | "
            f"{row['grasp_retry_exhausted']} | {row['forced_replans']} |"
            if deadline_rate is not None
            else f"| {row['task_id']} | {row['initial_state_index']} | {row['control_freq']} | "
            f"{row['open_loop_steps']} | {row['success']} | {close} | {transfer} | "
            f"{row['boundary_check_before_transfer']} | N/A | {row['grasp_checks']} | "
            f"{row['empty_grasp_replans']} | "
            f"{row['grasp_retry_exhausted']} | {row['forced_replans']} |"
        )
    report_lines.extend(
        [
            "",
            "> `First post-close transfer` excludes motion bundled with the close command; both labels remain action-log heuristics, not ground-truth grasp labels.",
            "> `Boundary check before transfer=True` means the first post-close transfer is action 0 of a later chunk, so an end-of-chunk grasp check can stop it. Otherwise interception at the close action (or a shorter open-loop prefix) is required.",
            "> A 50 Hz claim is supported only when `deadline_met_rate` is near 100% on the target GPU.",
        ]
    )
    (args.output_dir / "analysis.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'episode_summary.csv'}")
    print(f"Wrote {args.output_dir / 'chunk_summary.csv'}")
    print(f"Wrote {args.output_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()
