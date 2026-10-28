#!/usr/bin/env python3
"""Minimal regression checks for grasp-to-transfer boundary classification."""

import json
import tempfile
from pathlib import Path

from analyze_action_chunks import analyze_episode


def action(dx, gripper):
    return [dx, 0, 0, 0, 0, 0, gripper]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def run_case(transfer_action_index):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "episode_summary.json").write_text(json.dumps({"task_id": 4, "initial_state_index": 32}), encoding="utf-8")
        write_jsonl(
            root / "chunks.jsonl",
            [
                {"chunk_index": 0, "processed_actions": [action(0, -1)] * 7 + [action(0, 1)], "open_loop_steps": 8},
                {"chunk_index": 1, "processed_actions": [action(0.03 if i == transfer_action_index else 0, 1) for i in range(8)], "open_loop_steps": 8},
            ],
        )
        write_jsonl(
            root / "steps.jsonl",
            [
                {"chunk_index": 0, "action_index_in_chunk": index} for index in range(8)
            ] + [{"chunk_index": 1, "action_index_in_chunk": index} for index in range(8)],
        )
        write_jsonl(root / "events.jsonl", [])
        return analyze_episode(root / "episode_summary.json", 0.02)[0]


if __name__ == "__main__":
    at_boundary = run_case(0)
    assert at_boundary["first_close_chunk"] == 0
    assert at_boundary["first_close_action_index"] == 7
    assert at_boundary["first_transfer_chunk"] == 1
    assert at_boundary["first_transfer_action_index"] == 0
    assert at_boundary["boundary_check_before_transfer"]

    inside_chunk = run_case(3)
    assert not inside_chunk["boundary_check_before_transfer"]
    print("grasp-to-transfer boundary classification: OK")
