#!/usr/bin/env python3
"""Extract targeted task/state outcomes and generate a DAgger collection manifest."""

import argparse
import csv
import re
from pathlib import Path


START_RE = re.compile(r"Starting targeted task_id=(\d+), initial_state_index=(\d+)")
SUCCESS_RE = re.compile(r"Success:\s*(True|False)")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    current = None
    rows = []
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        start_match = START_RE.search(line)
        if start_match:
            current = {
                "task_id": int(start_match.group(1)),
                "initial_state_index": int(start_match.group(2)),
            }
            continue
        success_match = SUCCESS_RE.search(line)
        if success_match and current is not None:
            rows.append({**current, "success": success_match.group(1) == "True"})
            current = None

    if not rows:
        raise SystemExit(f"No targeted episode outcomes found in {args.log}")

    result_rows = [
        {
            "task_id": row["task_id"],
            "initial_state_index": row["initial_state_index"],
            "success": row["success"],
        }
        for row in rows
    ]
    manifest_rows = [
        {
            "task_id": row["task_id"],
            "initial_state_index": row["initial_state_index"],
            "policy_success": row["success"],
            "needs_expert_correction": not row["success"],
            "failure_category": "TO_FILL",
            "expert_demo_status": "PENDING" if not row["success"] else "NOT_REQUIRED",
            "expert_demo_path": "",
            "notes": "",
        }
        for row in rows
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "targeted_results.csv", result_rows, list(result_rows[0].keys()))
    write_csv(args.output_dir / "dagger_collection_manifest.csv", manifest_rows, list(manifest_rows[0].keys()))

    successes = sum(row["success"] for row in rows)
    print(f"Episodes: {len(rows)}")
    print(f"Successes: {successes}")
    print(f"Failures: {len(rows) - successes}")
    print(f"Success rate: {successes / len(rows):.1%}")
    print(f"Manifest: {args.output_dir / 'dagger_collection_manifest.csv'}")


if __name__ == "__main__":
    main()
