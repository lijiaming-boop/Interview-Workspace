"""Compare paired LIBERO results by initial_state_index.

The evaluator must print both:
  Starting targeted task_id=4, initial_state_index=N ...
  Success: True|False
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


STATE_RE = re.compile(r"initial_state_index=(\d+)")
EPISODE_RE = re.compile(r"Starting episode\s+(\d+)")
SUCCESS_RE = re.compile(r"Success:\s*(True|False)")


def read_results(path: Path) -> tuple[dict[int, bool], str]:
    results: dict[int, bool] = {}
    current_state: int | None = None
    key_type = "state_index"
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = STATE_RE.search(line)
        if match:
            current_state = int(match.group(1))
        elif current_state is None:
            match = EPISODE_RE.search(line)
            if match:
                # 旧版评估器只记录 1-based episode 编号，没有 state index。
                current_state = int(match.group(1))
                key_type = "episode_order"
        match = SUCCESS_RE.search(line)
        if match and current_state is not None:
            results[current_state] = match.group(1) == "True"
            current_state = None
    if not results:
        raise ValueError(f"{path}: 没有找到 initial_state_index/episode + Success 配对记录")
    return results, key_type


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-log", required=True, type=Path)
    parser.add_argument("--fine-log", required=True, type=Path)
    args = parser.parse_args()

    base, base_key = read_results(args.base_log)
    fine, fine_key = read_results(args.fine_log)
    states = sorted(set(base) & set(fine))
    if not states:
        raise ValueError("基座和 head-only 没有共同的 initial_state_index")

    base_wins = sum(base[s] and not fine[s] for s in states)
    fine_wins = sum(fine[s] and not base[s] for s in states)
    both_success = sum(base[s] and fine[s] for s in states)
    both_fail = sum(not base[s] and not fine[s] for s in states)

    print("state 数量:", len(states))
    print(f"基座成功: {sum(base[s] for s in states)}/{len(states)}")
    print(f"head-only 成功: {sum(fine[s] for s in states)}/{len(states)}")
    print("基座独胜:", base_wins)
    print("head-only 独胜:", fine_wins)
    print("同时成功:", both_success)
    print("同时失败:", both_fail)
    if base_key == "episode_order" or fine_key == "episode_order":
        print("对齐方式: episode 顺序（旧版日志未记录 initial_state_index，不能证明是相同物理初始状态）")
    else:
        print("对齐方式: initial_state_index")
    print()
    if fine_wins > base_wins and sum(fine[s] for s in states) > sum(base[s] for s in states):
        print("结论: head-only 在该配对测试中表现更好，但仍需扩大重复测试确认。")
    elif base_wins > fine_wins or sum(base[s] for s in states) > sum(fine[s] for s in states):
        print("结论: 基座模型表现更好，继续作为权威模型。")
    else:
        print("结论: 两者无实质差异，基座模型继续作为正式基线。")


if __name__ == "__main__":
    main()
