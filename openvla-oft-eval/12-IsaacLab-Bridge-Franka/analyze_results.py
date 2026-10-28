"""Summarize paired domain-gap runs without inventing missing metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    required = {"n", "successes", "success_rate"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path} missing fields: {sorted(missing)}")
    return data


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = z * ((p * (1 - p) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def make_report(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    before_rate = float(before["success_rate"])
    after_rate = float(after["success_rate"])
    drop_pp = (after_rate - before_rate) * 100.0
    b_lo, b_hi = wilson_interval(int(before["successes"]), int(before["n"]))
    a_lo, a_hi = wilson_interval(int(after["successes"]), int(after["n"]))
    direction = "补救后提升" if drop_pp >= 0 else "补救后仍掉点"
    before_stages = before.get("failure_stage_counts", {})
    after_stages = after.get("failure_stage_counts", {})
    return f"""# Isaac Lab × Bridge 域差距报告

> 该报告由 `analyze_results.py` 从实际 `summary.json` 生成；没有输入的指标不会被估算。

| 条件 | n | 成功 | 成功率 | Wilson 95% CI |
|---|---:|---:|---:|---:|
| 迁移前 / raw | {before['n']} | {before['successes']} | {before_rate:.1%} | [{b_lo:.1%}, {b_hi:.1%}] |
| 迁移后 / aligned | {after['n']} | {after['successes']} | {after_rate:.1%} | [{a_lo:.1%}, {a_hi:.1%}] |

## 失败阶段计数

| 条件 | grasp | place_or_release | success |
|---|---:|---:|---:|
| raw | {before_stages.get('grasp', 0)} | {before_stages.get('place_or_release', 0)} | {before_stages.get('success', 0)} |
| aligned | {after_stages.get('grasp', 0)} | {after_stages.get('place_or_release', 0)} | {after_stages.get('success', 0)} |

## 掉点 / 补救

- 绝对变化：`{drop_pp:+.2f} pp`。
- 判定：**{direction}**。
- 只在相同 seed、checkpoint、episode 初始状态、控制频率和 `open_loop_steps` 下解释该差值。

## 需要结合 episode JSONL 检查的失败阶段

1. `ever_lifted=False`：抓取/闭合阶段失败，优先查相机视角、物体轮廓、动作缩放和夹爪符号。
2. `ever_lifted=True, success=False`：搬运/放置阶段失败，优先查目标位置、IK 坐标系、动作 chunk 长度和释放时机。
3. 同 seed 只有 raw 失败：视觉域差距较强；优先比较 raw RGB、aligned RGB 和模型输入三张图。
4. 两组都失败：先做 controller-only heuristic smoke，再查 action normalization、相机朝向和 IK 限幅。

## 不允许的结论

不能把 `camera_aligned` 或 `bridge_aligned` 的成功率提升直接归因于微调本身；需要保留原始 RGB、随机化参数、推理时延和失败帧，才能把视觉补救与控制补救分开。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = make_report(load_summary(args.before), load_summary(args.after))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
