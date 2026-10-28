#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_collect_failure_cases.py —— 扫描评测视频，生成失败 case 清单与人工标注模板

官方评测脚本每个 episode 保存一个视频（libero_utils.save_rollout_video，已核对）：
  $OFT_ROOT/rollouts/<日期>/<时间>--openvla_oft--episode=<序号>--success=<True|False>--task=<任务名>.mp4
文件名自带 成功/失败 与 任务名 —— 直接从文件名解析，无需改动官方代码。

输出（默认 results/failure_cases/）：
  manifest.csv            全部 episode 清单（date_dir/suite/task/episode/success/时长/路径）
  failures_by_task.md     每任务成败统计，按失败数排序（定位系统性弱项）
  annotation_template.csv 失败 case 人工标注模板（预填定位信息，留空标注列）
                          —— 分类体系与填法见 docs/03_failure_analysis.md

用法：
  python scripts/05_collect_failure_cases.py                    # 汇总所有日期
  python scripts/05_collect_failure_cases.py --date-dir 2026_08_27   # 只看某次运行
"""
import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR.parent / "results"
DEFAULT_OFT_ROOT = Path(__import__("os").environ.get("OFT_ROOT", str(Path.home() / "openvla-oft")))

SPLIT = "--openvla_oft--"
RE_REST = re.compile(r"^episode=(\d+)--success=(True|False)--task=(.+)\.mp4$")

ANNOTATION_COLS = [
    "failure_stage",        # reach / grasp / transfer / place / long_horizon / timeout
    "category",             # 感知 / 控制 / 任务理解 / 待定
    "what_happened",        # 客观现象描述（带时间点，如 "第14s 抓空后持续原地空夹"）
    "root_cause_hypothesis",# 根因假设
    "evidence_timestamp",   # 视频中的证据时间点
    "followup_experiment",  # 建议的验证实验
]


def parse_video(path: Path):
    stem = path.name
    if SPLIT not in stem:
        return None
    ts, rest = stem.split(SPLIT, 1)
    m = RE_REST.match(rest)
    if not m:
        return None
    return {"ts": ts, "episode": int(m.group(1)),
            "success": m.group(2) == "True", "task": m.group(3)}


def probe_duration(path: Path):
    """用 ffprobe 读取视频时长（秒）；没有 ffprobe 就留空"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return round(float(out.stdout.strip()), 1)
    except Exception:
        pass
    return None


def norm_date(s: str) -> str:
    """'2026-08-27' 与 '2026_08_27' 统一成 '20260827'，用于对齐 runs.jsonl 与视频目录"""
    return re.sub(r"\D", "", s)[:8]


def suite_by_date(runs_jsonl: Path):
    """date(数字串) -> suite 名。同一天跑过多个 suite 时无法区分，返回空串"""
    m = defaultdict(set)
    if runs_jsonl.exists():
        for line in runs_jsonl.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("smoke"):        # 冒烟测试不计入归属判断
                continue
            d, s = r.get("date"), r.get("suite")
            if d and s:
                m[norm_date(d)].add(s)
    return {k: (v.pop() if len(v) == 1 else "") for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser(description="生成失败 case 清单与标注模板")
    ap.add_argument("--rollouts", type=Path, default=DEFAULT_OFT_ROOT / "rollouts",
                    help="官方评测视频目录")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="results 目录")
    ap.add_argument("--date-dir", default=None,
                    help="只分析指定日期目录（如 2026_08_27），默认全部")
    ap.add_argument("--top", type=int, default=10, help="控制台打印失败最多的前 N 个任务")
    args = ap.parse_args()

    if not args.rollouts.exists():
        print(f"!! 视频目录不存在: {args.rollouts}（先跑 03_run_libero_eval.sh）")
        return 1

    suite_map = suite_by_date(args.results / "runs.jsonl")

    # ---------- 收集 ----------
    rows = []
    date_dirs = sorted(args.rollouts.glob("*"))
    if args.date_dir:
        date_dirs = [d for d in date_dirs if d.name == args.date_dir]
        if not date_dirs:
            print(f"!! 未找到日期目录 {args.date_dir}；现有: "
                  f"{[d.name for d in sorted(args.rollouts.glob('*'))]}")
            return 1
    for vdir in date_dirs:
        if not vdir.is_dir():
            continue
        suite = suite_map.get(norm_date(vdir.name), "")
        for vid in sorted(vdir.glob("*.mp4")):
            info = parse_video(vid)
            if info is None:
                continue
            info.update(date_dir=vdir.name, suite=suite,
                        video=str(vid), size_mb=round(vid.stat().st_size / 1e6, 2),
                        duration_s=probe_duration(vid))
            rows.append(info)
    if not rows:
        print("!! 没有解析到任何评测视频（检查文件名是否为官方 save_rollout_video 格式）")
        return 1

    # ---------- 统计 ----------
    stat = defaultdict(lambda: {"total": 0, "fail": 0, "suite": ""})
    for r in rows:
        key = (r["suite"], r["task"])
        stat[key]["total"] += 1
        stat[key]["fail"] += 0 if r["success"] else 1
        stat[key]["suite"] = r["suite"]

    out_dir = args.results / "failure_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- manifest.csv ----------
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date_dir", "suite", "task", "episode", "success",
                    "duration_s", "size_mb", "video"])
        for r in sorted(rows, key=lambda x: (x["date_dir"], x["task"], x["episode"])):
            w.writerow([r["date_dir"], r["suite"], r["task"], r["episode"],
                        r["success"], r["duration_s"], r["size_mb"], r["video"]])

    # ---------- failures_by_task.md ----------
    total = len(rows)
    fails = [r for r in rows if not r["success"]]
    md = ["# 失败 case 统计（按任务）\n"]
    scope = f"日期目录 {args.date_dir}" if args.date_dir else "全部日期目录"
    md.append(f"- 范围：{scope}")
    md.append(f"- episodes 总数：{total}，失败：{len(fails)} "
              f"（整体成功率 {100 * (total - len(fails)) / total:.1f}%）")
    md.append("- 注意：同一日期目录可能混有当天多次运行（如冒烟+全量），解读时结合 manifest.csv 的 episode 号。\n")
    md.append("| Suite | 任务 | episodes | 失败数 | 失败率 |")
    md.append("|---|---|---|---|---|")
    for (suite, task), s in sorted(stat.items(), key=lambda kv: -kv[1]["fail"]):
        md.append(f"| {suite or '(未知)'} | {task} | {s['total']} | {s['fail']} | "
                  f"{100 * s['fail'] / s['total']:.1f}% |")
    (out_dir / "failures_by_task.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------- annotation_template.csv ----------
    tpl_path = out_dir / "annotation_template.csv"
    with tpl_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date_dir", "suite", "task", "episode", "duration_s", "video"] + ANNOTATION_COLS)
        for r in sorted(fails, key=lambda x: (x["date_dir"], x["task"], x["episode"])):
            w.writerow([r["date_dir"], r["suite"], r["task"], r["episode"],
                        r["duration_s"], r["video"]] + [""] * len(ANNOTATION_COLS))

    # ---------- 控制台摘要 ----------
    print(f"\n===== 失败 case 收集结果 =====")
    print(f"  episodes: {total}   失败: {len(fails)}   成功率: {100 * (total - len(fails)) / total:.1f}%")
    print(f"\n  失败最多的前 {args.top} 个任务：")
    ranked = sorted(stat.items(), key=lambda kv: -kv[1]["fail"])[: args.top]
    for (suite, task), s in ranked:
        if s["fail"] == 0:
            break
        print(f"    [{suite or '?':>8}] {task:<50} {s['fail']}/{s['total']}")
    print(f"\n已生成:")
    print(f"  {manifest_path}")
    print(f"  {out_dir / 'failures_by_task.md'}")
    print(f"  {tpl_path}")
    print(f"\n下一步: 按 docs/03_failure_analysis.md 的工作流，观看失败视频并填写 annotation_template.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
