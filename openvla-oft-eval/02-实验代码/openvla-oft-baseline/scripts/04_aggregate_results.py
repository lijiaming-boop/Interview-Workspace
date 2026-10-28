#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_aggregate_results.py —— 汇总 LIBERO 评测结果，生成 baseline 成功率表格

数据来源（优先级从高到低）：
  1) results/runs.jsonl               —— 03_run_libero_eval.sh 每次运行自动记录
  2) $OFT_ROOT/experiments/logs/*.txt —— 直接扫描官方评测日志（EVAL-*.txt）

日志格式（官方 run_libero_eval.py 输出，已核对）：
  "Total episodes: 500"
  "Total successes: 486"
  "Overall success rate: 0.9720 (97.2%)"
  "Current task success rate: 0.94"      （其上方应有任务名行，尽力解析）

输出：
  results/baseline_success_rates.md   —— 正式 baseline 表格（每个 suite 取 trials 最多、
                                          最近一次的运行），含论文参考值对比
  results/baseline_success_rates.csv  —— 同内容 CSV
  控制台同步打印表格摘要

用法：
  python scripts/04_aggregate_results.py [--include-smoke]
  python scripts/04_aggregate_results.py --results results
"""
import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR.parent / "results"
DEFAULT_OFT_ROOT = Path(os.environ.get("OFT_ROOT", str(Path.home() / "openvla-oft")))

SHORT2NAME = {"spatial": "libero_spatial", "object": "libero_object",
              "goal": "libero_goal", "10": "libero_10"}
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABEL = {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object",
               "libero_goal": "LIBERO-Goal", "libero_10": "LIBERO-10 (Long)"}
# 论文参考值（seed=7, 50 trials/task, A100, 单 suite checkpoint；均值 97.1 为官方 README 背书）
REF_RATE = {"libero_spatial": 97.2, "libero_object": 97.4,
            "libero_goal": 95.8, "libero_10": 98.0}

RE_OVERALL = re.compile(r"Overall success rate:\s*([0-9.]+)\s*\(([0-9.]+)%\)")
RE_TOTAL_EP = re.compile(r"Total episodes:\s*(\d+)")
RE_TOTAL_SU = re.compile(r"Total successes:\s*(\d+)")
RE_TASK_HDR = re.compile(r"Evaluating task:\s*(.+?)\s*$")
RE_TASK_RATE = re.compile(r"Current task success rate:\s*([0-9.]+)")


def parse_log(path: Path):
    """解析单个官方评测日志，返回 dict 或 None（无整体成功率行时）"""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    run = {"tasks": {}, "episodes": None, "successes": None, "rate": None}
    cur_task = None
    for ln in text.splitlines():
        m = RE_OVERALL.search(ln)
        if m:
            run["rate"] = float(m.group(2))
            continue
        m = RE_TOTAL_EP.search(ln)
        if m:
            run["episodes"] = int(m.group(1))
            continue
        m = RE_TOTAL_SU.search(ln)
        if m:
            run["successes"] = int(m.group(1))
            continue
        m = RE_TASK_RATE.search(ln)
        if m and cur_task:
            r = float(m.group(1))
            run["tasks"][cur_task] = round(r * 100, 1) if r <= 1.0 else round(r, 1)
            cur_task = None
            continue
        m = RE_TASK_HDR.search(ln)
        if m:
            cur_task = m.group(1).strip()
    if run["rate"] is None:
        return None
    return run


def load_runs(results_dir: Path, oft_root: Path, include_smoke: bool):
    runs = []
    jr = results_dir / "runs.jsonl"
    if jr.exists():
        for line in jr.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                meta = json.loads(line)
            except json.JSONDecodeError:
                continue
            if meta.get("smoke") and not include_smoke:
                continue
            log = Path(str(meta.get("log", "")))
            if not log.exists():
                print(f"[warn] 日志不存在，跳过: {log}")
                continue
            parsed = parse_log(log)
            if parsed is None:
                print(f"[warn] 日志缺少整体成功率行，跳过: {log}")
                continue
            parsed.update(
                suite=SHORT2NAME.get(meta.get("suite"), meta.get("suite")),
                ckpt=meta.get("ckpt", "?"),
                trials=meta.get("trials"),
                seed=meta.get("seed"),
                ts=str(meta.get("ts", "")),
                log=str(log),
            )
            runs.append(parsed)
    else:
        print("[info] 未找到 runs.jsonl，改为直接扫描日志目录")
        logs_dir = oft_root / "experiments" / "logs"
        if logs_dir.exists():
            for lg in sorted(logs_dir.glob("EVAL-*.txt")):
                m = re.match(r"EVAL-(libero_[a-z0-9]+)-openvla-", lg.name)
                if not m:
                    continue
                parsed = parse_log(lg)
                if parsed is None:
                    continue
                parsed.update(suite=m.group(1), ckpt="?", trials=None,
                              seed=None, ts=lg.stem, log=str(lg))
                runs.append(parsed)
    return runs


def pick_baseline(runs):
    """每个 suite 取 trials 最多、时间最近的一次运行作为 baseline 行"""
    best = {}
    for r in runs:
        if r.get("suite") not in SUITE_LABEL:
            continue
        rank = (r.get("trials") or 0, r.get("ts", ""))
        key = r["suite"]
        if key not in best or rank > best[key][0]:
            best[key] = (rank, r)
    return [best[k][1] for k in SUITE_ORDER if k in best]


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return "unknown"


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="汇总 LIBERO 评测结果为 baseline 表格")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="results 目录")
    ap.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT, help="openvla-oft 仓库路径")
    ap.add_argument("--include-smoke", action="store_true", help="把冒烟测试(2 trials)也算进候选")
    args = ap.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results, args.oft_root, args.include_smoke)
    if not runs:
        print("!! 没有可用的运行记录——先跑 03_run_libero_eval.sh，或检查 OFT_ROOT/experiments/logs")
        return 1
    baseline = pick_baseline(runs)
    if not baseline:
        print("!! 候选运行里没有四个标准 suite 的记录")
        return 1

    gpu = gpu_name()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------- CSV ----------
    csv_path = args.results / "baseline_success_rates.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["suite", "checkpoint", "trials_per_task", "seed",
                    "episodes", "successes", "success_rate_pct",
                    "paper_ref_pct", "delta_pct", "log_file"])
        for r in baseline:
            ref = REF_RATE.get(r["suite"])
            delta = None if ref is None or r["rate"] is None else round(r["rate"] - ref, 1)
            w.writerow([r["suite"], r.get("ckpt"), r.get("trials"), r.get("seed"),
                        r.get("episodes"), r.get("successes"), r.get("rate"),
                        ref, delta, r.get("log")])

    # ---------- Markdown ----------
    md = []
    md.append("# OpenVLA-OFT LIBERO Baseline 成功率\n")
    md.append(f"- 生成时间：{now}    GPU：{gpu}")
    md.append(f"- 运行环境：官方 openvla-oft 仓库 + 官方 checkpoint（详见 results/runs.jsonl）")
    md.append(f"- 选取规则：每个 suite 取 trials/task 最多、最近一次的运行\n")
    md.append("| Suite | Checkpoint | Trials/task | Seed | Episodes | 成功率 | 论文参考 | 偏差 |")
    md.append("|---|---|---|---|---|---|---|---|")
    rates = []
    for r in baseline:
        ref = REF_RATE.get(r["suite"])
        delta = "-" if ref is None or r["rate"] is None else f"{r['rate'] - ref:+.1f}"
        rates.append(r["rate"])
        md.append(f"| {SUITE_LABEL[r['suite']]} | {r.get('ckpt')} | {r.get('trials')} | "
                  f"{r.get('seed')} | {r.get('episodes')} | **{fmt(r['rate'])}%** | "
                  f"{fmt(ref)}% | {delta} |")
    if rates:
        avg = sum(rates) / len(rates)
        md.append(f"| **平均（{len(rates)} 个 suite）** | | | | | **{avg:.1f}%** | 97.1% | {avg - 97.1:+.1f} |")
    md.append("\n> 论文参考值：seed=7、50 trials/task、A100、py3.10.14 + torch 2.2.0 + transformers fork。")
    md.append("> 官方提示不同 GPU 结果可能有轻微波动；偏差在 ±2% 内视为复现成功。\n")

    # ---------- per-task 明细（尽力解析） ----------
    any_task = any(r["tasks"] for r in baseline)
    if any_task:
        md.append("## 每任务成功率明细（来自日志，供失败分析定位弱项）\n")
        for r in baseline:
            if not r["tasks"]:
                continue
            md.append(f"### {SUITE_LABEL[r['suite']]}（{r.get('ckpt')}）\n")
            md.append("| 任务 | 成功率 |")
            md.append("|---|---|")
            for task, rate in sorted(r["tasks"].items(), key=lambda kv: kv[1]):
                md.append(f"| {task} | {rate}% |")
            md.append("")

    # ---------- 所有运行记录 ----------
    md.append("## 附录：全部运行记录\n")
    md.append("| 时间 | Suite | Checkpoint | Trials | Seed | 成功率 | 日志 |")
    md.append("|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda x: str(x.get("ts"))):
        md.append(f"| {r.get('ts')} | {r.get('suite')} | {r.get('ckpt')} | "
                  f"{r.get('trials')} | {r.get('seed')} | {fmt(r['rate'])}% | `{r.get('log')}` |")

    md_path = args.results / "baseline_success_rates.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------- 控制台摘要 ----------
    print(f"\n===== Baseline 成功率摘要（{gpu}）=====")
    for r in baseline:
        print(f"  {SUITE_LABEL[r['suite']]:<18} {fmt(r['rate'])}%  "
              f"(参考 {fmt(REF_RATE.get(r['suite']))}%)  trials={r.get('trials')} seed={r.get('seed')}")
    if rates:
        print(f"  {'平均':<18} {sum(rates)/len(rates):.1f}%  (参考 97.1%)")
    print(f"\n已生成:\n  {md_path}\n  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
