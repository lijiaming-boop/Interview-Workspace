#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回退样本分析器 —— 找出「BF16 成功 → INT8 失败」的样本，并判定它们**是什么类型**。

为什么需要它
------------
完整 500 episodes 配对实验给出的是**聚合数字**（+0.4 pp、wins/regressions = 2/4）。
聚合数字回答"差了多少"，但回答不了"**差在哪一类样本上**" ——
而后者才决定改进方向：是动作数值精度敏感（夹爪阈值事件）？是视觉特征退化？
还是纯噪声边缘样本？

本脚本**只做归因的第一步：分类**。它不预设结论，而是把回退样本的
可观测特征全部摊开，让人**先看清是什么类型的样本，再决定后续规划**。

字段来源（权威，已核对上游源码，不是猜的）
------------------------------------------
`episode_summary.json` 由 `experiments/robot/libero/run_libero_eval.py:742-763` 写出，
真实字段（**注意：没有 num_steps / total_reward / failure_mode**）：

    task_id, initial_state_index, task_description, success, control_freq,
    num_open_loop_steps, target_motion_replan_threshold_m, verify_grasp,
    grasp_check_count, empty_grasp_replan_count, grasp_retry_count,
    grasp_retry_exhausted, replan_count, forced_replan_count,
    executed_control_steps, simulated_duration_seconds

三个 jsonl（同目录，`dump_action_chunks=True` 时才有）：
    chunks.jsonl  每次 replan：chunk_index/control_step/inference_latency_ms/
                  chunk_deadline_met/raw_actions/processed_actions/
                  raw_gripper_commands/processed_gripper_commands/proprio/
                  sim_snapshot_before_chunk
    steps.jsonl   实际执行的每一步：chunk_index/action_index_in_chunk/
                  processed_action/simulation_intervention/intervention_phase
    events.jsonl  干预事件：grasp_recovery_opened / target_motion_forced_replan

**关键概念**：本实验开着 `verify_grasp` 干预。**`executed_control_steps`（实际执行步）
≠ `replan_count × num_open_loop_steps`** —— 干预会丢弃队列里的动作。所以
"步数"必须用 `executed_control_steps`，而不是拿 chunk 数去乘。

设计原则
--------
1. **不猜字段名。** 先扫描真实 schema 并从候选别名取值；取不到标 `N/A`，不编数字。
2. **只读。** 不修改任何实验产物。
3. **离线可测。** `--self-test` 造合成产物验证全链路，不需要实例。

用法
----
    python analyze_regressions.py --run-dir /mnt/workspace/openvla-oft/experiments/failure_study_runs/<RUN_TAG>
    python analyze_regressions.py --run-dir <RUN_TAG> --json     # 额外导出机器可读结果
    python analyze_regressions.py --self-test                    # 离线自检

产出（默认 `<RUN_TAG>/regression_analysis/`）
--------------------------------------------
    regression_pasteblock.md   可直接粘回对话的**完整分析块**（主产出）
    regressions.csv            逐条明细，可直接贴报告 §9
    regressions.md             分类汇总 + 每类判读建议
    schema_report.txt          实际读到的字段与别名命中情况（排查用）
    regressions.json           --json 时额外输出
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------- 字段别名表
# 首选名 = 上游 run_libero_eval.py 实际写出的名字（已核对源码）；
# 后面挂历史/变体名做兼容。取不到就 N/A。
ALIASES = {
    "success":      ["success", "is_success", "task_success", "succeeded"],
    "task_id":      ["task_id", "task_idx", "task_index"],
    "state_idx":    ["initial_state_index", "init_state_index", "state_index",
                     "initial_state_idx"],
    "task_name":    ["task_description", "task_name", "task_language",
                     "instruction", "language"],
    # 真实字段：实际执行的控制步（不含 wait_steps）—— 不是"规划步"
    "steps":        ["executed_control_steps", "num_steps", "total_steps",
                     "episode_length"],
    "replan_count": ["replan_count", "num_replans", "policy_queries"],
    "forced_replan": ["forced_replan_count"],
    "duration":     ["simulated_duration_seconds", "elapsed_time", "duration"],
    "grasp_checks": ["grasp_check_count"],
    "empty_grasp":  ["empty_grasp_replan_count"],
    "retry_count":  ["grasp_retry_count"],
    "retry_exhausted": ["grasp_retry_exhausted"],
    "open_loop":    ["num_open_loop_steps"],
    "control_freq": ["control_freq"],
    "verify_grasp": ["verify_grasp"],
    "motion_thresh": ["target_motion_replan_threshold_m"],
}

GROUPS = {"bf16": "bf16", "int8": "int8-mixed"}

# 夹爪开合阈值：processed action 的末维是二值/近二值（负=闭合方向，非负=张开方向）
CLOSE_EPS = 0.0
# 「明显平移」阈值（与上游 analyze_action_chunks.py 的默认值保持一致）
MOTION_EPS = 0.02


def pick(d: dict, key: str):
    """按别名表取值；取不到返回 None（而不是猜）。"""
    for name in ALIASES.get(key, [key]):
        if name in d and d[name] is not None:
            return d[name]
    return None


def as_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "success", "succeeded"):
        return True
    if s in ("false", "0", "no", "n", "fail", "failure", "failed"):
        return False
    return None


def as_int(v):
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def as_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


# ---------------------------------------------------------------- 索引构建
def load_group(actions_dir: Path):
    """读一个组的全部 episode_summary.json → (records, schema)。"""
    records: dict = {}
    schema: dict = defaultdict(set)
    if not actions_dir.is_dir():
        return records, schema

    for p in sorted(actions_dir.rglob("episode_summary.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                schema[k].add(str(v)[:40])

        tid = as_int(pick(d, "task_id"))
        sid = as_int(pick(d, "state_idx"))
        records[(tid, sid)] = {"summary": d, "path": p, "ep_dir": p.parent}
    return records, schema


# ---------------------------------------------------------------- 动作块分析
def episode_chunk_features(ep_dir: Path) -> dict:
    """从 chunks/steps/events.jsonl 提取**用于分类**的可观测特征。

    全部字段名均已对照上游源码。文件缺失 → 返回附 'missing' 标记的空 dict。
    """
    out: dict = {}
    chunks = read_jsonl(ep_dir / "chunks.jsonl")
    steps = read_jsonl(ep_dir / "steps.jsonl")
    events = read_jsonl(ep_dir / "events.jsonl")
    if not chunks:
        out["missing"] = True
        return out

    out["n_chunks"] = len(chunks)
    out["n_executed_steps"] = len(steps)

    # ---- 1) 夹爪事件：首次「闭」、首次「闭后再明显平移」 ----
    # steps.jsonl 是真正执行的序列，用它判定夹爪转变（与上游一致）
    first_close = None            # (chunk_idx, action_idx)
    first_transfer = None         # 闭之后再出现明显平移
    gripper_seq = []
    prev = -1.0
    prev_closed = False
    for st in steps:
        act = st.get("processed_action")
        if not act:
            continue
        g = as_float(act[-1])
        if g is None:
            continue
        gripper_seq.append(g)
        if prev < CLOSE_EPS <= g and first_close is None:
            first_close = (as_int(st.get("chunk_index")),
                           as_int(st.get("action_index_in_chunk")))
            prev_closed = True
        elif prev_closed and g >= CLOSE_EPS:
            tnorm = math.sqrt(sum(float(x) ** 2 for x in act[:3]))
            if tnorm >= MOTION_EPS and first_transfer is None:
                first_transfer = (as_int(st.get("chunk_index")),
                                  as_int(st.get("action_index_in_chunk")))
        prev = g

    out["first_close"] = first_close
    out["first_transfer"] = first_transfer
    out["grasp_reached"] = first_close is not None
    # 闭后一直没有平移 → 抓到了但搬不动（或没抓到）
    out["closed_but_no_transfer"] = bool(first_close and not first_transfer)
    # 全程未出现闭合命令 → 压根没走"抓"这个动作
    out["never_closed"] = first_close is None
    out["gripper_switches"] = sum(
        1 for a, b in zip(gripper_seq, gripper_seq[1:]) if (a < 0) != (b < 0))

    # ---- 2) 干预介入程度（这是本实验的关键产能消耗项） ----
    out["intervention_steps"] = sum(
        1 for st in steps if st.get("simulation_intervention"))
    out["intervention_rate"] = (out["intervention_steps"] / len(steps)) if steps else None
    out["events"] = Counter(
        str(e.get("event")) for e in events if isinstance(e, dict) and e.get("event"))

    # ---- 3) deadline / 延迟（顺带给出，用于排除"INT8 是超时输的"） ----
    lats = [as_float(c.get("inference_latency_ms")) for c in chunks]
    lats = [x for x in lats if x is not None]
    if lats:
        out["latency_mean_ms"] = statistics.mean(lats)
        out["latency_max_ms"] = max(lats)
    dl = [c.get("chunk_deadline_met") for c in chunks]
    dl = [as_bool(x) for x in dl if x is not None]
    if dl:
        out["deadline_met_rate"] = sum(1 for x in dl if x) / len(dl)

    # ---- 4) 动作数值幅度（量化误差最直观的体现） ----
    raws = []
    for c in chunks:
        ra = c.get("raw_actions")
        if isinstance(ra, list):
            for row in ra:
                if isinstance(row, list) and len(row) >= 3:
                    raws.append(row)
    if raws:
        trans = [math.sqrt(sum(float(x) ** 2 for x in row[:3])) for row in raws]
        out["raw_trans_mean"] = statistics.mean(trans)
        out["raw_trans_p95"] = sorted(trans)[int(0.95 * (len(trans) - 1))]
        out["n_raw_actions"] = len(raws)
    return out


# ---------------------------------------------------------------- 分类
def classify(rec: dict) -> str:
    """给单个回退样本打**一个**主类型标签。

    规则按「先具体后宽泛」排列，命中即返回。
    **每一类都必须有可观测依据**，不能凭印象贴标签。
    """
    f = rec.get("feat", {}) or {}
    b_steps, i_steps = rec.get("bf16_steps"), rec.get("int8_steps")

    # ① 根本没有动作块数据 → 只能靠汇总字段
    if f.get("missing"):
        if b_steps is not None and i_steps is not None and b_steps > 0:
            if i_steps <= max(3, b_steps * 0.35):
                return "极早失败（步数远少于 BF16，疑早期发散）"
            if i_steps / b_steps >= 1.5:
                return "反复尝试后失败（步数显著多于 BF16）"
        return "待判（无 chunks.jsonl，信息不足）"

    # ② 干预耗尽 / 抓取重试打满 → 不是"动作精度"问题，是"抓取干预"问题
    if f.get("events", {}).get("grasp_recovery_opened", 0) > 0:
        if rec.get("int8_retry_exhausted"):
            return "抓取干预重试打满（grasp_retry_exhausted）"
    if f.get("closed_but_no_transfer"):
        return "抓到了但没搬动（闭爪后无平移）"
    if f.get("never_closed"):
        return "全程未闭合夹爪（未进入抓取阶段）"

    # ③ 步数形态（用 executed_control_steps，不是 chunk 数乘出来的）
    if b_steps and i_steps is not None:
        if i_steps <= max(3, b_steps * 0.35):
            return "极早失败（步数远少于 BF16，疑早期发散）"
        if i_steps / b_steps >= 1.5:
            return "反复尝试后失败（步数显著多于 BF16）"
        if i_steps == b_steps:
            # 同长度失败 → 差异在动作质量，是最"干净"的量化误差信号
            if f.get("first_transfer"):
                return "同长度失败且抓取搬运完成（差异在收尾判定）"
            return "同长度失败（步数一致，差异在动作质量）"

    # ④ 干预占比很高 → 轨迹被外部干预主导
    ir = f.get("intervention_rate")
    if ir is not None and ir >= 0.3:
        return f"干预主导（{ir:.0%} 的步被仿真干预接管）"

    return "待判（需人工看 chunks 动作块偏差）"


def classification_basis(kind: str) -> str:
    """给出该分类的**判读含义**（写进 pasteblock，帮用户决定下一步）。"""
    table = {
        "极早失败": "前几步就发散 → 看第 1~2 个 chunk 的动作是否与 BF16 明显不同；"
                    "指向**早期动作数值误差**，而非感知问题。",
        "反复尝试后失败": "INT8 在反复补齐 → 单步动作质量下降但没崩；"
                          "指向**动作精度退化**，可用动作块偏差量化。",
        "同长度失败": "两组走了一样多的步，只是结果不同 → 最干净的量化误差信号。",
        "抓取干预重试打满": "`grasp_retry_exhausted=True` → 与夹爪阈值事件的敏感度有关，"
                            "属**动作数值精度**问题。",
        "抓到了但没搬动": "夹爪闭合并成功，但之后没有明显平移 → 可能抓空/滑脱。",
        "全程未闭合夹爪": "未进入抓取阶段 → 高层决策或视觉特征出错，"
                          "**须优先怀疑视觉编码器**（它也在量化范围内，见 01 §5.1）。",
        "干预主导": "大部分步被仿真干预接管 → 统计上更难归因，先看干预事件类型。",
        "待判": "信息不足，需人工读 chunks.jsonl 的动作块偏差（见 analyze_action_chunks.py）。",
    }
    for k, v in table.items():
        if kind.startswith(k):
            return v
    return table["待判"]


# ---------------------------------------------------------------- schema 报告
def render_schema(groups_schema: dict) -> str:
    lines = ["=" * 68,
             "episode_summary.json 实际 schema（字段 → 观测到的取值样例）",
             "=" * 68, ""]
    all_keys = sorted(set().union(*[set(s) for s in groups_schema.values()])
                      if groups_schema else set())
    if not all_keys:
        lines.append("⚠️ 未读到任何字段 —— 检查 --run-dir 与 actions 目录是否存在。")
    for k in all_keys:
        vals = set()
        for s in groups_schema.values():
            vals |= s.get(k, set())
        if not vals:
            continue
        shown = sorted(vals)[:6]
        more = f" …(+{len(vals) - len(shown)})" if len(vals) > len(shown) else ""
        lines.append(f"  {k:<32} {len(vals):>5} 种取值   例: {', '.join(shown)}{more}")

    lines += ["", "-" * 68, "别名解析结果（能否按别名表取到值）", "-" * 68]
    for key, names in ALIASES.items():
        hits = [n for n in names if any(n in s for s in groups_schema.values())]
        mark = "✅" if hits else "❌"
        lines.append(f"  {mark} {key:<16} 命中: "
                     f"{', '.join(hits) if hits else '无（该指标将标 N/A）'}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 主流程
def analyze(run_dir: Path, out_dir: Path, want_json: bool) -> int:
    roots = {k: run_dir / sub for k, sub in GROUPS.items()}
    bf16, schema_b = load_group(roots["bf16"] / "actions")
    int8, schema_i = load_group(roots["int8"] / "actions")

    out_dir.mkdir(parents=True, exist_ok=True)
    schema_txt = render_schema({"bf16": schema_b, "int8": schema_i})
    (out_dir / "schema_report.txt").write_text(schema_txt, encoding="utf-8")

    print(schema_txt)

    if not bf16 or not int8:
        print("❌ 无法构建配对集合。")
        print(f"   bf16 读到 {len(bf16)} 个 episode_summary.json")
        print(f"   int8 读到 {len(int8)} 个")
        print(f"   检查路径: {roots['bf16'] / 'actions'}")
        print(f"             {roots['int8'] / 'actions'}")
        return 1

    common = sorted(set(bf16) & set(int8))
    regressions, wins, both_ok, both_fail = [], [], 0, 0
    for k in common:
        b = as_bool(pick(bf16[k]["summary"], "success"))
        i = as_bool(pick(int8[k]["summary"], "success"))
        if b is None or i is None:
            continue
        if b and not i:
            regressions.append(k)
        elif not b and i:
            wins.append(k)
        elif b and i:
            both_ok += 1
        else:
            both_fail += 1

    print("=" * 68)
    print("回退样本分析（BF16 成功 → INT8 失败）")
    print("=" * 68)
    print(f"run_dir      : {run_dir}")
    print(f"配对样本数   : {len(common)}")
    print(f"  两组都成功 : {both_ok}")
    print(f"  两组都失败 : {both_fail}")
    print(f"  INT8 反超  : {len(wins)}")
    print(f"  ⚠️ 回退     : {len(regressions)}")
    print()

    if not regressions:
        msg = "# 回退样本分析\n\n本次配对实验未发现「BF16 成功 → INT8 失败」的样本。\n"
        (out_dir / "regressions.md").write_text(msg, encoding="utf-8")
        (out_dir / "regression_pasteblock.md").write_text(msg, encoding="utf-8")
        print("没有回退样本 —— 无需归因。")
        return 0

    # ---- 组装逐条明细 ----
    rows = []
    for k in regressions:
        bs, is_ = bf16[k]["summary"], int8[k]["summary"]
        rec = {
            "key": k,
            "bf16_steps": as_int(pick(bs, "steps")),
            "int8_steps": as_int(pick(is_, "steps")),
            "bf16_replans": as_int(pick(bs, "replan_count")),
            "int8_replans": as_int(pick(is_, "replan_count")),
            "int8_forced": as_int(pick(is_, "forced_replan")) or 0,
            "int8_grasp_checks": as_int(pick(is_, "grasp_checks")) or 0,
            "int8_empty_grasp": as_int(pick(is_, "empty_grasp")) or 0,
            "int8_retry_count": as_int(pick(is_, "retry_count")) or 0,
            "int8_retry_exhausted": as_bool(pick(is_, "retry_exhausted")) or False,
            "int8_verify_grasp": as_bool(pick(is_, "verify_grasp")),
            "bf16_dur": as_float(pick(bs, "duration")),
            "int8_dur": as_float(pick(is_, "duration")),
            "open_loop": as_int(pick(is_, "open_loop")),
            "control_freq": as_int(pick(is_, "control_freq")),
            "task_name": str(pick(bs, "task_name") or pick(is_, "task_name") or ""),
        }
        rec["feat"] = episode_chunk_features(int8[k]["ep_dir"])
        rec["kind"] = classify(rec)
        rows.append(rec)

    # ---- 逐条明细 ----
    print("── 逐条明细 " + "─" * 50)
    print(f"{'task':>4} {'state':>5}  {'执行步(bf16→int8)':>18}  {'任务名':<30} 类型")
    print("-" * 108)
    for r in rows:
        tid, sid = r["key"]
        st = (f"{r['bf16_steps']}→{r['int8_steps']}"
              if r["bf16_steps"] is not None and r["int8_steps"] is not None else "N/A")
        tn = r["task_name"]
        tn = (tn[:28] + "..") if len(tn) > 30 else (tn or "—")
        print(f"{tid:>4} {sid:>5}  {st:>18}  {tn:<30} {r['kind']}")
    print()

    # ---- 分类汇总 ----
    kinds = Counter(r["kind"] for r in rows)
    print("── 类型分布 " + "─" * 50)
    for kind, n in kinds.most_common():
        print(f"  {n:>2} × {kind}")
    print()

    # ---- 任务集中度 ----
    task_counts = Counter(r["key"][0] for r in rows)
    print("── 任务集中度 " + "─" * 46)
    for tid, n in task_counts.most_common():
        names = {r["task_name"] for r in rows if r["key"][0] == tid and r["task_name"]}
        print(f"  task {tid:<3} {n} 例   {next(iter(names)) if names else ''}")
    concentrated = len(task_counts) == 1
    if concentrated:
        print("  ⚠️ 全部回退集中在同一任务 → 指向任务类型特异性，而非普遍退化。")
    else:
        print(f"  分散在 {len(task_counts)} 个任务上 → 不指向单一任务类型。")
    print()

    # ---- 夹爪/干预特征汇总（本次分析的核心新维度）----
    have_feat = [r for r in rows if not r["feat"].get("missing")]
    if have_feat:
        print("── 夹爪 / 干预特征 " + "─" * 42)
        print(f"  {'task':>4} {'state':>5}  {'闭合?':>5} {'搬动?':>5} "
              f"{'干预步占比':>10} {'deadline达标':>12} {'延迟均值':>9}")
        print("  " + "-" * 70)
        for r in have_feat:
            f = r["feat"]
            tid, sid = r["key"]
            dl = f.get("deadline_met_rate")
            lm = f.get("latency_mean_ms")
            ir = f.get("intervention_rate")
            ir_s = f"{ir:.0%}" if ir is not None else "N/A"
            dl_s = f"{dl:.0%}" if dl is not None else "N/A"
            lm_s = f"{lm:.0f}ms" if lm is not None else "N/A"
            print(f"  {tid:>4} {sid:>5}  "
                  f"{'是' if f.get('grasp_reached') else '否':>5} "
                  f"{'是' if f.get('first_transfer') else '否':>5} "
                  f"{ir_s:>10} {dl_s:>12} {lm_s:>9}")
        print()
        # 关键判读：INT8 是不是"超时输的"？
        dls = [r["feat"].get("deadline_met_rate") for r in have_feat]
        dls = [x for x in dls if x is not None]
        if dls and max(dls) > 0:
            print("  ⚠️ 有回退样本的 chunk 达标了 deadline —— 说明不是纯粹超时输的，"
                  "动作质量本身也参与了。")
        elif dls:
            print("  ℹ️ 这些回退样本在仿真里全部未达 deadline（仿真无实时约束，正常）。")
        print()

    # ---- 步数统计 ----
    step_pairs = [(r["bf16_steps"], r["int8_steps"]) for r in rows
                  if r["bf16_steps"] is not None and r["int8_steps"] is not None]
    if step_pairs:
        b_all = [b for b, _ in step_pairs]
        i_all = [i for _, i in step_pairs]
        print("── 执行步对比 " + "─" * 48)
        print(f"  BF16 均值 {statistics.mean(b_all):.1f}  |  "
              f"INT8 均值 {statistics.mean(i_all):.1f}")
        ratios = [i / b for b, i in step_pairs if b]
        if ratios:
            print(f"  比值(INT8/BF16) 均值 {statistics.mean(ratios):.2f}")
        print()

    # ---- CSV ----
    csv_path = out_dir / "regressions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task_id", "state_idx", "task_name",
                    "bf16_success", "int8_success",
                    "bf16_exec_steps", "int8_exec_steps", "步数比",
                    "int8_replans", "int8_forced_replans",
                    "grasp_checks", "empty_grasp_replans",
                    "retry_count", "retry_exhausted",
                    "夹爪闭合", "闭后搬动", "干预步占比",
                    "deadline达标率", "延迟均值ms", "分类"])
        for r in rows:
            tid, sid = r["key"]
            b, i = r["bf16_steps"], r["int8_steps"]
            f = r["feat"]
            w.writerow([
                tid, sid, r["task_name"], "True", "False",
                b if b is not None else "", i if i is not None else "",
                f"{i / b:.2f}" if (b and i is not None) else "",
                r["int8_replans"] if r["int8_replans"] is not None else "",
                r["int8_forced"], r["int8_grasp_checks"], r["int8_empty_grasp"],
                r["int8_retry_count"], r["int8_retry_exhausted"],
                f.get("grasp_reached", ""), bool(f.get("first_transfer")),
                f"{f['intervention_rate']:.3f}" if f.get("intervention_rate") is not None else "",
                f"{f['deadline_met_rate']:.3f}" if f.get("deadline_met_rate") is not None else "",
                f"{f['latency_mean_ms']:.1f}" if f.get("latency_mean_ms") is not None else "",
                r["kind"],
            ])

    # ---- Markdown 汇总 ----
    md = [
        "# 回退样本分析（BF16 成功 → INT8 失败）",
        "",
        f"- 结果目录：`{run_dir}`",
        f"- 配对样本：{len(common)}（都成功 {both_ok} / 都失败 {both_fail} "
        f"/ INT8 反超 {len(wins)} / **回退 {len(regressions)}**）",
        "",
        "## 逐条明细",
        "",
        "| task_id | state_idx | 任务名 | BF16 执行步 | INT8 执行步 | 步数比 | 分类 |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in rows:
        tid, sid = r["key"]
        b, i = r["bf16_steps"], r["int8_steps"]
        ratio = f"{i / b:.2f}" if (b and i is not None) else "—"
        md.append(f"| {tid} | {sid} | {r['task_name'] or '—'} | "
                  f"{b if b is not None else '—'} | {i if i is not None else '—'} | "
                  f"{ratio} | {r['kind']} |")

    md += ["", "## 类型分布", ""]
    for kind, n in kinds.most_common():
        md.append(f"- **{n} 例** — {kind}")
        md.append(f"  - 判读：{classification_basis(kind)}")

    md += ["", "## 任务集中度", ""]
    for tid, n in task_counts.most_common():
        names = {r["task_name"] for r in rows if r["key"][0] == tid and r["task_name"]}
        md.append(f"- `task {tid}`：{n} 例 — "
                  f"{next(iter(names)) if names else '（任务名未记录）'}")
    md.append("")
    if concentrated:
        md.append("> ⚠️ **全部回退集中在同一任务** → 指向该任务的任务类型特异性，"
                  "而非量化导致的普遍性能力退化。")
    else:
        md.append(f"> 回退分散在 {len(task_counts)} 个不同任务上 → 不指向单一任务类型。")

    md += ["", "---", "", "## 后续规划的自查问题", "",
           "分类结果用来回答下面 4 个问题；**在回答完之前不要急着调参**：", "",
           "1. **集中还是分散？** 集中在 1 个任务 → 看该任务的物理属性"
           "（是否涉及精细抓取 / 旋转 / 夹爪阈值事件）。",
           "2. **步数变长还是变短？** 变长 = 反复试错（动作质量下降）；"
           "变短 = 早期发散（前几步就错）。两者修法不同。",
           "3. **夹爪走到哪一步了？** 全程未闭合 → 怀疑感知/高层决策，"
           "**须优先怀疑视觉编码器**（它也在量化范围内）；"
           "闭了但没搬动 → 抓取精度问题，更像动作数值误差。",
           "4. **是不是纯粹超时输的？** 若回退样本的动作质量本身没问题、"
           "只是 INT8 慢 → 那是 G2 的已知代价，不算新增能力损失。",
           "",
           "> 本文件由 `analyze_regressions.py` 生成。分类是**基于可观测字段的启发式**，"
           "不是结论 —— 请结合 `chunks.jsonl` 的动作块偏差做最终归因。"]
    (out_dir / "regressions.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- pasteblock：给用户直接粘回对话的块 ----
    paste = build_pasteblock(run_dir, common, both_ok, both_fail, wins,
                            rows, kinds, task_counts, concentrated)
    (out_dir / "regression_pasteblock.md").write_text(paste, encoding="utf-8")

    if want_json:
        payload = {
            "run_dir": str(run_dir),
            "paired": len(common), "both_ok": both_ok, "both_fail": both_fail,
            "wins": [[t, s] for t, s in wins],
            "regressions": [{
                "task_id": r["key"][0], "state_idx": r["key"][1],
                "task_name": r["task_name"], "kind": r["kind"],
                "bf16_steps": r["bf16_steps"], "int8_steps": r["int8_steps"],
                "int8_replans": r["int8_replans"], "int8_forced": r["int8_forced"],
                "grasp_checks": r["int8_grasp_checks"],
                "empty_grasp_replans": r["int8_empty_grasp"],
                "retry_count": r["int8_retry_count"],
                "retry_exhausted": r["int8_retry_exhausted"],
                "feat": {k: (dict(v) if isinstance(v, Counter) else v)
                         for k, v in r["feat"].items()},
            } for r in rows],
            "kind_counts": dict(kinds),
            "task_counts": {str(k): v for k, v in task_counts.items()},
        }
        (out_dir / "regressions.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("── 产出 " + "─" * 54)
    for name in ("regression_pasteblock.md", "regressions.csv",
                 "regressions.md", "schema_report.txt"):
        p = out_dir / name
        print(f"  {name:<28} {p.stat().st_size if p.exists() else 0:>8} B")
    if want_json:
        p = out_dir / "regressions.json"
        print(f"  {'regressions.json':<28} {p.stat().st_size if p.exists() else 0:>8} B")
    print(f"\n  目录: {out_dir}")
    print("\n👉 把 regression_pasteblock.md 的内容整段粘回对话，即可继续规划。")
    return 0


def build_pasteblock(run_dir, common, both_ok, both_fail, wins,
                     rows, kinds, task_counts, concentrated) -> str:
    """生成可直接粘回对话的紧凑分析块（不含长篇判读，只给事实+分类）。"""
    L = []
    L.append("### 回退样本分析结果（analyze_regressions.py 产出）")
    L.append("")
    L.append(f"run_dir: {run_dir}")
    L.append(f"配对 {len(common)}  |  都成功 {both_ok}  |  都失败 {both_fail}  |  "
             f"INT8 反超 {len(wins)}  |  回退 {len(rows)}")
    L.append("")
    L.append("| task_id | state_idx | 任务名 | BF16执行步 | INT8执行步 | 步数比 | "
             "闭合 | 搬动 | 干预步占比 | 分类 |")
    L.append("|---:|---:|---|---:|---:|---:|:--:|:--:|---:|---|")
    for r in rows:
        tid, sid = r["key"]
        b, i = r["bf16_steps"], r["int8_steps"]
        f = r["feat"]
        ratio = f"{i / b:.2f}" if (b and i is not None) else "—"
        ir = f.get("intervention_rate")
        L.append(f"| {tid} | {sid} | {r['task_name'] or '—'} | "
                 f"{b if b is not None else '—'} | {i if i is not None else '—'} | "
                 f"{ratio} | {'是' if f.get('grasp_reached') else '否'} | "
                 f"{'是' if f.get('first_transfer') else '否'} | "
                 f"{f'{ir:.0%}' if ir is not None else '—'} | {r['kind']} |")
    L.append("")
    L.append("**类型分布**")
    for kind, n in kinds.most_common():
        L.append(f"- {n} × {kind}")
    L.append("")
    L.append("**任务集中度**")
    for tid, n in task_counts.most_common():
        names = {r["task_name"] for r in rows if r["key"][0] == tid and r["task_name"]}
        L.append(f"- task {tid}: {n} 例 — {next(iter(names)) if names else '（无任务名）'}")
    L.append(f"- 结论：{'全部集中在同一任务' if concentrated else f'分散在 {len(task_counts)} 个任务'}")
    L.append("")
    L.append("**干预 / 抓取统计**")
    for r in rows:
        tid, sid = r["key"]
        L.append(f"- task{tid}/state{sid}: replans={r['int8_replans']} "
                 f"forced={r['int8_forced']} grasp_checks={r['int8_grasp_checks']} "
                 f"empty_grasp={r['int8_empty_grasp']} "
                 f"retry={r['int8_retry_count']} "
                 f"exhausted={r['int8_retry_exhausted']}")
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 自检
def self_test() -> int:
    """造一套**严格模拟真实 schema** 的合成产物，验证全链路。"""
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="regr_selftest_"))
    run = tmp / "run"
    ok = True

    # 4 个回退，落在 3 个任务上；制造两类形态：早期发散 / 反复尝试
    # ⚠️ 注意 state 索引必须落在下面循环实际生成的范围内（0..3），否则样本不存在
    early_keys = {(0, 1), (0, 2)}          # 极早失败：步数远少于 BF16
    retry_keys = {(3, 1), (4, 0)}          # 反复尝试 + 抓取重试打满
    regress_keys = early_keys | retry_keys
    win_keys = {(1, 1)}

    for group in ("bf16", "int8-mixed"):
        for tid in range(5):
            for sid in range(4):
                key = (tid, sid)
                is_reg = key in regress_keys
                is_win = key in win_keys
                if group == "bf16":
                    succ = not is_win
                    steps, replans, dur = 40, 5, 2.0
                    retry_exhausted = False
                    n_chunks = 5
                else:
                    succ = not is_reg
                    if key in early_keys:
                        steps, replans, dur = 8, 1, 0.4       # 早期发散
                    elif is_reg:
                        steps, replans, dur = 70, 9, 3.5      # 反复尝试
                    else:
                        steps, replans, dur = 40, 5, 2.0
                    retry_exhausted = key in retry_keys
                    n_chunks = replans

                ep = run / group / "actions" / f"task{tid}" / f"ep{sid}"
                ep.mkdir(parents=True, exist_ok=True)

                # --- 真实 schema 的 episode_summary ---
                (ep / "episode_summary.json").write_text(json.dumps({
                    "task_id": tid, "initial_state_index": sid,
                    "task_description": f"pick up the black bowl variant {tid}",
                    "success": succ, "control_freq": 20, "num_open_loop_steps": 8,
                    "target_motion_replan_threshold_m": 0.0,
                    "verify_grasp": True, "grasp_check_count": 2 if is_reg else 1,
                    "empty_grasp_replan_count": 1 if is_reg else 0,
                    "grasp_retry_count": 3 if retry_exhausted else 0,
                    "grasp_retry_exhausted": retry_exhausted,
                    "replan_count": replans, "forced_replan_count": 0,
                    "executed_control_steps": steps,
                    "simulated_duration_seconds": dur,
                }), encoding="utf-8")

                # --- chunks.jsonl（真实字段名） ---
                lines = []
                for c in range(n_chunks):
                    # 回退样本里，让早期发散的那个第 1 个 chunk 完全没闭合命令
                    close_here = not (is_reg and key in early_keys and c == 0)
                    processed = [
                        [0.1, 0.05, -0.02, 0.0, 0.0, 0.0,
                         (1.0 if (i < 4 or not close_here) else -1.0)]
                        for i in range(8)
                    ]
                    lines.append(json.dumps({
                        "chunk_index": c, "control_step": c * 8,
                        "sim_time_seconds": c * 8 / 20,
                        "control_freq": 20,
                        "inference_latency_ms": 555.0,
                        "chunk_deadline_ms": 400.0,
                        "chunk_deadline_met": False,
                        "predicted_shape": [1, 8, 7], "open_loop_steps": 8,
                        "executed_action_indices_planned": list(range(8)),
                        "raw_actions": [
                            [0.12, 0.04, -0.01, 0.0, 0.0, 0.0, 1.0]
                            for _ in range(8)],
                        "processed_actions": processed,
                        "raw_gripper_commands": [1.0] * 8,
                        "processed_gripper_commands": [a[-1] for a in processed],
                        "proprio": [0.0] * 8,
                        "sim_snapshot_before_chunk": {
                            "objects": {"black_bowl_1": {"position": [0.1, 0.2, 0.9]}},
                            "contacts": [],
                        },
                    }, ensure_ascii=False))
                (ep / "chunks.jsonl").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")

                # --- steps.jsonl（真实字段名） ---
                s_lines = []
                for s in range(min(steps, n_chunks * 8)):
                    c_idx, a_idx = divmod(s, 8)
                    close_here = not (is_reg and key in early_keys and c_idx == 0)
                    g = 1.0 if (a_idx < 4 or not close_here) else -1.0
                    s_lines.append(json.dumps({
                        "chunk_index": c_idx, "action_index_in_chunk": a_idx,
                        "control_step": s, "sim_time_seconds": s / 20,
                        "processed_action": [0.1, 0.05, -0.02, 0.0, 0.0, 0.0, g],
                        "queue_remaining_before_step": max(0, 8 - a_idx - 1),
                        "simulation_intervention": bool(is_reg and s % 3 == 0),
                        "intervention_phase": "close_hold" if (is_reg and s % 3 == 0) else None,
                    }, ensure_ascii=False))
                (ep / "steps.jsonl").write_text("\n".join(s_lines) + "\n",
                                                encoding="utf-8")

                # --- events.jsonl ---
                e_lines = []
                if retry_exhausted:
                    e_lines.append(json.dumps({
                        "event": "grasp_recovery_opened",
                        "target_name": "black_bowl_1", "control_step": 20,
                        "retry_count": 3,
                    }))
                (ep / "events.jsonl").write_text("\n".join(e_lines) + "\n",
                                                 encoding="utf-8")

    out = tmp / "out"
    rc = analyze(run, out, want_json=True)
    if rc != 0:
        print("[FAIL] analyze() 返回非 0")
        ok = False

    csv_txt = (out / "regressions.csv").read_text(encoding="utf-8-sig")
    n_rows = len([l for l in csv_txt.splitlines()[1:] if l.strip()])
    md = (out / "regressions.md").read_text(encoding="utf-8")
    pb = (out / "regression_pasteblock.md").read_text(encoding="utf-8")
    sj = json.loads((out / "regressions.json").read_text(encoding="utf-8"))

    kinds = sj["kind_counts"]
    checks = [
        ("regressions.csv 生成", (out / "regressions.csv").exists()),
        ("regressions.md 生成", (out / "regressions.md").exists()),
        ("regression_pasteblock.md 生成", (out / "regression_pasteblock.md").exists()),
        ("schema_report.txt 生成", (out / "schema_report.txt").exists()),
        ("regressions.json 生成", (out / "regressions.json").exists()),
        ("CSV 有 4 条回退记录", n_rows == 4),
        ("真实字段名被识别（executed_control_steps）",
         "executed_control_steps" in
         (out / "schema_report.txt").read_text(encoding="utf-8")),
        ("步数列非空（不是 N/A）", "N/A" not in [
            r.split(",")[5] for r in csv_txt.splitlines()[1:] if r.strip()][0]),
        ("分类不是全部『待判』", not all(k.startswith("待判") for k in kinds)),
        ("识别出早期发散类", any("极早失败" in k for k in kinds)),
        ("识别出抓取重试打满类", any("重试打满" in k for k in kinds)),
        ("夹爪闭合被检出", "是" in pb),
        ("干预统计出现在 pasteblock", "grasp_checks" in pb),
        ("MD 含后续规划段", "后续规划" in md),
        ("pasteblock 含类型分布", "类型分布" in pb),
    ]

    print()
    for name, good in checks:
        print(f"  [{'OK' if good else 'FAIL'}] {name}")
        ok &= good

    # 空目录兜底
    empty = tmp / "empty"
    empty.mkdir()
    rc2 = analyze(empty, tmp / "out2", want_json=False)
    good = (rc2 == 1)
    print(f"  [{'OK' if good else 'FAIL'}] 空目录返回 1（不崩）")
    ok &= good

    shutil.rmtree(tmp, ignore_errors=True)
    print()
    print("[OK] 自检全部通过" if ok else "[FAIL] 自检有问题")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="分析「BF16 成功 → INT8 失败」的回退样本并分类（只读）")
    ap.add_argument("--run-dir", type=Path,
                    help="RUN_TAG 目录，需含 bf16/ 与 int8-mixed/")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="产出目录（默认 <run-dir>/regression_analysis）")
    ap.add_argument("--json", action="store_true", help="额外导出 regressions.json")
    ap.add_argument("--self-test", action="store_true", help="离线自检（造合成数据）")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.run_dir:
        ap.print_help()
        print("\n错误：需要 --run-dir（或用 --self-test 离线自检）")
        return 2
    rd = args.run_dir.expanduser().resolve()
    if not rd.is_dir():
        print(f"错误：不是目录 {rd}")
        return 2
    out = args.out_dir or (rd / "regression_analysis")
    return analyze(rd, out, args.json)


if __name__ == "__main__":
    sys.exit(main())
