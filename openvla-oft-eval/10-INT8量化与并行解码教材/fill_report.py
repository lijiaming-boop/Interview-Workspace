#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_report.py —— 把 05-INT8实验结果报告模板.md 里的 {{占位符}} 用真实产物填满

用法（实例上、实验跑完后执行）：
  python fill_report.py \
      --run-dir /mnt/workspace/openvla-oft/experiments/failure_study_runs/int8-full-20260914-195951

  # 自定义模板 / 输出
  python fill_report.py --run-dir <DIR> \
      --template /mnt/workspace/05-INT8实验结果报告模板.md \
      --out      /mnt/workspace/INT8实验报告.md

数据来源：
  <run-dir>/summary/metrics.csv     两组全部指标
  <run-dir>/summary/report.md       官方汇总（原样附到报告末尾）
  <run-dir>/{bf16,int8-mixed}/run_config.txt        配对一致性
  <run-dir>/{bf16,int8-mixed}/console.log           起止时间 / NaN / OOM 扫描
  <run-dir>/{bf16,int8-mixed}/actions/**/chunks.jsonl   deadline 达标率

说明：deadline 达标率官方汇总脚本**不计算**，本脚本按
      budget_ms = open_loop / control_freq * 1000 额外补算（G2 要求 ≥95%）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WARMUP_CHUNKS = 3
GROUPS = {"bf16": "bf16", "int8": "int8-mixed"}
PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def read_config(root: Path) -> dict:
    p = root / "run_config.txt"
    if not p.exists():
        return {}
    cfg = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def read_metrics(run_dir: Path) -> dict:
    """读 summary/metrics.csv -> {'bf16': {...}, 'int8-mixed': {...}}"""
    p = run_dir / "summary" / "metrics.csv"
    if not p.exists():
        return {}
    out = {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row.get("precision", "")] = row
    return out


def fnum(v, digits=2, default="—"):
    try:
        f = float(v)
        if f != f:  # NaN
            return default
        return f"{f:.{digits}f}"
    except (TypeError, ValueError):
        return default


def latencies(root: Path):
    """按 summarize 脚本同样的顺序收集 latency，丢掉前 WARMUP_CHUNKS 个。"""
    recs = []
    for path in sorted((root / "actions").rglob("chunks.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("inference_latency_ms") is not None:
                    recs.append(float(r["inference_latency_ms"]))
        except Exception:
            continue
    return recs[WARMUP_CHUNKS:], len(recs)


def deadline_rate(root: Path, budget_ms: float):
    lats, _ = latencies(root)
    if not lats:
        return None, 0
    ok = sum(1 for x in lats if x <= budget_ms)
    return ok / len(lats) * 100.0, len(lats)


def extract_times(console: Path):
    if not console.exists():
        return "—", "—"
    txt = console.read_text(encoding="utf-8", errors="ignore")
    stamps = re.findall(r"\d{2}/\d{2} \[\d{2}:\d{2}:\d{2}\]", txt)
    if not stamps:
        return "—", "—"
    return stamps[0], stamps[-1]


def scan_bad(console: Path):
    """扫 NaN / Inf / OOM 之类异常痕迹。"""
    if not console.exists():
        return "日志缺失", False
    txt = console.read_text(encoding="utf-8", errors="ignore")
    bad = []
    for pat, label in [
        (r"\bNaN\b", "NaN"),
        (r"CUDA out of memory", "CUDA OOM"),
        (r"dtype mismatch", "dtype mismatch"),
        (r"Traceback \(most recent call last\)", "Traceback"),
    ]:
        n = len(re.findall(pat, txt, flags=re.IGNORECASE))
        if n:
            bad.append(f"{label}×{n}")
    return ("、".join(bad) if bad else "未见异常"), bool(bad)


QNT_RE = re.compile(r"\b(?:int8|quant|8bit|8-bit|bnb|bitsandbytes)\b", re.IGNORECASE)


def scan_quant_audit(console: Path):
    """从 console.log 里尽力抽取量化覆盖证据（G1 审计）。

    评测脚本若打印过 Linear8bitLt 计数 / 覆盖模块清单，这里可以自动填；
    若什么都没打印，则明确返回「日志未打印」并给出补审计命令 —— 不猜、不编。
    """
    if not console.exists():
        return None
    txt = console.read_text(encoding="utf-8", errors="ignore")

    # 1) 直接的层数声明，例如 "Linear8bitLt: 224" / "int8 layers = 224"
    m = re.search(
        r"(?:Linear8bitLt|int8[_ ]?layers?|量化线性层)\D{0,12}(\d+)",
        txt, flags=re.IGNORECASE)
    n_layers = m.group(1) if m else None

    # 2) 覆盖模块名（按顶层模块归组的痕迹）
    mods = sorted(set(re.findall(
        r"\b(vision_backbone|language_model|llm_backbone|projector|action_head)\b",
        txt, flags=re.IGNORECASE)))
    return {"layers": n_layers, "modules": mods, "logged": bool(n_layers or mods)}


# audit_int8_coverage.py 产物里的关键行
_COV_NUM_RE = re.compile(r"Linear8bitLt（被量化）\s*[:：]\s*(\d+)")
_COV_PLAIN_RE = re.compile(r"普通 nn\.Linear（未量化）\s*[:：]\s*(\d+)")
_COV_RATE_RE = re.compile(r"量化占比\s*([\d.]+)%")
_COV_GRP_RE = re.compile(r"^\s{2,}([A-Za-z_][\w]*)\s+(\d+)\s*$", re.MULTILINE)
_COV_HEAD_RE = re.compile(r"\[2/3\]\s*加载 action head[\s\S]{0,200}?期望 bf16\s*→\s*(✅|❌)")
_COV_PROP_RE = re.compile(r"\[3/3\]\s*加载 proprio projector[\s\S]{0,200}?期望 bf16\s*→\s*(✅|❌)")


def read_coverage_evidence(run_dir: Path):
    """读取 `audit_int8_coverage.py` 的产物，把 G1 表自动填满。

    搜索顺序（先具体后宽泛）：
      1) <run_dir>/audit_int8_coverage.txt
      2) <run_dir>/**/audit_int8_coverage.txt
      3) <repo 根>/audit_int8_coverage.txt（脚本默认落点）
    读不到、或读到的内容不完整 → 返回 None，让上层退回「待审计」。
    绝不根据文件名猜数值。
    """
    cands: list[Path] = []
    if run_dir:
        cands.append(run_dir / "audit_int8_coverage.txt")
        cands += sorted(run_dir.rglob("audit_int8_coverage.txt"))
    # 仓库根 / 脚本所在目录
    cands.append(Path(__file__).resolve().parent / "audit_int8_coverage.txt")
    cands.append(Path(__file__).resolve().parent.parent / "audit_int8_coverage.txt")

    txt = None
    for p in cands:
        try:
            if p.exists() and p.is_file():
                txt = p.read_text(encoding="utf-8", errors="ignore")
                break
        except OSError:
            continue
    if not txt:
        return None

    d: dict = {}

    m = _COV_NUM_RE.search(txt)
    if not m:
        return None
    d["n_int8"] = m.group(1)

    m = _COV_PLAIN_RE.search(txt)
    d["n_plain"] = m.group(1) if m else "—"

    m = _COV_RATE_RE.search(txt)
    if m:
        rate = float(m.group(1))
        d["rate"] = f"{rate:.1f}%（{d['n_int8']} / "
        n_int8_i = int(d["n_int8"])
        n_plain_i = int(d["n_plain"]) if str(d["n_plain"]).isdigit() else 0
        total = n_int8_i + n_plain_i
        d["rate"] = f"{n_int8_i} / {total} = {rate:.1f}%" if total else f"{rate:.1f}%"
        d["rate_ok"] = rate >= 90.0
    else:
        d["rate"], d["rate_ok"] = "—", False

    # 归组：只看「按顶层模块归组（INT8）」那一段，避免把其它缩进行收进来
    seg = re.search(r"按顶层模块归组（INT8）[\s\S]{0,400}?(?:关键判读|第|={4,})",
                    txt)
    groups = re.findall(r"^\s{2,}([A-Za-z_][\w]*)\s+(\d+)\s*$",
                        seg.group(0) if seg else "", flags=re.MULTILINE)
    d["modules"] = [g for g, _ in groups]
    d["group_counts"] = {g: int(n) for g, n in groups}

    m = _COV_HEAD_RE.search(txt)
    d["head_bf16"] = True if (m and m.group(1) == "✅") else (False if m else None)
    m = _COV_PROP_RE.search(txt)
    d["prop_bf16"] = True if (m and m.group(1) == "✅") else (False if m else None)

    return d


def env_info() -> dict:
    info = {
        "hostname": "—", "gpu_name": "—", "driver": "—",
        "python_ver": "—", "torch_ver": "—", "cuda_ver": "—",
        "numpy_ver": "—", "bnb_ver": "—", "tf_ver": "—", "fa_ver": "—",
    }
    import platform
    info["hostname"] = platform.node() or "—"
    info["python_ver"] = platform.python_version()

    def ver(mod, attr="__version__"):
        try:
            m = __import__(mod, fromlist=["__file__"])
            return getattr(m, attr, "—")
        except Exception:
            return "—"

    info["torch_ver"] = ver("torch")
    info["numpy_ver"] = ver("numpy")
    info["bnb_ver"] = ver("bitsandbytes")
    info["tf_ver"] = ver("transformers")
    info["fa_ver"] = ver("flash_attn")
    try:
        import torch
        info["cuda_ver"] = str(torch.version.cuda)
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, errors="ignore", timeout=15,
            ).stdout.strip().splitlines()
            if out:
                parts = [x.strip() for x in out[0].split(",")]
                info["gpu_name"] = parts[0] if parts else "—"
                info["driver"] = parts[1] if len(parts) > 1 else "—"
        except Exception:
            pass
    return info


def gpu_others() -> str:
    if not shutil.which("nvidia-smi"):
        return "无 nvidia-smi"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, errors="ignore", timeout=15,
        ).stdout.strip()
        return out.replace("\n", " / ") if out else "无（采样时干净）"
    except Exception:
        return "—"


def yesno(ok: bool) -> str:
    return "✅" if ok else "❌"


# --------------------------------------------------------------------------
def build(run_dir: Path, template: Path) -> str:
    tpl = template.read_text(encoding="utf-8")
    v: dict[str, str] = {}

    v["run_tag"] = run_dir.name
    v["report_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    v["result_base"] = str(run_dir)

    roots = {k: run_dir / sub for k, sub in GROUPS.items()}
    cfgs = {k: read_config(p) for k, p in roots.items()}
    mets = read_metrics(run_dir)
    int8_key = GROUPS["int8"]

    # ---- 环境 ----
    v.update(env_info())
    v["gpu_others"] = gpu_others()
    v["ckpt"] = cfgs["bf16"].get("checkpoint") or cfgs["int8"].get("checkpoint") or "—"
    v["run_start"], v["run_end"] = extract_times(roots["bf16"] / "console.log")
    i_start, i_end = extract_times(roots["int8"] / "console.log")
    if i_start != "—":
        v["run_end"] = i_end

    # ---- 配对一致性 ----
    pair_map = [
        ("suite", "task_suite"), ("seed", "seed"), ("chunk", "chunk_size"),
        ("ol", "open_loop"), ("freq", "control_freq"), ("targets", "targets"),
        ("8bit", "load_in_8bit"), ("head", "action_head_dtype"),
    ]
    pair_ok = True
    for short, key in pair_map:
        b = cfgs["bf16"].get(key, "—")
        i = cfgs["int8"].get(key, "—")
        v[f"cfg_{short}_bf16"] = b
        v[f"cfg_{short}_int8"] = i
        same = (b == i)
        if short not in ("8bit", "head"):
            pair_ok = pair_ok and same
            v[f"cfg_{short}_same"] = yesno(same)
        else:
            v[f"cfg_{short}_same"] = "—"

    # ---- 指标 ----
    # metrics.csv 的真实列名（来自 summarize_int8_benchmark.py）与模板占位符后缀不同，
    # 因此这里显式给出 (列名, 占位符后缀) 映射，避免用错列填出 "—"。
    def put(prefix, row, col, suffix=None, digits=2):
        v[f"{prefix}_{suffix or col}"] = fnum(row.get(col) if row else None, digits)

    for k in ("bf16", "int8"):
        row = mets.get(GROUPS[k], {})
        v[f"{k}_episodes"] = (str(int(float(row["episodes"])))
                              if row.get("episodes") else "—")
        sc, ep = row.get("success_count"), row.get("episodes")
        v[f"{k}_success"] = (f"{int(float(sc))}/{int(float(ep))}"
                             if sc and ep else "—")
        sr = row.get("success_rate")
        v[f"{k}_success_pct"] = (f"{float(sr)*100:.1f}%" if sr else "—")
        put(k, row, "latency_mean_ms", "lat_mean")
        put(k, row, "latency_p50_ms", "lat_p50")
        put(k, row, "latency_p95_ms", "lat_p95")
        put(k, row, "policy_query_fps", "pqfps")
        put(k, row, "predicted_action_fps", "afps")
        put(k, row, "executed_action_capacity_fps", "eafps")
        put(k, row, "memory_min_mib", "mem_min", 0)
        put(k, row, "memory_peak_mib", "mem_peak", 0)
        put(k, row, "memory_delta_mib", "mem_delta", 0)
        put(k, row, "gpu_util_mean_percent", "gpu_util", 1)
        v[f"{k}_samples"] = (str(int(float(row["latency_samples_measured"])))
                             if row.get("latency_samples_measured") else "—")
        v[f"{k}_memsamples"] = (str(int(float(row["gpu_samples"])))
                                if row.get("gpu_samples") else "—")

    # ---- deadline 达标率（官方汇总不含，这里补算）----
    ol = float(cfgs["bf16"].get("open_loop", 8) or 8)
    freq = float(cfgs["bf16"].get("control_freq", 20) or 20)
    budget = ol / freq * 1000.0
    dl = {}
    for k in ("bf16", "int8"):
        rate, n = deadline_rate(roots[k], budget)
        dl[k] = rate
        v[f"{k}_deadline"] = f"{rate:.1f}%" if rate is not None else "—"
    v["budget_ms"] = f"{budget:.0f}"

    # ---- 变化量 ----
    def num(key, d=None):
        try:
            return float(mets.get(GROUPS[d], {}).get(key))
        except (TypeError, ValueError, AttributeError):
            return None

    def diff(key, digits=2, unit=""):
        b, i = num(key, "bf16"), num(key, "int8")
        if b is None or i is None:
            return "—"
        pct = (i - b) / b * 100.0 if b else float("nan")
        return f"{i-b:+.{digits}f}{unit} ({pct:+.1f}%)"

    for key, label in [("latency_mean_ms", "lat_mean"), ("latency_p50_ms", "lat_p50"),
                       ("latency_p95_ms", "lat_p95")]:
        v[f"{label}_chg"] = diff(key)
    for key, label in [("policy_query_fps", "pqfps"), ("predicted_action_fps", "afps"),
                       ("executed_action_capacity_fps", "eafps")]:
        v[f"{label}_chg"] = diff(key)
    v["mem_chg"] = diff("memory_peak_mib", 0, " MiB")

    # ---- 汇总三项 ----
    b_lat, i_lat = num("latency_mean_ms", "bf16"), num("latency_mean_ms", "int8")
    b_mem, i_mem = num("memory_peak_mib", "bf16"), num("memory_peak_mib", "int8")
    b_sr, i_sr = num("success_rate", "bf16"), num("success_rate", "int8")
    speedup = (b_lat / i_lat) if (b_lat and i_lat) else None
    mem_red = (b_mem - i_mem) if (b_mem is not None and i_mem is not None) else None
    loss_pp = ((b_sr - i_sr) * 100.0) if (b_sr is not None and i_sr is not None) else None

    v["speedup"] = f"{speedup:.3f}" if speedup else "—"
    v["mem_reduction"] = f"{mem_red:.0f}" if mem_red is not None else "—"
    v["loss_pp"] = f"{loss_pp:+.1f}" if loss_pp is not None else "—"

    # ---- paired wins / regressions ----
    wins = regs = None
    try:
        def succ(root):
            d = {}
            for p in sorted((root / "actions").rglob("episode_summary.json")):
                s = json.loads(p.read_text(encoding="utf-8"))
                d[(int(s["task_id"]), int(s["initial_state_index"]))] = bool(s.get("success"))
            return d
        sb, si = succ(roots["bf16"]), succ(roots["int8"])
        if sb and si and set(sb) == set(si):
            wins = sum(1 for k in sb if not sb[k] and si[k])
            regs = sum(1 for k in sb if sb[k] and not si[k])
    except Exception:
        pass
    v["wins"] = str(wins) if wins is not None else "—"
    v["regressions"] = str(regs) if regs is not None else "—"

    # ---- 产物完整性 ----
    checks = {
        "f_bf16_cfg": roots["bf16"] / "run_config.txt",
        "f_bf16_log": roots["bf16"] / "console.log",
        "f_bf16_mem": roots["bf16"] / "gpu_memory.csv",
        "f_bf16_chunks": next((roots["bf16"] / "actions").rglob("chunks.jsonl"), None),
        "f_bf16_eps": next((roots["bf16"] / "actions").rglob("episode_summary.json"), None),
        "f_int8_cfg": roots["int8"] / "run_config.txt",
        "f_int8_log": roots["int8"] / "console.log",
        "f_int8_mem": roots["int8"] / "gpu_memory.csv",
        "f_int8_chunks": next((roots["int8"] / "actions").rglob("chunks.jsonl"), None),
        "f_int8_eps": next((roots["int8"] / "actions").rglob("episode_summary.json"), None),
        "f_sum_metrics": run_dir / "summary" / "metrics.csv",
        "f_sum_report": run_dir / "summary" / "report.md",
    }
    all_present = True
    for key, p in checks.items():
        ok = bool(p and Path(p).exists())
        all_present = all_present and ok
        v[key] = yesno(ok)

    # ---- 量化路径审计字段 ----
    v["g1_load8"] = cfgs["int8"].get("load_in_8bit", "—")
    v["g1_load4"] = "False"   # 脚本固定 --load_in_4bit False
    v["g1_head"] = cfgs["int8"].get("action_head_dtype", "—")
    v["g1_prop"] = "bf16"     # proprietary projector 不在 run_config 中，按设计声明
    v["g1_load8_v"] = yesno(v["g1_load8"] == "True")
    v["g1_load4_v"] = yesno(v["g1_load4"] == "False")
    v["g1_head_v"] = yesno(v["g1_head"] == "bf16")
    v["g1_prop_v"] = "✍️ 需运行时审计确认"

    # ---- 量化覆盖审计（G1）：能从日志抽到就自动填，抽不到就明确标"待审计"，不猜 ----
    audit = scan_quant_audit(roots["int8"] / "console.log")
    AUDIT_CMD = ("import types\n"
                 "import bitsandbytes as bnb\n"
                 "from collections import Counter\n"
                 "from experiments.robot.openvla_utils import get_vla\n"
                 "\n"
                 "cfg = types.SimpleNamespace(\n"
                 "    pretrained_checkpoint=CKPT, load_in_8bit=True, load_in_4bit=False,\n"
                 "    use_film=False, num_images_in_input=2,\n"
                 "    use_l1_regression=True, use_diffusion=False)\n"
                 "vla = get_vla(cfg)          # 注意：get_vla() 返回 vla.model\n"
                 "\n"
                 "n = [k for k, m in vla.named_modules() if isinstance(m, bnb.nn.Linear8bitLt)]\n"
                 "print('量化线性层数量:', len(n))\n"
                 "print('按顶层模块归组:', Counter(k.split('.')[0] for k in n))\n"
                 "print('视觉编码器在内?', any('vision' in k.lower() for k in n))")
    if audit and audit["layers"]:
        v["g1_nlayers"] = audit["layers"]
        v["g1_nlayers_v"] = "✅"
        v["g1_cover"] = "、".join(audit["modules"]) if audit["modules"] else "✍️ 见 console.log"
        v["g1_cover_v"] = "✅" if audit["modules"] else "✍️"
    else:
        # 日志里没有 → 尝试读 audit_int8_coverage.py 的产物（放在 run 目录或工作区）
        ev = read_coverage_evidence(run_dir)
        if ev:
            v["g1_nlayers"] = ev["n_int8"]
            v["g1_nlayers_v"] = "✅"
            v["g1_nplain"] = ev["n_plain"]
            # n_plain 是正则捕获的字符串（可能是 "—"），必须安全转 int 再比较
            try:
                n_plain_i = int(str(ev["n_plain"]).strip())
            except (TypeError, ValueError):
                n_plain_i = None
            v["g1_nplain_v"] = ("✅" if n_plain_i <= 1 else "⚠️") if n_plain_i is not None else "—"
            v["g1_rate"] = ev["rate"]
            v["g1_rate_v"] = "✅" if ev["rate_ok"] else "⚠️"
            v["g1_cover"] = "、".join(ev["modules"])
            v["g1_cover_v"] = "✅"
            if ev["head_bf16"] is True:
                v["g1_head_v"] = "✅"
            if ev["prop_bf16"] is True:
                v["g1_prop_v"] = "✅"
            v["g1_audit_cmd"] = AUDIT_CMD
            v["_g1_audited"] = True
        else:
            v["g1_nlayers"] = "**待审计**（日志未打印）"
            v["g1_nlayers_v"] = "✍️"
            v["g1_nplain"] = "**待审计**"
            v["g1_nplain_v"] = "✍️"
            v["g1_rate"] = "**待审计**"
            v["g1_rate_v"] = "✍️"
            v["g1_cover"] = "**待审计**（日志未打印）"
            v["g1_cover_v"] = "✍️"
            v["g1_audit_cmd"] = AUDIT_CMD
    v["g1_audit_cmd"] = v.get("g1_audit_cmd", AUDIT_CMD)

    bad_txt, bad = scan_bad(roots["int8"] / "console.log")
    v["g1_scan"] = bad_txt
    v["g1_scan_v"] = "❌" if bad else "✅"
    v["g1_ep_v"] = yesno(v["bf16_episodes"] == v["int8_episodes"]
                         and v["bf16_episodes"] != "—")

    # ---- 判定 ----
    v["pair_ok"] = yesno(pair_ok)
    g0 = v["python_ver"].startswith("3.10") and v["torch_ver"] != "—"
    # G1：配置正确 + 无异常 + episode 数一致；若审计产物存在，还要求覆盖率达标。
    # 审计产物缺席不判失败（日志未打印属可接受的证据缺口），但会在表中标「待审计」。
    g1 = (v["g1_load8"] == "True" and v["g1_head"] == "bf16"
          and not bad and v["bf16_episodes"] == v["int8_episodes"]
          and v["bf16_episodes"] != "—")
    if v.get("_g1_audited"):
        g1 = g1 and bool(v.get("g1_rate_v") == "✅")
    dl_ok = dl.get("int8") is not None and dl["int8"] >= 95.0
    g2 = (speedup is not None and speedup >= 1.0 and dl_ok)
    g3 = (mem_red is not None and mem_red >= 0)
    # 注：门槛只看「掉点幅度」|loss_pp|（loss_pp<0 表示 INT8 反超，同样算通过）。
    g4 = loss_pp is not None and loss_pp <= 1.0
    g5 = pair_ok and all_present

    v["verdict_g0"] = "✅ 通过" if g0 else "❌ 未通过"
    v["verdict_g1"] = "✅ 通过" if g1 else "❌ 未通过（详见 G1 表）"
    v["verdict_g2"] = ("✅ 通过" if g2 else
                       ("⚠️ 待复核" if speedup is None else "❌ 未通过"))
    v["verdict_g3"] = "✅ 通过" if g3 else "❌ 未通过"
    v["verdict_mem"] = v["verdict_g3"]
    # loss_pp = (BF16 成功率 − INT8 成功率) × 100。
    # 正值 = INT8 掉点（变差）；负值 = INT8 反而更好。
    # 门槛只看「掉点幅度」|loss_pp|，否则 INT8 胜出时会跨越全部门槛落成「失败（>2pp）」。
    if loss_pp is None:
        v["verdict_g4"], v["verdict_loss"] = "❌ 待跑", "待跑"
    elif loss_pp <= 1.0:
        v["verdict_g4"], v["verdict_loss"] = "✅ 通过", "通过（≤1.0pp）"
    elif loss_pp < 0:
        v["verdict_g4"] = "✅ 通过"
        v["verdict_loss"] = f"通过（INT8 反超 {abs(loss_pp):.1f}pp）"
    elif loss_pp <= 2.0:
        v["verdict_g4"], v["verdict_loss"] = "❌ 需复核", "需复核（1~2pp）"
    else:
        v["verdict_g4"], v["verdict_loss"] = "❌ 失败", "失败（>2pp）"
    v["loss_mark_pass"] = ("← 本次" if (loss_pp is not None and loss_pp <= 1.0) else "")
    v["loss_mark_review"] = ("← 本次" if (loss_pp is not None and 1.0 < loss_pp <= 2.0)
                             else "")
    v["loss_mark_fail"] = ("← 本次" if (loss_pp is not None and loss_pp > 2.0) else "")

    # ---- 评测范围（完整 500 / 配对子集）----
    def _n(key):
        try:
            return int(float(v.get(key)))
        except (TypeError, ValueError):
            return None

    n_b16, n_i8 = _n("bf16_episodes"), _n("int8_episodes")
    if n_b16 and n_i8 and n_b16 == n_i8:
        if n_b16 >= 500:
            v["scope_note"] = ("完整 LIBERO-Spatial（10 任务 × 50 初始状态 = "
                               f"{n_b16} 个 episode），同 seed、同 state 顺序。")
            v["limitation_partial"] = "5. 本次为完整 500 episode 评测。"
        else:
            v["scope_note"] = (
                f"**配对子集 N={n_b16}**：两组取 task/state 集合的交集后对比，"
                "**不是完整评测**。同 seed、同 state 顺序。")
            v["limitation_partial"] = (
                f"5. **本次为部分评测**：只用了双方共有的 {n_b16} 个 task/state"
                "（原计划 500）。结论**不能**外推到完整 LIBERO-Spatial；"
                "补跑满 500 后应替换本报告。")
    else:
        v["scope_note"] = "⚠️ 两组 episode 数不一致，配对无效。"
        v["limitation_partial"] = "5. 两组 episode 数不一致，配对无效。"
    v["verdict_speedup"] = ("性能收益" if (speedup and speedup > 1.0)
                            else ("无回退" if (speedup and speedup >= 0.98) else "有回退"))
    v["verdict_g5"] = "✅ 通过" if g5 else "❌ 未通过（缺件或配对不一致）"

    gate_flags = {"通过": True, "待跑": False, "需复核": False, "失败": False}
    overall = all([g0, g1, g2, g3, g4, g5])
    if overall:
        v["verdict_overall"] = "✅ 通过（可进入重复性复核）"
    elif loss_pp is None:
        v["verdict_overall"] = "⏳ 待跑（汇总产物不完整）"
    else:
        fail = [n for n, ok in (("G0", g0), ("G1", g1), ("G2", g2),
                                ("G4", g4)) if not ok]
        v["verdict_overall"] = "❌ 未通过" + (f"（{'/'.join(fail)}）" if fail else "")

    # ---- 附原始汇总 ----
    rp = run_dir / "summary" / "report.md"
    v["raw_summary"] = rp.read_text(encoding="utf-8") if rp.exists() else "（尚未生成）"

    # ---- 渲染 ----
    missing = []

    def sub(m):
        k = m.group(1)
        if k not in v:
            missing.append(k)
            return m.group(0)
        return v[k]

    out = PLACEHOLDER_RE.sub(sub, tpl)
    if missing:
        print(f"[WARN] 模板里有未提供值的占位符: {sorted(set(missing))}",
              file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="RUN_TAG 目录，例 .../failure_study_runs/int8-full-20260914-195951")
    ap.add_argument("--template",
                    default=str(Path(__file__).with_name("05-INT8实验结果报告模板.md")))
    ap.add_argument("--out", default=None,
                    help="默认写到 <run-dir>/INT8实验报告.md")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"不是目录: {run_dir}")
    tpl = Path(args.template)
    if not tpl.exists():
        raise SystemExit(f"找不到模板: {tpl}")

    text = build(run_dir, tpl)
    out = Path(args.out) if args.out else run_dir / "INT8实验报告.md"
    out.write_text(text, encoding="utf-8")
    print(f"[OK] 报告已生成: {out}")
    print(f"     行数 {len(text.splitlines())}，字符 {len(text)}")
    print("     标 ✍️ 的段落仍需人工填写（量化覆盖模块、失败案例明细、局限性）。")


if __name__ == "__main__":
    main()
