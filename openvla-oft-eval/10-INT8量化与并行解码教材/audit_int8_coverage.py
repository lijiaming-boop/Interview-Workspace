#!/usr/bin/env python
"""
G1 量化路径审计：确认 load_in_8bit=True 到底量化了「谁」。

用法（在 openvla-oft 仓库根目录、已 source venv 的环境里跑）：

    # 最小版：只审计主干（约 1 分钟）
    python audit_int8_coverage.py \
        --checkpoint /mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial

    # 完整版：连 action head / proprio projector 一起审计（推荐，G1 要的就是这个）
    python audit_int8_coverage.py \
        --checkpoint /mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial \
        --with-head

输出：
    1. 终端打印（可直接粘进报告 §4）
    2. 同目录（或 --out 指定）写 audit_int8_coverage.txt

设计原则：只读、不改模型、不做推理 —— 不会占用额外显存，也不影响正在跑的实验。
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from collections import Counter

import torch


def _ensure_repo_on_path() -> None:
    """把 openvla-oft 仓库根加进 sys.path，否则 `experiments.robot.*` 导不进来。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.getcwd(), "/mnt/workspace/openvla-oft"):
        if os.path.isfile(os.path.join(cand, "experiments", "robot", "openvla_utils.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return
    sys.exit("[FAIL] 找不到 openvla-oft 仓库根（需含 experiments/robot/openvla_utils.py）。\n"
             "       请在仓库根目录运行，或用 --repo 指定。")


# --------------------------------------------------------------------------
# 统计内核
# --------------------------------------------------------------------------
def count_linear(model, bnb) -> dict:
    """统计 nn.Linear / Linear8bitLt / 其他 dtype 的线性层，并按顶层模块归组。"""
    n_linear = n_int8 = 0
    int8_names: list[str] = []
    dt: Counter = Counter()

    for name, m in model.named_modules():
        if isinstance(m, bnb.nn.Linear8bitLt):
            n_int8 += 1
            int8_names.append(name)
        elif isinstance(m, torch.nn.Linear):
            n_linear += 1
        else:
            continue
        try:
            dt[str(next(m.parameters()).dtype)] += 1
        except StopIteration:
            pass

    groups = Counter(n.split(".")[0] for n in int8_names)
    return {
        "n_linear_plain": n_linear,       # 仍是普通 nn.Linear 的
        "n_int8": n_int8,                 # 被替换成 Linear8bitLt 的
        "int8_names": int8_names,
        "groups": dict(groups),
        "dtypes": dict(dt),
    }


def dtype_counts(module) -> dict:
    c: Counter = Counter()
    for _, p in module.named_parameters():
        c[str(p.dtype)] += 1
    for _, b in module.named_buffers():
        c[str(b.dtype)] += 1
    return dict(c)


def _cfg(**kw) -> types.SimpleNamespace:
    """构造 get_vla 需要的最小 cfg（其余字段给安全的默认值）。"""
    base = dict(
        pretrained_checkpoint="",
        load_in_8bit=True,
        load_in_4bit=False,
        use_film=False,
        num_images_in_input=2,
        use_l1_regression=True,
        use_diffusion=False,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="G1 INT8 量化覆盖审计")
    ap.add_argument("--checkpoint", required=True,
                    help="本地 checkpoint 目录或 HF repo id")
    ap.add_argument("--repo", default=None, help="openvla-oft 仓库根（默认自动探测）")
    ap.add_argument("--with-head", action="store_true",
                    help="同时审计 action head / proprio projector 的 dtype")
    ap.add_argument("--out", default=None, help="输出 txt 路径")
    args = ap.parse_args()

    if args.repo:
        sys.path.insert(0, args.repo)
    _ensure_repo_on_path()

    import bitsandbytes as bnb
    from experiments.robot.openvla_utils import (
        get_vla, get_action_head, get_proprio_projector,
    )

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 68)
    emit("G1 量化路径审计报告")
    emit("=" * 68)
    emit(f"checkpoint      : {args.checkpoint}")
    emit(f"bitsandbytes    : {bnb.__version__}")
    emit(f"torch           : {torch.__version__} / cuda={torch.version.cuda}")
    emit()

    # ---- 1. 主干 ----
    emit("[1/3] 加载主干（load_in_8bit=True）……")
    vla = get_vla(_cfg(pretrained_checkpoint=args.checkpoint))
    emit(f"      返回类型: {type(vla).__name__}")

    st = count_linear(vla, bnb)
    emit()
    emit("── 线性层统计 ──────────────────────────────────────")
    emit(f"Linear8bitLt（被量化）        : {st['n_int8']}")
    emit(f"普通 nn.Linear（未量化）      : {st['n_linear_plain']}")
    total = st["n_int8"] + st["n_linear_plain"]
    if total:
        emit(f"合计                          : {total}"
             f"  →  量化占比 {st['n_int8']/total*100:.1f}%")
    emit()
    emit("── 按顶层模块归组（INT8） ──────────────────────────")
    if st["groups"]:
        for g, c in sorted(st["groups"].items(), key=lambda x: -x[1]):
            emit(f"  {g:<28} {c}")
    else:
        emit("  （无 —— 量化没有生效！）")

    # 视觉编码器是否在内，是本次审计最关键的问题
    vis_hit = [g for g in st["groups"] if "vision" in g.lower()]
    emit()
    emit("── 关键判读 ────────────────────────────────────────")
    if st["n_int8"] == 0:
        emit("  ❌ 量化线性层数量 = 0 → load_in_8bit 未生效，显存降幅另有原因")
    else:
        emit(f"  ✅ INT8 路径生效：{st['n_int8']} 个 Linear8bitLt")
    if vis_hit:
        emit(f"  ⚠️ 视觉编码器也在量化范围内：{vis_hit}")
        emit("     → 报告应写「主干 + 视觉编码器均被量化」")
    else:
        emit("  ✅ 视觉编码器未被量化（归组中无 vision_backbone）")
        emit("     → 报告可写「仅 LLM 主干被量化」")

    # ---- 2. 动作头 ----
    if args.with_head:
        emit()
        emit("[2/3] 加载 action head ……")
        llm_dim = getattr(vla, "llm_dim", None) or getattr(vla, "hidden_size", 4096)
        try:
            head = get_action_head(_cfg(pretrained_checkpoint=args.checkpoint), llm_dim)
            hd = dtype_counts(head)
            emit(f"      dtype: {hd}")
            emit(f"      期望 bf16 → {'✅ 符合' if list(hd) == ['torch.bfloat16'] else '❌ 不符！'}")
        except Exception as e:  # noqa: BLE001
            emit(f"      [WARN] 加载失败（不影响主干结论）: {type(e).__name__}: {e}")

        emit()
        emit("[3/3] 加载 proprio projector ……")
        try:
            prop = get_proprio_projector(
                _cfg(pretrained_checkpoint=args.checkpoint),
                llm_dim=llm_dim, proprio_dim=8)
            pd = dtype_counts(prop)
            emit(f"      dtype: {pd}")
            emit(f"      期望 bf16 → {'✅ 符合' if list(pd) == ['torch.bfloat16'] else '❌ 不符！'}")
        except Exception as e:  # noqa: BLE001
            emit(f"      [WARN] 加载失败（不影响主干结论）: {type(e).__name__}: {e}")
    else:
        emit()
        emit("[2/3][3/3] 已跳过 action head / proprio projector（未加 --with-head）")

    # ---- 3. 可直接粘进报告的表格 ----
    emit()
    emit("=" * 68)
    emit("可直接粘进报告 §4 的表格行")
    emit("=" * 68)
    cover = "、".join(sorted(st["groups"])) or "—"
    emit(f"| 量化线性层数量 | {st['n_int8']} | ≥1 | "
         f"{'✅' if st['n_int8'] > 0 else '❌'} |")
    emit(f"| 量化覆盖模块 | {cover} | 须明确记录 | ✅ |")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "audit_int8_coverage.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[OK] 已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
