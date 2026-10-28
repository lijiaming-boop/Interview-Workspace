"""Analyze LIBERO base-model action traces.

Trace format (.npz):
  actions: float array [T, 7]
  optional: step [T], chunk [T], state_index, success, model

CSV format:
  step,chunk,dx,dy,dz,drx,dry,drz,gripper

The base model is the only authoritative reference in this experiment.
This script intentionally does not infer physical collision from actions alone;
it reports motion that must be checked against the video or contact state.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ACTION_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")


def load_trace(path: Path) -> tuple[np.ndarray, dict]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "actions" not in data:
                raise ValueError(f"{path}: 缺少 actions 数组")
            actions = np.asarray(data["actions"], dtype=np.float64)
            meta = {}
            for key in ("step", "chunk", "state_index", "success", "model"):
                if key in data:
                    value = data[key]
                    meta[key] = value.tolist() if value.ndim else value.item()
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        missing = [name for name in ACTION_NAMES if name not in rows[0]] if rows else list(ACTION_NAMES)
        if missing:
            raise ValueError(f"{path}: CSV 缺少列 {missing}")
        actions = np.asarray(
            [[float(row[name]) for name in ACTION_NAMES] for row in rows], dtype=np.float64
        )
        meta = {}
        for key in ("step", "chunk", "state_index", "success", "model"):
            if rows and key in rows[0]:
                values = [row[key] for row in rows]
                meta[key] = values
    else:
        raise ValueError(f"仅支持 .npz 或 .csv: {path}")

    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"{path}: actions 应为 [T,7]，实际为 {actions.shape}")
    if len(actions) == 0:
        raise ValueError(f"{path}: actions 为空")
    return actions, meta


def as_chunks(actions: np.ndarray, meta: dict, chunk_size: int) -> np.ndarray:
    if "chunk" in meta:
        chunks = np.asarray(meta["chunk"])
        if len(chunks) == len(actions):
            return chunks.astype(int)
    return np.arange(len(actions)) // chunk_size


def fmt_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x): .4f}" for x in values) + "]"


def summarize(label: str, actions: np.ndarray, meta: dict, chunk_size: int) -> list[str]:
    chunks = as_chunks(actions, meta, chunk_size)
    position = actions[:, :3].cumsum(axis=0)
    lines = [
        f"### {label}",
        f"- steps: {len(actions)}",
        f"- action range: min={fmt_vector(actions.min(axis=0))}, max={fmt_vector(actions.max(axis=0))}",
        f"- mean absolute action: {fmt_vector(np.abs(actions).mean(axis=0))}",
        f"- cumulative position delta: {fmt_vector(position[-1])}",
        f"- max single-step position delta: {fmt_vector(actions[np.abs(actions[:, :3]).sum(axis=1).argmax(), :3])}",
        f"- gripper unique rounded values: {np.unique(np.round(actions[:, 6], 4)).tolist()}",
        "",
        "前三步动作（基座作为权威参考）:",
        "| step | chunk | dx | dy | dz | drx | dry | drz | gripper |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i in range(min(3, len(actions))):
        row = actions[i]
        lines.append(
            f"| {i} | {int(chunks[i])} | "
            + " | ".join(f"{float(x):.5f}" for x in row)
            + " |"
        )
    return lines


def build_report(base_path: Path, chunk_size: int) -> str:
    base, base_meta = load_trace(base_path)
    lines = [
        "# 失败任务动作分析",
        "",
        "> 分析原则：当前实验只使用基座模型；不使用微调模型作为参考。",
        "> 动作数值本身不能证明碰撞，必须结合失败视频或环境 contact/state 日志判断。",
        "",
        f"- base trace: `{base_path}`",
    ]
    lines += ["", *summarize("基座模型（权威）", base, base_meta, chunk_size)]
    lines += ["", "### 当前结论", "- 本报告只对基座失败轨迹作权威分析，不引入微调模型结论。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="基座动作轨迹 .npz/.csv；必填")
    parser.add_argument("--out", type=Path, help="输出 Markdown 路径；不填则只打印")
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()

    report = build_report(args.base, args.chunk_size)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"已写入: {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
