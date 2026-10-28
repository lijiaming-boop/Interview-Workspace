#!/usr/bin/env python3
"""Block another unsafe OpenVLA-OFT continuation run before it consumes GPU time."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def tfds_examples(dataset_dir: Path) -> int | None:
    infos = list(dataset_dir.rglob("dataset_info.json"))
    if not infos:
        return None
    info = json.loads(infos[-1].read_text(encoding="utf-8"))
    total = 0
    for split in info.get("splits", []):
        lengths = split.get("shardLengths") or split.get("shard_lengths") or []
        total += sum(int(x) for x in lengths)
    return total or None


def audit_tfds(dataset_dir: Path, errors: list[str], notes: list[str]) -> None:
    """Inspect one stored RLDS episode without requiring its Python builder."""
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        errors.append("环境缺少 tensorflow_datasets，无法核验 RLDS schema")
        return
    infos = list(dataset_dir.rglob("dataset_info.json"))
    if not infos:
        return
    try:
        builder = tfds.builder_from_directory(str(infos[-1].parent))
        episode = next(iter(builder.as_dataset(split="train").take(1)))
        step = next(iter(episode["steps"].take(1)))
        observation = step.get("observation", {})
        action = step.get("action")
        if action is None or tuple(action.shape) != (7,):
            errors.append(f"修正 RLDS 的 action 应为 [7]，实际 {getattr(action, 'shape', None)}")
        if not {"image", "wrist_image"}.issubset(observation):
            errors.append("修正 RLDS observation 缺少 image/wrist_image")
        if "state" not in observation and not {"EEF_state", "gripper_state"}.issubset(observation):
            errors.append("修正 RLDS observation 缺少 8D state 或 EEF_state+gripper_state")
        if "language_instruction" not in step and "language_instruction" not in observation:
            errors.append("修正 RLDS 缺少逐步 language_instruction")
        notes.append(f"RLDS schema: step={sorted(step)}, observation={sorted(observation)}")
    except Exception as exc:
        errors.append(f"无法读取修正 RLDS: {type(exc).__name__}: {exc}")


def audit_hdf5(path: Path, min_demos: int, errors: list[str], notes: list[str]) -> None:
    try:
        import h5py
        import numpy as np
    except ImportError:
        errors.append("SOURCE_HDF5 已设置，但环境缺少 h5py/numpy")
        return

    with h5py.File(path, "r") as f:
        if "data" not in f:
            errors.append("HDF5 缺少 /data")
            return
        demos = list(f["data"].keys())
        if len(demos) < min_demos:
            errors.append(f"专家示教只有 {len(demos)} 条，最低要求 {min_demos} 条")
        actions = []
        for name in demos:
            group = f["data"][name]
            action_key = "action" if "action" in group else "actions" if "actions" in group else None
            if action_key is None:
                errors.append(f"{name} 缺少 action/actions")
                continue
            value = np.asarray(group[action_key])
            if value.ndim != 2 or value.shape[1] != 7:
                errors.append(f"{name}/action 形状应为 [T,7]，实际 {value.shape}")
                continue
            if not np.isfinite(value).all():
                errors.append(f"{name}/action 含 NaN/Inf")
                continue
            actions.append(value)
        if not actions:
            return
        all_actions = np.concatenate(actions)
        ranges = np.ptp(all_actions, axis=0)
        # 上次脚本把旋转三维全部写成 0；这不是可接受的 OFT 续训数据。
        if np.count_nonzero(ranges[3:6] > 1e-3) < 2:
            errors.append(f"旋转动作缺乏覆盖，dims 3:6 range={ranges[3:6].round(6).tolist()}")
        if np.max(np.abs(all_actions[:, :6])) > 1.001:
            errors.append("OSC 动作超出归一化范围 [-1,1]")
        notes.append(f"HDF5: demos={len(demos)}, steps={len(all_actions)}, action_range={ranges.round(4).tolist()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, required=True)
    parser.add_argument("--base-step", type=int, default=150000)
    parser.add_argument("--base-dataset", default="libero_spatial_no_noops")
    parser.add_argument("--correction-dataset", default="libero_spatial_top_drawer_dagger")
    parser.add_argument("--mixture", default="libero_spatial_plus_top_drawer_dagger")
    parser.add_argument("--source-hdf5", type=Path)
    parser.add_argument("--min-demos", type=int, default=50)
    args = parser.parse_args()

    errors: list[str] = []
    notes: list[str] = []
    checkpoint = args.checkpoint.resolve()
    data_root = args.data_root.resolve()

    for name in (
        f"action_head--{args.base_step}_checkpoint.pt",
        f"proprio_projector--{args.base_step}_checkpoint.pt",
    ):
        if not (checkpoint / name).is_file():
            errors.append(f"基座 checkpoint 缺少 {name}；不能安全续训")
    if not (checkpoint / "config.json").is_file():
        errors.append("基座 checkpoint 缺少 config.json")
    if not any(checkpoint.glob("*.safetensors")) and not any(checkpoint.glob("pytorch_model*.bin")):
        errors.append("基座 checkpoint 缺少完整 VLA 权重")

    for dataset in (args.base_dataset, args.correction_dataset):
        path = data_root / dataset
        if not path.is_dir():
            errors.append(f"缺少 RLDS 数据集目录: {path}")
        elif not list(path.rglob("*.tfrecord*")):
            errors.append(f"RLDS 数据集没有 tfrecord: {path}")
        else:
            notes.append(f"RLDS {dataset}: examples={tfds_examples(path) or 'unknown'}")
            if dataset == args.correction_dataset:
                audit_tfds(path, errors, notes)

    mixtures = args.oft_root / "prismatic/vla/datasets/rlds/oxe/mixtures.py"
    text = mixtures.read_text(encoding="utf-8") if mixtures.is_file() else ""
    for token in (args.mixture, args.base_dataset, args.correction_dataset):
        if token not in text:
            errors.append(f"mixtures.py 未注册 {token}")

    stats_file = checkpoint / "dataset_statistics.json"
    if not stats_file.is_file():
        errors.append("基座 checkpoint 缺少 dataset_statistics.json")
    else:
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        if args.base_dataset not in stats and "libero_spatial" not in stats:
            errors.append("基座归一化统计中没有 libero_spatial(_no_noops)")

        def visit(value):
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
            elif isinstance(value, (int, float)) and not math.isfinite(value):
                errors.append("基座归一化统计含 NaN/Inf")

        visit(stats)

    if args.source_hdf5:
        if not args.source_hdf5.is_file():
            errors.append(f"找不到 SOURCE_HDF5: {args.source_hdf5}")
        else:
            audit_hdf5(args.source_hdf5, args.min_demos, errors, notes)
    else:
        errors.append("未提供 --source-hdf5，无法核验示教条数和旋转动作覆盖")

    for line in notes:
        print(f"[OK] {line}")
    if errors:
        for line in dict.fromkeys(errors):
            print(f"[FAIL] {line}")
        return 2
    print("[PASS] 可进入 200-step 小规模续训；这不代表允许直接发布模型")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
