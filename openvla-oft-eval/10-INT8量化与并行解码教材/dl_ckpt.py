#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dl_ckpt.py —— 用 snapshot_download API 拉 OpenVLA-OFT libero-spatial checkpoint
（不依赖 `python -m modelscope`，因为很多 pip 版没有 __main__.py）

用法（在实例上、已 source venv 的终端里）：
    python dl_ckpt.py                # 下载
    python dl_ckpt.py --force        # 强制重下已存在文件
    python dl_ckpt.py --probe        # 只列出远端文件，不下载

断点续传：snapshot_download 会跳过已存在的完整文件，中断后直接重跑本脚本即可。
"""
import argparse
import json
import os
import sys
import time
import urllib.request

# ---- 关键：清掉阿里云 OSM 端点（它只镜像 damo/iic 官方模型，社区模型一律 404）----
for _k in ("MODELSCOPE_ENDPOINT", "MODELSCOPE_CACHE"):
    os.environ.pop(_k, None)

MODEL = os.environ.get("MODEL", "a11111ad/openvla-7b-oft-finetuned-libero-spatial")
DEST = os.environ.get(
    "DEST",
    "/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial",
)

# 远端权威清单（2026-09-14 走官方 API 核对）
EXPECTED = {
    "model-00001-of-00004.safetensors": 4_925_122_448,
    "model-00002-of-00004.safetensors": 4_947_392_496,
    "model-00003-of-00004.safetensors": 4_947_417_456,
    "model-00004-of-00004.safetensors": 262_668_432,
    "action_head--150000_checkpoint.pt": 302_242_674,
    "proprio_projector--150000_checkpoint.pt": 67_275_256,
    "dataset_statistics.json": 2_957,
}

BIG = {k for k in EXPECTED if k.endswith((".safetensors", ".pt"))}


def human(n):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or u == "TiB":
            return f"{n:.2f} {u}"
        n /= 1024


def probe():
    url = (f"https://www.modelscope.cn/api/v1/models/{MODEL}"
           f"/repo/files?Revision=master&Recursive=True")
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)
    files = d.get("Data", {}).get("Files", [])
    if not files:
        print("[FAIL] 未取到清单；仓库可能不存在或为私有", file=sys.stderr)
        return 1
    total = 0
    for f in sorted(files, key=lambda x: x["Path"]):
        if f["Type"] == "tree":
            continue
        total += f["Size"]
        print(f"{f['Size']:>15,}  {f['Path']}{'  [LFS]' if f['IsLFS'] else ''}")
    print(f"\n合计 {human(total)}；必需文件检查：")
    for n in EXPECTED:
        hit = any(f["Path"] == n for f in files)
        print(f"  {'[OK]  ' if hit else '[FAIL]'} {n}")
        if not hit:
            return 1
    return 0


def download(force: bool):
    try:
        from modelscope import snapshot_download
    except ImportError:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError:
            print("[FAIL] 装 modelscope：python -m pip install -U modelscope "
                  "-i https://mirrors.aliyun.com/pypi/simple/", file=sys.stderr)
            return 1

    os.makedirs(DEST, exist_ok=True)
    print(f"==> 模型: {MODEL}")
    print(f"==> 目标: {DEST}")
    print(f"==> 端点: 官方 www.modelscope.cn（已清除 MODELSCOPE_ENDPOINT）")
    print("==> 开始下载；中断后重跑本脚本可续传。\n")

    # 后台线程每 20s 报一次已落盘体积
    import threading

    stop = threading.Event()

    def ticker():
        t0 = time.time()
        last_sz, last_t = 0, t0
        while not stop.wait(20):
            sz = sum(
                os.path.getsize(os.path.join(dp, fn))
                for dp, _, fns in os.walk(DEST)
                for fn in fns
                if not fn.endswith(".tmp")
            )
            dt = time.time() - last_t
            spd = (sz - last_sz) / dt if dt > 0 else 0
            pct = sz / 15_500_000_000 * 100
            el = time.time() - t0
            print(f"    [{el:6.0f}s] {human(sz):>10}  {pct:5.1f}%  "
                  f"{spd/1024**2:6.2f} MiB/s", flush=True)
            last_sz, last_t = sz, time.time()

    th = threading.Thread(target=ticker, daemon=True)
    th.start()
    try:
        snapshot_download(MODEL, local_dir=DEST, force_download=force)
    except TypeError:
        # 老版本参数名不同
        snapshot_download(MODEL, cache_dir=DEST, force_download=force)
    finally:
        stop.set()
        th.join(timeout=2)
    return 0


def verify():
    print("\n==> 校验落盘完整性")
    rc = 0
    total = 0
    for name, want in EXPECTED.items():
        p = os.path.join(DEST, name)
        if not os.path.exists(p):
            print(f"  [FAIL] 缺 {name}")
            rc = 1
            continue
        got = os.path.getsize(p)
        total += got
        if got == want:
            print(f"  [OK]   {name}  {human(got)}")
        else:
            print(f"  [FAIL] {name}  实得 {got:,} != 期望 {want:,}")
            rc = 1

    # norm_stats 键校验（缺它会除零出 nan，500 episodes 全废）
    p = os.path.join(DEST, "dataset_statistics.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            keys = [k for k in d if "libero" in k.lower()]
            print(f"\n  norm_stats 键: {list(d.keys())}")
            if any("spatial" in k.lower() for k in keys):
                print("  [OK] 含 spatial 统计量")
            else:
                print(f"  [FAIL] 无 spatial 统计量：{keys}")
                rc = 1
        except Exception as e:
            print(f"  [FAIL] dataset_statistics.json 解析失败: {e}")
            rc = 1

    print(f"\n  落盘合计 {human(total)}")
    if rc == 0:
        print("\n[OK] 校验通过。下一步：\n"
              "  cd /mnt/workspace/openvla-oft\n"
              "  PYTHON_BIN=python bash experiments/robot/libero/failure_study/"
              "verify_oft_step.sh checkpoint\n"
              "  PYTHON_BIN=python bash experiments/robot/libero/failure_study/"
              "verify_oft_step.sh model")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.probe:
        return probe()
    if a.verify:
        return verify()
    rc = download(a.force)
    if rc == 0:
        rc = verify()
    return rc


if __name__ == "__main__":
    sys.exit(main())
