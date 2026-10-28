"""Sweep camera position/orientation to find the best Bridge-matching viewpoint.

Renders a grid of camera views at different positions and saves comparison images
against a reference Bridge dataset frame.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep camera poses for Bridge alignment")
    parser.add_argument("--output_dir", default="12-IsaacLab-Bridge-Franka/outputs/camera-sweep")
    parser.add_argument("--reference_image", default="", help="Path to a real Bridge dataset frame for comparison")
    parser.add_argument("--x_range", nargs=3, type=float, default=[0.90, 1.20, 0.05], help="x min max step")
    parser.add_argument("--y_range", nargs=3, type=float, default=[-1.20, -0.90, 0.05], help="y min max step")
    parser.add_argument("--z_range", nargs=3, type=float, default=[0.70, 1.00, 0.05], help="z min max step")
    parser.add_argument("--seed", type=int, default=7)
    try:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    except ImportError:
        parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _arange_inclusive(start: float, stop: float, step: float) -> List[float]:
    values = []
    v = start
    while v <= stop + 1e-9:
        values.append(round(v, 4))
        v += step
    return values


def main() -> None:
    args = parse_args()
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    import gymnasium as gym
    import torch
    import register_env  # noqa: F401
    from bridge_pick_place_env_cfg import BridgePickPlaceEnvCfg, BRIDGE_CAMERA_ROT
    from observation import camera_rgb

    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = _arange_inclusive(*args.x_range)
    ys = _arange_inclusive(*args.y_range)
    zs = _arange_inclusive(*args.z_range)
    sweep_points = list(itertools.product(xs, ys, zs))
    print(f"Total sweep points: {len(sweep_points)}")

    results = []
    ref_image = None
    if args.reference_image:
        from PIL import Image
        ref_image = np.array(Image.open(args.reference_image).convert("RGB"))

    for idx, (x, y, z) in enumerate(sweep_points):
        env_cfg = BridgePickPlaceEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args.device
        env_cfg.scene.bridge_camera.offset.pos = (x, y, z)
        env_cfg.scene.bridge_camera.offset.rot = BRIDGE_CAMERA_ROT

        env = gym.make("Isaac-Bridge-PickPlace-Franka-IK-Rel-v0", cfg=env_cfg)
        env.reset(seed=args.seed)
        # Step a few times to let physics settle
        zero_action = torch.zeros(1, 7, device=args.device)
        for _ in range(5):
            env.step(zero_action)

        rgb = camera_rgb(env.unwrapped, env_id=0)
        fname = f"cam_x{x:.2f}_y{y:.2f}_z{z:.2f}.png"
        from PIL import Image
        Image.fromarray(rgb).save(out_dir / fname)

        # Compute simple similarity to reference if available
        similarity = -1.0
        if ref_image is not None:
            ref_resized = np.array(Image.fromarray(ref_image).resize((rgb.shape[1], rgb.shape[0])))
            # Use normalized cross-correlation as similarity
            a = rgb.astype(np.float32).ravel()
            b = ref_resized.astype(np.float32).ravel()
            a_norm = a - a.mean()
            b_norm = b - b.mean()
            ncc = float(np.dot(a_norm, b_norm) / (np.linalg.norm(a_norm) * np.linalg.norm(b_norm) + 1e-8))
            similarity = ncc

        results.append({"x": x, "y": y, "z": z, "file": fname, "ncc_similarity": similarity})
        print(f"[{idx+1}/{len(sweep_points)}] ({x}, {y}, {z}) sim={similarity:.4f}")

        env.close()

    # Save results and find best
    results_sorted = sorted(results, key=lambda r: r["ncc_similarity"], reverse=True)
    (out_dir / "sweep_results.json").write_text(
        json.dumps(results_sorted, indent=2), encoding="utf-8"
    )
    if results_sorted[0]["ncc_similarity"] > 0:
        best = results_sorted[0]
        print(f"\nBest camera pose: ({best['x']}, {best['y']}, {best['z']}) NCC={best['ncc_similarity']:.4f}")

    simulation_app.close()


if __name__ == "__main__":
    main()
