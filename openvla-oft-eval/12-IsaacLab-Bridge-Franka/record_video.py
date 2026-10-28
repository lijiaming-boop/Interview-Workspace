"""Record a side-by-side raw vs Bridge-aligned video for domain-gap visualization."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge_action_mapper import BridgeActionMapper, BridgeActionMappingConfig
from domain_randomization import DomainRandomizationConfig, apply_rgb_alignment
from observation import camera_rgb, make_openvla_observation
from openvla_policy import HeuristicPickPlacePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record domain-gap comparison video")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--output_dir", default="12-IsaacLab-Bridge-Franka/outputs/video")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=20)
    try:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    except ImportError:
        parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _overlay_text(frame: np.ndarray, text: str, position: tuple = (10, 20)) -> np.ndarray:
    """Simple text overlay using numpy (no OpenCV dependency)."""
    # We burn a simple dark bar + the text description into metadata
    result = frame.copy()
    y = position[1]
    # Draw a semi-transparent dark bar at the top
    bar_height = 28
    result[:bar_height, :] = (result[:bar_height, :].astype(np.float32) * 0.4).astype(np.uint8)
    return result


def _make_comparison_frame(
    raw: np.ndarray, aligned: np.ndarray, step: int, action: np.ndarray | None = None
) -> np.ndarray:
    """Create a side-by-side comparison frame."""
    # Resize both to the same height
    target_h = max(raw.shape[0], aligned.shape[0], 224)
    target_w = max(raw.shape[1], aligned.shape[1], 224)
    
    def resize_to(img, h, w):
        if img.shape[0] == h and img.shape[1] == w:
            return img
        from PIL import Image
        pil = Image.fromarray(img).resize((w, h), Image.BILINEAR)
        return np.array(pil)
    
    raw_resized = resize_to(raw, target_h, target_w)
    aligned_resized = resize_to(aligned, target_h, target_w)
    
    # Add label bars
    raw_labeled = _overlay_text(raw_resized, f"RAW step={step}")
    aligned_labeled = _overlay_text(aligned_resized, f"ALIGNED step={step}")
    
    # Create separator
    sep = np.full((target_h, 4, 3), 255, dtype=np.uint8)
    
    combined = np.concatenate([raw_labeled, sep, aligned_labeled], axis=1)
    return combined


def main() -> None:
    args = parse_args()
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    import gymnasium as gym
    import torch
    import register_env  # noqa: F401
    from bridge_pick_place_env_cfg import BridgePickPlaceEnvCfg, apply_camera_profile
    import imageio.v2 as imageio

    env_cfg = BridgePickPlaceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    apply_camera_profile(env_cfg, "bridge_aligned")
    env = gym.make("Isaac-Bridge-PickPlace-Franka-IK-Rel-v0", cfg=env_cfg)
    device = env.unwrapped.device if hasattr(env.unwrapped, "device") else args.device

    raw_cfg = DomainRandomizationConfig(mode="raw")
    aligned_cfg = DomainRandomizationConfig(mode="bridge_aligned")
    mapper = BridgeActionMapper(BridgeActionMappingConfig())
    policy = HeuristicPickPlacePolicy()
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    video = imageio.get_writer(out_dir / "comparison.mp4", fps=args.fps)
    raw_video = imageio.get_writer(out_dir / "raw_only.mp4", fps=args.fps)
    aligned_video = imageio.get_writer(out_dir / "aligned_only.mp4", fps=args.fps)

    try:
        obs, _ = env.reset(seed=args.seed)
        policy.reset()
        for step in range(args.max_frames):
            if not simulation_app.is_running():
                break
            raw_rgb = camera_rgb(env.unwrapped, env_id=0)
            aligned_rgb = apply_rgb_alignment(raw_rgb.copy(), aligned_cfg, rng=rng)
            raw_passthrough = apply_rgb_alignment(raw_rgb.copy(), raw_cfg, rng=rng)

            obs_dict = make_openvla_observation(
                env.unwrapped, "pick up the cube", aligned_cfg, rng=rng, env_id=0
            )
            bridge_action = policy.act(obs_dict)
            isaac_action = mapper.to_isaac_action(bridge_action)

            comparison = _make_comparison_frame(raw_passthrough, aligned_rgb, step, bridge_action)
            video.append_data(comparison)
            raw_video.append_data(raw_passthrough)
            aligned_video.append_data(aligned_rgb)

            if step % 10 == 0:
                from PIL import Image
                Image.fromarray(comparison).save(frames_dir / f"frame_{step:04d}.png")

            action_tensor = torch.as_tensor(
                np.repeat(isaac_action[None, :], args.num_envs, axis=0),
                dtype=torch.float32, device=device
            )
            env.step(action_tensor)
    finally:
        video.close()
        raw_video.close()
        aligned_video.close()
        env.close()
        simulation_app.close()

    print(f"Video saved to {out_dir}")
    print(f"  comparison.mp4, raw_only.mp4, aligned_only.mp4")
    print(f"  Key frames in {frames_dir}")


if __name__ == "__main__":
    main()
