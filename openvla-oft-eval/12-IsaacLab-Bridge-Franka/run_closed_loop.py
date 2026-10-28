"""Run a closed-loop RGB -> OpenVLA -> IK -> Isaac Lab rollout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge_action_mapper import BridgeActionMapper, BridgeActionMappingConfig
from domain_randomization import (
    DomainRandomizationConfig,
    apply_stage_randomization,
    sample_scene_parameters,
)
from observation import make_openvla_observation
from openvla_policy import HeuristicPickPlacePolicy, OpenVLAAdapter, OpenVLAConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["openvla", "heuristic"], default="openvla")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--openvla_root", default="")
    parser.add_argument("--unnorm_key", default="bridge_orig")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=240)
    parser.add_argument("--open_loop_steps", type=int, default=1)
    parser.add_argument(
        "--domain_mode", choices=["raw", "camera_aligned", "bridge_aligned"], default="bridge_aligned"
    )
    parser.add_argument("--output_dir", default="12-IsaacLab-Bridge-Franka/outputs")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    # Isaac Lab's AppLauncher arguments are added after this parser is created.
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except ImportError:
        parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _make_video_writer(path: Path):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("--video requires imageio in the Isaac Lab environment") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=20)


def _tensor_action(action: np.ndarray, num_envs: int, device: str):
    import torch

    return torch.as_tensor(np.repeat(action[None, :], num_envs, axis=0), dtype=torch.float32, device=device)


def _object_positions(env: Any) -> np.ndarray:
    value = env.scene["object"].data.root_pos_w
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _env_origins(env: Any) -> np.ndarray:
    value = env.scene.env_origins
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _success_state(
    env: Any, ever_lifted: np.ndarray, target_xyz=(0.68, -0.18, 0.055)
) -> tuple[np.ndarray, np.ndarray]:
    positions = _object_positions(env) - _env_origins(env)
    lifted_now = positions[:, 2] > 0.12
    ever_lifted |= lifted_now
    target = np.asarray(target_xyz, dtype=np.float32)
    placed = (np.linalg.norm(positions[:, :2] - target[:2], axis=1) < 0.075) & (
        np.abs(positions[:, 2] - target[2]) < 0.035
    )
    return ever_lifted.copy(), ever_lifted & placed


def main() -> None:
    args = parse_args()
    # AppLauncher must be instantiated before importing/creating the env.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    import gymnasium as gym
    import torch

    import register_env  # noqa: F401

    from bridge_pick_place_env_cfg import BridgePickPlaceEnvCfg, apply_camera_profile

    env_cfg = BridgePickPlaceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    apply_camera_profile(env_cfg, args.domain_mode)
    env = gym.make("Isaac-Bridge-PickPlace-Franka-IK-Rel-v0", cfg=env_cfg)
    device = env.unwrapped.device if hasattr(env.unwrapped, "device") else args.device

    domain_cfg = DomainRandomizationConfig(mode=args.domain_mode)
    mapper = BridgeActionMapper(BridgeActionMappingConfig())
    if args.policy == "openvla":
        policy = OpenVLAAdapter(
            OpenVLAConfig(
                checkpoint=args.checkpoint,
                openvla_root=args.openvla_root,
                unnorm_key=args.unnorm_key,
            )
        )
        task_description = "pick up the cube and place it on the green target"
        policies = [policy] * args.num_envs
    else:
        policies = [HeuristicPickPlacePolicy() for _ in range(args.num_envs)]
        task_description = "pick up the cube and place it on the green target"

    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    video = _make_video_writer(out_dir / "closed_loop_demo.mp4") if args.video else None
    records: List[Dict[str, Any]] = []
    episode_count = 0
    rng = np.random.default_rng(args.seed)
    try:
        while episode_count < args.episodes and simulation_app.is_running():
            obs, _ = env.reset(seed=args.seed + episode_count)
            for policy in policies:
                if hasattr(policy, "reset"):
                    policy.reset()
            num_envs = args.num_envs
            ever_lifted = np.zeros(num_envs, dtype=bool)
            success_seen = np.zeros(num_envs, dtype=bool)
            done = np.zeros(num_envs, dtype=bool)
            scene_params = sample_scene_parameters(domain_cfg, rng)
            apply_stage_randomization(env.unwrapped, scene_params)
            if hasattr(env.unwrapped, "_bridge_stage_randomization_error"):
                scene_params["stage_hook_error"] = getattr(
                    env.unwrapped, "_bridge_stage_randomization_error"
                )
            for step in range(args.max_steps):
                batch_actions = []
                for env_id, policy in enumerate(policies):
                    frame_obs = make_openvla_observation(
                        env.unwrapped,
                        task_description,
                        domain_cfg,
                        rng=rng,
                        env_id=env_id,
                    )
                    if video is not None and env_id == 0:
                        # Record the same RGB tensor that is sent to OpenVLA,
                        # including the selected domain-gap alignment.
                        video.append_data(frame_obs["full_image"])
                    bridge_action = policy.act(frame_obs)
                    current_quat = None
                    if mapper.config.ik_frame == "body":
                        current_quat = (
                            env.unwrapped.scene["ee_frame"]
                            .data.target_quat_w[env_id, 0]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    batch_actions.append(
                        mapper.to_isaac_action(
                            bridge_action, current_eef_quat_wxyz=current_quat
                        )
                    )
                action_tensor = torch.as_tensor(
                    np.asarray(batch_actions), dtype=torch.float32, device=device
                )
                for _ in range(max(1, args.open_loop_steps)):
                    _, _, terminated, truncated, _ = env.step(action_tensor)
                    terminated_np = terminated.detach().cpu().numpy().astype(bool)
                    truncated_np = truncated.detach().cpu().numpy().astype(bool)
                    ever_lifted, success_now = _success_state(env.unwrapped, ever_lifted)
                    # ManagerBasedRLEnv may auto-reset terminated env slots;
                    # keep success as an episode event rather than reading a
                    # possibly reset object pose at the end.
                    success_seen |= success_now
                    done |= terminated_np | truncated_np
                    if done.all():
                        break
                if done.all():
                    break
            success = success_seen
            for env_id in range(num_envs):
                records.append(
                    {
                        "episode": episode_count * num_envs + env_id,
                        "env_id": env_id,
                        "seed": args.seed + episode_count,
                        "domain_mode": args.domain_mode,
                        "success": bool(success[env_id]),
                        "ever_lifted": bool(ever_lifted[env_id]),
                        "failure_stage": (
                            "success"
                            if success[env_id]
                            else "place_or_release"
                            if ever_lifted[env_id]
                            else "grasp"
                        ),
                        "steps": step + 1,
                        "scene_params": scene_params,
                    }
                )
            episode_count += 1
    finally:
        if video is not None:
            video.close()
        env.close()
        simulation_app.close()

    jsonl = out_dir / "episodes.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    success_count = sum(int(record["success"]) for record in records)
    summary = {
        "n": len(records),
        "successes": success_count,
        "success_rate": success_count / max(len(records), 1),
        "ever_lifted_rate": sum(int(r["ever_lifted"]) for r in records) / max(len(records), 1),
        "failure_stage_counts": {
            stage: sum(1 for r in records if r["failure_stage"] == stage)
            for stage in {r["failure_stage"] for r in records}
        },
        "domain_mode": args.domain_mode,
        "policy": args.policy,
        "open_loop_steps": args.open_loop_steps,
        "episodes_jsonl": str(jsonl),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
