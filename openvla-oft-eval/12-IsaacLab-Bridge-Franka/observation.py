"""Isaac Lab sensor/state extraction for the Bridge OpenVLA interface."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:  # supports both package imports and isaaclab.sh script imports
    from .domain_randomization import DomainRandomizationConfig, apply_rgb_alignment
except ImportError:  # pragma: no cover - exercised by the standalone launcher
    from domain_randomization import DomainRandomizationConfig, apply_rgb_alignment


def quat_wxyz_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-12)
    if q[0] < 0:
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    scale = angle / max(np.sin(angle / 2.0), 1e-8)
    return (q[1:4] * scale).astype(np.float32)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def camera_rgb(env: Any, camera_name: str = "bridge_camera", env_id: int = 0) -> np.ndarray:
    """Read the first environment's RGB frame from a Camera sensor."""

    camera = env.scene[camera_name]
    output = camera.data.output["rgb"]
    image = _tensor_to_numpy(output)
    if image.ndim == 4:
        image = image[env_id]
    if image.shape[-1] == 4:
        image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _first_eef_pose(env: Any, env_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    frame = env.scene["ee_frame"].data
    pos = _tensor_to_numpy(frame.target_pos_w[env_id, 0])
    quat = _tensor_to_numpy(frame.target_quat_w[env_id, 0])
    # Bridge state is expressed in the local tabletop frame. Isaac Lab stores
    # sensor poses in world coordinates, so remove the vectorized env origin.
    origin = _tensor_to_numpy(env.scene.env_origins[env_id])
    pos = pos - origin
    return pos.astype(np.float32), quat.astype(np.float32)


def _first_gripper_state(env: Any, env_id: int = 0) -> float:
    robot = env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    finger_ids = [i for i, name in enumerate(joint_names) if "finger" in name]
    if not finger_ids:
        raise RuntimeError(f"could not find panda finger joints in {joint_names}")
    width = float(_tensor_to_numpy(robot.data.joint_pos[env_id, finger_ids]).mean())
    # Bridge `full_state` stores the absolute gripper scalar in [0, 1]
    # (closed -> open), matching the dataset transform and its normalization
    # mask. The action adapter uses a separate [-1, +1] command only because
    # Isaac Lab's BinaryJointAction consumes sign-valued inputs.
    return float(np.clip(width / 0.04, 0.0, 1.0))


def make_openvla_observation(
    env: Any,
    task_description: str,
    domain_config: DomainRandomizationConfig,
    *,
    rng: np.random.Generator,
    env_id: int = 0,
) -> Dict[str, Any]:
    """Build the exact dict expected by existing OpenVLA-OFT helpers."""

    raw_rgb = camera_rgb(env, env_id=env_id)
    rgb = apply_rgb_alignment(raw_rgb, domain_config, rng=rng)
    eef_pos, eef_quat = _first_eef_pose(env, env_id=env_id)
    state = np.concatenate(
        [eef_pos, quat_wxyz_to_axis_angle(eef_quat), [_first_gripper_state(env, env_id=env_id)]]
    )
    return {
        "full_image": rgb,
        "state": state.astype(np.float32),
        "task_description": task_description,
        "_raw_rgb": raw_rgb,
    }
