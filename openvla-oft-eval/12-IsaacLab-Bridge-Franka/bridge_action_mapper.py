"""Bridge 7D EEF action -> Isaac Lab relative IK + binary gripper.

The module intentionally has no Isaac Lab or PyTorch dependency so that the
contract can be tested on the existing OpenVLA workstation before installing
Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class BridgeActionMappingConfig:
    """Units and conventions at the Bridge/Isaac Lab boundary."""

    max_position_delta_m: float = 0.06
    max_rotation_delta_rad: float = 0.35
    bridge_frame: str = "world"
    ik_frame: str = "root"
    gripper_input: str = "absolute_01"
    open_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.max_position_delta_m <= 0 or self.max_rotation_delta_rad <= 0:
            raise ValueError("action limits must be positive")
        if self.bridge_frame not in {"world", "root", "body"}:
            raise ValueError("bridge_frame must be world, root, or body")
        if self.ik_frame not in {"root", "body"}:
            raise ValueError("ik_frame must be root or body")
        if self.gripper_input not in {"absolute_01", "signed_open_positive", "signed_close_positive"}:
            raise ValueError(
                "gripper_input must be absolute_01, signed_open_positive, or signed_close_positive"
            )


def _as_vector(value: Iterable[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def quat_wxyz_to_matrix(quat: Iterable[float]) -> np.ndarray:
    """Return a rotation matrix for an Isaac Lab wxyz quaternion."""

    q = _as_vector(quat, 4, "quat")
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class BridgeActionMapper:
    """Convert one already-unnormalized Bridge action to a 7D env action.

    Isaac Lab's official Franka IK config has a 6D pose action term and a
    separate binary gripper term. The manager concatenates them, so the value
    returned by :meth:`to_isaac_action` is `[arm_6d, gripper_1d]`.
    """

    def __init__(self, config: Optional[BridgeActionMappingConfig] = None) -> None:
        self.config = config or BridgeActionMappingConfig()

    def to_isaac_action(
        self,
        bridge_action: Iterable[float],
        *,
        current_eef_quat_wxyz: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        raw = _as_vector(bridge_action, 7, "bridge_action")
        delta_pos = np.clip(
            raw[:3], -self.config.max_position_delta_m, self.config.max_position_delta_m
        )
        delta_rot = np.clip(
            raw[3:6], -self.config.max_rotation_delta_rad, self.config.max_rotation_delta_rad
        )

        needs_body_conversion = (
            self.config.bridge_frame in {"world", "root"} and self.config.ik_frame == "body"
        ) or (self.config.bridge_frame == "body" and self.config.ik_frame == "root")
        if needs_body_conversion:
            if current_eef_quat_wxyz is None:
                raise ValueError("current_eef_quat_wxyz is required for frame conversion")
            rotation = quat_wxyz_to_matrix(current_eef_quat_wxyz)
            if self.config.bridge_frame in {"world", "root"} and self.config.ik_frame == "body":
                rotation = rotation.T
            delta_pos = rotation @ delta_pos
            delta_rot = rotation @ delta_rot

        if self.config.gripper_input == "absolute_01":
            # This is the encoding emitted by the existing Bridge transform:
            # 0 = closed and 1 = open; the last action dim is not Q01-Q99
            # normalized.
            gripper_is_open = raw[6] >= self.config.open_threshold
        elif self.config.gripper_input == "signed_open_positive":
            gripper_is_open = raw[6] > 0.0
        else:
            gripper_is_open = raw[6] < 0.0
        gripper = 1.0 if gripper_is_open else -1.0
        return np.concatenate([delta_pos, delta_rot, np.array([gripper])]).astype(np.float32)

    @staticmethod
    def to_torch_action(np_action: np.ndarray, device: str = "cuda:0"):
        """Convert to a batched torch tensor without importing torch at module load."""

        import torch

        return torch.as_tensor(np_action, dtype=torch.float32, device=device).reshape(1, 7)
