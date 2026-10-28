"""Bridge-vs-Isaac RGB alignment and controlled domain randomization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Conservative ranges measured in image space unless noted otherwise."""

    mode: str = "bridge_aligned"
    crop_aspect: Tuple[int, int] = (4, 3)
    brightness_range: Tuple[float, float] = (0.92, 1.08)
    contrast_range: Tuple[float, float] = (0.90, 1.10)
    saturation_range: Tuple[float, float] = (0.92, 1.08)
    sensor_noise_std: float = 1.5
    jpeg_probability: float = 0.20
    camera_position_jitter_m: float = 0.015
    camera_yaw_jitter_deg: float = 2.0
    light_intensity_range: Tuple[float, float] = (2200.0, 3400.0)
    table_color_range: Tuple[float, float] = (0.32, 0.48)
    object_color_range: Tuple[float, float] = (0.05, 0.15)

    def __post_init__(self) -> None:
        if self.mode not in {"raw", "camera_aligned", "bridge_aligned"}:
            raise ValueError("mode must be raw, camera_aligned, or bridge_aligned")
        if self.mode == "raw":
            return
        if self.sensor_noise_std < 0 or not 0 <= self.jpeg_probability <= 1:
            raise ValueError("noise and probability ranges are invalid")


def center_crop_to_aspect(image: np.ndarray, aspect: Tuple[int, int] = (4, 3)) -> np.ndarray:
    """Center-crop HWC RGB input to Bridge's 4:3 camera composition."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError(f"image must be HWC RGB/RGBA, got {array.shape}")
    height, width = array.shape[:2]
    target_ratio = aspect[0] / aspect[1]
    current_ratio = width / height
    if current_ratio > target_ratio:
        new_width = int(round(height * target_ratio))
        left = (width - new_width) // 2
        array = array[:, left : left + new_width]
    elif current_ratio < target_ratio:
        new_height = int(round(width / target_ratio))
        top = (height - new_height) // 2
        array = array[top : top + new_height]
    return array[..., :3].copy()


def _rgb_to_hsv_like(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = rgb.astype(np.float32) / 255.0
    max_value = values.max(axis=-1)
    min_value = values.min(axis=-1)
    delta = max_value - min_value
    saturation = np.where(max_value > 1e-6, delta / np.maximum(max_value, 1e-6), 0.0)
    return values, max_value, saturation


def apply_rgb_alignment(
    image: np.ndarray,
    config: DomainRandomizationConfig,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply deterministic Bridge composition and conservative appearance jitter.

    The function stays NumPy-only so the exact transform can also be applied
    to recorded Isaac frames during offline domain-gap analysis.
    """

    if config.mode == "raw":
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] not in {3, 4}:
            raise ValueError(f"image must be HWC RGB/RGBA, got {array.shape}")
        return array[..., :3].astype(np.uint8, copy=False)

    array = center_crop_to_aspect(image, config.crop_aspect)

    rng = rng or np.random.default_rng()
    values, max_value, saturation = _rgb_to_hsv_like(array)
    brightness = rng.uniform(*config.brightness_range)
    contrast = rng.uniform(*config.contrast_range)
    values = np.clip(values * brightness, 0.0, 1.0)
    values = np.clip((values - 0.5) * contrast + 0.5, 0.0, 1.0)
    if config.mode == "bridge_aligned":
        sat_scale = rng.uniform(*config.saturation_range)
        gray = values.mean(axis=-1, keepdims=True)
        values = np.clip(gray + (values - gray) * sat_scale, 0.0, 1.0)
        if config.sensor_noise_std:
            values += rng.normal(0.0, config.sensor_noise_std / 255.0, values.shape)
    return np.clip(values * 255.0, 0, 255).astype(np.uint8)


def sample_scene_parameters(
    config: DomainRandomizationConfig, rng: np.random.Generator
) -> Dict[str, Any]:
    """Sample parameters for stage-level randomization and log them per episode."""

    if config.mode != "bridge_aligned":
        return {"mode": config.mode}
    return {
        "mode": config.mode,
        "camera_position_jitter_m": rng.normal(0.0, config.camera_position_jitter_m, 3).tolist(),
        "camera_yaw_jitter_deg": float(rng.uniform(-config.camera_yaw_jitter_deg, config.camera_yaw_jitter_deg)),
        "light_intensity": float(rng.uniform(*config.light_intensity_range)),
        "table_gray": float(rng.uniform(*config.table_color_range)),
        "object_gray": float(rng.uniform(*config.object_color_range)),
    }


def apply_stage_randomization(env: Any, params: Dict[str, Any]) -> None:
    """Best-effort USD hook for stage lighting/material randomization.

    Isaac Lab assets differ slightly between Isaac Sim releases. The function
    deliberately probes attributes and leaves unsupported materials untouched;
    the sampled parameters are still recorded, so a failed hook is visible in
    the run metadata rather than silently changing the evaluation protocol.
    """

    if params.get("mode") != "bridge_aligned":
        return
    try:
        stage = env.sim.stage
        light = stage.GetPrimAtPath("/World/light")
        if light and light.IsValid():
            intensity = light.GetAttribute("inputs:intensity")
            if intensity:
                intensity.Set(float(params["light_intensity"]))
        material_values = {
            "Table": float(params["table_gray"]),
            "Object": float(params["object_gray"]),
        }
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            for token, gray in material_values.items():
                if f"/{token}" not in path:
                    continue
                for attr_name in ("inputs:diffuse_color_constant", "inputs:diffuse_color"):
                    attr = prim.GetAttribute(attr_name)
                    if attr and attr.IsValid():
                        attr.Set((gray, gray, gray))
    except Exception as exc:  # pragma: no cover - depends on Isaac Sim/USD
        setattr(env, "_bridge_stage_randomization_error", repr(exc))
