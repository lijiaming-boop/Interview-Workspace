"""OpenVLA-OFT adapter plus a controller-only heuristic smoke policy."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class OpenVLAConfig:
    checkpoint: str
    openvla_root: str
    unnorm_key: str = "bridge_orig"
    use_proprio: bool = True
    center_crop: bool = True
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 1
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_diffusion_steps_train: int = 10
    num_diffusion_steps_inference: int = 10
    dump_camera_inputs: bool = False


class OpenVLAAdapter:
    """Thin wrapper around this repository's tested OpenVLA-OFT utilities."""

    def __init__(self, config: OpenVLAConfig) -> None:
        if not config.checkpoint:
            raise ValueError("--checkpoint is required for --policy openvla")
        root = Path(config.openvla_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # The existing project selects ACTION_DIM/normalization by scanning
        # argv. Keep the Bridge contract explicit even if this launcher is
        # renamed or invoked through a wrapper that hides "bridge" in argv.
        if "bridge" not in " ".join(sys.argv).lower():
            sys.argv.append("--bridge")
        utils = importlib.import_module("experiments.robot.openvla_utils")
        self._utils = utils
        self.cfg = SimpleNamespace(**config.__dict__, model_family="openvla")
        self.model = utils.get_vla(self.cfg)
        self.processor = utils.get_processor(self.cfg)
        self.action_head = utils.get_action_head(self.cfg, self.model.llm_dim)
        self.proprio_projector = None
        if config.use_proprio:
            self.proprio_projector = utils.get_proprio_projector(
                self.cfg, self.model.llm_dim, proprio_dim=7
            )

    def act(self, observation: Dict[str, Any]) -> np.ndarray:
        chunk = self._utils.get_vla_action(
            cfg=self.cfg,
            vla=self.model,
            processor=self.processor,
            obs=observation,
            task_label=observation["task_description"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        first = np.asarray(chunk[0], dtype=np.float32).reshape(-1)
        if first.shape != (7,):
            raise RuntimeError(f"OpenVLA returned action shape {first.shape}, expected (7,)")
        return first


class HeuristicPickPlacePolicy:
    """RGB-independent controller used only to validate Isaac Lab plumbing.

    It moves to a fixed pre-grasp, closes, lifts, moves to the target, then
    opens. It is intentionally not used for model metrics.
    """

    def __init__(self, target_xyz: Iterable[float] = (0.68, -0.18, 0.08)) -> None:
        self.target = np.asarray(target_xyz, dtype=np.float32)
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def act(self, observation: Dict[str, Any]) -> np.ndarray:
        state = np.asarray(observation["state"], dtype=np.float32)
        current = state[:3]
        if self.step < 25:
            target = current + np.array([0, 0, 0.08], dtype=np.float32)
            grip = 1.0
        elif self.step < 70:
            target = np.array([0.5, 0.0, 0.12], dtype=np.float32)
            grip = -1.0
        elif self.step < 110:
            target = np.array([0.5, 0.0, 0.32], dtype=np.float32)
            grip = -1.0
        elif self.step < 170:
            target = self.target + np.array([0, 0, 0.18], dtype=np.float32)
            grip = -1.0
        else:
            target = self.target
            grip = 1.0
        self.step += 1
        delta = np.clip(target - current, -0.03, 0.03)
        return np.concatenate([delta, np.zeros(3, dtype=np.float32), [grip]]).astype(np.float32)
