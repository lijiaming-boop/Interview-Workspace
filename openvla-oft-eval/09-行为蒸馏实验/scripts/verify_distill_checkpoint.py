import argparse
import json
import re
from pathlib import Path

import torch


def newest(root: Path, component: str) -> Path:
    files = list(root.glob(f"{component}--*_checkpoint.pt"))
    if not files:
        raise FileNotFoundError(f"{root}: no {component} checkpoint")
    return max(files, key=lambda p: int(re.search(r"--(\d+)_checkpoint", p.name).group(1)))


def load(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key.removeprefix("module."): value.float() for key, value in state.items()}


parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True, type=Path)
parser.add_argument("--candidate", required=True, type=Path)
parser.add_argument("--max-relative-fc2-drift", type=float, default=0.02)
args = parser.parse_args()

base_path = newest(args.base, "action_head")
candidate_path = args.candidate if args.candidate.is_file() else newest(args.candidate, "action_head")
base, candidate = load(base_path), load(candidate_path)
if base.keys() != candidate.keys():
    raise SystemExit("FAIL: state_dict keys differ")

upstream = [key for key in base if not key.startswith("model.fc2.")]
upstream_diff = max((candidate[key] - base[key]).abs().max().item() for key in upstream)
gripper_weight_diff = (candidate["model.fc2.weight"][6] - base["model.fc2.weight"][6]).abs().max().item()
gripper_bias_diff = (candidate["model.fc2.bias"][6] - base["model.fc2.bias"][6]).abs().max().item()
motion_delta = candidate["model.fc2.weight"][:6] - base["model.fc2.weight"][:6]
relative_fc2_drift = motion_delta.norm().item() / max(base["model.fc2.weight"][:6].norm().item(), 1e-12)

report = {
    "base": str(base_path),
    "candidate": str(candidate_path),
    "upstream_max_abs_diff": upstream_diff,
    "gripper_weight_max_abs_diff": gripper_weight_diff,
    "gripper_bias_abs_diff": gripper_bias_diff,
    "relative_motion_fc2_drift": relative_fc2_drift,
}
print(json.dumps(report, indent=2, ensure_ascii=False))

passed = (
    upstream_diff == 0.0
    and gripper_weight_diff == 0.0
    and gripper_bias_diff == 0.0
    and relative_fc2_drift <= args.max_relative_fc2_drift
)
print("PASS" if passed else "FAIL")
raise SystemExit(0 if passed else 2)
