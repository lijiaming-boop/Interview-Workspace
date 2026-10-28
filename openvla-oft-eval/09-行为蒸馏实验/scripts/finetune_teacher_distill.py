"""Conservative teacher-distillation for the OpenVLA-OFT L1 action head.

The VLA backbone, proprio projector, and frozen teacher head come from the
official LIBERO-Spatial checkpoint.  The student starts from the same head and
only the first six rows of the final linear layer receive gradients.  The
gripper row and every upstream parameter remain bit-identical to the teacher.
"""

import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import draccus
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

# Import the RLDS stack before the first CUDA operation.  OpenVLA disables the
# TensorFlow GPU here; importing it later causes "Visible devices cannot be
# modified after being initialized".
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


@dataclass
class Config:
    base_checkpoint: str = "/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial"
    data_root_dir: str = "/mnt/workspace/datasets/rlds_head_only_clean_20260906"
    dataset_name: str = "libero_spatial_top_drawer_expert"
    base_stats_key: str = "libero_spatial_no_noops"
    output_dir: str = "/mnt/workspace/runs/topdrawer_teacher_distill_v1"
    num_images_in_input: int = 2
    use_proprio: bool = True
    image_aug: bool = False
    batch_size: int = 1
    grad_accumulation_steps: int = 4
    learning_rate: float = 2e-6
    max_steps: int = 100
    save_steps: str = "25,50,100"
    shuffle_buffer_size: int = 1000
    expert_weight: float = 1.0
    kd_weight: float = 2.0
    correction_kd_weight: float = 0.25
    correction_threshold: float = 0.12
    parameter_anchor_weight: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 7


def _rebase_bounds_q99(array: np.ndarray, source: dict, target: dict) -> np.ndarray:
    """Convert q99-normalized values from source statistics to target statistics."""
    value = np.asarray(array, dtype=np.float32)
    source_low = np.asarray(source["q01"], dtype=np.float32)
    source_high = np.asarray(source["q99"], dtype=np.float32)
    target_low = np.asarray(target["q01"], dtype=np.float32)
    target_high = np.asarray(target["q99"], dtype=np.float32)
    target_min = np.asarray(target["min"], dtype=np.float32)
    target_max = np.asarray(target["max"], dtype=np.float32)
    source_mask = np.asarray(source.get("mask", np.ones_like(source_low)), dtype=bool)
    target_mask = np.asarray(target.get("mask", np.ones_like(target_low)), dtype=bool)

    raw = np.where(
        source_mask,
        (value + 1.0) * 0.5 * (source_high - source_low) + source_low,
        value,
    )
    normalized = np.clip(2.0 * (raw - target_low) / (target_high - target_low + 1e-8) - 1.0, -1.0, 1.0)
    normalized = np.where(target_mask, normalized, raw)
    normalized = np.where(target_min == target_max, 0.0, normalized)
    return normalized.astype(np.float32)


class RebasedRLDSBatchTransform(RLDSBatchTransform):
    """Build prompts and tensors only after moving actions/proprio to base coordinates."""

    def __init__(self, *args, target_stats: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_stats = None
        self.target_stats = target_stats

    def set_source_stats(self, source_stats: dict) -> None:
        self.source_stats = source_stats
        action_mask = np.asarray(source_stats["action"].get("mask", [True] * ACTION_DIM), dtype=bool)
        target_mask = np.asarray(self.target_stats["action"].get("mask", [True] * ACTION_DIM), dtype=bool)
        if action_mask.shape != (ACTION_DIM,) or target_mask.shape != (ACTION_DIM,):
            raise ValueError("unexpected action normalization mask")
        if action_mask[-1] or target_mask[-1]:
            raise ValueError("gripper must be excluded from q99 normalization")

    def __call__(self, rlds_batch):
        if self.source_stats is None:
            raise RuntimeError("source dataset statistics were not initialized")
        sample = dict(rlds_batch)
        sample["action"] = _rebase_bounds_q99(
            sample["action"], self.source_stats["action"], self.target_stats["action"]
        )
        if "proprio" in sample["observation"]:
            observation = dict(sample["observation"])
            observation["proprio"] = _rebase_bounds_q99(
                observation["proprio"], self.source_stats["proprio"], self.target_stats["proprio"]
            )
            sample["observation"] = observation
        return super().__call__(sample)


def _checkpoint(root: str, component: str) -> Path:
    files = list(Path(root).glob(f"{component}--*_checkpoint.pt"))
    if not files:
        raise FileNotFoundError(f"{root}: no {component} checkpoint")

    def step(path: Path) -> int:
        match = re.search(r"--(\d+)_checkpoint\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(files, key=step)


def _state_dict(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key.removeprefix("module."): value for key, value in state.items()}


def _save(student, output_dir: Path, step: int, cfg: Config, metrics: dict) -> None:
    path = output_dir / f"action_head--{step}_checkpoint.pt"
    torch.save(student.state_dict(), path)
    payload = {"step": step, **metrics}
    with (output_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"[SAVE] {path}", flush=True)


@draccus.wrap()
def main(cfg: Config) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if cfg.grad_accumulation_steps < 1 or cfg.max_steps < 1:
        raise ValueError("steps must be positive")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "distill_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    stats_path = Path(cfg.base_checkpoint) / "dataset_statistics.json"
    all_base_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if cfg.base_stats_key not in all_base_stats:
        raise KeyError(f"{cfg.base_stats_key!r} not found in {stats_path}")
    base_stats = all_base_stats[cfg.base_stats_key]

    processor = AutoProcessor.from_pretrained(cfg.base_checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.base_checkpoint,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.requires_grad_(False).eval()

    llm_dim = vla.llm_dim
    projector = None
    if cfg.use_proprio:
        projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)
        projector.load_state_dict(_state_dict(_checkpoint(cfg.base_checkpoint, "proprio_projector")))
        projector = projector.to(device=device, dtype=dtype)
        projector.requires_grad_(False).eval()

    teacher = L1RegressionActionHead(llm_dim, llm_dim, ACTION_DIM)
    teacher_state = _state_dict(_checkpoint(cfg.base_checkpoint, "action_head"))
    teacher.load_state_dict(teacher_state)
    teacher = teacher.to(device=device, dtype=dtype)
    teacher.requires_grad_(False).eval()

    student = L1RegressionActionHead(llm_dim, llm_dim, ACTION_DIM)
    student.load_state_dict(teacher_state)
    # Keep the trainable student in FP32.  At lr=2e-6, BF16 parameter updates
    # can fall below one representable ULP and silently become no-ops.
    student = student.to(device=device, dtype=torch.float32)
    student.requires_grad_(False)
    student.model.fc2.weight.requires_grad_(True)
    student.model.fc2.bias.requires_grad_(True)
    student.train()

    # Row 6 is the gripper.  Zeroing that row's gradients guarantees that its
    # weights and bias remain exactly equal to the base head.
    student.model.fc2.weight.register_hook(
        lambda grad: torch.cat((grad[:6], torch.zeros_like(grad[6:])), dim=0)
    )
    student.model.fc2.bias.register_hook(
        lambda grad: torch.cat((grad[:6], torch.zeros_like(grad[6:])), dim=0)
    )
    initial_weight = student.model.fc2.weight.detach().float().clone()
    initial_bias = student.model.fc2.bias.detach().float().clone()

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RebasedRLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=cfg.num_images_in_input > 1,
        use_proprio=cfg.use_proprio,
        target_stats=base_stats,
    )
    dataset = RLDSDataset(
        Path(cfg.data_root_dir),
        cfg.dataset_name,
        transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    source_stats_map = dataset.dataset_statistics
    if len(source_stats_map) != 1:
        raise ValueError(f"expected one source dataset, got {list(source_stats_map)}")
    source_name, source_stats = next(iter(source_stats_map.items()))
    transform.set_source_stats(source_stats)
    print(f"[STATS] source={source_name} -> target={cfg.base_stats_key}", flush=True)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0, collate_fn=collator)

    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        num_patches += 1

    params = [student.model.fc2.weight, student.model.fc2.bias]
    optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    save_steps = {int(item) for item in cfg.save_steps.split(",") if item.strip()}
    if cfg.max_steps not in save_steps:
        save_steps.add(cfg.max_steps)

    update_step = 0
    running = {"loss": 0.0, "expert_l1": 0.0, "kd_l1": 0.0, "anchor": 0.0, "teacher_l1": 0.0}
    running_count = 0

    for batch_idx, batch in enumerate(loader):
        ground_truth = batch["actions"].to(device=device, dtype=dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            output = vla(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device=device, dtype=dtype),
                labels=batch["labels"].to(device),
                output_hidden_states=True,
                proprio=batch["proprio"].to(device) if cfg.use_proprio else None,
                proprio_projector=projector,
                use_film=False,
            )
            labels = batch["labels"][:, 1:].to(device)
            mask = get_current_action_mask(labels) | get_next_actions_mask(labels)
            text_hidden = output.hidden_states[-1][:, num_patches:-1]
            hidden = text_hidden[mask].reshape(
                batch["input_ids"].shape[0], NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            ).to(dtype)
            teacher_action = teacher.predict_action(hidden)

        with torch.autocast("cuda", enabled=False):
            student_action = student.predict_action(hidden.detach().float())
            # Gripper prediction is excluded from the trainable objective.
            student_motion = student_action[..., :6]
            teacher_motion = teacher_action[..., :6]
            expert_motion = ground_truth[..., :6]
            expert_l1 = F.l1_loss(student_motion.float(), expert_motion.float())
            teacher_error = (teacher_motion.float() - expert_motion.float()).abs().mean(dim=-1, keepdim=True)
            kd_scale = torch.where(
                teacher_error > cfg.correction_threshold,
                torch.full_like(teacher_error, cfg.correction_kd_weight),
                torch.full_like(teacher_error, cfg.kd_weight),
            )
            kd_l1 = (kd_scale * (student_motion.float() - teacher_motion.float()).abs()).mean()
            weight_delta = student.model.fc2.weight[:6].float() - initial_weight[:6]
            bias_delta = student.model.fc2.bias[:6].float() - initial_bias[:6]
            anchor = weight_delta.square().mean() + bias_delta.square().mean()
            loss = (
                cfg.expert_weight * expert_l1
                + kd_l1
                + cfg.parameter_anchor_weight * anchor
            )

        (loss / cfg.grad_accumulation_steps).backward()
        values = {
            "loss": loss.item(),
            "expert_l1": expert_l1.item(),
            "kd_l1": kd_l1.item(),
            "anchor": anchor.item(),
            "teacher_l1": F.l1_loss(teacher_motion.float(), expert_motion.float()).item(),
        }
        for key, value in values.items():
            running[key] += value
        running_count += 1

        if (batch_idx + 1) % cfg.grad_accumulation_steps != 0:
            continue

        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

        mean = {key: value / running_count for key, value in running.items()}
        if update_step == 1 or update_step % 10 == 0:
            print(
                f"step={update_step:4d} loss={mean['loss']:.5f} "
                f"expert={mean['expert_l1']:.5f} kd={mean['kd_l1']:.5f} "
                f"teacher={mean['teacher_l1']:.5f}",
                flush=True,
            )
        if update_step in save_steps:
            _save(student, output_dir, update_step, cfg, mean)
        running = {key: 0.0 for key in running}
        running_count = 0
        if update_step >= cfg.max_steps:
            break

    if update_step != cfg.max_steps:
        raise RuntimeError(f"dataset ended at update {update_step}, expected {cfg.max_steps}")

    # Hard safety checks before declaring the run usable.
    state = {key: value.detach().cpu() for key, value in student.state_dict().items()}
    unchanged = [key for key in state if not key.startswith("model.fc2.")]
    max_upstream_diff = max((state[key].float() - teacher_state[key].float()).abs().max().item() for key in unchanged)
    gripper_weight_diff = (state["model.fc2.weight"][6] - teacher_state["model.fc2.weight"][6]).abs().max().item()
    gripper_bias_diff = (state["model.fc2.bias"][6] - teacher_state["model.fc2.bias"][6]).abs().max().item()
    print(f"[VERIFY] upstream_max_diff={max_upstream_diff:.9g}")
    print(f"[VERIFY] gripper_weight_max_diff={gripper_weight_diff:.9g}")
    print(f"[VERIFY] gripper_bias_max_diff={gripper_bias_diff:.9g}")
    if max_upstream_diff != 0.0 or gripper_weight_diff != 0.0 or gripper_bias_diff != 0.0:
        raise RuntimeError("frozen-parameter invariant was violated")


if __name__ == "__main__":
    main()
