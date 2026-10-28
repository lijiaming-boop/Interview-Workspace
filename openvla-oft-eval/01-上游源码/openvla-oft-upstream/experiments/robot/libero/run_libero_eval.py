"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero.failure_study.grasp_verification import (
    grasp_contact_is_valid,
    nearest_named_object,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}
BASE_CONTROL_FREQ = 20


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    wrist_image_weight: float = 1.0                  # Scale wrist-camera patch features (1.0 keeps baseline behavior)

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    target_episodes: str = ""                        # Optional zero-based task:state pairs, e.g. "4:32,6:20,7:27"
    dump_camera_inputs: bool = False                 # Save raw/resized/final camera inputs at every policy replan
    camera_dump_dir: str = "./experiments/camera_dumps"  # Root directory for camera diagnostics
    dump_action_chunks: bool = False                 # Save each predicted 8x7 chunk and executed gripper commands
    action_dump_dir: str = "./experiments/action_dumps"  # Root directory for action/simulator diagnostics
    target_motion_replan_threshold_m: float = 0.0    # Privileged sim ablation; 0 disables target-motion replanning
    verify_grasp: bool = False                       # Simulator-only close/verify/open/retry state machine
    grasp_target_fragment: str = "bowl"             # Object-name fragment used by privileged grasp verification
    grasp_verify_max_distance_m: float = 0.12        # Maximum target-center to gripper-site distance
    grasp_close_hold_steps: int = 2                  # Extra zero-motion close steps before checking contact
    grasp_open_recovery_steps: int = 4               # Zero-motion open steps after an empty grasp
    grasp_max_retries: int = 3                       # Stop the episode after this many empty grasps
    control_freq: int = BASE_CONTROL_FREQ            # LIBERO controller frequency; baseline is 20 Hz

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert 1 <= cfg.num_open_loop_steps <= NUM_ACTIONS_CHUNK, (
        f"num_open_loop_steps must be in [1, {NUM_ACTIONS_CHUNK}], got {cfg.num_open_loop_steps}"
    )
    assert cfg.wrist_image_weight > 0, "wrist_image_weight must be positive!"
    assert cfg.target_motion_replan_threshold_m >= 0, "target_motion_replan_threshold_m must be non-negative!"
    assert cfg.grasp_target_fragment.strip(), "grasp_target_fragment must not be empty!"
    assert cfg.grasp_verify_max_distance_m > 0, "grasp_verify_max_distance_m must be positive!"
    assert cfg.grasp_close_hold_steps > 0, "grasp_close_hold_steps must be positive!"
    assert cfg.grasp_open_recovery_steps > 0, "grasp_open_recovery_steps must be positive!"
    assert cfg.grasp_max_retries > 0, "grasp_max_retries must be positive!"
    assert cfg.control_freq > 0, "control_freq must be positive!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def parse_target_episodes(spec: str):
    """Parse comma-separated zero-based task:state pairs while preserving requested order."""
    if not spec.strip():
        return None

    targets = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            task_id_text, episode_idx_text = item.split(":", maxsplit=1)
            task_id, episode_idx = int(task_id_text), int(episode_idx_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid target_episodes item {item!r}; expected zero-based task:state, e.g. '4:32,6:20'"
            ) from exc
        if task_id < 0 or episode_idx < 0:
            raise ValueError(f"target_episodes values must be non-negative, got {item!r}")
        targets.setdefault(task_id, []).append(episode_idx)
    return targets


def append_jsonl(path: Path, payload) -> None:
    """Append one JSON-serializable record and flush it for crash-safe diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def collect_sim_snapshot(env):
    """Collect privileged object poses and MuJoCo contacts for diagnosis only."""
    wrapped_env = getattr(env, "env", env)
    sim = getattr(wrapped_env, "sim", None)
    snapshot = {"objects": {}, "contacts": []}
    if sim is None:
        return snapshot

    for collection_name in ("objects_dict", "fixtures_dict"):
        collection = getattr(wrapped_env, collection_name, {}) or {}
        for object_name, obj in collection.items():
            body_name = getattr(obj, "root_body", None)
            if body_name is None:
                continue
            try:
                body_id = sim.model.body_name2id(body_name)
                snapshot["objects"][str(object_name)] = {
                    "kind": collection_name,
                    "body_name": str(body_name),
                    "position": np.asarray(sim.data.body_xpos[body_id]).astype(float).tolist(),
                    "quaternion": np.asarray(sim.data.body_xquat[body_id]).astype(float).tolist(),
                }
            except Exception:
                continue

    try:
        for contact_idx in range(int(sim.data.ncon)):
            contact = sim.data.contact[contact_idx]
            geom1 = sim.model.geom_id2name(int(contact.geom1))
            geom2 = sim.model.geom_id2name(int(contact.geom2))
            snapshot["contacts"].append([str(geom1), str(geom2)])
    except Exception:
        pass
    return snapshot


def max_named_object_motion(before, after, name_fragment="bowl"):
    """Return the largest position change for matching objects in two privileged snapshots."""
    before_objects, after_objects = before.get("objects", {}), after.get("objects", {})
    motions = {}
    for name, before_state in before_objects.items():
        if name_fragment.lower() not in name.lower() or name not in after_objects:
            continue
        before_pos = np.asarray(before_state["position"], dtype=float)
        after_pos = np.asarray(after_objects[name]["position"], dtype=float)
        motions[name] = float(np.linalg.norm(after_pos - before_pos))
    if not motions:
        return 0.0, None
    moved_name = max(motions, key=motions.get)
    return motions[moved_name], moved_name


def collect_sim_grasp_state(env, target_name):
    """Return robosuite's two-finger grasp label and gripper-to-target distance."""
    wrapped_env = getattr(env, "env", env)
    target = (getattr(wrapped_env, "objects_dict", {}) or {}).get(target_name)
    robots = getattr(wrapped_env, "robots", [])
    if target is None or not robots:
        return None
    try:
        gripper = robots[0].gripper
        return {
            "two_finger_contact": bool(wrapped_env._check_grasp(gripper=gripper, object_geoms=target)),
            "gripper_target_distance_m": float(
                wrapped_env._gripper_to_target(gripper, target, return_distance=True)
            ),
        }
    except Exception:
        return None


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Apply an explicit feature-scale ablation to non-primary (wrist) camera patch embeddings.
    if hasattr(model.vision_backbone, "set_wrist_image_weight"):
        model.vision_backbone.set_wrist_image_weight(cfg.wrist_image_weight)
    elif cfg.wrist_image_weight != 1.0:
        raise AttributeError("This vision backbone does not support wrist_image_weight")

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "_camera_debug": {
            "raw_full_image": img,
            "raw_wrist_image": wrist_img,
        },
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    task_id: int = -1,
    episode_idx: int = -1,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    # Keep an unbounded queue and explicitly enqueue the FIRST N actions below. Using
    # deque(maxlen=N).extend(all_8_actions) would silently retain the LAST N actions.
    action_queue = deque()

    # Setup
    t = 0
    replay_images = []
    frequency_scale = cfg.control_freq / BASE_CONTROL_FREQ
    max_steps = int(round(TASK_MAX_STEPS[cfg.task_suite_name] * frequency_scale))
    wait_steps = int(round(cfg.num_steps_wait * frequency_scale))
    replan_idx = 0
    active_chunk_idx = -1
    chunk_start_snapshot = None
    diagnostic_dir = Path(cfg.action_dump_dir) / f"task_{task_id:02d}" / f"state_{episode_idx:02d}"
    chunks_path = diagnostic_dir / "chunks.jsonl"
    steps_path = diagnostic_dir / "steps.jsonl"
    events_path = diagnostic_dir / "events.jsonl"
    if cfg.dump_action_chunks:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        for diagnostic_path in (chunks_path, steps_path, events_path):
            diagnostic_path.write_text("", encoding="utf-8")
    forced_replans = 0
    grasp_checks = 0
    empty_grasp_replans = 0
    previous_gripper_command = -1.0
    grasp_phase = None
    grasp_steps_remaining = 0
    grasp_target_name = None
    grasp_close_chunk_idx = -1
    grasp_close_action_idx = -1
    grasp_discarded_actions = 0
    grasp_retry_count = 0
    grasp_retry_exhausted = False

    # Run episode
    success = False
    try:
        while t < max_steps + wait_steps:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < wait_steps:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if grasp_phase is None and len(action_queue) == 0:
                if cfg.dump_camera_inputs:
                    dump_dir = Path(cfg.camera_dump_dir) / f"task_{task_id:02d}" / f"state_{episode_idx:02d}"
                    observation["_camera_debug"].update(
                        {
                            "dump_dir": str(dump_dir),
                            "prefix": f"replan_{replan_idx:03d}_step_{t - wait_steps:03d}",
                            "task_description": task_description,
                            "task_id": task_id,
                            "episode_idx": episode_idx,
                        }
                    )
                # Query model to get action
                inference_started = time.perf_counter()
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
                actions_array = np.asarray(actions, dtype=float)
                if actions_array.ndim != 2 or actions_array.shape[1] != 7:
                    raise ValueError(f"Expected action chunk shaped (N, 7), got {actions_array.shape}")
                queued_count = min(len(actions), cfg.num_open_loop_steps)
                action_queue.extend((action_idx, actions[action_idx]) for action_idx in range(queued_count))
                active_chunk_idx = replan_idx
                chunk_start_snapshot = collect_sim_snapshot(env)
                if cfg.dump_action_chunks:
                    processed_actions = [
                        process_action(np.asarray(chunk_action).copy(), cfg.model_family).astype(float).tolist()
                        for chunk_action in actions
                    ]
                    append_jsonl(
                        chunks_path,
                        {
                            "chunk_index": replan_idx,
                            "control_step": t - wait_steps,
                            "sim_time_seconds": (t - wait_steps) / cfg.control_freq,
                            "control_freq": cfg.control_freq,
                            "inference_latency_ms": inference_latency_ms,
                            "chunk_deadline_ms": cfg.num_open_loop_steps / cfg.control_freq * 1000.0,
                            "chunk_deadline_met": inference_latency_ms
                            <= cfg.num_open_loop_steps / cfg.control_freq * 1000.0,
                            "predicted_shape": list(actions_array.shape),
                            "open_loop_steps": cfg.num_open_loop_steps,
                            "executed_action_indices_planned": list(range(queued_count)),
                            "raw_actions": actions_array.tolist(),
                            "processed_actions": processed_actions,
                            "raw_gripper_commands": actions_array[:, -1].tolist(),
                            "processed_gripper_commands": [action[-1] for action in processed_actions],
                            "proprio": np.asarray(observation["state"], dtype=float).tolist(),
                            "sim_snapshot_before_chunk": chunk_start_snapshot,
                        },
                    )
                log_message(
                    f"Replan {replan_idx}: predicted {len(actions)} actions; "
                    f"queued first {min(len(actions), cfg.num_open_loop_steps)}; "
                    f"inference {inference_latency_ms:.1f} ms",
                    log_file,
                )
                replan_idx += 1

            intervention_phase = grasp_phase
            if intervention_phase is not None:
                action_idx_in_chunk = -1
                action = np.zeros(7, dtype=float)
                action[-1] = 1.0 if intervention_phase == "close_hold" else -1.0
            else:
                action_idx_in_chunk, action = action_queue.popleft()
                action = process_action(np.asarray(action).copy(), cfg.model_family)

                if cfg.verify_grasp and previous_gripper_command < 0 <= action[-1]:
                    close_snapshot = collect_sim_snapshot(env)
                    grasp_target_name = nearest_named_object(
                        close_snapshot, obs["robot0_eef_pos"], cfg.grasp_target_fragment
                    )
                    if grasp_target_name is not None:
                        grasp_close_chunk_idx = active_chunk_idx
                        grasp_close_action_idx = action_idx_in_chunk
                        grasp_discarded_actions = len(action_queue)
                        action_queue.clear()
                        grasp_phase = "close_hold"
                        intervention_phase = "close_hold"
                        # The current policy close is executed once; the configured zero-motion
                        # holds follow it so a final approach encoded with close is not discarded.
                        grasp_steps_remaining = cfg.grasp_close_hold_steps + 1
                        event = {
                            "event": "grasp_close_intercepted",
                            "target_name": grasp_target_name,
                            "chunk_index": active_chunk_idx,
                            "action_index_in_chunk": action_idx_in_chunk,
                            "control_step": t - wait_steps,
                            "discarded_queued_actions": grasp_discarded_actions,
                            "close_hold_steps": cfg.grasp_close_hold_steps,
                            "retry_number": grasp_retry_count + 1,
                        }
                        if cfg.dump_action_chunks:
                            append_jsonl(events_path, event)
                        log_message(
                            f"Intercepted close for {grasp_target_name}; discarded "
                            f"{grasp_discarded_actions} queued actions and holding closed for "
                            f"{cfg.grasp_close_hold_steps} extra steps",
                            log_file,
                        )

            if cfg.dump_action_chunks:
                append_jsonl(
                    steps_path,
                    {
                        "chunk_index": active_chunk_idx,
                        "action_index_in_chunk": action_idx_in_chunk,
                        "control_step": t - wait_steps,
                        "sim_time_seconds": (t - wait_steps) / cfg.control_freq,
                        "processed_action": np.asarray(action, dtype=float).tolist(),
                        "queue_remaining_before_step": len(action_queue),
                        "simulation_intervention": intervention_phase is not None,
                        "intervention_phase": intervention_phase,
                    },
                )

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            previous_gripper_command = float(action[-1])
            if intervention_phase is not None:
                grasp_steps_remaining -= 1
                if grasp_steps_remaining == 0 and intervention_phase == "close_hold":
                    grasp_state = collect_sim_grasp_state(env, grasp_target_name)
                    if grasp_state is None:
                        log_message(
                            f"Grasp contact check unavailable for {grasp_target_name}; disabling intervention",
                            log_file,
                        )
                        grasp_phase = None
                        grasp_target_name = None
                    else:
                        verified = grasp_contact_is_valid(grasp_state, cfg.grasp_verify_max_distance_m)
                        grasp_checks += 1
                        attempt_number = grasp_retry_count + 1
                        event = {
                            "event": "grasp_verified" if verified else "empty_grasp_forced_replan",
                            "target_name": grasp_target_name,
                            "close_chunk_index": grasp_close_chunk_idx,
                            "close_action_index_in_chunk": grasp_close_action_idx,
                            "control_step": t - wait_steps,
                            "discarded_queued_actions": grasp_discarded_actions,
                            "retry_number": attempt_number,
                            "verified": verified,
                            **grasp_state,
                        }
                        if cfg.dump_action_chunks:
                            append_jsonl(events_path, event)
                        log_message(
                            f"Grasp contact check for {grasp_target_name}: verified={verified}, "
                            f"two_finger_contact={grasp_state['two_finger_contact']}, "
                            f"distance={grasp_state['gripper_target_distance_m']:.4f} m, "
                            f"attempt={attempt_number}/{cfg.grasp_max_retries}",
                            log_file,
                        )
                        if verified:
                            grasp_phase = None
                            grasp_target_name = None
                            grasp_retry_count = 0
                        else:
                            forced_replans += 1
                            empty_grasp_replans += 1
                            grasp_retry_count += 1
                            if grasp_retry_count >= cfg.grasp_max_retries:
                                grasp_retry_exhausted = True
                                exhausted_event = {
                                    "event": "grasp_retry_exhausted",
                                    "target_name": grasp_target_name,
                                    "control_step": t - wait_steps,
                                    "retry_count": grasp_retry_count,
                                }
                                if cfg.dump_action_chunks:
                                    append_jsonl(events_path, exhausted_event)
                                log_message(
                                    f"Grasp recovery exhausted after {grasp_retry_count} attempts; ending episode",
                                    log_file,
                                )
                                break
                            grasp_phase = "open_hold"
                            grasp_steps_remaining = cfg.grasp_open_recovery_steps
                            log_message(
                                f"Empty grasp detected; opening in place for "
                                f"{cfg.grasp_open_recovery_steps} steps before replanning",
                                log_file,
                            )
                elif grasp_steps_remaining == 0 and intervention_phase == "open_hold":
                    if cfg.dump_action_chunks:
                        append_jsonl(
                            events_path,
                            {
                                "event": "grasp_recovery_opened",
                                "target_name": grasp_target_name,
                                "control_step": t - wait_steps,
                                "retry_count": grasp_retry_count,
                            },
                        )
                    log_message("Recovery open-hold complete; replanning from the new observation", log_file)
                    grasp_phase = None
                    grasp_target_name = None
                    action_queue.clear()
            if cfg.target_motion_replan_threshold_m > 0 and action[-1] < 0 and chunk_start_snapshot is not None:
                current_snapshot = collect_sim_snapshot(env)
                target_motion, moved_target = max_named_object_motion(chunk_start_snapshot, current_snapshot, "bowl")
                if target_motion >= cfg.target_motion_replan_threshold_m and len(action_queue) > 0:
                    discarded_actions = len(action_queue)
                    action_queue.clear()
                    forced_replans += 1
                    event = {
                        "event": "target_motion_forced_replan",
                        "chunk_index": active_chunk_idx,
                        "action_index_in_chunk": action_idx_in_chunk,
                        "control_step": t - wait_steps,
                        "target_name": moved_target,
                        "target_motion_m": target_motion,
                        "threshold_m": cfg.target_motion_replan_threshold_m,
                        "discarded_queued_actions": discarded_actions,
                        "gripper_command": float(action[-1]),
                    }
                    if cfg.dump_action_chunks:
                        append_jsonl(events_path, event)
                    log_message(
                        f"Target moved {target_motion:.4f} m while gripper was open; "
                        f"discarded {discarded_actions} queued actions and will replan",
                        log_file,
                    )
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    if cfg.dump_action_chunks:
        summary = {
            "task_id": task_id,
            "initial_state_index": episode_idx,
            "task_description": task_description,
            "success": success,
            "control_freq": cfg.control_freq,
            "num_open_loop_steps": cfg.num_open_loop_steps,
            "target_motion_replan_threshold_m": cfg.target_motion_replan_threshold_m,
            "verify_grasp": cfg.verify_grasp,
            "grasp_check_count": grasp_checks,
            "empty_grasp_replan_count": empty_grasp_replans,
            "grasp_retry_count": grasp_retry_count,
            "grasp_retry_exhausted": grasp_retry_exhausted,
            "replan_count": replan_idx,
            "forced_replan_count": forced_replans,
            "executed_control_steps": max(0, t - wait_steps),
            "simulated_duration_seconds": max(0, t - wait_steps) / cfg.control_freq,
        }
        (diagnostic_dir / "episode_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    episode_indices=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(
        task, cfg.model_family, resolution=cfg.env_img_res, control_freq=cfg.control_freq
    )

    # Start episodes
    task_episodes, task_successes = 0, 0
    if episode_indices is None:
        episode_indices = list(range(cfg.num_trials_per_task))
    for episode_idx in tqdm.tqdm(episode_indices):
        if episode_idx >= len(initial_states):
            raise IndexError(
                f"Task {task_id} requested initial-state index {episode_idx}, but only {len(initial_states)} exist"
            )
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(
            f"Starting targeted task_id={task_id}, initial_state_index={episode_idx} "
            f"(task trial {episode_idx + 1})...",
            log_file,
        )

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            task_id,
            episode_idx,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    target_episodes = parse_target_episodes(cfg.target_episodes)

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Open-loop execution steps: {cfg.num_open_loop_steps}/{NUM_ACTIONS_CHUNK}", log_file)
    log_message(f"Wrist image feature weight: {cfg.wrist_image_weight}", log_file)
    log_message(f"Target episodes (zero-based task:state): {target_episodes or 'ALL'}", log_file)
    log_message(f"Camera input dump: {cfg.dump_camera_inputs} -> {cfg.camera_dump_dir}", log_file)
    log_message(f"Action chunk dump: {cfg.dump_action_chunks} -> {cfg.action_dump_dir}", log_file)
    log_message(f"Target-motion replan threshold: {cfg.target_motion_replan_threshold_m} m", log_file)
    log_message(
        f"Privileged grasp verification: {cfg.verify_grasp} "
        f"(target={cfg.grasp_target_fragment!r}, close_hold={cfg.grasp_close_hold_steps}, "
        f"open_hold={cfg.grasp_open_recovery_steps}, retries={cfg.grasp_max_retries}, "
        f"max_distance={cfg.grasp_verify_max_distance_m} m)",
        log_file,
    )
    log_message(f"LIBERO control frequency: {cfg.control_freq} Hz", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_ids = list(target_episodes.keys()) if target_episodes is not None else list(range(num_tasks))
    for task_id in tqdm.tqdm(task_ids):
        if task_id >= num_tasks:
            raise IndexError(f"Requested task_id {task_id}, but suite only has {num_tasks} tasks")
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            target_episodes[task_id] if target_episodes is not None else None,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
