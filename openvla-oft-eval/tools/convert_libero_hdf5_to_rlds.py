"""Convert one official LIBERO HDF5 file to an RLDS/TFDS dataset.

Usage:
  LIBERO_HDF5=/path/to/demo.hdf5 python -m tools.convert_libero_hdf5_to_rlds \
    --output_dir /mnt/workspace/datasets/rlds
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image


TASK = "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
IMAGE_SIZE = (256, 256)


def resize_rgb(value: np.ndarray) -> np.ndarray:
    """Resize one RGB frame without changing its uint8 representation."""
    if value.shape != (128, 128, 3):
        raise ValueError(f"Expected (128, 128, 3), got {value.shape}")
    return np.asarray(Image.fromarray(value).resize(IMAGE_SIZE, Image.Resampling.BILINEAR), dtype=np.uint8)


class LiberoSpatialTopDrawer(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Official LIBERO top-drawer expert demonstrations."}

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,), dtype=np.uint8
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,), dtype=np.uint8
                                    ),
                                    # Raw input for libero_dataset_transform.
                                    "state": tfds.features.Tensor(
                                        shape=(8,), dtype=np.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,), dtype=np.float32
                            ),
                            "discount": tfds.features.Scalar(np.float32),
                            "is_first": tfds.features.Scalar(np.bool_),
                            "is_last": tfds.features.Scalar(np.bool_),
                            "is_terminal": tfds.features.Scalar(np.bool_),
                            "language_instruction": tfds.features.Text(),
                            "reward": tfds.features.Scalar(np.float32),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "language_instruction": tfds.features.Text(),
                            "source": tfds.features.Text(),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        path = os.environ.get("LIBERO_HDF5")
        if not path:
            raise RuntimeError("Set LIBERO_HDF5 to the official LIBERO .hdf5 file")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)

        with h5py.File(path, "r") as source:
            demos = source["data"]
            names = sorted(demos, key=lambda name: int(name.split("_")[-1]))
            for index, name in enumerate(names):
                demo = demos[name]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                rewards = np.asarray(demo["rewards"], dtype=np.float32)
                dones = np.asarray(demo["dones"], dtype=np.bool_)
                obs = demo["obs"]
                ee_states = np.asarray(obs["ee_states"], dtype=np.float32)
                gripper_states = np.asarray(obs["gripper_states"], dtype=np.float32)
                agentview = np.asarray(obs["agentview_rgb"])
                wrist = np.asarray(obs["eye_in_hand_rgb"])

                n = len(actions)
                shapes = {
                    len(rewards), len(dones), len(ee_states), len(gripper_states),
                    len(agentview), len(wrist),
                }
                if shapes != {n}:
                    raise ValueError(f"Length mismatch in {name}: {shapes}, actions={n}")

                steps = []
                for t in range(n):
                    terminal = bool(dones[t]) or t == n - 1
                    steps.append(
                        {
                            "observation": {
                                "image": resize_rgb(agentview[t]),
                                "wrist_image": resize_rgb(wrist[t]),
                                "state": np.concatenate(
                                    (ee_states[t], gripper_states[t])
                                ).astype(np.float32),
                            },
                            # Official LIBERO HDF5 already uses -1=open,
                            # +1=close; preserve the recorded command.
                            "action": actions[t],
                            "discount": np.float32(1.0),
                            "is_first": t == 0,
                            "is_last": terminal,
                            "is_terminal": terminal,
                            "language_instruction": TASK,
                            "reward": rewards[t],
                        }
                    )

                yield index, {
                    "steps": steps,
                    "episode_metadata": {
                        "language_instruction": TASK,
                        "source": str(path),
                    },
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    os.environ["LIBERO_HDF5"] = args.hdf5
    builder = LiberoSpatialTopDrawer(data_dir=args.output_dir)
    builder.download_and_prepare()
    print(builder.info)


if __name__ == "__main__":
    main()
