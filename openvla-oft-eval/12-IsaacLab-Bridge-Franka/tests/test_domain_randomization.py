import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain_randomization import (
    DomainRandomizationConfig,
    apply_rgb_alignment,
    center_crop_to_aspect,
)


class DomainRandomizationTest(unittest.TestCase):
    def test_center_crop_is_4_by_3(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        cropped = center_crop_to_aspect(image)
        self.assertEqual(cropped.shape, (100, 133, 3))

    def test_raw_mode_is_deterministic(self):
        image = np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3)
        config = DomainRandomizationConfig(mode="raw")
        first = apply_rgb_alignment(image, config, rng=np.random.default_rng(1))
        second = apply_rgb_alignment(image, config, rng=np.random.default_rng(2))
        np.testing.assert_array_equal(first, second)

    def test_aligned_mode_preserves_image_contract(self):
        image = np.full((80, 120, 3), 128, dtype=np.uint8)
        config = DomainRandomizationConfig(mode="bridge_aligned")
        aligned = apply_rgb_alignment(image, config, rng=np.random.default_rng(3))
        self.assertEqual(aligned.shape, (80, 107, 3))
        self.assertEqual(aligned.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
