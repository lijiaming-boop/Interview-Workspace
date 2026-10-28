import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge_action_mapper import BridgeActionMapper, BridgeActionMappingConfig


class ActionMapperTest(unittest.TestCase):
    def test_metric_action_and_gripper_sign(self):
        mapper = BridgeActionMapper()
        result = mapper.to_isaac_action([0.2, -0.2, 0.01, 0.4, 0.0, -0.4, 1.0])
        np.testing.assert_allclose(result, [0.06, -0.06, 0.01, 0.35, 0.0, -0.35, 1.0])
        closed = mapper.to_isaac_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(closed[-1], -1.0)

    def test_body_to_root_frame(self):
        mapper = BridgeActionMapper(
            BridgeActionMappingConfig(bridge_frame="body", ik_frame="root")
        )
        # 90 degrees around z: body x maps to root y.
        quat_z_90 = [0.70710678, 0.0, 0.0, 0.70710678]
        result = mapper.to_isaac_action(
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            current_eef_quat_wxyz=quat_z_90,
        )
        np.testing.assert_allclose(result[:3], [0.0, 0.01, 0.0], atol=1e-6)
        self.assertEqual(result[-1], -1.0)

    def test_frame_conversion_requires_pose(self):
        mapper = BridgeActionMapper(
            BridgeActionMappingConfig(bridge_frame="world", ik_frame="body")
        )
        with self.assertRaises(ValueError):
            mapper.to_isaac_action(np.zeros(7))


if __name__ == "__main__":
    unittest.main()
