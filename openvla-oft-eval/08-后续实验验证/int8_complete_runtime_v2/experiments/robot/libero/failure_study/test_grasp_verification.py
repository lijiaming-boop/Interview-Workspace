#!/usr/bin/env python3
"""Small check for the simulator-only grasp contact rule."""

from grasp_verification import grasp_contact_is_valid


if __name__ == "__main__":
    assert grasp_contact_is_valid(
        {"two_finger_contact": True, "gripper_target_distance_m": 0.06}, 0.12
    )
    assert not grasp_contact_is_valid(
        {"two_finger_contact": False, "gripper_target_distance_m": 0.06}, 0.12
    )
    assert not grasp_contact_is_valid(
        {"two_finger_contact": True, "gripper_target_distance_m": 0.13}, 0.12
    )
    print("grasp verification rule: OK")
