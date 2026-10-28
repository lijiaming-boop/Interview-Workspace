"""Privileged simulator-only grasp verification helpers."""

import numpy as np


def nearest_named_object(snapshot, point, name_fragment="bowl"):
    """Return the matching object nearest to an end-effector position."""
    point = np.asarray(point, dtype=float)
    candidates = {
        name: float(np.linalg.norm(np.asarray(state["position"], dtype=float) - point))
        for name, state in snapshot.get("objects", {}).items()
        if name_fragment.lower() in name.lower()
    }
    return min(candidates, key=candidates.get) if candidates else None


def grasp_contact_is_valid(grasp_state, max_distance):
    """Require robosuite's two-finger contact and a bounded target distance."""
    return bool(
        grasp_state
        and grasp_state["two_finger_contact"]
        and grasp_state["gripper_target_distance_m"] <= max_distance
    )
