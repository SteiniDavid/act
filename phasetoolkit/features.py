"""Environment-relative features for phase detection and tracking."""

from __future__ import annotations

import numpy as np


def identity_features(
    observations: np.ndarray,
) -> np.ndarray:
    """Identity transform - return observations/actions as-is.

    This is the default transform for clustering directly on actions or
    observations without any task-specific feature engineering. This is
    appropriate for most robotics tasks where the action space itself
    naturally captures behavioral phases.
    """
    obs = np.asarray(observations, dtype=np.float64)
    return obs


def goal_relative_features(
    observations: np.ndarray,
    *,
    world_size: float,
) -> np.ndarray:
    """Convert raw Nav2D observations to goal-relative feature vectors.

    This feature definition produces goal-relative, environment-invariant
    signals used for phase detection and canonical policy conditioning. It
    encodes the direction of approach and alignment rather than absolute
    positions.

    NOTE: This is task-specific for Nav2D navigation and kept for backward
    compatibility. Most tasks should use identity_features instead.
    """

    obs = np.asarray(observations, dtype=np.float32)
    squeeze = False
    if obs.ndim == 1:
        obs = obs[None]
        squeeze = True

    scale = float(world_size)
    agent_pos = obs[..., 0:2] / scale
    goal_pos = obs[..., 2:4] / scale
    rel_pos = goal_pos - agent_pos  # direction to goal
    rel_dist = np.linalg.norm(rel_pos, axis=-1, keepdims=True)
    time_progress = obs[..., 5:6]
    heading = obs[..., 6:8]

    # Goal direction unit vector (approach direction)
    unit_rel = rel_pos / np.maximum(rel_dist, 1e-6)

    # Alignment metrics between current heading and goal direction
    heading_dot_goal = np.sum(heading * unit_rel, axis=-1, keepdims=True)
    heading_cross_goal = (
        heading[..., 0:1] * unit_rel[..., 1:2] - heading[..., 1:2] * unit_rel[..., 0:1]
    )

    features = np.concatenate(
        (
            rel_pos,         # 2 dims
            unit_rel,        # 2 dims (direction of approach)
            rel_dist,        # 1 dim (distance to goal)
            time_progress,   # 1 dim (episode phase)
            heading,         # 2 dims (current orientation)
            heading_dot_goal,    # 1 dim (alignment toward goal)
            heading_cross_goal,  # 1 dim (steering direction)
        ),
        axis=-1,
    )

    if squeeze:
        return features[0]
    return features


DEFAULT_FEATURE_TRANSFORM = "identity"

FEATURE_TRANSFORMS = {
    "identity": identity_features,
    "goal_relative_v1": goal_relative_features,
}
