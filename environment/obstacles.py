"""Obstacle layouts. Boxes are real collision geoms (pool of 6 in the XML)."""

LAYOUTS = {
    "none": [],
    # a post between the base and the typical workspace
    "simple": [
        {"pos": (0.35, 0.25, 0.20), "half": (0.04, 0.04, 0.20)},
        {"pos": (0.35, -0.25, 0.15), "half": (0.04, 0.04, 0.15)},
    ],
    # scattered boxes around the workspace
    "cluttered": [
        {"pos": (0.40, 0.00, 0.10), "half": (0.05, 0.05, 0.10)},
        {"pos": (0.25, 0.35, 0.12), "half": (0.05, 0.05, 0.12)},
        {"pos": (0.25, -0.35, 0.12), "half": (0.05, 0.05, 0.12)},
        {"pos": (-0.35, 0.20, 0.15), "half": (0.06, 0.06, 0.15)},
    ],
}


def create_obstacle_layout(env, layout_type="simple"):
    """Activate a named layout on ``env`` (replaces any current obstacles)."""
    return env.set_obstacles(LAYOUTS[layout_type])
