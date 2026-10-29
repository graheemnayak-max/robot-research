"""Dynamics-drift helpers. Each one mutates the real MuJoCo model through RobotEnv hooks.

(A fixed-base arm has no locomotion terrain, so "slope" is modelled as an inclined
mounting -- gravity tilted in the arm's frame -- and "uneven terrain" is dropped.)
"""


def modify_ground_friction(env, multiplier):
    """Scale floor friction (affects fingers/links touching the floor and obstacles)."""
    return env.set_floor_friction(multiplier)


def add_slope(env, degrees):
    """Mount the arm on an incline: gravity tilts by ``degrees`` about the y axis."""
    return env.set_gravity_tilt(degrees)


def add_payload(env, mass_kg):
    return env.set_payload(mass_kg)


def scale_joint_damping(env, scale):
    return env.set_joint_damping_scale(scale)
