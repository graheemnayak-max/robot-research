"""Continual-learning stages A-D. Each function configures a RobotEnv in place."""
from . import obstacles, terrain


def stage_A_baseline(env):
    """Nominal robot: default friction/damping/gravity, no obstacles, clean sensors."""
    return env.restore_nominal()


def stage_B_dynamics_drift(env, friction=0.5, slope_deg=8.0, payload_kg=0.8, damping_scale=1.3):
    """Dynamics drift: friction, inclined mount, unmodelled payload, joint damping."""
    env.restore_nominal()
    terrain.modify_ground_friction(env, friction)
    terrain.add_slope(env, slope_deg)
    terrain.add_payload(env, payload_kg)
    terrain.scale_joint_damping(env, damping_scale)
    return env


def stage_C_sensor_perception_drift(env, qpos_std=0.005, qvel_std=0.05, pos_std=0.01,
                                    lighting=0.4, layout="simple"):
    """Sensor drift: observation noise, dimmer light (camera renders), new obstacles."""
    env.restore_nominal()
    env.set_observation_noise(qpos_std, qvel_std, pos_std)
    env.set_lighting(lighting)
    obstacles.create_obstacle_layout(env, layout)
    return env


def stage_D_cross_robot_transfer(env, damping_scale=1.5, range_scale=0.8, payload_kg=0.5):
    """A mechanically different arm: heavier damping, smaller joint ranges, extra tool mass."""
    env.restore_nominal()
    env.set_joint_damping_scale(damping_scale)
    env.set_joint_range_scale(range_scale)
    env.set_payload(payload_kg)
    return env
