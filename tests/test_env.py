"""Tests for RobotEnv.  Run:  python -m pytest tests -q     (or: python tests/test_env.py)

Headless rendering: set MUJOCO_GL=egl (Linux) before running; Windows/macOS work as-is.
"""
import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment import obstacles, scenarios  # noqa: E402
from environment.robot_env import RobotEnv  # noqa: E402
from experiments.policies import JacobianReachPolicy  # noqa: E402


@pytest.fixture
def env():
    e = RobotEnv(max_steps=100)
    yield e
    e.close()


def _contacts(env, name):
    m = env.model
    return sum(1 for c in env.data.contact[:env.data.ncon]
               if name in m.geom(c.geom1).name + m.geom(c.geom2).name)


# ---------------------------------------------------------------- API / spaces
def test_spaces_and_dtypes(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (22,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (7,) and env.action_space.dtype == np.float32
    assert {"is_success", "distance", "ee_pos", "target_pos"} <= set(info)


def test_gymnasium_check_env():
    from gymnasium.utils.env_checker import check_env
    e = RobotEnv(max_steps=20)
    check_env(e, skip_render_check=True)
    e.close()


def test_step_returns_python_bools(env):
    env.reset(seed=1)
    _, r, term, trunc, _ = env.step(env.action_space.sample())
    assert type(term) is bool and type(trunc) is bool and np.isfinite(r)


def test_truncates_at_max_steps():
    e = RobotEnv(max_steps=5, target_randomize=False)
    e.reset(seed=0, options={"target_pos": [0.0, 0.0, 2.0]})  # unreachable
    for i in range(5):
        _, _, term, trunc, _ = e.step(np.zeros(7))
    assert trunc and not term
    e.close()


# ---------------------------------------------------- the bug that voided v1
def test_ee_site_is_attached_to_the_arm(env):
    m = env.model
    body = m.site_bodyid[env.ee_site_id]
    assert m.body_weldid[body] != 0, "ee_tip must not be welded to the world"


def test_ee_moves_with_joints(env):
    env.reset(seed=0)
    p0 = env.ee_pos
    for _ in range(40):
        env.step(np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float32))  # swing pan
    assert np.linalg.norm(env.ee_pos - p0) > 0.1


def test_start_pose_is_collision_free_and_above_floor(env):
    for s in range(10):
        env.reset(seed=s)
        assert env.data.ncon == 0, "arm must not start in contact"
        assert env.ee_pos[2] > 0.05


# ----------------------------------------------------- task is solvable
@pytest.mark.parametrize("mode", ["absolute", "delta"])
def test_oracle_solves_task(mode):
    e = RobotEnv(action_mode=mode, max_steps=300)
    pol = JacobianReachPolicy(e)
    wins = 0
    for s in range(20):
        obs, _ = e.reset(seed=s)
        for _ in range(300):
            obs, r, term, trunc, info = e.step(pol(obs))
            if term or trunc:
                break
        wins += term
    e.close()
    assert wins >= 19, f"oracle solved only {wins}/20 in {mode} mode"


def test_targets_are_reachable_and_valid(env):
    for s in range(50):
        env.reset(seed=s)
        t = env.target_pos
        assert t[2] >= 0.10 and np.linalg.norm(t[:2]) >= 0.12


def test_success_terminates_and_rewards(env):
    env.reset(seed=1)
    env.target_pos = env.ee_pos.copy()
    _, r, term, _, info = env.step(env.action_for_target(env.q_target))
    assert term is True and info["is_success"] is True
    assert r > 0.5  # success bonus outweighs the tiny residual distance


def test_reward_is_higher_when_closer(env):
    pol = JacobianReachPolicy(env)
    obs, _ = env.reset(seed=3)
    d0 = env._dist()
    rewards = []
    for _ in range(10):
        obs, r, *_ = env.step(pol(obs))
        rewards.append(r)
    assert env._dist() < d0 and rewards[-1] > rewards[0] - 1.0
    assert all(np.isfinite(rewards)) and max(np.abs(rewards)) < 10


# ------------------------------------------------------------- actuators
def test_position_servo_tracks_target(env):
    env.reset(seed=0)
    a = env.action_for_target(env.q_target + np.array([0.3, 0, 0, 0, 0, 0]))
    for _ in range(60):
        env.step(a)
    q = env.data.qpos[env.arm_qpos_ids]
    assert abs(q[0] - env.q_target[0]) < 0.02


def test_gravity_sag_is_small(env):
    env.reset(seed=0)
    a = np.zeros(7, dtype=np.float32)
    a[1] = 0.75
    for _ in range(100):
        env.step(a)
    assert abs(env.data.qpos[env.arm_qpos_ids[1]] - env.q_target[1]) < 0.12


def test_ctrl_never_exceeds_ctrlrange(env):
    env.reset(seed=0)
    env.step(np.full(7, 5.0))  # out-of-range action gets clipped
    lo, hi = env.model.actuator_ctrlrange.T
    assert np.all(env.data.ctrl >= lo - 1e-9) and np.all(env.data.ctrl <= hi + 1e-9)


def test_torque_limits_respected(env):
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    lim = env.model.actuator_forcerange[:, 1]
    for _ in range(100):
        env.step(rng.choice([-1.0, 1.0], 7))
        assert np.all(np.abs(env.data.actuator_force) <= lim + 1e-6)


def test_gripper_polarity_and_symmetry(env):
    env.reset(seed=5)
    g = lambda: env.data.qpos[env.gripper_qpos_ids].copy()  # noqa: E731
    close = np.zeros(7, dtype=np.float32)
    close[6] = -1.0
    for _ in range(50):
        env.step(close)
    q_closed = g()
    op = close.copy()
    op[6] = 1.0
    for _ in range(50):
        env.step(op)
    q_open = g()
    assert np.all(q_closed < 0.003), q_closed          # -1 closes
    assert np.all(q_open > 0.027), q_open              # +1 opens
    assert abs(q_open[0] - q_open[1]) < 1e-3 and abs(q_closed[0] - q_closed[1]) < 1e-3


def test_stable_under_bang_bang_actions(env):
    rng = np.random.default_rng(0)
    for s in range(10):
        env.reset(seed=s)
        for _ in range(100):
            obs, r, term, trunc, info = env.step(rng.choice([-1.0, 1.0], 7))
            assert np.all(np.isfinite(obs)) and "unstable" not in info
            assert env.observation_space.contains(obs)


# -------------------------------------------------------- determinism
def test_seed_determinism():
    a, b = RobotEnv(), RobotEnv()
    oa, _ = a.reset(seed=123)
    ob, _ = b.reset(seed=123)
    assert np.allclose(oa, ob)
    act = np.linspace(-1, 1, 7, dtype=np.float32)
    for _ in range(20):
        oa, *_ = a.step(act)
        ob, *_ = b.step(act)
    assert np.allclose(oa, ob)


# ------------------------------------------------------ drift hooks are REAL
def test_friction_hook_mutates_and_restores(env):
    g = env.floor_geom_id
    base = env.model.geom_friction[g].copy()
    env.set_floor_friction(2.0)
    assert np.allclose(env.model.geom_friction[g], base * 2.0)
    env.set_floor_friction(2.0)  # idempotent, not compounding
    assert np.allclose(env.model.geom_friction[g], base * 2.0)
    env.reset_floor_friction()
    assert np.allclose(env.model.geom_friction[g], base)


def test_payload_changes_dynamics(env):
    def sag(payload):
        env.restore_nominal()
        env.set_payload(payload)
        env.reset(seed=0)
        a = np.zeros(7, dtype=np.float32)
        a[1], a[2] = 0.7, 0.2
        for _ in range(100):
            env.step(a)
        return env.data.qpos[env.arm_qpos_ids[1]] - env.q_target[1]
    assert abs(sag(1.5)) > abs(sag(0.0)) + 0.005


def test_gravity_tilt_changes_gravity(env):
    env.set_gravity_tilt(20.0)
    g = env.model.opt.gravity
    assert abs(np.linalg.norm(g) - 9.81) < 1e-6 and abs(g[0]) > 3.0
    env.restore_nominal()
    assert np.allclose(env.model.opt.gravity, [0, 0, -9.81])


def test_range_scale_is_respected_by_actions_and_targets(env):
    env.set_joint_range_scale(0.5)
    r = env.arm_ranges
    env.reset(seed=0)
    env.step(np.ones(7))
    assert np.all(env.q_target <= r[:, 1] + 1e-9)
    env.restore_nominal()
    assert np.all(env.arm_ranges[:, 1] > r[:, 1])


def test_obstacles_actually_collide(env):
    env.reset(seed=0)
    p = env.data.xpos[env.model.body("forearm").id] + [0, 0, 0.1]
    env.set_obstacles([{"pos": p, "half": (0.05, 0.05, 0.05)}])
    mujoco.mj_forward(env.model, env.data)
    assert _contacts(env, "obstacle_0") > 0
    env.reset(seed=0)  # obstacle persists across reset
    assert _contacts(env, "obstacle_0") > 0
    env.restore_nominal()
    mujoco.mj_forward(env.model, env.data)
    assert env.data.ncon == 0


def test_obstacle_layouts_load():
    e = RobotEnv()
    for name in obstacles.LAYOUTS:
        obstacles.create_obstacle_layout(e, name)
        e.reset(seed=0)
        assert np.all(np.isfinite(e.step(np.zeros(7))[0]))
    e.close()


def test_observation_noise_affects_obs_not_state(env):
    env.reset(seed=0)
    clean = env._get_obs()
    env.set_observation_noise(0.05, 0.1, 0.02)
    noisy = env._get_obs()
    assert not np.allclose(clean, noisy)
    assert np.allclose(env.ee_pos, clean[16:19])  # ground truth untouched
    env.restore_nominal()
    assert np.allclose(env._get_obs(), clean)


def test_all_scenarios_run():
    e = RobotEnv(max_steps=20)
    for fn in (scenarios.stage_A_baseline, scenarios.stage_B_dynamics_drift,
               scenarios.stage_C_sensor_perception_drift, scenarios.stage_D_cross_robot_transfer):
        fn(e)
        e.reset(seed=0)
        for _ in range(5):
            assert np.all(np.isfinite(e.step(e.action_space.sample())[0]))
    scenarios.stage_A_baseline(e)  # A restores nominal
    assert np.allclose(e.model.dof_damping, e._nominal["dof_damping"])
    e.close()


# ------------------------------------------------------------- rendering
def test_rgb_array_shows_the_arm():
    e = RobotEnv(render_mode="rgb_array")
    e.reset(seed=1)
    f0 = e.render()
    assert f0.shape == (480, 640, 3) and f0.dtype == np.uint8
    for _ in range(40):
        e.step(np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float32))
    f1 = e.render()
    assert (np.abs(f1.astype(int) - f0.astype(int)).mean()) > 0.05, "camera does not see the arm move"
    e.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
