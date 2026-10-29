# robot-research: MuJoCo 6-DOF arm reaching environment

Gymnasium environment for continual-learning experiments (stages A-D) on a simulated
6-DOF arm with a parallel gripper. The task is **reaching**: move the end-effector tip to a
random target. (The gripper is actuated and observed, but there is no graspable object yet.)

## Layout
```
simulator/simple_6dof_arm.xml   MJCF: arm, gripper, floor, target marker, obstacle pool, camera
environment/robot_env.py        RobotEnv (reset/step/render + drift hooks)
environment/scenarios.py        Stage A-D configurations (act on the real model)
environment/terrain.py          dynamics-drift helpers (friction, incline, payload, damping)
environment/obstacles.py        obstacle layouts (real collision boxes)
experiments/common.py           seeded runner -> results/<stage>_results.csv
experiments/policies.py         RandomPolicy, JacobianReachPolicy (scripted reference)
experiments/{baseline,dynamics_drift,sensor_drift,cross_robot_transfer}.py
tests/test_env.py               28 tests (pytest)
```

## Install / run
```bash
python -m venv venv && venv\Scripts\activate      # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python -m pytest tests -q                         # headless Linux: MUJOCO_GL=egl python -m pytest tests -q
python experiments/baseline.py                    # also dynamics_drift / sensor_drift / cross_robot_transfer
```
Use your own policy by adding a callable `obs -> action` to `experiments/common.py:POLICIES`.

## Environment
| | |
|---|---|
| Observation (22,) | qpos(8) + qvel(8) + ee_pos(3) + target_pos(3); SI units, radians; clipped to the space |
| Action (7,) in [-1,1] | 6 arm joints + gripper (-1 closed, +1 open) |
| `action_mode="absolute"` | -1..1 maps to each joint's range (default) |
| `action_mode="delta"` | action x `max_delta` (0.1 rad) added to the joint target |
| Control | 200 Hz physics, `frame_skip=4` -> 50 Hz policy, 300 steps = 6 s |
| Reward | `-dist + progress_gain*(prev_dist - dist) - 1e-3*|a|^2 (+1 on success)` |
| Termination | `terminated` at < 5 cm; `truncated` at `max_steps`; NaN guard resets and truncates |
| Reset | home pose + noise; target sampled by forward kinematics, so it is always reachable |

### Actuators
Position servos (`kp`, `kv`, torque limits `forcerange`) on every joint; ctrl is in radians
(metres for fingers). Joints carry armature, damping and friction loss; `implicitfast`
integrator. No gravity compensation, so a payload really perturbs tracking (~0.05 rad sag
at full lift).

### Drift hooks (real model mutations, all undone by `env.restore_nominal()`)
`set_floor_friction`, `set_joint_damping_scale`, `set_joint_range_scale`, `set_payload`,
`set_gravity_tilt`, `set_lighting`, `set_obstacles`, `set_observation_noise`.

## Reference results (20 episodes, seed 0; `results/`)
| Stage | scripted `jacobian` | `random` |
|---|---|---|
| A baseline | 100% | 15% |
| B dynamics drift | 100% | 25% |
| C sensor drift | 100% | 15% |
| D cross-robot | 100% | 30% |

The scripted policy reads ground-truth sim state: it proves the task is solvable, but it is
immune to observation noise and ignores obstacles, so these tables show the environment works,
not how drift hurts. Train a learned policy on A, then evaluate on B-D to measure drift.

## Known limitations
- Reaching only; no grasp/place object.
- Random success is non-trivial (15-30%) because absolute-mode random targets sweep a small workspace.
- Stage C lighting only affects rendered frames, not the state observation.
