#!/usr/bin/env python3
"""
Gymnasium environment: MuJoCo 6-DOF arm + parallel gripper, reaching task.

Task
    Move the end-effector tip (``ee_tip`` site, attached to the gripper) to a
    randomly placed target within ``success_threshold`` metres.

Action  (7,) float32 in [-1, 1]
    [0:6]  arm joints. Meaning depends on ``action_mode``:
             "absolute": -1..1 maps linearly onto each joint's range
             "delta":    a * max_delta radians added to the current joint target
    [6]    gripper opening: -1 = closed, +1 = fully open

Observation (22,) float32 (clipped to the observation space)
    qpos (8) | qvel (8) | ee_pos (3) | target_pos (3)
    qpos/qvel in rad, rad/s (arm) and m, m/s (fingers); positions in metres.

Reward (per step)
    -dist  +  progress_gain * (prev_dist - dist)  -  action_reg * |a|^2
           +  success_bonus on success

Everything that "drifts" (friction, damping, joint range, payload, gravity tilt,
obstacles, sensor noise, lighting) is a real mutation of the MuJoCo model or the
observation pipeline, and can be undone with ``restore_nominal()``.
"""

import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class RobotEnv(gym.Env):
    ARM_JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow", "wrist1", "wrist2", "wrist3",
    ]
    GRIPPER_JOINT_NAMES = ["left_finger_joint", "right_finger_joint"]
    EE_SITE_NAME = "ee_tip"
    TARGET_SITE_NAME = "target"
    FLOOR_GEOM_NAME = "floor"
    PAYLOAD_BODY_NAME = "wrist_3"
    N_OBSTACLES = 6

    # Home pose (rad): elbow-up, tip pointing forward/down, away from singularities.
    HOME_QPOS = np.array([0.0, -0.6, 1.2, 0.0, 0.5, 0.0])

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=300, frame_skip=4,
                 action_mode="absolute", max_delta=0.1, target_randomize=True,
                 action_regularization=1e-3, progress_gain=5.0,
                 success_bonus=1.0, success_threshold=0.05):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array"), render_mode
        assert action_mode in ("absolute", "delta"), action_mode

        xml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "simulator",
            "simple_6dof_arm.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._fk_data = mujoco.MjData(self.model)  # scratch for reachability sampling

        # ---- ids resolved by name (fail loudly if the XML drifts) ----
        def _id(obj, name):
            i = mujoco.mj_name2id(self.model, obj, name)
            if i < 0:
                raise ValueError(f"'{name}' not found in {xml_path}")
            return i

        J, A = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
        self.arm_joint_ids = [_id(J, n) for n in self.ARM_JOINT_NAMES]
        self.gripper_joint_ids = [_id(J, n) for n in self.GRIPPER_JOINT_NAMES]
        self.arm_qpos_ids = np.array([self.model.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.arm_dof_ids = np.array([self.model.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.gripper_qpos_ids = np.array([self.model.jnt_qposadr[j] for j in self.gripper_joint_ids])
        self.arm_actuator_ids = [_id(A, n) for n in self.ARM_JOINT_NAMES]
        self.gripper_actuator_ids = [_id(A, n) for n in self.GRIPPER_JOINT_NAMES]
        self.ee_site_id = _id(mujoco.mjtObj.mjOBJ_SITE, self.EE_SITE_NAME)
        self.target_site_id = _id(mujoco.mjtObj.mjOBJ_SITE, self.TARGET_SITE_NAME)
        self.floor_geom_id = _id(mujoco.mjtObj.mjOBJ_GEOM, self.FLOOR_GEOM_NAME)
        self.payload_body_id = _id(mujoco.mjtObj.mjOBJ_BODY, self.PAYLOAD_BODY_NAME)
        self.obstacle_geom_ids = [
            _id(mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{i}") for i in range(self.N_OBSTACLES)]
        self.obstacle_mocap_ids = [
            int(self.model.body_mocapid[self.model.geom_bodyid[g]]) for g in self.obstacle_geom_ids]
        if self.model.body_weldid[self.model.site_bodyid[self.ee_site_id]] == 0:
            raise ValueError("ee_tip site is attached to the world; it must ride on the gripper")

        self.grip_max = float(self.model.jnt_range[self.gripper_joint_ids[0], 1])

        # ---- timing ----
        self.frame_skip = int(frame_skip)
        self.dt = self.model.opt.timestep * self.frame_skip
        self.metadata = {**self.metadata, "render_fps": int(round(1.0 / self.dt))}

        # ---- spaces ----
        self.action_mode = action_mode
        self.max_delta = float(max_delta)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

        nq, nv = self.model.nq, self.model.nv
        margin = 0.2
        q_lo = self.model.jnt_range[:, 0] - margin
        q_hi = self.model.jnt_range[:, 1] + margin
        v_b, p_b = np.full(nv, 50.0), np.full(3, 2.0)
        low = np.concatenate([q_lo, -v_b, -p_b, -p_b]).astype(np.float32)
        high = np.concatenate([q_hi, v_b, p_b, p_b]).astype(np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        assert self.observation_space.shape == (nq + nv + 6,)

        # ---- reward / termination config ----
        self.action_regularization = action_regularization
        self.progress_gain = progress_gain
        self.success_bonus = success_bonus
        self.success_threshold = success_threshold
        self.target_randomize = target_randomize
        self.max_steps = int(max_steps)

        # ---- state ----
        self.target_pos = np.array([0.5, 0.0, 0.5])
        self.q_target = self.HOME_QPOS.copy()
        self.grip_target = self.grip_max
        self._prev_dist = None
        self.current_step = 0
        self.obstacles = []
        self.obs_noise = (0.0, 0.0, 0.0)  # (qpos std, qvel std, ee/target-pos std)

        # ---- rendering ----
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None

        self._snapshot_nominal()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    @property
    def arm_ranges(self):
        return self.model.jnt_range[self.arm_joint_ids]  # live: honours range drift

    def _dist(self):
        return float(np.linalg.norm(self.data.site_xpos[self.ee_site_id] - self.target_pos))

    def action_for_target(self, q_des, grip=1.0):
        """Action that commands arm joint targets ``q_des`` (rad) in the current action mode."""
        lo, hi = self.arm_ranges.T
        if self.action_mode == "absolute":
            a_arm = 2.0 * (np.clip(q_des, lo, hi) - lo) / (hi - lo) - 1.0
        else:
            a_arm = (np.clip(q_des, lo, hi) - self.q_target) / self.max_delta
        return np.concatenate([np.clip(a_arm, -1, 1), [grip]]).astype(np.float32)

    def _sample_target(self):
        """Sample a target by forward kinematics of a random pose, so it is always reachable."""
        lo, hi = self.arm_ranges.T
        fk = self._fk_data
        for _ in range(100):
            q = self.np_random.uniform(0.6 * lo, 0.6 * hi)
            fk.qpos[:] = 0.0
            fk.qpos[self.arm_qpos_ids] = q
            mujoco.mj_kinematics(self.model, fk)
            p = fk.site_xpos[self.ee_site_id].copy()
            if p[2] < 0.10 or np.linalg.norm(p[:2]) < 0.12:
                continue
            if np.linalg.norm(p - self.ee_pos) < 3 * self.success_threshold:
                continue
            if any(np.all(np.abs(p - o["pos"]) < o["half"] + 0.04) for o in self.obstacles):
                continue
            return p
        return np.array([0.45, 0.0, 0.45])

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        d = self.data
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
        ee, tgt = d.site_xpos[self.ee_site_id].copy(), self.target_pos.copy()
        sq, sv, sp = self.obs_noise
        if sq or sv or sp:
            qpos += self.np_random.normal(0, sq, qpos.shape)
            qvel += self.np_random.normal(0, sv, qvel.shape)
            ee += self.np_random.normal(0, sp, 3)
            tgt += self.np_random.normal(0, sp, 3)
        obs = np.concatenate([qpos, qvel, ee, tgt])
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _info(self):
        dist = self._dist()
        return {"is_success": bool(dist < self.success_threshold), "distance": dist,
                "ee_pos": self.ee_pos, "target_pos": self.target_pos.copy()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)
        for i, o in enumerate(self.obstacles):  # mj_resetData parks mocap bodies; restore them
            self.data.mocap_pos[self.obstacle_mocap_ids[i]] = o["pos"]

        lo, hi = self.arm_ranges.T
        q0 = np.clip(self.HOME_QPOS + self.np_random.uniform(-0.05, 0.05, 6), lo, hi)
        self.data.qpos[self.arm_qpos_ids] = q0
        self.data.qpos[self.gripper_qpos_ids] = self.grip_max
        self.q_target = q0.copy()
        self.grip_target = self.grip_max
        self._apply_ctrl()  # hold the start pose: no initial jerk
        mujoco.mj_forward(self.model, self.data)

        if "target_pos" in options:
            self.target_pos = np.asarray(options["target_pos"], dtype=np.float64)
        elif self.target_randomize:
            self.target_pos = self._sample_target()
        self.model.site_pos[self.target_site_id] = self.target_pos

        self.current_step = 0
        self._prev_dist = self._dist()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._info()

    def _apply_ctrl(self):
        lo, hi = self.model.actuator_ctrlrange[self.arm_actuator_ids].T
        self.data.ctrl[self.arm_actuator_ids] = np.clip(self.q_target, lo, hi)
        self.data.ctrl[self.gripper_actuator_ids] = self.grip_target

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        lo, hi = self.arm_ranges.T
        if self.action_mode == "absolute":
            self.q_target = lo + (action[:6] + 1.0) * 0.5 * (hi - lo)
        else:
            self.q_target = np.clip(self.q_target + action[:6] * self.max_delta, lo, hi)
        self.grip_target = float((action[6] + 1.0) * 0.5 * self.grip_max)
        self._apply_ctrl()

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)  # make site_xpos consistent with qpos
        self.current_step += 1

        info = self._info()
        stable = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        if not stable:  # never leak NaNs to the learner
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            info = {**self._info(), "unstable": True}
            return self._get_obs(), -10.0, False, True, info

        dist = info["distance"]
        reward = -dist + self.progress_gain * (self._prev_dist - dist)
        reward -= self.action_regularization * float(action @ action)
        terminated = info["is_success"]
        if terminated:
            reward += self.success_bonus
        self._prev_dist = dist
        truncated = bool(self.current_step >= self.max_steps) and not terminated

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Drift / scenario hooks (all real model mutations)
    # ------------------------------------------------------------------ #
    def _snapshot_nominal(self):
        m = self.model
        self._nominal = {
            "geom_friction": m.geom_friction.copy(), "dof_damping": m.dof_damping.copy(),
            "jnt_range": m.jnt_range.copy(), "body_mass": m.body_mass.copy(),
            "body_inertia": m.body_inertia.copy(), "opt_gravity": m.opt.gravity.copy(),
            "light_diffuse": m.light_diffuse.copy(),
            "geom_size": m.geom_size.copy(), "geom_rgba": m.geom_rgba.copy(),
            "geom_rbound": m.geom_rbound.copy(), "geom_aabb": m.geom_aabb.copy(),
        }

    def restore_nominal(self):
        """Undo every drift hook (friction, damping, range, payload, gravity, light, obstacles, noise)."""
        m = self.model
        m.geom_friction[:] = self._nominal["geom_friction"]
        m.dof_damping[:] = self._nominal["dof_damping"]
        m.jnt_range[:] = self._nominal["jnt_range"]
        m.body_mass[:] = self._nominal["body_mass"]
        m.body_inertia[:] = self._nominal["body_inertia"]
        m.opt.gravity[:] = self._nominal["opt_gravity"]
        m.light_diffuse[:] = self._nominal["light_diffuse"]
        for k in ("geom_size", "geom_rgba", "geom_rbound", "geom_aabb"):
            getattr(m, k)[:] = self._nominal[k]
        self.obstacles = []
        self.data.mocap_pos[self.obstacle_mocap_ids] = [0.0, 0.0, -5.0]
        self.obs_noise = (0.0, 0.0, 0.0)
        return self

    def set_floor_friction(self, multiplier):
        self.model.geom_friction[self.floor_geom_id] = \
            self._nominal["geom_friction"][self.floor_geom_id] * multiplier
        return self

    def reset_floor_friction(self):
        return self.set_floor_friction(1.0)

    def set_joint_damping_scale(self, scale):
        self.model.dof_damping[:] = self._nominal["dof_damping"] * scale
        return self

    def set_joint_range_scale(self, scale):
        """Shrink/expand each joint range about its midpoint (cross-robot variation)."""
        r0 = self._nominal["jnt_range"]
        mid, half = r0.mean(axis=1), (r0[:, 1] - r0[:, 0]) / 2.0
        self.model.jnt_range[:, 0] = mid - half * scale
        self.model.jnt_range[:, 1] = mid + half * scale
        return self

    def set_payload(self, mass_kg):
        """Add a point mass to the wrist (inertia scaled with mass). Unmodelled by the servo."""
        b = self.payload_body_id
        m0 = self._nominal["body_mass"][b]
        self.model.body_mass[b] = m0 + mass_kg
        self.model.body_inertia[b] = self._nominal["body_inertia"][b] * (m0 + mass_kg) / m0
        return self

    def set_gravity_tilt(self, degrees):
        """Tilt gravity about +y: models the arm mounted on an incline of ``degrees``."""
        th = np.deg2rad(degrees)
        self.model.opt.gravity[:] = 9.81 * np.array([np.sin(th), 0.0, -np.cos(th)])
        return self

    def set_observation_noise(self, qpos_std=0.0, qvel_std=0.0, pos_std=0.0):
        self.obs_noise = (float(qpos_std), float(qvel_std), float(pos_std))
        return self

    def set_lighting(self, diffuse):
        self.model.light_diffuse[:] = float(diffuse)
        return self

    def set_obstacles(self, obstacles):
        """Activate up to 6 static boxes. obstacles: [{'pos': (x,y,z), 'half': (hx,hy,hz)}]"""
        assert len(obstacles) <= self.N_OBSTACLES, f"max {self.N_OBSTACLES} obstacles"
        m = self.model
        self.obstacles = []
        for i, gid in enumerate(self.obstacle_geom_ids):
            if i < len(obstacles):
                pos = np.asarray(obstacles[i]["pos"], dtype=np.float64)
                half = np.asarray(obstacles[i]["half"], dtype=np.float64)
                self.data.mocap_pos[self.obstacle_mocap_ids[i]] = pos
                m.geom_size[gid] = half
                m.geom_rbound[gid] = float(np.linalg.norm(half))
                m.geom_aabb[gid] = np.concatenate([[0, 0, 0], half])
                m.geom_rgba[gid] = [0.8, 0.3, 0.2, 1.0]
                self.obstacles.append({"pos": pos, "half": half})
            else:
                self.data.mocap_pos[self.obstacle_mocap_ids[i]] = [0, 0, -5]  # parked below floor
                m.geom_size[gid] = [0.05, 0.05, 0.05]
                m.geom_rgba[gid] = [0.8, 0.3, 0.2, 0.0]
        return self

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, 480, 640)
            self.renderer.update_scene(self.data, camera="track")
            return self.renderer.render()
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer as mj_viewer  # lazy: needs a display (mjpython on macOS)
                self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
