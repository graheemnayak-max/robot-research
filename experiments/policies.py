"""Reference policies for sanity-checking the environment and giving experiments a real baseline."""
import mujoco
import numpy as np


class RandomPolicy:
    name = "random"

    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        return self.env.action_space.sample()


class JacobianReachPolicy:
    """Damped-least-squares resolved-rate controller (privileged: reads sim state, not obs).

    Proves the task is solvable and gives a non-trivial upper-ish reference. Works in both
    action modes via ``env.action_for_target``. Uses ground-truth state, so it is
    unaffected by observation-noise drift.
    """
    name = "jacobian"

    def __init__(self, env, gain=1.0, damping=0.05, max_step=0.08):
        self.env, self.gain, self.damping, self.max_step = env, gain, damping, max_step
        self._jacp = np.zeros((3, env.model.nv))

    def __call__(self, obs):
        e = self.env
        mujoco.mj_jacSite(e.model, e.data, self._jacp, None, e.ee_site_id)
        J = self._jacp[:, e.arm_dof_ids]
        err = e.target_pos - e.data.site_xpos[e.ee_site_id]
        dq = J.T @ np.linalg.solve(J @ J.T + self.damping ** 2 * np.eye(3), self.gain * err)
        n = np.linalg.norm(dq)
        if n > self.max_step:
            dq *= self.max_step / n
        return e.action_for_target(e.q_target + dq, grip=1.0)
