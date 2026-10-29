#!/usr/bin/env python3
"""Stage C: observation noise, lighting, and new obstacles.

The scripted ``jacobian`` policy reads ground-truth state, so it is immune to observation
noise; it shows what obstacles alone cost. Learned policies are what feel the noise.
"""
from common import run_stage

from environment import scenarios


def sample(rng, ep):
    return {"qpos_std": rng.uniform(0.001, 0.01), "qvel_std": rng.uniform(0.01, 0.1),
            "pos_std": rng.uniform(0.001, 0.02), "lighting": rng.uniform(0.2, 0.8),
            "layout": ["simple", "cluttered"][ep % 2]}


def run_sensor_drift_experiment(**kw):
    return run_stage("sensor_drift",
                     lambda env, p: scenarios.stage_C_sensor_perception_drift(
                         env, p["qpos_std"], p["qvel_std"], p["pos_std"], p["lighting"], p["layout"]),
                     sample, **kw)


if __name__ == "__main__":
    run_sensor_drift_experiment()
