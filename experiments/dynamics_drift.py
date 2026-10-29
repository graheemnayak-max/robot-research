#!/usr/bin/env python3
"""Stage B: dynamics drift (floor friction, inclined mount, payload, joint damping)."""
from common import run_stage

from environment import scenarios


def sample(rng, ep):
    return {"friction": rng.uniform(0.3, 2.0), "slope_deg": rng.uniform(0, 15),
            "payload_kg": rng.uniform(0, 1.5), "damping_scale": rng.uniform(0.7, 2.0)}


def run_dynamics_drift_experiment(**kw):
    return run_stage("dynamics_drift",
                     lambda env, p: scenarios.stage_B_dynamics_drift(
                         env, p["friction"], p["slope_deg"], p["payload_kg"], p["damping_scale"]),
                     sample, **kw)


if __name__ == "__main__":
    run_dynamics_drift_experiment()
