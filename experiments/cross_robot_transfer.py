#!/usr/bin/env python3
"""Stage D: a mechanically different arm (more damping, smaller joint ranges, extra tool mass)."""
from common import run_stage

from environment import scenarios


def run_cross_robot_transfer_experiment(**kw):
    return run_stage("cross_robot_transfer",
                     lambda env, p: scenarios.stage_D_cross_robot_transfer(env), **kw)


if __name__ == "__main__":
    run_cross_robot_transfer_experiment()
