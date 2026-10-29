#!/usr/bin/env python3
"""Stage A: nominal robot, no drift."""
from common import run_stage

from environment import scenarios


def run_baseline_experiment(**kw):
    return run_stage("baseline", lambda env, p: scenarios.stage_A_baseline(env), **kw)


if __name__ == "__main__":
    run_baseline_experiment()
