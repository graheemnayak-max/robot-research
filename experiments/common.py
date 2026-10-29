"""Shared experiment runner: seeded, logs success/reward/length/timing/memory to CSV."""
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment.robot_env import RobotEnv  # noqa: E402
from experiments.policies import JacobianReachPolicy, RandomPolicy  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
POLICIES = {"random": RandomPolicy, "jacobian": JacobianReachPolicy}


def run_stage(name, configure, sample_params=None, num_episodes=20, max_steps=300,
              policies=("random", "jacobian"), seed=0, render=False, verbose=True):
    """Run ``num_episodes`` per policy.

    configure(env, params)       -> applies the stage's drift to ``env`` (real model mutations)
    sample_params(rng, episode)  -> dict of drift parameters logged to the CSV (optional)
    Episode e uses the SAME env seed and drift parameters for every policy (paired comparison).
    """
    from tqdm import tqdm

    env = RobotEnv(render_mode="human" if render else None, max_steps=max_steps)
    proc = psutil.Process(os.getpid())
    rows = []
    try:
        for pname in policies:
            policy = POLICIES[pname](env)
            rng = np.random.default_rng(seed)
            env.action_space.seed(seed)
            for ep in tqdm(range(num_episodes), desc=f"{name}/{pname}", disable=not verbose):
                params = sample_params(rng, ep) if sample_params else {}
                configure(env, params)
                obs, info = env.reset(seed=seed + ep)
                mem0, t0, total, steps = proc.memory_info().rss / 2**20, time.perf_counter(), 0.0, 0
                for _ in range(max_steps):
                    obs, r, terminated, truncated, info = env.step(policy(obs))
                    total += r
                    steps += 1
                    if terminated or truncated:
                        break
                dur = time.perf_counter() - t0
                rows.append({
                    "policy": pname, "episode": ep, **params,
                    "success": bool(info["is_success"]),
                    "final_distance": info["distance"],
                    "episode_reward": total, "episode_steps": steps,
                    "episode_duration_sec": dur,
                    "memory_used_mb": proc.memory_info().rss / 2**20 - mem0,
                    "steps_per_second": steps / dur if dur > 0 else 0.0,
                })
    finally:
        env.restore_nominal()
        env.close()

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}_results.csv")
    df.to_csv(path, index=False)
    if verbose:
        print(f"\n{name} summary")
        print(df.groupby("policy").agg(
            success_rate=("success", "mean"), mean_reward=("episode_reward", "mean"),
            mean_steps=("episode_steps", "mean"), final_dist=("final_distance", "mean"),
            steps_per_sec=("steps_per_second", "mean")).round(3).to_string())
        print(f"saved: {os.path.abspath(path)}")
    return df
