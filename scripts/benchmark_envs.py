#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.vector import VectorizedMultiAgentEnv, select_device


def random_action_dict(n_agents: int, n_envs: int, device: torch.device) -> Dict[int, torch.Tensor]:
    actions = {}
    for agent_id in range(n_agents):
        side = torch.rand((n_envs, 1), device=device)
        offset = torch.rand((n_envs, 1), device=device) * 10.0 - 5.0
        qty = torch.rand((n_envs, 1), device=device) * 49.0 + 1.0
        actions[agent_id] = torch.cat([side, offset, qty], dim=1)
    return actions


def benchmark_vector(n_agents: int, n_envs: int, steps: int, device_name: str) -> float:
    device = select_device(device_name)
    env = VectorizedMultiAgentEnv(
        n_agents=n_agents,
        n_envs=n_envs,
        max_steps=steps + 1,
        device=str(device),
        return_tensors=True,
    )
    env.reset(seed=123)
    actions = random_action_dict(n_agents, n_envs, device)

    start = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    env.close()
    return elapsed


def benchmark_parallel(n_agents: int, n_envs: int, steps: int) -> float:
    from src.multi_agent_env import MultiAgentExchangeEnv
    from src.parallel_env import ParallelEnv

    env_fns = [lambda: MultiAgentExchangeEnv(n_agents=n_agents, max_steps=steps + 1) for _ in range(n_envs)]
    env = ParallelEnv(env_fns, n_envs=n_envs)
    env.reset(seed=123)
    action_space = env.action_space

    actions: List[Dict[int, np.ndarray]] = []
    for _ in range(n_envs):
        actions.append({agent_id: action_space.sample() for agent_id in range(n_agents)})

    start = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    elapsed = time.perf_counter() - start
    env.close()
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark parallel vs vector exchange envs")
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n_envs", type=int, nargs="+", default=[8, 32, 64, 128])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--skip_parallel", action="store_true")
    args = parser.parse_args()

    print("backend,n_envs,seconds,steps_per_sec,env_steps_per_sec,speedup_vs_parallel")
    for n_envs in args.n_envs:
        parallel_elapsed = None
        if not args.skip_parallel:
            try:
                parallel_elapsed = benchmark_parallel(args.n_agents, n_envs, args.steps)
                print(
                    f"parallel,{n_envs},{parallel_elapsed:.4f},"
                    f"{args.steps / parallel_elapsed:.2f},{args.steps * n_envs / parallel_elapsed:.2f},1.00"
                )
            except ModuleNotFoundError as exc:
                print(f"parallel,{n_envs},missing_dependency:{exc.name},nan,nan,nan")

        vector_elapsed = benchmark_vector(args.n_agents, n_envs, args.steps, args.device)
        speedup = parallel_elapsed / vector_elapsed if parallel_elapsed else float("nan")
        print(
            f"vector,{n_envs},{vector_elapsed:.4f},"
            f"{args.steps / vector_elapsed:.2f},{args.steps * n_envs / vector_elapsed:.2f},{speedup:.2f}"
        )


if __name__ == "__main__":
    main()
