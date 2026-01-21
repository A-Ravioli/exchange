#!/usr/bin/env python3
"""Test parallel environment wrapper."""

import sys
import time
import numpy as np
from parallel_env import ParallelEnv, SubprocVecEnv
from multi_agent_env import MultiAgentExchangeEnv


def test_basic_parallel():
    """Test basic parallel environment functionality."""
    print("Testing basic parallel environment...")
    
    n_envs = 4
    n_agents = 2
    
    # Create environment functions
    env_fns = [lambda: MultiAgentExchangeEnv(n_agents=n_agents, max_steps=10) 
               for _ in range(n_envs)]
    
    # Create parallel environment
    with ParallelEnv(env_fns, n_envs=n_envs) as par_env:
        # Test reset
        obs = par_env.reset(seed=42)
        print(f"✓ Reset successful")
        print(f"  Observation shape per agent: {obs[0].shape}")  # (n_envs, obs_dim)
        assert obs[0].shape[0] == n_envs, "Wrong batch size"
        
        # Test step
        actions = {i: par_env.action_space.sample() for i in range(n_agents)}
        obs, rewards, dones, truncs, infos = par_env.step(actions)
        
        print(f"✓ Step successful")
        print(f"  Rewards shape: {rewards[0].shape}")  # (n_envs,)
        assert rewards[0].shape[0] == n_envs, "Wrong reward batch size"
        
        # Run a few more steps
        for _ in range(5):
            actions = {i: par_env.action_space.sample() for i in range(n_agents)}
            obs, rewards, dones, truncs, infos = par_env.step(actions)
        
        print(f"✓ Multiple steps successful")
    
    print("\n✅ Basic parallel environment test passed!\n")


def test_speed_comparison():
    """Compare speed of parallel vs sequential execution."""
    print("Testing speed comparison...")
    
    n_envs = 8
    n_agents = 4
    n_steps = 20
    
    # Sequential execution
    print(f"Running {n_envs} environments sequentially...")
    env_fns = [lambda: MultiAgentExchangeEnv(n_agents=n_agents, max_steps=n_steps) 
               for _ in range(n_envs)]
    
    start = time.time()
    for env_fn in env_fns:
        env = env_fn()
        obs, _ = env.reset(seed=42)
        for _ in range(n_steps):
            actions = {i: env.action_space.sample() for i in range(n_agents)}
            env.step(actions)
    sequential_time = time.time() - start
    print(f"  Sequential time: {sequential_time:.2f}s")
    
    # Parallel execution
    print(f"Running {n_envs} environments in parallel...")
    env_fns = [lambda: MultiAgentExchangeEnv(n_agents=n_agents, max_steps=n_steps) 
               for _ in range(n_envs)]
    
    start = time.time()
    with ParallelEnv(env_fns, n_envs=n_envs) as par_env:
        obs = par_env.reset(seed=42)
        for _ in range(n_steps):
            actions = {i: par_env.action_space.sample() for i in range(n_agents)}
            par_env.step(actions)
    parallel_time = time.time() - start
    print(f"  Parallel time: {parallel_time:.2f}s")
    
    speedup = sequential_time / parallel_time
    print(f"\n✓ Speedup: {speedup:.2f}x")
    
    if speedup > 1.5:
        print(f"✅ Good speedup achieved!")
    else:
        print(f"⚠️  Speedup lower than expected (overhead from process creation)")
    
    print()


def test_data_consistency():
    """Test that parallel environments produce consistent results."""
    print("Testing data consistency...")
    
    n_envs = 4
    n_agents = 2
    seed = 42
    
    # Run single environment
    env = MultiAgentExchangeEnv(n_agents=n_agents, max_steps=10)
    obs_single, _ = env.reset(seed=seed)
    obs_single_0 = obs_single[0].copy()
    
    # Run parallel environments with same seed
    env_fns = [lambda: MultiAgentExchangeEnv(n_agents=n_agents, max_steps=10) 
               for _ in range(n_envs)]
    
    with ParallelEnv(env_fns, n_envs=n_envs) as par_env:
        obs_parallel = par_env.reset(seed=seed)
        obs_parallel_0_0 = obs_parallel[0][0]  # First env, first agent
    
    # Check first observation matches
    np.testing.assert_array_almost_equal(
        obs_single_0, obs_parallel_0_0,
        err_msg="Observations don't match between single and parallel envs"
    )
    
    print("✓ Data consistency verified")
    print("✅ Parallel and single environments produce consistent results!\n")


if __name__ == "__main__":
    try:
        test_basic_parallel()
        test_data_consistency()
        test_speed_comparison()
        
        print("=" * 60)
        print("🎉 ALL PARALLEL ENVIRONMENT TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
