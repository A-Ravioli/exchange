#!/usr/bin/env python3
"""quick test of multi-agent environment"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from multi_agent_env import MultiAgentExchangeEnv

print("testing multi-agent competitive environment...\n")

env = MultiAgentExchangeEnv(n_agents=3, max_steps=50)
obs_all, _ = env.reset(seed=42)

print(f"observation space: {env.observation_space.shape}")
print(f"action space: {env.action_space.shape}")
print(f"number of agents: {env.n_agents}\n")

total_rewards = [0.0] * env.n_agents

for step in range(50):
    # random actions for all agents
    actions = {i: env.action_space.sample() for i in range(env.n_agents)}
    
    obs_all, rewards, dones, truncs, infos = env.step(actions)
    
    for i in range(env.n_agents):
        total_rewards[i] += rewards[i]
    
    if step % 10 == 0:
        print(f"step {step}:")
        for i in range(env.n_agents):
            inv = env.agents[i].inventory
            cash = env.agents[i].cash
            print(f"  agent {i}: inv={inv:3d}, cash=${cash:7.2f}, reward={rewards[i]:6.2f}")

print(f"\nfinal total rewards: {[f'{r:.2f}' for r in total_rewards]}")
print(f"final pnls: {[f'${env.agents[i].cash:.2f}' for i in range(env.n_agents)]}")
print("\n✓ multi-agent environment working!")
