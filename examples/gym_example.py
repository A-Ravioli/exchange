#!/usr/bin/env python3
"""minimal example of using the exchange gym environment"""

import sys
sys.path.insert(0, '../src')

from gym_env import ExchangeEnv

# create environment
env = ExchangeEnv(max_steps=50)

# reset and run random actions
obs, info = env.reset(seed=42)
print("Initial observation shape:", obs.shape)

total_reward = 0
for i in range(10):
    # random action: [side, price_offset, quantity]
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    print(f"Step {i+1}: reward={reward:.2f}, inventory={env.inventory}, cash=${env.cash:.2f}")
    
    if terminated or truncated:
        break

print(f"\nTotal reward: {total_reward:.2f}")

# render final state
env.render()
