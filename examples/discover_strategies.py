#!/usr/bin/env python3
"""discover hft strategies through self-play and evolution"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from evolve import RuleBasedAgent, evolve_strategies, compete_agents

print("=" * 60)
print("HFT STRATEGY DISCOVERY")
print("=" * 60)

# phase 1: evolve initial strategies
print("\n[phase 1] evolving strategies with evolutionary algorithms...")
best_agent, history = evolve_strategies(
    pop_size=20,
    n_generations=30,
    mutation_rate=0.15
)

print(f"\nbest fitness over time: {history[::5]}")
print(f"final parameters: {best_agent.params}")

# phase 2: create variants and compete
print("\n[phase 2] creating variants and running tournament...")
variants = [
    best_agent,
    best_agent.mutate(0.05),  # small mutation
    best_agent.mutate(0.2),   # large mutation
    RuleBasedAgent(),          # random baseline
]

scores = compete_agents(variants, n_rounds=10)

winner_idx = np.argmax(scores)
print(f"\nwinner: agent {winner_idx} with score {scores[winner_idx]/10:.2f}")

# phase 3: can also train RL agents and compete them
print("\n[phase 3] hybrid approach: compete evolved vs random...")
print("(for RL training, run train_rl.py separately - takes longer)")

print("\n✓ strategy discovery complete")
print("next steps:")
print("  - run train_rl.py to train neural policies")
print("  - compete evolved strategies vs RL policies")
print("  - analyze winning strategy parameters")
print("  - iterate and refine")
