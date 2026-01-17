#!/usr/bin/env python3
"""evolutionary strategies for trading agents"""

import numpy as np
from multi_agent_env import MultiAgentExchangeEnv
from typing import List, Tuple

# evolve simple rule-based strategies through tournament selection

class RuleBasedAgent:
    """parameterized trading strategy"""
    def __init__(self, params: np.ndarray = None):
        # 12 parameters defining strategy
        self.params = params if params is not None else np.random.randn(12) * 0.5
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """obs -> action using simple rules"""
        bid_prices = obs[0:5]
        bid_vols = obs[5:10]
        ask_prices = obs[10:15]
        ask_vols = obs[15:20]
        inventory = obs[20]
        
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        bid_pressure = np.sum(bid_vols)
        ask_pressure = np.sum(ask_vols)
        imbalance = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + 1e-8)
        
        # simple rule: buy if imbalance is positive, sell if negative
        # modulate based on inventory and parameters
        
        # params[0:6] = buy thresholds and sizing
        # params[6:12] = sell thresholds and sizing
        
        # decision logic
        buy_signal = (
            self.params[0] * imbalance + 
            self.params[1] * (1 / (spread + 0.01)) +
            self.params[2] * (-inventory / 50)  # inventory mean reversion
        )
        
        sell_signal = (
            self.params[6] * (-imbalance) +
            self.params[7] * (1 / (spread + 0.01)) +
            self.params[8] * (inventory / 50)
        )
        
        # choose side
        if buy_signal > sell_signal:
            side = 0.0  # buy
            price_offset = self.params[3] * spread + self.params[4]
            quantity = max(1, int(abs(self.params[5] * 20)))
        else:
            side = 1.0  # sell
            price_offset = self.params[9] * spread + self.params[10]
            quantity = max(1, int(abs(self.params[11] * 20)))
        
        return np.array([side, price_offset, quantity], dtype=np.float32)
    
    def mutate(self, mutation_rate: float = 0.1) -> 'RuleBasedAgent':
        """create mutated copy"""
        noise = np.random.randn(len(self.params)) * mutation_rate
        new_params = self.params + noise
        return RuleBasedAgent(new_params)
    
    def crossover(self, other: 'RuleBasedAgent') -> 'RuleBasedAgent':
        """create child from two parents"""
        mask = np.random.rand(len(self.params)) < 0.5
        child_params = np.where(mask, self.params, other.params)
        return RuleBasedAgent(child_params)


def evaluate_agent(agent: RuleBasedAgent, n_episodes: int = 5, seed: int = 42) -> float:
    """run agent in environment and return fitness (average pnl)"""
    env = MultiAgentExchangeEnv(n_agents=1, max_steps=300)
    
    total_pnl = 0.0
    for ep in range(n_episodes):
        obs_all, _ = env.reset(seed=seed + ep)
        done = False
        
        while not done:
            action = agent.get_action(obs_all[0])
            obs_all, rewards, dones, _, infos = env.step({0: action})
            done = dones[0]
        
        total_pnl += infos[0]["pnl"]
    
    return total_pnl / n_episodes


def tournament_select(population: List[RuleBasedAgent], fitnesses: List[float], 
                      tournament_size: int = 5) -> RuleBasedAgent:
    """select agent via tournament"""
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_idx]


def evolve_strategies(pop_size: int = 50, n_generations: int = 100, 
                     mutation_rate: float = 0.1):
    """evolutionary algorithm for discovering strategies"""
    
    # initialize population
    population = [RuleBasedAgent() for _ in range(pop_size)]
    
    best_fitness_history = []
    
    for gen in range(n_generations):
        # evaluate all agents
        fitnesses = [evaluate_agent(agent, n_episodes=3, seed=gen*1000) 
                    for agent in population]
        
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        mean_fitness = np.mean(fitnesses)
        
        best_fitness_history.append(best_fitness)
        
        print(f"Gen {gen}: best={best_fitness:.2f}, mean={mean_fitness:.2f}")
        
        # create next generation
        new_population = [population[best_idx]]  # elitism
        
        while len(new_population) < pop_size:
            # tournament selection
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)
            
            # crossover and mutation
            if np.random.rand() < 0.7:  # crossover probability
                child = parent1.crossover(parent2)
            else:
                child = parent1
            
            child = child.mutate(mutation_rate)
            new_population.append(child)
        
        population = new_population
    
    # return best agent
    final_fitnesses = [evaluate_agent(agent, n_episodes=10) for agent in population]
    best_idx = np.argmax(final_fitnesses)
    
    return population[best_idx], best_fitness_history


def compete_agents(agents: List[RuleBasedAgent], n_rounds: int = 10):
    """have multiple agents compete in same environment"""
    env = MultiAgentExchangeEnv(n_agents=len(agents), max_steps=500)
    
    scores = [0.0] * len(agents)
    
    for round_num in range(n_rounds):
        obs_all, _ = env.reset(seed=round_num)
        done = False
        
        while not done:
            actions = {i: agents[i].get_action(obs_all[i]) for i in range(len(agents))}
            obs_all, rewards, dones, _, infos = env.step(actions)
            done = dones[0]
        
        # accumulate scores
        for i in range(len(agents)):
            scores[i] += infos[i]["pnl"]
        
        print(f"Round {round_num}: PnLs = {[f'{infos[i]["pnl"]:.2f}' for i in range(len(agents))]}")
    
    print(f"\nFinal scores: {[f'{s/n_rounds:.2f}' for s in scores]}")
    return scores


if __name__ == "__main__":
    print("evolving trading strategies...")
    best_agent, history = evolve_strategies(pop_size=30, n_generations=50, mutation_rate=0.15)
    
    print(f"\nBest fitness trajectory: {history[-10:]}")
    print(f"Best agent parameters: {best_agent.params}")
    
    # save best agent
    np.save("best_agent_params.npy", best_agent.params)
    
    print("\n--- testing competitive play ---")
    # create diverse agents and compete
    agents = [best_agent] + [RuleBasedAgent() for _ in range(3)]
    compete_agents(agents, n_rounds=5)
