#!/usr/bin/env python3
"""evolutionary strategies for trading agents"""

import numpy as np
from multi_agent_env import MultiAgentExchangeEnv
from typing import List, Tuple
import wandb
import os
from datetime import datetime

# evolve simple rule-based strategies through tournament selection with wandb tracking

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
    """run agent against background traders and return risk-adjusted fitness"""
    env = MultiAgentExchangeEnv(n_agents=1, max_steps=300)
    
    total_pnl = 0.0
    total_sharpe = 0.0
    
    for ep in range(n_episodes):
        obs_all, _ = env.reset(seed=seed + ep)
        done = False
        episode_rewards = []
        
        while not done:
            action = agent.get_action(obs_all[0])
            obs_all, rewards, dones, _, infos = env.step({0: action})
            done = dones[0]
            episode_rewards.append(rewards[0])
        
        final_pnl = infos[0]["pnl"]
        
        # Calculate Sharpe ratio (risk-adjusted returns)
        if len(episode_rewards) > 1:
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            sharpe = mean_reward / (std_reward + 1e-8)
        else:
            sharpe = 0.0
        
        total_pnl += final_pnl
        total_sharpe += sharpe
    
    avg_pnl = total_pnl / n_episodes
    avg_sharpe = total_sharpe / n_episodes
    
    # Fitness = PnL + sharpe bonus (rewards consistent profitable strategies)
    # The sharpe component prevents "lucky" strategies that just sit still
    return avg_pnl + 10.0 * avg_sharpe


def tournament_select(population: List[RuleBasedAgent], fitnesses: List[float], 
                      tournament_size: int = 5) -> RuleBasedAgent:
    """select agent via tournament"""
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_idx]


def evolve_strategies(pop_size: int = 50, n_generations: int = 100, 
                     mutation_rate: float = 0.1, use_wandb: bool = True):
    """evolutionary algorithm for discovering strategies"""
    
    # initialize wandb
    if use_wandb:
        wandb.init(
            project="exchange-evolution",
            config={
                "pop_size": pop_size,
                "n_generations": n_generations,
                "mutation_rate": mutation_rate,
            },
            name=f"evolution_{pop_size}pop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # initialize population
    population = [RuleBasedAgent() for _ in range(pop_size)]
    
    best_fitness_history = []
    checkpoint_dir = "checkpoints/evolution"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_fitness_ever = -float('inf')
    
    for gen in range(n_generations):
        # evaluate all agents
        fitnesses = [evaluate_agent(agent, n_episodes=3, seed=gen*1000) 
                    for agent in population]
        
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        
        best_fitness_history.append(best_fitness)
        
        # log to wandb
        if use_wandb:
            wandb.log({
                "generation": gen,
                "best_fitness": best_fitness,
                "mean_fitness": mean_fitness,
                "std_fitness": std_fitness,
                "worst_fitness": np.min(fitnesses)
            })
        
        print(f"Gen {gen}: best={best_fitness:.2f}, mean={mean_fitness:.2f}, std={std_fitness:.2f}")
        
        # save checkpoint (only keep best)
        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            # delete old checkpoint
            for f in os.listdir(checkpoint_dir):
                if f.startswith("best_agent_"):
                    os.remove(os.path.join(checkpoint_dir, f))
            # save new checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"best_agent_gen_{gen}_fit_{best_fitness:.2f}.npy")
            np.save(checkpoint_path, population[best_idx].params)
            print(f"💾 saved checkpoint: {checkpoint_path}")
        
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
    
    if use_wandb:
        wandb.finish()
    
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
        
        pnls_str = [f'{infos[i]["pnl"]:.2f}' for i in range(len(agents))]
        print(f"Round {round_num}: PnLs = {pnls_str}")
    
    print(f"\nFinal scores: {[f'{s/n_rounds:.2f}' for s in scores]}")
    return scores


if __name__ == "__main__":
    print("🧬 evolving trading strategies...")
    print("this will run for a LONG time. go touch grass.")
    
    # LONG evolution run
    best_agent, history = evolve_strategies(
        pop_size=100,  # larger population
        n_generations=50000,  # 50k generations
        mutation_rate=0.15,
        use_wandb=True
    )
    
    print(f"\n✅ evolution complete!")
    print(f"Best agent parameters: {best_agent.params}")
