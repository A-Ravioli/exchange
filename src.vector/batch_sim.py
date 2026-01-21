# batched simulation for running multiple envs in parallel
# uses list of books for now (full gpu vectorization is complex)

import torch
import numpy as np
from typing import List

try:
    from .exchange_vector import VectorizedOrderBook
except ImportError:
    from exchange_vector import VectorizedOrderBook


class BatchedSimulation:
    # runs n_envs simulations with vectorized order books
    # hybrid approach - vectorized books but loops over envs
    
    def __init__(self, n_envs: int, tick_size: float = 0.01, device: str = 'cuda'):
        self.n_envs = n_envs
        self.device = device
        self.tick_size = tick_size
        
        # create vectorized order book for each environment
        self.books = [
            VectorizedOrderBook(tick_size=tick_size, device=device)
            for _ in range(n_envs)
        ]
        
        # simulation times
        self.current_times = torch.zeros(n_envs, device=device)
    
    def reset_all(self):
        # reset all environments
        for book in self.books:
            book.reset()
        self.current_times.zero_()
    
    def step_batch(self, actions_batch: List[dict]) -> tuple:
        # step all environments with given actions
        results = []
        
        for env_id, actions in enumerate(actions_batch):
            # process actions for this environment
            book = self.books[env_id]
            
            # convert actions to tensor format
            if len(actions) > 0:
                agent_ids = list(actions.keys())
                n_agents = len(agent_ids)
                
                # extract action components
                trader_ids = torch.tensor(agent_ids, device=self.device)
                sides = torch.zeros(n_agents, device=self.device)
                prices = torch.zeros(n_agents, device=self.device)
                quantities = torch.zeros(n_agents, device=self.device)
                timestamps = torch.full((n_agents,), self.current_times[env_id].item(), device=self.device)
                
                for i, agent_id in enumerate(agent_ids):
                    action = actions[agent_id]
                    sides[i] = 0 if action[0] < 0.5 else 1
                    # convert price offset to absolute price
                    best_bid, best_ask = book.get_best_bid_ask()
                    mid = (best_bid + best_ask) / 2 if not np.isnan(best_bid) else 100.0
                    prices[i] = float(mid + action[1])
                    quantities[i] = float(max(1, int(action[2])))
                
                # submit orders
                order_ids, executed_qtys = book.submit_order_batch(
                    trader_ids, sides, prices, quantities, timestamps
                )
                
                results.append((order_ids, executed_qtys))
            else:
                results.append((None, None))
            
            # advance time
            self.current_times[env_id] += 1.0
        
        return results
    
    def get_all_depths(self, levels: int = 5) -> List[dict]:
        # get book depth for all environments
        return [book.get_book_depth(levels) for book in self.books]


def test_batched_sim():
    # test batched simulation
    print("Testing Batched Simulation...")
    
    n_envs = 4
    device = 'cpu'
    
    sim = BatchedSimulation(n_envs=n_envs, device=device)
    sim.reset_all()
    
    print(f"✓ Created batched simulation with {n_envs} environments")
    
    # test batch step with random actions
    actions_batch = []
    for env_id in range(n_envs):
        actions = {
            0: np.array([0.3, 0.5, 10.0]),  # agent 0: buy
            1: np.array([0.7, -0.3, 15.0])  # agent 1: sell
        }
        actions_batch.append(actions)
    
    results = sim.step_batch(actions_batch)
    print(f"✓ Batch step executed")
    print(f"  Results for env 0: {len(results[0][0]) if results[0][0] is not None else 0} orders")
    
    # get depths
    depths = sim.get_all_depths(levels=3)
    print(f"✓ Got depths for all {len(depths)} environments")
    
    print("\n✅ Batched simulation test passed!")


if __name__ == "__main__":
    test_batched_sim()
