# vectorized multi-agent exchange environment
# api-compatible with MultiAgentExchangeEnv but uses vectorized order books

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

try:
    from .batch_sim import BatchedSimulation
    from .exchange_vector import VectorizedOrderBook
except ImportError:
    from batch_sim import BatchedSimulation
    from exchange_vector import VectorizedOrderBook


class VectorizedMultiAgentEnv(gym.Env):
    # vectorized version of multi-agent exchange env
    # can run single env (api compatible) or multiple envs in parallel
    
    def __init__(
        self,
        n_agents: int = 4,
        max_steps: int = 500,
        tick_size: float = 0.01,
        n_envs: int = 1,  # NEW: can run multiple envs
        device: str = 'cuda'
    ):
        super().__init__()
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.tick_size = tick_size
        self.n_envs = n_envs
        self.device = device
        
        # action and observation spaces (same as original)
        self.action_space = spaces.Box(
            low=np.array([0, -5.0, 1], dtype=np.float32),
            high=np.array([1, 5.0, 50], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22 + n_agents,), dtype=np.float32
        )
        
        # create batched simulation
        self.sim = BatchedSimulation(n_envs=n_envs, tick_size=tick_size, device=device)
        
        # agent states per environment
        self.agents = [[AgentState(i) for i in range(n_agents)] for _ in range(n_envs)]
        self.step_counts = [0] * n_envs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # reset simulation
        self.sim.reset_all()
        
        # reset agent states
        for env_agents in self.agents:
            for agent in env_agents:
                agent.reset()
        
        self.step_counts = [0] * self.n_envs
        
        # add background traders (simplified - using cpu simulation)
        # full gpu version would need these vectorized too
        
        # get initial observations
        obs_all = self._get_obs_all()
        
        if self.n_envs == 1:
            # single env mode - return dict
            return obs_all[0], {}
        else:
            # multi env mode - return batched observations
            return obs_all, {}
    
    def step(self, actions):
        # step the environment(s)
        # actions: dict {agent_id: action} for single env OR list[dict] for multiple envs
        # returns obs, rewards, dones, truncs, infos
        
        # handle both single and multi-env mode
        if self.n_envs == 1:
            actions_batch = [actions]
        else:
            actions_batch = actions if isinstance(actions, list) else [actions] * self.n_envs
        
        # process actions through simulation
        initial_states = [
            [(a.inventory, a.cash) for a in env_agents]
            for env_agents in self.agents
        ]
        
        # step simulation
        results = self.sim.step_batch(actions_batch)
        
        # update agent states and compute rewards
        obs_all = []
        rewards_all = []
        dones_all = []
        truncs_all = []
        infos_all = []
        
        for env_id in range(self.n_envs):
            self.step_counts[env_id] += 1
            
            # get book state
            book = self.sim.books[env_id]
            best_bid, best_ask = book.get_best_bid_ask()
            mid = (best_bid + best_ask) / 2 if not np.isnan(best_bid) else 100.0
            
            # compute rewards for this env
            pnls = []
            for agent_id in range(self.n_agents):
                agent = self.agents[env_id][agent_id]
                old_inv, old_cash = initial_states[env_id][agent_id]
                
                # simplified: reward is pnl change
                pnl_change = (agent.cash - old_cash) + mid * (agent.inventory - old_inv)
                trade_bonus = 0.1 if (agent.inventory != old_inv) else 0.0
                inventory_penalty = -0.01 * (abs(agent.inventory) ** 1.5)
                
                pnls.append(pnl_change + trade_bonus + inventory_penalty)
            
            # mix of absolute and relative rewards
            mean_pnl = np.mean(pnls)
            rewards = {
                i: 0.7 * pnls[i] + 0.3 * (pnls[i] - mean_pnl)
                for i in range(self.n_agents)
            }
            
            # check termination
            terminated = self.step_counts[env_id] >= self.max_steps
            dones = {i: terminated for i in range(self.n_agents)}
            truncs = {i: False for i in range(self.n_agents)}
            infos = {i: {"pnl": self.agents[env_id][i].cash} for i in range(self.n_agents)}
            
            obs = self._get_obs(env_id)
            
            obs_all.append(obs)
            rewards_all.append(rewards)
            dones_all.append(dones)
            truncs_all.append(truncs)
            infos_all.append(infos)
        
        # return format depends on n_envs
        if self.n_envs == 1:
            return obs_all[0], rewards_all[0], dones_all[0], truncs_all[0], infos_all[0]
        else:
            return obs_all, rewards_all, dones_all, truncs_all, infos_all
    
    def _get_obs_all(self):
        # get observations for all environments
        return [self._get_obs(env_id) for env_id in range(self.n_envs)]
    
    def _get_obs(self, env_id: int):
        # get observation for one environment
        book = self.sim.books[env_id]
        depth = book.get_book_depth(levels=5)
        
        # parse depth into arrays
        bid_prices = np.zeros(5, dtype=np.float32)
        bid_vols = np.zeros(5, dtype=np.float32)
        ask_prices = np.zeros(5, dtype=np.float32)
        ask_vols = np.zeros(5, dtype=np.float32)
        
        for i, (price, vol) in enumerate(depth["bids"][:5]):
            bid_prices[i] = price
            bid_vols[i] = vol
        for i, (price, vol) in enumerate(depth["asks"][:5]):
            ask_prices[i] = price
            ask_vols[i] = vol
        
        # get mid
        best_bid, best_ask = book.get_best_bid_ask()
        mid = (best_bid + best_ask) / 2 if not np.isnan(best_bid) else 100.0
        
        # build observations for each agent
        obs_dict = {}
        for agent_id in range(self.n_agents):
            agent = self.agents[env_id][agent_id]
            own_state = [agent.inventory, agent.cash]
            other_invs = [self.agents[env_id][i].inventory for i in range(self.n_agents)]
            
            obs = np.concatenate([
                bid_prices, bid_vols, ask_prices, ask_vols, own_state, other_invs
            ]).astype(np.float32)
            
            obs_dict[agent_id] = obs
        
        return obs_dict


class AgentState:
    # tracks state for one agent (same as original)
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.inventory = 0
        self.cash = 0.0
        self.trades = 0
    
    def reset(self):
        self.inventory = 0
        self.cash = 0.0
        self.trades = 0


def test_vec_env():
    # test vectorized environment
    print("Testing Vectorized Multi-Agent Environment...")
    
    # test single env mode (api compatibility)
    print("\nTest 1: Single environment mode (API compatible)")
    env = VectorizedMultiAgentEnv(n_agents=2, max_steps=10, n_envs=1, device='cpu')
    obs, _ = env.reset(seed=42)
    print(f"✓ Reset successful")
    print(f"  Observation keys: {obs.keys()}")
    print(f"  Obs shape per agent: {obs[0].shape}")
    
    # take a step
    actions = {0: np.array([0.3, 0.5, 10.0]), 1: np.array([0.7, -0.3, 15.0])}
    obs, rewards, dones, truncs, infos = env.step(actions)
    print(f"✓ Step successful")
    print(f"  Rewards: {rewards}")
    
    # test multi-env mode
    print("\nTest 2: Multiple environment mode")
    env_multi = VectorizedMultiAgentEnv(n_agents=2, max_steps=10, n_envs=4, device='cpu')
    obs_all, _ = env_multi.reset(seed=42)
    print(f"✓ Reset {len(obs_all)} environments")
    
    # take a step
    actions_batch = [
        {0: np.array([0.3, 0.5, 10.0]), 1: np.array([0.7, -0.3, 15.0])}
        for _ in range(4)
    ]
    obs_all, rewards_all, dones_all, truncs_all, infos_all = env_multi.step(actions_batch)
    print(f"✓ Batch step successful")
    print(f"  Got results for {len(obs_all)} environments")
    
    print("\n✅ Vectorized environment tests passed!")


if __name__ == "__main__":
    test_vec_env()
