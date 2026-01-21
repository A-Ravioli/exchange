from __future__ import annotations

import numpy as np
from typing import Dict, List
import gymnasium as gym
from gymnasium import spaces

from exchange import init_exchange, submit_order, get_best_bid_ask, get_book_depth, Order
from sim import init_sim

# multi-agent competitive environment for discovering trading strategies

class MultiAgentExchangeEnv(gym.Env):
    """multiple agents compete in the same order book"""
    
    def __init__(self, n_agents: int = 4, max_steps: int = 500, tick_size: float = 0.01):
        super().__init__()
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.tick_size = tick_size
        
        # action: [side, price_offset, quantity]
        self.action_space = spaces.Box(
            low=np.array([0, -5.0, 1], dtype=np.float32),
            high=np.array([1, 5.0, 50], dtype=np.float32),
            dtype=np.float32
        )
        
        # observation: book depth + position + other agents' inventories
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22 + n_agents,), dtype=np.float32
        )
        
        self.book = None
        self.sim = None
        self.step_count = 0
        self.agents = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # create fresh book and sim
        self.book = init_exchange(tick_size=self.tick_size, initial_mid_price=100.0)
        self.sim = init_sim(end_time=self.max_steps * 10)
        
        # initialize agents
        self.agents = [AgentState(i) for i in range(self.n_agents)]
        self.step_count = 0
        
        # ADD CONTINUOUS LIQUIDITY PROVIDERS (critical for learning!)
        from algorithms import RandomTrader, MarketMaker
        
        # Add 8 random traders for continuous order flow
        for i in range(8):
            RandomTrader(
                algo_id=800_000 + i,
                book=self.book,
                sim=self.sim,
                seed=(seed + i) if seed else i,
                interval=0.3,  # trade every 0.3 seconds
                max_qty=20,
                price_band=3.0
            )
        
        # Add 3 market makers for continuous liquidity
        for i in range(3):
            MarketMaker(
                algo_id=900_000 + i,
                book=self.book,
                sim=self.sim,
                spread=0.2 + i * 0.1,  # varying spreads (0.2, 0.3, 0.4)
                size=15,
                refresh_interval=0.5
            )
        
        # seed initial liquidity
        self._seed_liquidity()
        self.sim.run_until(5.0)
        
        return self._get_obs_all(), {}
    
    def step(self, actions: Dict[int, np.ndarray]):
        """actions is a dict mapping agent_id -> action"""
        initial_states = [(a.inventory, a.cash) for a in self.agents]
        
        # submit all orders simultaneously (same timestamp)
        for agent_id, action in actions.items():
            self._submit_agent_order(agent_id, action)
        
        # run sim forward
        self.sim.run_until(self.sim.current_time + 1.0)
        self.step_count += 1
        
        # calculate rewards (mix of absolute and relative)
        best_bid, best_ask = get_best_bid_ask(self.book)
        mid = (best_bid + best_ask) / 2 if not np.isnan(best_bid) and not np.isnan(best_ask) else 100.0
        
        pnls = []
        for i, agent in enumerate(self.agents):
            old_inv, old_cash = initial_states[i]
            pnl_change = (agent.cash - old_cash) + mid * (agent.inventory - old_inv)
            
            # Encourage trading (prevents "do nothing" strategy)
            trade_happened = (agent.inventory != old_inv)
            trade_bonus = 0.1 if trade_happened else 0.0
            
            # Penalize excessive inventory (but not too harshly)
            inventory_penalty = -0.01 * (abs(agent.inventory) ** 1.5)
            
            pnls.append(pnl_change + trade_bonus + inventory_penalty)
        
        # Mix of absolute (70%) and relative (30%) rewards
        # This allows agents to profit from background traders while still competing
        mean_pnl = np.mean(pnls)
        rewards = {
            i: 0.7 * pnls[i] + 0.3 * (pnls[i] - mean_pnl)
            for i in range(len(pnls))
        }
        
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        obs_all = self._get_obs_all()
        infos = {i: {"pnl": agent.cash} for i, agent in enumerate(self.agents)}
        
        return obs_all, rewards, {i: terminated for i in range(self.n_agents)}, \
               {i: truncated for i in range(self.n_agents)}, infos
    
    def _submit_agent_order(self, agent_id: int, action: np.ndarray):
        agent = self.agents[agent_id]
        
        # parse action
        side = "buy" if action[0] < 0.5 else "sell"
        price_offset = float(action[1])
        quantity = max(1, int(action[2]))
        
        # get current mid and create order
        best_bid, best_ask = get_best_bid_ask(self.book)
        mid = (best_bid + best_ask) / 2 if not np.isnan(best_bid) and not np.isnan(best_ask) else 100.0
        price = max(self.tick_size, mid + price_offset)
        
        order = Order(
            id=agent.next_order_id(),
            trader_id=agent.agent_id,
            side=side,
            order_type="limit",
            price=price,
            quantity=quantity,
            timestamp=self.sim.current_time,
            time_in_force="ioc" if abs(price_offset) < 0.1 else "gtc"  # aggressive = ioc
        )
        
        # submit and track fills
        trades = submit_order(self.book, order, self.sim.current_time)
        for trade in trades:
            if trade.buy_order_id == order.id:
                agent.inventory += trade.quantity
                agent.cash -= trade.quantity * trade.price
                agent.trades += 1
            elif trade.sell_order_id == order.id:
                agent.inventory -= trade.quantity
                agent.cash += trade.quantity * trade.price
                agent.trades += 1
    
    def _seed_liquidity(self):
        """add some initial orders to bootstrap the book"""
        for i in range(20):
            side = "buy" if i < 10 else "sell"
            offset = -0.5 - i * 0.1 if side == "buy" else 0.5 + (i-10) * 0.1
            order = Order(
                id=999_000_000 + i,
                trader_id=999_999,
                side=side,
                order_type="limit",
                price=100.0 + offset,
                quantity=10,
                timestamp=0.0
            )
            submit_order(self.book, order, 0.0)
    
    def _get_obs_all(self) -> Dict[int, np.ndarray]:
        """get observations for all agents"""
        return {i: self._get_obs(i) for i in range(self.n_agents)}
    
    def _get_obs(self, agent_id: int) -> np.ndarray:
        # get book depth
        depth = get_book_depth(self.book, levels=5)
        
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
        best_bid, best_ask = get_best_bid_ask(self.book)
        mid = (best_bid + best_ask) / 2 if not np.isnan(best_bid) and not np.isnan(best_ask) else 100.0
        
        # own state
        agent = self.agents[agent_id]
        own_state = [agent.inventory, agent.cash]
        
        # other agents' inventories (partial observability)
        other_invs = [self.agents[i].inventory for i in range(self.n_agents)]
        
        obs = np.concatenate([
            bid_prices, bid_vols, ask_prices, ask_vols, own_state, other_invs
        ]).astype(np.float32)
        
        return obs


class AgentState:
    """tracks state for one agent"""
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.inventory = 0
        self.cash = 0.0
        self.trades = 0
        self._next_order_id = agent_id * 1_000_000
    
    def next_order_id(self) -> int:
        self._next_order_id += 1
        return self._next_order_id
    
    def reset(self):
        self.inventory = 0
        self.cash = 0.0
        self.trades = 0
