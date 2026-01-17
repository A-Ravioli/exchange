from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from exchange import init_exchange, submit_order, get_best_bid_ask, get_book_depth, Order
from sim import init_sim
from algorithms import RandomTrader

# minimal gym wrapper for the exchange

class ExchangeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, max_steps: int = 1000, tick_size: float = 0.01):
        super().__init__()
        self.max_steps = max_steps
        self.tick_size = tick_size
        
        # action: [side, price_offset, quantity] where side in {0=buy, 1=sell}
        self.action_space = spaces.Box(
            low=np.array([0, -10.0, 1], dtype=np.float32),
            high=np.array([1, 10.0, 100], dtype=np.float32),
            dtype=np.float32
        )
        
        # observation: flattened book depth (5 levels each side) + position info
        # [bid_prices(5), bid_vols(5), ask_prices(5), ask_vols(5), inventory, cash, mid_price]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )
        
        self.book = None
        self.sim = None
        self.step_count = 0
        self.agent_id = 999
        self.inventory = 0
        self.cash = 0.0
        self.next_order_id = self.agent_id * 1_000_000
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # create fresh book and sim
        self.book = init_exchange(tick_size=self.tick_size, initial_mid_price=100.0)
        self.sim = init_sim(end_time=self.max_steps)
        
        # add random traders for liquidity
        for i in range(3):
            RandomTrader(i, self.book, self.sim, seed=seed+i if seed else i, 
                        interval=0.5, max_qty=10, price_band=2.0)
        
        # run sim a bit to build initial book
        self.sim.run_until(5.0)
        
        self.step_count = 0
        self.inventory = 0
        self.cash = 0.0
        
        return self._get_obs(), {}
    
    def step(self, action):
        # parse action
        side = "buy" if action[0] < 0.5 else "sell"
        price_offset = float(action[1])
        quantity = max(1, int(action[2]))
        
        # get current mid and create order
        best_bid, best_ask = get_best_bid_ask(self.book)
        mid = (best_bid + best_ask) / 2 if not np.isnan(best_bid) and not np.isnan(best_ask) else 100.0
        price = max(self.tick_size, mid + price_offset)
        
        order = Order(
            id=self.next_order_id,
            trader_id=self.agent_id,
            side=side,
            order_type="limit",
            price=price,
            quantity=quantity,
            timestamp=self.sim.current_time
        )
        self.next_order_id += 1
        
        # track initial state for reward
        initial_cash = self.cash
        initial_inventory = self.inventory
        
        # submit order and update position
        trades = submit_order(self.book, order, self.sim.current_time)
        for trade in trades:
            if trade.buy_order_id == order.id:
                self.inventory += trade.quantity
                self.cash -= trade.quantity * trade.price
            elif trade.sell_order_id == order.id:
                self.inventory -= trade.quantity
                self.cash += trade.quantity * trade.price
        
        # run sim forward
        self.sim.run_until(self.sim.current_time + 1.0)
        self.step_count += 1
        
        # calculate reward (pnl change + inventory penalty)
        pnl_change = (self.cash - initial_cash) + mid * (self.inventory - initial_inventory)
        inventory_penalty = -0.01 * abs(self.inventory)
        reward = pnl_change + inventory_penalty
        
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _get_obs(self):
        # get book depth
        depth = get_book_depth(self.book, levels=5)
        
        # extract prices and volumes (pad with zeros if needed)
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
        
        # get mid price
        best_bid, best_ask = get_best_bid_ask(self.book)
        mid = (best_bid + best_ask) / 2 if not np.isnan(best_bid) and not np.isnan(best_ask) else 100.0
        
        # combine into observation
        obs = np.concatenate([
            bid_prices, bid_vols, ask_prices, ask_vols,
            [self.inventory, self.cash, mid]
        ]).astype(np.float32)
        
        return obs
    
    def render(self):
        if self.book is not None:
            from visualizer import visualize_book_simple
            visualize_book_simple(self.book, levels=5)
            print(f"\nAgent - Inventory: {self.inventory}, Cash: ${self.cash:.2f}, PnL: ${self.cash:.2f}")
