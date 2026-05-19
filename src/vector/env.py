from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    class _Env:
        def reset(self, seed=None, options=None):
            return None

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = shape if shape is not None else np.asarray(low).shape
            self.dtype = dtype

        def sample(self):
            low = np.broadcast_to(self.low, self.shape).astype(np.float32)
            high = np.broadcast_to(self.high, self.shape).astype(np.float32)
            low = np.where(np.isfinite(low), low, -1.0)
            high = np.where(np.isfinite(high), high, 1.0)
            return np.random.uniform(low, high).astype(self.dtype)

    class _Spaces:
        Box = _Box

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()


ArrayLikeAction = Union[np.ndarray, torch.Tensor, List[float], Tuple[float, float, float]]


def select_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


@dataclass
class ExecutionResult:
    fill_qty: torch.Tensor
    avg_price: torch.Tensor
    passive_agent_fill_qty: torch.Tensor
    passive_agent_cash: torch.Tensor


class BatchedTrainingBook:
    """Aggregate tensor book for simplified high-throughput RL rollouts."""

    def __init__(
        self,
        n_envs: int,
        n_agents: int,
        tick_size: float,
        device: torch.device,
        n_price_levels: int = 2001,
        base_price: float = 100.0,
        background_depth: float = 120.0,
    ) -> None:
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.tick_size = tick_size
        self.device = device
        self.n_price_levels = n_price_levels
        self.base_price = base_price
        self.center_idx = n_price_levels // 2
        self.background_depth = background_depth

        self.levels = torch.arange(n_price_levels, device=device)
        self.prices = base_price + (self.levels.float() - self.center_idx) * tick_size

        shape = (n_envs, n_agents, n_price_levels)
        self.agent_bids = torch.zeros(shape, device=device)
        self.agent_asks = torch.zeros(shape, device=device)
        self.background_bids = torch.zeros((n_envs, n_price_levels), device=device)
        self.background_asks = torch.zeros((n_envs, n_price_levels), device=device)
        self.last_mid_idx = torch.full((n_envs,), self.center_idx, dtype=torch.long, device=device)

    def reset(self, generator: Optional[torch.Generator] = None) -> None:
        self.agent_bids.zero_()
        self.agent_asks.zero_()
        self.background_bids.zero_()
        self.background_asks.zero_()
        self.last_mid_idx.fill_(self.center_idx)
        self._seed_background(generator)

    def _seed_background(self, generator: Optional[torch.Generator] = None) -> None:
        distance = (self.levels - self.center_idx).abs().float()
        profile = torch.clamp(1.0 - distance / 240.0, min=0.0) * self.background_depth
        bids = torch.where(self.levels < self.center_idx, profile, torch.zeros_like(profile))
        asks = torch.where(self.levels > self.center_idx, profile, torch.zeros_like(profile))
        self.background_bids[:] = bids.unsqueeze(0)
        self.background_asks[:] = asks.unsqueeze(0)
        noise_shape = self.background_bids.shape
        bid_noise = torch.rand(noise_shape, generator=generator, device=self.device) * 4.0
        ask_noise = torch.rand(noise_shape, generator=generator, device=self.device) * 4.0
        self.background_bids.add_(torch.where(self.background_bids > 0, bid_noise, 0.0))
        self.background_asks.add_(torch.where(self.background_asks > 0, ask_noise, 0.0))

    def aggregate_bids(self) -> torch.Tensor:
        return self.background_bids + self.agent_bids.sum(dim=1)

    def aggregate_asks(self) -> torch.Tensor:
        return self.background_asks + self.agent_asks.sum(dim=1)

    def best_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        bids = self.aggregate_bids()
        asks = self.aggregate_asks()

        bid_mask = bids > 1e-6
        ask_mask = asks > 1e-6

        bid_candidates = torch.where(bid_mask, self.levels.unsqueeze(0), torch.zeros_like(bids, dtype=torch.long))
        ask_candidates = torch.where(
            ask_mask,
            self.levels.unsqueeze(0),
            torch.full_like(asks, self.n_price_levels - 1, dtype=torch.long),
        )

        best_bid = bid_candidates.max(dim=1).values
        best_ask = ask_candidates.min(dim=1).values
        best_bid = torch.where(bid_mask.any(dim=1), best_bid, self.last_mid_idx - 1)
        best_ask = torch.where(ask_mask.any(dim=1), best_ask, self.last_mid_idx + 1)
        best_bid = best_bid.clamp(0, self.n_price_levels - 1)
        best_ask = best_ask.clamp(0, self.n_price_levels - 1)
        return best_bid, best_ask

    def mid_indices(self) -> torch.Tensor:
        best_bid, best_ask = self.best_indices()
        mid = ((best_bid + best_ask) // 2).clamp(0, self.n_price_levels - 1)
        self.last_mid_idx = mid
        return mid

    def add_background_flow(self, generator: Optional[torch.Generator] = None) -> None:
        mid = self.mid_indices()
        env_idx = torch.arange(self.n_envs, device=self.device)

        passive_size = torch.rand((self.n_envs,), generator=generator, device=self.device) * 10.0 + 8.0
        bid_idx = (mid - torch.randint(4, 24, (self.n_envs,), generator=generator, device=self.device)).clamp(0, self.n_price_levels - 1)
        ask_idx = (mid + torch.randint(4, 24, (self.n_envs,), generator=generator, device=self.device)).clamp(0, self.n_price_levels - 1)
        self.background_bids[env_idx, bid_idx] += passive_size
        self.background_asks[env_idx, ask_idx] += passive_size

        market_side = torch.rand((self.n_envs,), generator=generator, device=self.device) < 0.5
        market_qty = torch.randint(1, 12, (self.n_envs,), generator=generator, device=self.device).float()
        buy_price = (mid + 80).clamp(0, self.n_price_levels - 1)
        sell_price = (mid - 80).clamp(0, self.n_price_levels - 1)
        sides = torch.where(market_side, torch.zeros_like(mid), torch.ones_like(mid))
        prices = torch.where(market_side, buy_price, sell_price)
        self.execute_batch(sides, prices, market_qty, taker_agent=None)

        self.background_bids.mul_(0.997)
        self.background_asks.mul_(0.997)

    def execute_batch(
        self,
        sides: torch.Tensor,
        price_indices: torch.Tensor,
        quantities: torch.Tensor,
        taker_agent: Optional[int],
    ) -> ExecutionResult:
        sides = sides.long()
        price_indices = price_indices.long().clamp(0, self.n_price_levels - 1)
        quantities = quantities.float().clamp_min(0.0)

        is_buy = sides == 0
        price_grid = self.levels.unsqueeze(0)

        ask_total = self.background_asks + self.agent_asks.sum(dim=1)
        bid_total = self.background_bids + self.agent_bids.sum(dim=1)
        if taker_agent is not None:
            ask_total = (ask_total - self.agent_asks[:, taker_agent, :]).clamp_min(0.0)
            bid_total = (bid_total - self.agent_bids[:, taker_agent, :]).clamp_min(0.0)
        matchable_asks = ask_total * (price_grid <= price_indices.unsqueeze(1)).float()
        matchable_bids = bid_total * (price_grid >= price_indices.unsqueeze(1)).float()

        available = torch.where(is_buy, matchable_asks.sum(dim=1), matchable_bids.sum(dim=1))
        fill_qty = torch.minimum(quantities, available)
        ratio = torch.where(available > 1e-6, fill_qty / available.clamp_min(1e-6), torch.zeros_like(fill_qty))

        ask_alloc = matchable_asks * (ratio * is_buy.float()).unsqueeze(1)
        bid_alloc = matchable_bids * (ratio * (~is_buy).float()).unsqueeze(1)

        ask_value = (ask_alloc * self.prices.unsqueeze(0)).sum(dim=1)
        bid_value = (bid_alloc * self.prices.unsqueeze(0)).sum(dim=1)
        total_value = torch.where(is_buy, ask_value, bid_value)
        avg_price = torch.where(fill_qty > 1e-6, total_value / fill_qty.clamp_min(1e-6), self.prices[price_indices])

        bg_ask_share = torch.where(ask_total > 1e-6, self.background_asks / ask_total.clamp_min(1e-6), torch.zeros_like(ask_total))
        bg_bid_share = torch.where(bid_total > 1e-6, self.background_bids / bid_total.clamp_min(1e-6), torch.zeros_like(bid_total))
        bg_ask_remove = torch.minimum(self.background_asks, ask_alloc * bg_ask_share)
        bg_bid_remove = torch.minimum(self.background_bids, bid_alloc * bg_bid_share)
        self.background_asks.sub_(bg_ask_remove).clamp_(min=0.0)
        self.background_bids.sub_(bg_bid_remove).clamp_(min=0.0)

        passive_qty = torch.zeros((self.n_envs, self.n_agents), device=self.device)
        passive_cash = torch.zeros((self.n_envs, self.n_agents), device=self.device)

        for agent_id in range(self.n_agents):
            agent_ask = self.agent_asks[:, agent_id, :]
            agent_bid = self.agent_bids[:, agent_id, :]
            ask_share = torch.where(ask_total > 1e-6, agent_ask / ask_total.clamp_min(1e-6), torch.zeros_like(agent_ask))
            bid_share = torch.where(bid_total > 1e-6, agent_bid / bid_total.clamp_min(1e-6), torch.zeros_like(agent_bid))
            agent_ask_remove = torch.minimum(agent_ask, ask_alloc * ask_share)
            agent_bid_remove = torch.minimum(agent_bid, bid_alloc * bid_share)

            if taker_agent == agent_id:
                agent_ask_remove.zero_()
                agent_bid_remove.zero_()

            ask_qty = agent_ask_remove.sum(dim=1)
            bid_qty = agent_bid_remove.sum(dim=1)
            ask_cash = (agent_ask_remove * self.prices.unsqueeze(0)).sum(dim=1)
            bid_cash = (agent_bid_remove * self.prices.unsqueeze(0)).sum(dim=1)

            passive_qty[:, agent_id] += bid_qty - ask_qty
            passive_cash[:, agent_id] += ask_cash - bid_cash

            agent_ask.sub_(agent_ask_remove).clamp_(min=0.0)
            agent_bid.sub_(agent_bid_remove).clamp_(min=0.0)

        return ExecutionResult(fill_qty, avg_price, passive_qty, passive_cash)

    def place_quotes(
        self,
        agent_id: int,
        sides: torch.Tensor,
        price_indices: torch.Tensor,
        quantities: torch.Tensor,
        filled: torch.Tensor,
    ) -> None:
        self.agent_bids[:, agent_id, :].zero_()
        self.agent_asks[:, agent_id, :].zero_()

        remaining = (quantities - filled).clamp_min(0.0)
        rest_mask = remaining > 1e-6
        if not rest_mask.any():
            return

        env_idx = torch.arange(self.n_envs, device=self.device)[rest_mask]
        price_idx = price_indices[rest_mask]
        qty = remaining[rest_mask]
        sides = sides[rest_mask]

        buy_mask = sides == 0
        if buy_mask.any():
            self.agent_bids[env_idx[buy_mask], agent_id, price_idx[buy_mask]] = qty[buy_mask]
        if (~buy_mask).any():
            self.agent_asks[env_idx[~buy_mask], agent_id, price_idx[~buy_mask]] = qty[~buy_mask]

    def top_depth(self, levels: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bids = self.aggregate_bids()
        asks = self.aggregate_asks()
        bid_prices = torch.zeros((self.n_envs, levels), device=self.device)
        bid_vols = torch.zeros((self.n_envs, levels), device=self.device)
        ask_prices = torch.zeros((self.n_envs, levels), device=self.device)
        ask_vols = torch.zeros((self.n_envs, levels), device=self.device)

        neg_inf = torch.full_like(bids, -1.0)
        bid_scores = torch.where(bids > 1e-6, self.levels.float().unsqueeze(0), neg_inf)
        bid_idx = torch.topk(bid_scores, k=levels, dim=1).indices
        bid_vols = bids.gather(1, bid_idx)
        bid_prices = self.prices[bid_idx]
        bid_prices = torch.where(bid_vols > 1e-6, bid_prices, torch.zeros_like(bid_prices))

        large = torch.full_like(asks, float(self.n_price_levels + 1))
        ask_scores = torch.where(asks > 1e-6, self.levels.float().unsqueeze(0), large)
        ask_idx = torch.topk(-ask_scores, k=levels, dim=1).indices
        ask_vols = asks.gather(1, ask_idx)
        ask_prices = self.prices[ask_idx]
        ask_prices = torch.where(ask_vols > 1e-6, ask_prices, torch.zeros_like(ask_prices))
        return bid_prices, bid_vols, ask_prices, ask_vols


class VectorizedMultiAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_agents: int = 4,
        max_steps: int = 500,
        tick_size: float = 0.01,
        n_envs: int = 64,
        device: str = "auto",
        return_tensors: bool = False,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.tick_size = tick_size
        self.n_envs = n_envs
        self.device = select_device(device)
        self.return_tensors = return_tensors

        self.action_space = spaces.Box(
            low=np.array([0, -5.0, 1], dtype=np.float32),
            high=np.array([1, 5.0, 50], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22 + n_agents,), dtype=np.float32
        )

        self.book = BatchedTrainingBook(n_envs, n_agents, tick_size, self.device)
        self.inventory = torch.zeros((n_envs, n_agents), device=self.device)
        self.cash = torch.zeros((n_envs, n_agents), device=self.device)
        self.trades = torch.zeros((n_envs, n_agents), device=self.device)
        self.step_count = 0
        self.generator: Optional[torch.Generator] = None
        self.last_execution: Optional[Dict[str, torch.Tensor]] = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.device.type == "cpu":
            self.generator = torch.Generator(device="cpu")
            if seed is not None:
                self.generator.manual_seed(int(seed))
        else:
            self.generator = None
            if seed is not None:
                torch.manual_seed(int(seed))

        self.inventory.zero_()
        self.cash.zero_()
        self.trades.zero_()
        self.step_count = 0
        self.book.reset(self.generator)
        self.book.add_background_flow(self.generator)
        return self._format_obs(self._observations()), {}

    def step(self, actions):
        action_tensor = self._actions_to_tensor(actions)
        old_value = self._portfolio_value()

        passive_qty_total = torch.zeros_like(self.inventory)
        passive_cash_total = torch.zeros_like(self.cash)
        taker_fill_qty = torch.zeros_like(self.inventory)
        taker_avg_price = torch.zeros_like(self.inventory)

        for agent_id in range(self.n_agents):
            agent_actions = action_tensor[:, agent_id, :]
            sides = (agent_actions[:, 0] >= 0.5).long()
            mid = self.book.mid_indices()
            price_offsets = agent_actions[:, 1].clamp(-5.0, 5.0)
            price_indices = (mid + torch.round(price_offsets / self.tick_size).long()).clamp(
                0, self.book.n_price_levels - 1
            )
            quantities = agent_actions[:, 2].clamp(1.0, 50.0).round()

            result = self.book.execute_batch(sides, price_indices, quantities, taker_agent=agent_id)
            buy_mask = sides == 0
            self.inventory[:, agent_id] += torch.where(buy_mask, result.fill_qty, -result.fill_qty)
            self.cash[:, agent_id] += torch.where(
                buy_mask,
                -result.fill_qty * result.avg_price,
                result.fill_qty * result.avg_price,
            )
            self.inventory += result.passive_agent_fill_qty
            self.cash += result.passive_agent_cash
            self.trades[:, agent_id] += (result.fill_qty > 1e-6).float()
            passive_trades = (result.passive_agent_fill_qty.abs() > 1e-6).float()
            self.trades += passive_trades

            passive_qty_total += result.passive_agent_fill_qty
            passive_cash_total += result.passive_agent_cash
            taker_fill_qty[:, agent_id] = result.fill_qty
            taker_avg_price[:, agent_id] = result.avg_price
            self.book.place_quotes(agent_id, sides, price_indices, quantities, result.fill_qty)

        self.book.add_background_flow(self.generator)
        self.step_count += 1

        new_value = self._portfolio_value()
        raw_pnl = new_value - old_value
        trade_bonus = 0.05 * ((taker_fill_qty > 1e-6).float() + (passive_qty_total.abs() > 1e-6).float())
        inventory_penalty = -0.002 * self.inventory.abs().pow(1.3)
        pnls = raw_pnl + trade_bonus + inventory_penalty
        mean_pnl = pnls.mean(dim=1, keepdim=True)
        rewards_tensor = torch.clamp(0.7 * pnls + 0.3 * (pnls - mean_pnl), -100.0, 100.0)

        terminated = self.step_count >= self.max_steps
        dones_tensor = torch.full((self.n_envs, self.n_agents), terminated, device=self.device, dtype=torch.bool)
        truncs_tensor = torch.zeros_like(dones_tensor)

        self.last_execution = {
            "taker_fill_qty": taker_fill_qty.detach(),
            "taker_avg_price": taker_avg_price.detach(),
            "passive_agent_fill_qty": passive_qty_total.detach(),
            "passive_agent_cash": passive_cash_total.detach(),
        }

        obs = self._format_obs(self._observations())
        rewards = self._format_agent_dict(rewards_tensor)
        dones = self._format_agent_dict(dones_tensor)
        truncs = self._format_agent_dict(truncs_tensor)
        infos = {
            i: {
                "pnl": self._to_output(new_value[:, i]),
                "inventory": self._to_output(self.inventory[:, i]),
                "cash": self._to_output(self.cash[:, i]),
            }
            for i in range(self.n_agents)
        }
        return obs, rewards, dones, truncs, infos

    def close(self) -> None:
        pass

    def _actions_to_tensor(self, actions) -> torch.Tensor:
        if isinstance(actions, dict):
            stacked = []
            for agent_id in range(self.n_agents):
                action = actions[agent_id]
                if isinstance(action, torch.Tensor):
                    tensor = action.to(self.device, dtype=torch.float32)
                else:
                    tensor = torch.as_tensor(action, device=self.device, dtype=torch.float32)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0).expand(self.n_envs, -1)
                stacked.append(tensor)
            return torch.stack(stacked, dim=1)

        per_env = []
        for env_actions in actions:
            per_env.append([env_actions[i] for i in range(self.n_agents)])
        return torch.as_tensor(per_env, device=self.device, dtype=torch.float32)

    def _portfolio_value(self) -> torch.Tensor:
        mid_idx = self.book.mid_indices()
        mid_price = self.book.prices[mid_idx].unsqueeze(1)
        return self.cash + self.inventory * mid_price

    def _observations(self) -> Dict[int, torch.Tensor]:
        bid_prices, bid_vols, ask_prices, ask_vols = self.book.top_depth(levels=5)
        cash_norm = torch.clamp(self.cash / 1000.0, -100.0, 100.0)
        inv_clip = torch.clamp(self.inventory, -100.0, 100.0)
        observations = {}
        for agent_id in range(self.n_agents):
            own_state = torch.stack([inv_clip[:, agent_id], cash_norm[:, agent_id]], dim=1)
            obs = torch.cat(
                [bid_prices, bid_vols, ask_prices, ask_vols, own_state, inv_clip],
                dim=1,
            )
            observations[agent_id] = torch.nan_to_num(obs, nan=0.0, posinf=1000.0, neginf=-1000.0).clamp(
                -1000.0, 1000.0
            )
        return observations

    def _format_obs(self, obs: Dict[int, torch.Tensor]) -> Dict[int, Union[np.ndarray, torch.Tensor]]:
        return {agent_id: self._to_output(value) for agent_id, value in obs.items()}

    def _format_agent_dict(self, values: torch.Tensor) -> Dict[int, Union[np.ndarray, torch.Tensor]]:
        return {agent_id: self._to_output(values[:, agent_id]) for agent_id in range(self.n_agents)}

    def _to_output(self, value: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
        value = value.detach()
        if self.return_tensors:
            return value
        return value.cpu().numpy()
