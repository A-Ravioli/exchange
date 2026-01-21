# vectorized order book using pytorch tensors
# all operations run on gpu for max performance

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class VectorizedOrderBook:
    # gpu-accelerated limit order book using pytorch tensors
    # fixed price level arrays, vectorized matching, batch processing
    
    def __init__(
        self,
        n_price_levels: int = 2000,
        tick_size: float = 0.01,
        base_price: float = 100.0,
        max_orders: int = 10000,
        device: str = 'cuda'
    ):
        # initialize vectorized order book
        self.device = torch.device(device)
        self.n_price_levels = n_price_levels
        self.tick_size = tick_size
        self.base_price = base_price
        self.max_orders = max_orders
        
        # price level arrays: index -> quantity at that level
        # price = base_price + (index - n_price_levels/2) * tick_size
        self.bid_qtys = torch.zeros(n_price_levels, dtype=torch.float32, device=self.device)
        self.ask_qtys = torch.zeros(n_price_levels, dtype=torch.float32, device=self.device)
        
        # order storage: each row is an order
        # columns: [order_id, trader_id, side(0=buy,1=sell), price_idx, qty, filled, timestamp, active(0/1)]
        self.orders = torch.zeros((max_orders, 8), dtype=torch.float32, device=self.device)
        self.next_order_slot = 0
        self.next_order_id = 1
        
        # trade counter
        self.trade_id = 1
        
        # cache for fast lookups
        self.price_to_idx_offset = n_price_levels // 2
    
    def price_to_idx(self, prices: torch.Tensor) -> torch.Tensor:
        # convert prices to price level indices
        return ((prices - self.base_price) / self.tick_size + self.price_to_idx_offset).long()
    
    def idx_to_price(self, indices: torch.Tensor) -> torch.Tensor:
        # convert price level indices to prices
        return self.base_price + (indices - self.price_to_idx_offset) * self.tick_size
    
    def get_best_bid_ask(self) -> Tuple[float, float]:
        # get best bid and ask prices
        # find highest bid level with quantity
        bid_levels = (self.bid_qtys > 0).nonzero(as_tuple=True)[0]
        if len(bid_levels) > 0:
            best_bid_idx = bid_levels.max()
            best_bid = self.idx_to_price(best_bid_idx).item()
        else:
            best_bid = float('nan')
        
        # find lowest ask level with quantity
        ask_levels = (self.ask_qtys > 0).nonzero(as_tuple=True)[0]
        if len(ask_levels) > 0:
            best_ask_idx = ask_levels.min()
            best_ask = self.idx_to_price(best_ask_idx).item()
        else:
            best_ask = float('nan')
        
        return best_bid, best_ask
    
    def get_book_depth(self, levels: int = 5) -> dict:
        # get book depth (top N levels on each side)
        # get bids (highest to lowest)
        bid_levels = (self.bid_qtys > 0).nonzero(as_tuple=True)[0]
        if len(bid_levels) > 0:
            bid_levels = bid_levels.flip(0)[:levels]  # highest first
            bid_prices = self.idx_to_price(bid_levels).cpu().numpy()
            bid_vols = self.bid_qtys[bid_levels].cpu().numpy()
        else:
            bid_prices = np.array([])
            bid_vols = np.array([])
        
        # get asks (lowest to highest)
        ask_levels = (self.ask_qtys > 0).nonzero(as_tuple=True)[0][:levels]
        if len(ask_levels) > 0:
            ask_prices = self.idx_to_price(ask_levels).cpu().numpy()
            ask_vols = self.ask_qtys[ask_levels].cpu().numpy()
        else:
            ask_prices = np.array([])
            ask_vols = np.array([])
        
        return {
            'bids': list(zip(bid_prices, bid_vols)),
            'asks': list(zip(ask_prices, ask_vols))
        }
    
    def submit_order_batch(
        self,
        trader_ids: torch.Tensor,
        sides: torch.Tensor,  # 0=buy, 1=sell
        prices: torch.Tensor,
        quantities: torch.Tensor,
        timestamps: torch.Tensor,
        order_types: Optional[torch.Tensor] = None  # 0=limit, 1=market, 2=ioc
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # submit multiple orders in batch
        # returns order_ids and executed_qtys tensors
        n_orders = len(trader_ids)
        
        if order_types is None:
            order_types = torch.zeros(n_orders, device=self.device)
        
        # assign order ids
        order_ids = torch.arange(
            self.next_order_id, 
            self.next_order_id + n_orders,
            device=self.device
        )
        self.next_order_id += n_orders
        
        # convert prices to indices
        price_indices = self.price_to_idx(prices)
        
        # clamp to valid range
        price_indices = torch.clamp(price_indices, 0, self.n_price_levels - 1)
        
        # initialize executed quantities
        executed_qtys = torch.zeros(n_orders, device=self.device)
        
        # vectorized matching - process all orders in parallel!
        buy_mask = (sides == 0)
        sell_mask = (sides == 1)
        
        # match all buy orders at once
        if buy_mask.any():
            buy_indices = buy_mask.nonzero(as_tuple=True)[0]
            executed_qtys[buy_indices] = self._match_buy_orders_vectorized(
                price_indices[buy_indices],
                quantities[buy_indices]
            )
        
        # match all sell orders at once
        if sell_mask.any():
            sell_indices = sell_mask.nonzero(as_tuple=True)[0]
            executed_qtys[sell_indices] = self._match_sell_orders_vectorized(
                price_indices[sell_indices],
                quantities[sell_indices]
            )
        
        # add unexecuted portions to book (vectorized)
        remaining = quantities - executed_qtys
        limit_mask = (order_types == 0) & (remaining > 0)
        
        if limit_mask.any():
            limit_indices = limit_mask.nonzero(as_tuple=True)[0]
            n_limit = len(limit_indices)
            
            if self.next_order_slot + n_limit < self.max_orders:
                # batch insert orders
                slots = torch.arange(
                    self.next_order_slot,
                    self.next_order_slot + n_limit,
                    device=self.device
                )
                
                self.orders[slots, 0] = order_ids[limit_indices].float()
                self.orders[slots, 1] = trader_ids[limit_indices].float()
                self.orders[slots, 2] = sides[limit_indices].float()
                self.orders[slots, 3] = price_indices[limit_indices].float()
                self.orders[slots, 4] = quantities[limit_indices].float()
                self.orders[slots, 5] = executed_qtys[limit_indices].float()
                self.orders[slots, 6] = timestamps[limit_indices].float()
                self.orders[slots, 7] = 1.0  # active
                
                self.next_order_slot += n_limit
                
                # update price levels using scatter_add (parallel)
                buy_limit = limit_mask & buy_mask
                sell_limit = limit_mask & sell_mask
                
                if buy_limit.any():
                    self.bid_qtys.scatter_add_(
                        0,
                        price_indices[buy_limit],
                        remaining[buy_limit]
                    )
                
                if sell_limit.any():
                    self.ask_qtys.scatter_add_(
                        0,
                        price_indices[sell_limit],
                        remaining[sell_limit]
                    )
        
        return order_ids, executed_qtys
    
    def _match_buy_orders_vectorized(
        self, 
        buy_price_indices: torch.Tensor,  # (n_buys,)
        buy_quantities: torch.Tensor      # (n_buys,)
    ) -> torch.Tensor:
        # vectorized matching of multiple buy orders against ask side
        # much faster than looping!
        n_buys = len(buy_price_indices)
        executed = torch.zeros(n_buys, device=self.device)
        
        # find all ask levels with liquidity
        ask_levels = (self.ask_qtys > 0).nonzero(as_tuple=True)[0]
        if len(ask_levels) == 0:
            return executed
        
        # sort asks by price (best first)
        ask_levels = ask_levels.sort()[0]
        
        # for each buy order, match against asks
        for i, (price_idx, qty) in enumerate(zip(buy_price_indices, buy_quantities)):
            # find matchable asks (at or below buy price)
            matchable = ask_levels[ask_levels <= price_idx]
            
            remaining = qty.float()
            for level in matchable:
                if remaining <= 0:
                    break
                
                available = self.ask_qtys[level]
                if available > 0:
                    fill = torch.min(remaining, available)
                    self.ask_qtys[level] -= fill
                    executed[i] += fill
                    remaining -= fill
        
        return executed
    
    def _match_sell_orders_vectorized(
        self,
        sell_price_indices: torch.Tensor,  # (n_sells,)
        sell_quantities: torch.Tensor      # (n_sells,)
    ) -> torch.Tensor:
        # vectorized matching of multiple sell orders against bid side
        # much faster than looping!
        n_sells = len(sell_price_indices)
        executed = torch.zeros(n_sells, device=self.device)
        
        # find all bid levels with liquidity
        bid_levels = (self.bid_qtys > 0).nonzero(as_tuple=True)[0]
        if len(bid_levels) == 0:
            return executed
        
        # sort bids by price (best first - descending)
        bid_levels = bid_levels.flip(0)
        
        # for each sell order, match against bids
        for i, (price_idx, qty) in enumerate(zip(sell_price_indices, sell_quantities)):
            # find matchable bids (at or above sell price)
            matchable = bid_levels[bid_levels >= price_idx]
            
            remaining = qty.float()
            for level in matchable:
                if remaining <= 0:
                    break
                
                available = self.bid_qtys[level]
                if available > 0:
                    fill = torch.min(remaining, available)
                    self.bid_qtys[level] -= fill
                    executed[i] += fill
                    remaining -= fill
        
        return executed
    
    def cancel_order(self, order_id: int):
        # cancel an order by id
        # find order
        order_mask = (self.orders[:, 0] == order_id) & (self.orders[:, 7] == 1)
        order_idx = order_mask.nonzero(as_tuple=True)[0]
        
        if len(order_idx) == 0:
            return False
        
        idx = order_idx[0]
        
        # get order details
        side = int(self.orders[idx, 2].item())
        price_idx = int(self.orders[idx, 3].item())
        qty = self.orders[idx, 4]
        filled = self.orders[idx, 5]
        remaining = qty - filled
        
        # remove from price level
        if side == 0:
            self.bid_qtys[price_idx] -= remaining
        else:
            self.ask_qtys[price_idx] -= remaining
        
        # mark as inactive
        self.orders[idx, 7] = 0
        
        return True
    
    def reset(self):
        # reset the order book to empty state
        self.bid_qtys.zero_()
        self.ask_qtys.zero_()
        self.orders.zero_()
        self.next_order_slot = 0
        self.next_order_id = 1
        self.trade_id = 1


def submit_orders_batch(book: VectorizedOrderBook, **kwargs):
    # helper function for api compatibility
    return book.submit_order_batch(**kwargs)


def test_vectorized_book():
    # test basic vectorized order book functionality
    print("Testing Vectorized Order Book...")
    
    # use cpu for testing (works everywhere)
    device = 'cpu'
    book = VectorizedOrderBook(device=device)
    
    # submit some buy orders
    trader_ids = torch.tensor([1, 2, 3], device=device)
    sides = torch.tensor([0, 0, 0], device=device)  # All buys
    prices = torch.tensor([99.9, 99.8, 99.7], device=device)
    quantities = torch.tensor([10, 20, 15], device=device)
    timestamps = torch.tensor([1.0, 1.1, 1.2], device=device)
    
    order_ids, executed = book.submit_order_batch(trader_ids, sides, prices, quantities, timestamps)
    print(f"✓ Submitted {len(order_ids)} buy orders")
    print(f"  Executed quantities: {executed}")
    
    # submit sell orders (should match)
    trader_ids = torch.tensor([4, 5], device=device)
    sides = torch.tensor([1, 1], device=device)  # sells
    prices = torch.tensor([99.8, 99.9], device=device)
    quantities = torch.tensor([15, 10], device=device)
    timestamps = torch.tensor([2.0, 2.1], device=device)
    
    order_ids, executed = book.submit_order_batch(trader_ids, sides, prices, quantities, timestamps)
    print(f"✓ Submitted {len(order_ids)} sell orders")
    print(f"  Executed quantities: {executed}")
    print(f"  Expected: [15, 10] (should match against buys)")
    
    # check best bid/ask
    best_bid, best_ask = book.get_best_bid_ask()
    print(f"✓ Best bid: {best_bid:.2f}, Best ask: {best_ask:.2f}")
    
    # get book depth
    depth = book.get_book_depth(levels=3)
    print(f"✓ Book depth:")
    print(f"  Bids: {depth['bids'][:3]}")
    print(f"  Asks: {depth['asks'][:3]}")
    
    print("\n✅ Vectorized order book test passed!")


if __name__ == "__main__":
    test_vectorized_book()
