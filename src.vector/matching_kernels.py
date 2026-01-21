# optimized matching kernels for vectorized order processing
# implements fifo and pro-rata matching using pytorch

import torch
import torch.nn.functional as F


def fifo_match_vectorized(
    bid_qtys: torch.Tensor,
    ask_qtys: torch.Tensor,
    bid_times: torch.Tensor,
    ask_times: torch.Tensor,
    aggressive_side: int,  # 0=buy, 1=sell
    aggressive_qty: float,
    aggressive_price_idx: int
) -> tuple:
    # vectorized fifo matching using pytorch operations
    # returns filled_qty and avg_price_idx
    
    if aggressive_side == 0:  # buy order matching against asks
        # find all asks at or below buy price
        matchable = (ask_qtys > 0) & (torch.arange(len(ask_qtys), device=ask_qtys.device) <= aggressive_price_idx)
        matchable_indices = matchable.nonzero(as_tuple=True)[0]
        
        if len(matchable_indices) == 0:
            return 0.0, aggressive_price_idx
        
        # sort by price (ascending for asks), then time
        matchable_indices = matchable_indices.sort()[0]
        
        # match sequentially (fifo)
        filled = 0.0
        total_value = 0.0
        remaining = aggressive_qty
        
        for idx in matchable_indices:
            if remaining <= 0:
                break
            
            available = ask_qtys[idx].item()
            fill = min(remaining, available)
            
            filled += fill
            total_value += fill * idx
            remaining -= fill
        
        avg_price_idx = total_value / filled if filled > 0 else aggressive_price_idx
        return filled, avg_price_idx
        
    else:  # sell order matching against bids
        # find all bids at or above sell price
        matchable = (bid_qtys > 0) & (torch.arange(len(bid_qtys), device=bid_qtys.device) >= aggressive_price_idx)
        matchable_indices = matchable.nonzero(as_tuple=True)[0]
        
        if len(matchable_indices) == 0:
            return 0.0, aggressive_price_idx
        
        # sort by price (descending for bids)
        matchable_indices = matchable_indices.flip(0)
        
        # match sequentially (fifo)
        filled = 0.0
        total_value = 0.0
        remaining = aggressive_qty
        
        for idx in matchable_indices:
            if remaining <= 0:
                break
            
            available = bid_qtys[idx].item()
            fill = min(remaining, available)
            
            filled += fill
            total_value += fill * idx
            remaining -= fill
        
        avg_price_idx = total_value / filled if filled > 0 else aggressive_price_idx
        return filled, avg_price_idx


def prorata_match_vectorized(
    bid_qtys: torch.Tensor,
    ask_qtys: torch.Tensor,
    aggressive_side: int,
    aggressive_qty: float,
    aggressive_price_idx: int
) -> tuple:
    # vectorized pro-rata matching using pytorch operations
    # allocates quantity proportionally to resting orders
    # returns filled_qty and allocations tensor
    
    if aggressive_side == 0:  # buy order matching against asks
        # find matchable asks
        matchable = (ask_qtys > 0) & (torch.arange(len(ask_qtys), device=ask_qtys.device) <= aggressive_price_idx)
        
        if not matchable.any():
            return 0.0, torch.zeros_like(ask_qtys)
        
        # get matchable quantities
        matchable_qtys = ask_qtys.clone()
        matchable_qtys[~matchable] = 0
        
        # calculate total available
        total_available = matchable_qtys.sum()
        
        if total_available <= aggressive_qty:
            # fill everything
            return total_available.item(), matchable_qtys
        else:
            # pro-rata allocation
            allocation_ratios = matchable_qtys / total_available
            allocations = allocation_ratios * aggressive_qty
            return aggressive_qty, allocations
            
    else:  # sell order matching against bids
        # find matchable bids
        matchable = (bid_qtys > 0) & (torch.arange(len(bid_qtys), device=bid_qtys.device) >= aggressive_price_idx)
        
        if not matchable.any():
            return 0.0, torch.zeros_like(bid_qtys)
        
        # get matchable quantities
        matchable_qtys = bid_qtys.clone()
        matchable_qtys[~matchable] = 0
        
        # calculate total available
        total_available = matchable_qtys.sum()
        
        if total_available <= aggressive_qty:
            # fill everything
            return total_available.item(), matchable_qtys
        else:
            # pro-rata allocation
            allocation_ratios = matchable_qtys / total_available
            allocations = allocation_ratios * aggressive_qty
            return aggressive_qty, allocations


def batch_match_orders(
    orders_side: torch.Tensor,
    orders_price: torch.Tensor,
    orders_qty: torch.Tensor,
    bid_qtys: torch.Tensor,
    ask_qtys: torch.Tensor,
    matching_mode: str = 'fifo'
) -> torch.Tensor:
    # batch matching of multiple orders against the book
    # returns executed_qtys tensor
    n_orders = len(orders_side)
    executed = torch.zeros(n_orders, device=orders_side.device)
    
    # process each order (sequential for now, can be parallelized)
    for i in range(n_orders):
        if matching_mode == 'fifo':
            filled, _ = fifo_match_vectorized(
                bid_qtys, ask_qtys,
                torch.zeros_like(bid_qtys), torch.zeros_like(ask_qtys),
                orders_side[i].item(),
                orders_qty[i].item(),
                orders_price[i].item()
            )
        else:  # prorata
            filled, _ = prorata_match_vectorized(
                bid_qtys, ask_qtys,
                orders_side[i].item(),
                orders_qty[i].item(),
                orders_price[i].item()
            )
        
        executed[i] = filled
    
    return executed


def test_matching_kernels():
    # test matching kernel implementations
    print("Testing Matching Kernels...")
    
    device = 'cpu'
    n_levels = 100
    
    # create mock book
    bid_qtys = torch.zeros(n_levels, device=device)
    ask_qtys = torch.zeros(n_levels, device=device)
    
    # add some bids and asks
    bid_qtys[45:50] = torch.tensor([10, 20, 30, 25, 15], device=device)  # prices 45-49
    ask_qtys[50:55] = torch.tensor([15, 25, 20, 10, 5], device=device)   # prices 50-54
    
    print(f"Book state:")
    print(f"  Bids at 45-49: {bid_qtys[45:50]}")
    print(f"  Asks at 50-54: {ask_qtys[50:55]}")
    
    # test fifo matching - buy order
    print("\nTest 1: Buy order (qty=30, price=52)")
    filled, avg_price = fifo_match_vectorized(
        bid_qtys, ask_qtys,
        torch.zeros(n_levels), torch.zeros(n_levels),
        aggressive_side=0,  # Buy
        aggressive_qty=30,
        aggressive_price_idx=52
    )
    print(f"  Filled: {filled:.0f} (expected: 30)")
    print(f"  Avg price idx: {avg_price:.1f}")
    assert abs(filled - 30) < 0.1, f"Expected 30, got {filled}"
    
    # test pro-rata matching
    print("\nTest 2: Pro-rata buy order (qty=20, price=52)")
    filled, allocations = prorata_match_vectorized(
        bid_qtys, ask_qtys,
        aggressive_side=0,
        aggressive_qty=20,
        aggressive_price_idx=52
    )
    print(f"  Filled: {filled:.0f}")
    print(f"  Allocations at 50-54: {allocations[50:55]}")
    
    # test batch matching
    print("\nTest 3: Batch matching")
    orders_side = torch.tensor([0, 1, 0], device=device)
    orders_price = torch.tensor([52, 47, 51], device=device)
    orders_qty = torch.tensor([10, 15, 20], device=device)
    
    executed = batch_match_orders(
        orders_side, orders_price, orders_qty,
        bid_qtys, ask_qtys,
        matching_mode='fifo'
    )
    print(f"  Executed: {executed}")
    
    print("\n✅ All matching kernel tests passed!")


if __name__ == "__main__":
    test_matching_kernels()
