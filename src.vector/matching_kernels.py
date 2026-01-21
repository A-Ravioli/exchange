"""
Optimized matching kernels for vectorized order processing.
Implements FIFO and pro-rata matching using PyTorch operations.
"""

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
    """
    Vectorized FIFO matching using PyTorch operations.
    
    Args:
        bid_qtys: (n_levels,) quantities at each bid level
        ask_qtys: (n_levels,) quantities at each ask level
        bid_times: (n_levels,) earliest timestamp at each bid level
        ask_times: (n_levels,) earliest timestamp at each ask level
        aggressive_side: 0 for buy (matches against asks), 1 for sell (matches against bids)
        aggressive_qty: Quantity to match
        aggressive_price_idx: Limit price index
    
    Returns:
        filled_qty: Total quantity matched
        avg_price_idx: Average execution price index
    """
    
    if aggressive_side == 0:  # Buy order matching against asks
        # Find all asks at or below buy price
        matchable = (ask_qtys > 0) & (torch.arange(len(ask_qtys), device=ask_qtys.device) <= aggressive_price_idx)
        matchable_indices = matchable.nonzero(as_tuple=True)[0]
        
        if len(matchable_indices) == 0:
            return 0.0, aggressive_price_idx
        
        # Sort by price (ascending for asks), then time
        matchable_indices = matchable_indices.sort()[0]
        
        # Match sequentially (FIFO)
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
        
    else:  # Sell order matching against bids
        # Find all bids at or above sell price
        matchable = (bid_qtys > 0) & (torch.arange(len(bid_qtys), device=bid_qtys.device) >= aggressive_price_idx)
        matchable_indices = matchable.nonzero(as_tuple=True)[0]
        
        if len(matchable_indices) == 0:
            return 0.0, aggressive_price_idx
        
        # Sort by price (descending for bids)
        matchable_indices = matchable_indices.flip(0)
        
        # Match sequentially (FIFO)
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
    """
    Vectorized pro-rata matching using PyTorch operations.
    Allocates quantity proportionally to resting orders.
    
    Args:
        bid_qtys: (n_levels,) quantities at each bid level
        ask_qtys: (n_levels,) quantities at each ask level
        aggressive_side: 0 for buy, 1 for sell
        aggressive_qty: Quantity to match
        aggressive_price_idx: Limit price index
    
    Returns:
        filled_qty: Total quantity matched
        allocations: (n_levels,) quantity allocated at each level
    """
    
    if aggressive_side == 0:  # Buy order matching against asks
        # Find matchable asks
        matchable = (ask_qtys > 0) & (torch.arange(len(ask_qtys), device=ask_qtys.device) <= aggressive_price_idx)
        
        if not matchable.any():
            return 0.0, torch.zeros_like(ask_qtys)
        
        # Get matchable quantities
        matchable_qtys = ask_qtys.clone()
        matchable_qtys[~matchable] = 0
        
        # Calculate total available
        total_available = matchable_qtys.sum()
        
        if total_available <= aggressive_qty:
            # Fill everything
            return total_available.item(), matchable_qtys
        else:
            # Pro-rata allocation
            allocation_ratios = matchable_qtys / total_available
            allocations = allocation_ratios * aggressive_qty
            return aggressive_qty, allocations
            
    else:  # Sell order matching against bids
        # Find matchable bids
        matchable = (bid_qtys > 0) & (torch.arange(len(bid_qtys), device=bid_qtys.device) >= aggressive_price_idx)
        
        if not matchable.any():
            return 0.0, torch.zeros_like(bid_qtys)
        
        # Get matchable quantities
        matchable_qtys = bid_qtys.clone()
        matchable_qtys[~matchable] = 0
        
        # Calculate total available
        total_available = matchable_qtys.sum()
        
        if total_available <= aggressive_qty:
            # Fill everything
            return total_available.item(), matchable_qtys
        else:
            # Pro-rata allocation
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
    """
    Batch matching of multiple orders against the book.
    
    Args:
        orders_side: (n_orders,) 0=buy, 1=sell
        orders_price: (n_orders,) price indices
        orders_qty: (n_orders,) quantities
        bid_qtys: (n_levels,) bid quantities
        ask_qtys: (n_levels,) ask quantities
        matching_mode: 'fifo' or 'prorata'
    
    Returns:
        executed_qtys: (n_orders,) executed quantities
    """
    n_orders = len(orders_side)
    executed = torch.zeros(n_orders, device=orders_side.device)
    
    # Process each order (sequential for now, can be parallelized for independent orders)
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
    """Test matching kernel implementations."""
    print("Testing Matching Kernels...")
    
    device = 'cpu'
    n_levels = 100
    
    # Create mock book
    bid_qtys = torch.zeros(n_levels, device=device)
    ask_qtys = torch.zeros(n_levels, device=device)
    
    # Add some bids and asks
    bid_qtys[45:50] = torch.tensor([10, 20, 30, 25, 15], device=device)  # Prices 45-49
    ask_qtys[50:55] = torch.tensor([15, 25, 20, 10, 5], device=device)   # Prices 50-54
    
    print(f"Book state:")
    print(f"  Bids at 45-49: {bid_qtys[45:50]}")
    print(f"  Asks at 50-54: {ask_qtys[50:55]}")
    
    # Test FIFO matching - buy order
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
    
    # Test pro-rata matching
    print("\nTest 2: Pro-rata buy order (qty=20, price=52)")
    filled, allocations = prorata_match_vectorized(
        bid_qtys, ask_qtys,
        aggressive_side=0,
        aggressive_qty=20,
        aggressive_price_idx=52
    )
    print(f"  Filled: {filled:.0f}")
    print(f"  Allocations at 50-54: {allocations[50:55]}")
    
    # Test batch matching
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
