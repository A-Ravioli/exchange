"""
Algorithms.jl - Trading Strategies

Implements trading algorithms (strategies) that read market data and send orders.
Agents subscribe to market data from Venue.jl and make trading decisions based
on observed market conditions.

Responsibilities:
- Subscribe to market data events (trades, book updates)
- Implement trading logic (market making, arbitrage, momentum, etc.)
- Generate order signals based on market state
- Send orders through Network.jl to Venue.jl
- Track agent-specific state (inventory, PnL, position limits)

Algorithms are event-driven: they react to market data updates by potentially
sending new orders or canceling existing ones. Each algorithm maintains its own
state and decision-making logic.

Key algorithm types (to be implemented):
- Market makers: Provide liquidity, profit from spread
- Arbitrageurs: Exploit price differences
- Momentum traders: Follow trends
- Mean reversion: Bet on price returning to mean

Key functions (to be implemented):
- `create_algorithm(strategy_type, params)`: Initialize algorithm
- `on_market_data(algorithm, event)`: Handle market data update
- `decide_action(algorithm, market_state)`: Generate trading decision
- `send_order(algorithm, order)`: Submit order via Network
- `get_algorithm_state(algorithm)`: Return algorithm's current state (PnL, inventory, etc.)
"""
module Agents

# Trading strategies will be implemented here

end # module

