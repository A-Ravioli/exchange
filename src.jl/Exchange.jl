"""
Exchange.jl - Exchange Simulator

Implements the exchange where trading occurs. Manages the limit order book
(LOB), order matching engine, and market data emission.

Responsibilities:
- Maintain limit order book state (bid/ask levels, order queues)
- Process incoming orders (limit orders, market orders, cancellations)
- Execute matching logic (price-time priority)
- Emit market data events (trades, book updates, top-of-book changes)
- Track order fills and partial fills
- Provide book state queries (best bid/ask, depth, spread)

The exchange operates on discrete events scheduled through Sim.jl. When an order
arrives (via Network.jl), it is processed immediately, potentially generating:
- Trade events (if matched)
- Book update events (if order is added to book)
- Market data broadcasts (to all subscribed algorithms)

Key data structures:
- Limit order book: Price levels with time-priority queues
- Order registry: Track all active orders and their states
- Trade log: Record all executed trades

Key functions (to be implemented):
- `submit_order(order)`: Process incoming order
- `cancel_order(order_id)`: Cancel existing order
- `get_book_state()`: Return current LOB state
- `get_best_bid_ask()`: Return top of book
- `emit_market_data()`: Broadcast updates to subscribers
"""
module Exchange

using DataStructures

export Order, Trade, LimitOrderBook, MarketDataEvent
export OrderAddedEvent, OrderExecutedEvent, OrderCancelledEvent, TradeEvent, BookUpdateEvent
export init_exchange, submit_order!, cancel_order!
export get_best_bid_ask, get_book_depth, get_spread, get_mid_price

# ============================================================================
# Data Structures
# ============================================================================

"""
Order represents a trading order with all necessary attributes.
Supports multiple order types: limit, market, IOC, FOK, post-only, iceberg, stop orders.
"""
mutable struct Order
    id::Int64
    trader_id::Int64
    side::Symbol  # :buy or :sell
    order_type::Symbol  # :limit, :market, :ioc, :fok, :post_only, :iceberg, :stop_limit, :stop_loss
    price::Float64  # limit price (NaN for market orders)
    quantity::Int64
    filled::Int64
    timestamp::Float64  # submission time
    
    # Special order attributes
    iceberg_display_qty::Union{Nothing, Int64}  # for iceberg orders
    iceberg_hidden_qty::Int64
    stop_price::Union{Nothing, Float64}  # for stop orders
    time_in_force::Symbol  # :gtc (good-til-cancel), :ioc, :fok, :day
    post_only::Bool  # reject if would cross spread
end

"""
Constructor for Order with sensible defaults.
"""
function Order(;
    id::Int64,
    trader_id::Int64,
    side::Symbol,
    order_type::Symbol,
    price::Float64,
    quantity::Int64,
    timestamp::Float64,
    iceberg_display_qty::Union{Nothing, Int64} = nothing,
    iceberg_hidden_qty::Int64 = 0,
    stop_price::Union{Nothing, Float64} = nothing,
    time_in_force::Symbol = :gtc,
    post_only::Bool = false
)
    Order(
        id, trader_id, side, order_type, price, quantity, 0, timestamp,
        iceberg_display_qty, iceberg_hidden_qty, stop_price, time_in_force, post_only
    )
end

"""
Trade represents an executed trade between two orders.
"""
struct Trade
    trade_id::Int64
    timestamp::Float64
    price::Float64
    quantity::Int64
    buy_order_id::Int64
    sell_order_id::Int64
    aggressor_side::Symbol  # :buy or :sell
end

"""
LimitOrderBook maintains the order book state with configurable matching mode.
Uses SortedDict for efficient price level management and Deque for FIFO queues.
"""
mutable struct LimitOrderBook
    bids::SortedDict{Float64, Deque{Order}, Base.Order.ReverseOrdering}  # descending price
    asks::SortedDict{Float64, Deque{Order}, Base.Order.ForwardOrdering}  # ascending price
    orders::Dict{Int64, Order}  # order_id -> Order (registry)
    stop_orders::Vector{Order}  # pending stop orders
    matching_mode::Symbol  # :fifo or :prorata
    tick_size::Float64
    last_trade_price::Float64
    next_trade_id::Int64
    prorata_top_order_pct::Float64  # for prorata mode (e.g., 0.4 = 40%)
    enable_self_match_prevention::Bool
    market_data_callback::Union{Nothing, Function}  # callback for market data events
end

# ============================================================================
# Market Data Events
# ============================================================================

"""
Abstract type for all market data events (L3 market data).
"""
abstract type MarketDataEvent end

"""
OrderAddedEvent: emitted when an order is added to the book.
"""
struct OrderAddedEvent <: MarketDataEvent
    timestamp::Float64
    order_id::Int64
    side::Symbol
    price::Float64
    quantity::Int64
end

"""
OrderExecutedEvent: emitted when an order is executed (fully or partially).
"""
struct OrderExecutedEvent <: MarketDataEvent
    timestamp::Float64
    order_id::Int64
    exec_quantity::Int64
    exec_price::Float64
    trade_id::Int64
end

"""
OrderCancelledEvent: emitted when an order is cancelled.
"""
struct OrderCancelledEvent <: MarketDataEvent
    timestamp::Float64
    order_id::Int64
    remaining_qty::Int64
end

"""
TradeEvent: emitted when a trade occurs.
"""
struct TradeEvent <: MarketDataEvent
    timestamp::Float64
    trade_id::Int64
    price::Float64
    quantity::Int64
    aggressor_side::Symbol
end

"""
BookUpdateEvent: emitted when the top of book changes.
"""
struct BookUpdateEvent <: MarketDataEvent
    timestamp::Float64
    best_bid::Float64
    best_ask::Float64
    bid_volume::Int64
    ask_volume::Int64
end

# ============================================================================
# Initialization
# ============================================================================

"""
Initialize a limit order book with specified parameters.

# Arguments
- `tick_size`: Minimum price increment (default: 0.01)
- `matching_mode`: :fifo or :prorata (default: :fifo)
- `initial_mid_price`: Starting mid price (default: 100.0)
- `prorata_top_order_pct`: Top order preference in prorata mode (default: 0.4)
- `enable_self_match_prevention`: Prevent trader from matching with self (default: true)
- `market_data_callback`: Function to call with market data events (default: nothing)

# Returns
- `LimitOrderBook`: Initialized order book
"""
function init_exchange(;
    tick_size::Float64 = 0.01,
    matching_mode::Symbol = :fifo,
    initial_mid_price::Float64 = 100.0,
    prorata_top_order_pct::Float64 = 0.4,
    enable_self_match_prevention::Bool = true,
    market_data_callback::Union{Nothing, Function} = nothing
)::LimitOrderBook
    
    @assert matching_mode in [:fifo, :prorata] "matching_mode must be :fifo or :prorata"
    @assert tick_size > 0 "tick_size must be positive"
    @assert 0.0 <= prorata_top_order_pct <= 1.0 "prorata_top_order_pct must be in [0, 1]"
    
    LimitOrderBook(
        SortedDict{Float64, Deque{Order}, Base.Order.ReverseOrdering}(Base.Order.Reverse),
        SortedDict{Float64, Deque{Order}, Base.Order.ForwardOrdering}(Base.Order.Forward),
        Dict{Int64, Order}(),
        Vector{Order}(),
        matching_mode,
        tick_size,
        initial_mid_price,
        1,  # next_trade_id
        prorata_top_order_pct,
        enable_self_match_prevention,
        market_data_callback
    )
end

# ============================================================================
# Book Query Functions
# ============================================================================

"""
Get the best bid and ask prices.

# Returns
- `(best_bid, best_ask)`: Tuple of best bid and ask prices (NaN if side is empty)
"""
function get_best_bid_ask(book::LimitOrderBook)::Tuple{Float64, Float64}
    best_bid = isempty(book.bids) ? NaN : first(book.bids)[1]
    best_ask = isempty(book.asks) ? NaN : first(book.asks)[1]
    return (best_bid, best_ask)
end

"""
Get the spread (ask - bid).

# Returns
- `spread`: Spread in price units (NaN if either side is empty)
"""
function get_spread(book::LimitOrderBook)::Float64
    best_bid, best_ask = get_best_bid_ask(book)
    if isnan(best_bid) || isnan(best_ask)
        return NaN
    end
    return best_ask - best_bid
end

"""
Get the mid price ((bid + ask) / 2).

# Returns
- `mid_price`: Mid price (NaN if either side is empty, otherwise uses last trade price if one side empty)
"""
function get_mid_price(book::LimitOrderBook)::Float64
    best_bid, best_ask = get_best_bid_ask(book)
    if isnan(best_bid) && isnan(best_ask)
        return book.last_trade_price
    elseif isnan(best_bid)
        return best_ask
    elseif isnan(best_ask)
        return best_bid
    else
        return (best_bid + best_ask) / 2.0
    end
end

"""
Get book depth for specified number of levels.

# Arguments
- `book`: The limit order book
- `levels`: Number of price levels to return (default: 5)

# Returns
- `NamedTuple`: (bids, asks) where each is a vector of (price, volume) tuples
"""
function get_book_depth(book::LimitOrderBook, levels::Int = 5)::NamedTuple
    bids = Vector{Tuple{Float64, Int64}}()
    asks = Vector{Tuple{Float64, Int64}}()
    
    # Get bid levels (already sorted descending)
    for (price, queue) in Iterators.take(book.bids, levels)
        total_qty = sum(order.quantity - order.filled for order in queue)
        push!(bids, (price, total_qty))
    end
    
    # Get ask levels (already sorted ascending)
    for (price, queue) in Iterators.take(book.asks, levels)
        total_qty = sum(order.quantity - order.filled for order in queue)
        push!(asks, (price, total_qty))
    end
    
    return (bids = bids, asks = asks)
end

"""
Get total volume at best bid and ask.

# Returns
- `(bid_volume, ask_volume)`: Total quantity at best bid and ask
"""
function get_top_of_book_volume(book::LimitOrderBook)::Tuple{Int64, Int64}
    bid_vol = 0
    ask_vol = 0
    
    if !isempty(book.bids)
        _, queue = first(book.bids)
        bid_vol = sum(order.quantity - order.filled for order in queue)
    end
    
    if !isempty(book.asks)
        _, queue = first(book.asks)
        ask_vol = sum(order.quantity - order.filled for order in queue)
    end
    
    return (bid_vol, ask_vol)
end

# ============================================================================
# Market Data Emission
# ============================================================================

"""
Emit a market data event via the callback if one is registered.
"""
function emit_market_data(book::LimitOrderBook, event::MarketDataEvent)
    if book.market_data_callback !== nothing
        book.market_data_callback(event)
    end
end

"""
Emit a book update event with current top of book state.
"""
function emit_book_update(book::LimitOrderBook, timestamp::Float64)
    best_bid, best_ask = get_best_bid_ask(book)
    bid_vol, ask_vol = get_top_of_book_volume(book)
    
    # Convert NaN to 0.0 for volumes if needed
    bid_vol = isnan(best_bid) ? 0 : bid_vol
    ask_vol = isnan(best_ask) ? 0 : ask_vol
    best_bid = isnan(best_bid) ? 0.0 : best_bid
    best_ask = isnan(best_ask) ? 0.0 : best_ask
    
    event = BookUpdateEvent(timestamp, best_bid, best_ask, bid_vol, ask_vol)
    emit_market_data(book, event)
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
Add an order to the appropriate side of the book.
"""
function add_order_to_book!(book::LimitOrderBook, order::Order)
    side_book = order.side == :buy ? book.bids : book.asks
    
    if !haskey(side_book, order.price)
        side_book[order.price] = Deque{Order}()
    end
    
    push!(side_book[order.price], order)
    book.orders[order.id] = order
    
    # Emit order added event
    emit_market_data(book, OrderAddedEvent(
        order.timestamp, order.id, order.side, order.price,
        order.quantity - order.filled
    ))
end

"""
Remove an order from the book.
"""
function remove_order_from_book!(book::LimitOrderBook, order::Order)
    side_book = order.side == :buy ? book.bids : book.asks
    
    if haskey(side_book, order.price)
        queue = side_book[order.price]
        # Find and remove the order from the queue (Deque doesn't support filter!)
        # Create new queue without the target order
        new_queue = Deque{Order}()
        for o in queue
            if o.id != order.id
                push!(new_queue, o)
            end
        end
        
        if isempty(new_queue)
            # Remove price level if empty
            delete!(side_book, order.price)
        else
            # Replace with filtered queue
            side_book[order.price] = new_queue
        end
    end
    
    delete!(book.orders, order.id)
end

"""
Check if an order would cross the spread (for post-only orders).
"""
function would_cross_spread(book::LimitOrderBook, order::Order)::Bool
    if order.side == :buy
        _, best_ask = get_best_bid_ask(book)
        return !isnan(best_ask) && order.price >= best_ask
    else  # sell
        best_bid, _ = get_best_bid_ask(book)
        return !isnan(best_bid) && order.price <= best_bid
    end
end

"""
Calculate how much of an order can be filled immediately.
"""
function calculate_fillable_quantity(book::LimitOrderBook, order::Order)::Int64
    contra_side = order.side == :buy ? book.asks : book.bids
    remaining = order.quantity - order.filled
    fillable = 0
    
    for (price, queue) in contra_side
        # Check price compatibility
        if order.side == :buy && price > order.price && order.order_type != :market
            break
        elseif order.side == :sell && price < order.price && order.order_type != :market
            break
        end
        
        for contra_order in queue
            # Skip self-matches if enabled
            if book.enable_self_match_prevention && contra_order.trader_id == order.trader_id
                continue
            end
            
            contra_remaining = contra_order.quantity - contra_order.filled
            fillable += min(remaining - fillable, contra_remaining)
            
            if fillable >= remaining
                return fillable
            end
        end
    end
    
    return fillable
end

# ============================================================================
# Order Matching (FIFO)
# ============================================================================

"""
Match an order using FIFO (price-time priority) matching.
Returns a vector of executed trades.
"""
function match_fifo!(book::LimitOrderBook, order::Order, current_time::Float64)::Vector{Trade}
    trades = Trade[]
    contra_side = order.side == :buy ? book.asks : book.bids
    remaining = order.quantity - order.filled
    
    while remaining > 0 && !isempty(contra_side)
        best_price, queue = first(contra_side)
        
        # Check if we can match at this price level
        if order.order_type != :market
            if order.side == :buy && best_price > order.price
                break
            elseif order.side == :sell && best_price < order.price
                break
            end
        end
        
        # Process orders in FIFO order at this price level
        orders_to_remove = Order[]
        matched_any = false  # Track if we matched anything at this level
        
        for contra_order in queue
            if remaining == 0
                break
            end
            
            # Self-match prevention
            if book.enable_self_match_prevention && contra_order.trader_id == order.trader_id
                continue
            end
            
            matched_any = true  # We found at least one valid counterparty
            
            # Calculate fill quantity
            contra_remaining = contra_order.quantity - contra_order.filled
            fill_qty = min(remaining, contra_remaining)
            
            # Execute trade
            trade = Trade(
                book.next_trade_id,
                current_time,
                best_price,
                fill_qty,
                order.side == :buy ? order.id : contra_order.id,
                order.side == :sell ? order.id : contra_order.id,
                order.side
            )
            push!(trades, trade)
            book.next_trade_id += 1
            book.last_trade_price = best_price
            
            # Update order fills
            order.filled += fill_qty
            contra_order.filled += fill_qty
            remaining -= fill_qty
            
            # Emit market data events
            emit_market_data(book, OrderExecutedEvent(
                current_time, order.id, fill_qty, best_price, trade.trade_id
            ))
            emit_market_data(book, OrderExecutedEvent(
                current_time, contra_order.id, fill_qty, best_price, trade.trade_id
            ))
            emit_market_data(book, TradeEvent(
                current_time, trade.trade_id, best_price, fill_qty, order.side
            ))
            
            # Mark contra order for removal if fully filled
            if contra_order.filled >= contra_order.quantity
                push!(orders_to_remove, contra_order)
            end
            
            # Handle iceberg order replenishment for contra order
            if contra_order.order_type == :iceberg && 
               contra_order.iceberg_display_qty !== nothing &&
               contra_order.filled < contra_order.quantity
                
                # Replenish display quantity from hidden
                replenish_qty = min(
                    contra_order.iceberg_display_qty - (contra_order.quantity - contra_order.filled),
                    contra_order.iceberg_hidden_qty
                )
                if replenish_qty > 0
                    contra_order.iceberg_hidden_qty -= replenish_qty
                    # The order stays in the book with updated quantities
                end
            end
        end
        
        # Remove fully filled orders (Deque doesn't support filter!)
        if !isempty(orders_to_remove)
            ids_to_remove = Set(o.id for o in orders_to_remove)
            new_queue = Deque{Order}()
            for o in queue
                if !(o.id in ids_to_remove)
                    push!(new_queue, o)
                end
            end
            
            if isempty(new_queue)
                delete!(contra_side, best_price)
            else
                contra_side[best_price] = new_queue
            end
            
            # Remove from order registry
            for order_to_remove in orders_to_remove
                delete!(book.orders, order_to_remove.id)
            end
        end
        
        # If we didn't match anything at this level due to self-match prevention,
        # break out to avoid infinite loop
        if !matched_any
            break
        end
    end
    
    return trades
end

# ============================================================================
# Order Matching (Pro-Rata)
# ============================================================================

"""
Match an order using pro-rata matching with top-order preference.
Returns a vector of executed trades.
"""
function match_prorata!(book::LimitOrderBook, order::Order, current_time::Float64)::Vector{Trade}
    trades = Trade[]
    contra_side = order.side == :buy ? book.asks : book.bids
    remaining = order.quantity - order.filled
    
    while remaining > 0 && !isempty(contra_side)
        best_price, queue = first(contra_side)
        
        # Check if we can match at this price level
        if order.order_type != :market
            if order.side == :buy && best_price > order.price
                break
            elseif order.side == :sell && best_price < order.price
                break
            end
        end
        
        # Calculate total available quantity at this level (excluding self-matches)
        eligible_orders = Order[]
        total_qty = 0
        
        for contra_order in queue
            if book.enable_self_match_prevention && contra_order.trader_id == order.trader_id
                continue
            end
            contra_remaining = contra_order.quantity - contra_order.filled
            if contra_remaining > 0
                push!(eligible_orders, contra_order)
                total_qty += contra_remaining
            end
        end
        
        if isempty(eligible_orders)
            # No eligible orders at this level, move to next level
            delete!(contra_side, best_price)
            continue
        end
        
        # Allocate fills pro-rata with top-order preference
        qty_to_allocate = min(remaining, total_qty)
        
        # First, give preference to the first order
        top_order = eligible_orders[1]
        top_allocation = floor(Int, qty_to_allocate * book.prorata_top_order_pct)
        top_remaining = top_order.quantity - top_order.filled
        top_fill = min(top_allocation, top_remaining, remaining)
        
        if top_fill > 0
            trade = Trade(
                book.next_trade_id,
                current_time,
                best_price,
                top_fill,
                order.side == :buy ? order.id : top_order.id,
                order.side == :sell ? order.id : top_order.id,
                order.side
            )
            push!(trades, trade)
            book.next_trade_id += 1
            book.last_trade_price = best_price
            
            order.filled += top_fill
            top_order.filled += top_fill
            remaining -= top_fill
            qty_to_allocate -= top_fill
            
            # Emit events
            emit_market_data(book, OrderExecutedEvent(
                current_time, order.id, top_fill, best_price, trade.trade_id
            ))
            emit_market_data(book, OrderExecutedEvent(
                current_time, top_order.id, top_fill, best_price, trade.trade_id
            ))
            emit_market_data(book, TradeEvent(
                current_time, trade.trade_id, best_price, top_fill, order.side
            ))
        end
        
        # Distribute remaining quantity pro-rata among all eligible orders
        if qty_to_allocate > 0 && total_qty > 0
            for contra_order in eligible_orders
                if remaining == 0
                    break
                end
                
                contra_remaining = contra_order.quantity - contra_order.filled
                if contra_remaining == 0
                    continue
                end
                
                # Pro-rata allocation based on order size
                allocation = floor(Int, qty_to_allocate * (contra_remaining / total_qty))
                fill_qty = min(allocation, contra_remaining, remaining)
                
                if fill_qty > 0
                    trade = Trade(
                        book.next_trade_id,
                        current_time,
                        best_price,
                        fill_qty,
                        order.side == :buy ? order.id : contra_order.id,
                        order.side == :sell ? order.id : contra_order.id,
                        order.side
                    )
                    push!(trades, trade)
                    book.next_trade_id += 1
                    book.last_trade_price = best_price
                    
                    order.filled += fill_qty
                    contra_order.filled += fill_qty
                    remaining -= fill_qty
                    
                    # Emit events
                    emit_market_data(book, OrderExecutedEvent(
                        current_time, order.id, fill_qty, best_price, trade.trade_id
                    ))
                    emit_market_data(book, OrderExecutedEvent(
                        current_time, contra_order.id, fill_qty, best_price, trade.trade_id
                    ))
                    emit_market_data(book, TradeEvent(
                        current_time, trade.trade_id, best_price, fill_qty, order.side
                    ))
                end
            end
        end
        
        # Clean up fully filled orders (Deque doesn't support filter!)
        orders_to_remove = Order[]
        for contra_order in queue
            if contra_order.filled >= contra_order.quantity
                push!(orders_to_remove, contra_order)
            end
        end
        
        if !isempty(orders_to_remove)
            ids_to_remove = Set(o.id for o in orders_to_remove)
            new_queue = Deque{Order}()
            for o in queue
                if !(o.id in ids_to_remove)
                    push!(new_queue, o)
                end
            end
            
            if isempty(new_queue)
                delete!(contra_side, best_price)
            else
                contra_side[best_price] = new_queue
            end
            
            # Remove from order registry
            for order_to_remove in orders_to_remove
                delete!(book.orders, order_to_remove.id)
            end
        end
    end
    
    return trades
end

"""
Dispatch to the appropriate matching function based on book configuration.
"""
function match_order!(book::LimitOrderBook, order::Order, current_time::Float64)::Vector{Trade}
    if book.matching_mode == :fifo
        return match_fifo!(book, order, current_time)
    elseif book.matching_mode == :prorata
        return match_prorata!(book, order, current_time)
    else
        error("Unknown matching mode: $(book.matching_mode)")
    end
end

# ============================================================================
# Stop Order Management
# ============================================================================

"""
Check if any stop orders should be triggered and convert them to regular orders.
Returns trades generated from triggered stop orders.
"""
function check_stop_orders!(book::LimitOrderBook, current_time::Float64)::Vector{Trade}
    all_trades = Trade[]
    triggered_orders = Order[]
    
    for stop_order in book.stop_orders
        should_trigger = false
        
        if stop_order.side == :buy
            # Buy stop: triggers when price rises to/above stop price
            should_trigger = book.last_trade_price >= stop_order.stop_price
        else  # sell
            # Sell stop: triggers when price falls to/below stop price
            should_trigger = book.last_trade_price <= stop_order.stop_price
        end
        
        if should_trigger
            push!(triggered_orders, stop_order)
            
            # Convert to market or limit order
            if stop_order.order_type == :stop_loss
                # Stop-loss becomes market order
                regular_order = Order(
                    id = stop_order.id,
                    trader_id = stop_order.trader_id,
                    side = stop_order.side,
                    order_type = :market,
                    price = NaN,
                    quantity = stop_order.quantity,
                    timestamp = current_time
                )
            else  # :stop_limit
                # Stop-limit becomes limit order at specified price
                regular_order = Order(
                    id = stop_order.id,
                    trader_id = stop_order.trader_id,
                    side = stop_order.side,
                    order_type = :limit,
                    price = stop_order.price,
                    quantity = stop_order.quantity,
                    timestamp = current_time
                )
            end
            
            # Submit the triggered order
            trades = submit_order!(book, regular_order, current_time)
            append!(all_trades, trades)
        end
    end
    
    # Remove triggered stop orders
    filter!(order -> !(order in triggered_orders), book.stop_orders)
    
    return all_trades
end

# ============================================================================
# Order Submission
# ============================================================================

"""
Submit an order to the exchange. Processes the order according to its type
and returns a vector of trades generated.

# Arguments
- `book`: The limit order book
- `order`: The order to submit
- `current_time`: Current simulation time

# Returns
- `Vector{Trade}`: Trades executed as a result of this order
"""
function submit_order!(
    book::LimitOrderBook,
    order::Order,
    current_time::Float64
)::Vector{Trade}
    
    trades = Trade[]
    
    # Handle stop orders specially - they don't match immediately
    if order.order_type in [:stop_loss, :stop_limit]
        push!(book.stop_orders, order)
        return trades
    end
    
    # Handle post-only orders
    if order.post_only && would_cross_spread(book, order)
        # Reject order - emit cancellation event
        emit_market_data(book, OrderCancelledEvent(
            current_time, order.id, order.quantity
        ))
        return trades
    end
    
    # Handle FOK (Fill-or-Kill) orders
    if order.order_type == :fok
        fillable = calculate_fillable_quantity(book, order)
        if fillable < order.quantity
            # Cannot fill entire order - reject it
            emit_market_data(book, OrderCancelledEvent(
                current_time, order.id, order.quantity
            ))
            return trades
        end
        # Otherwise, proceed to match (will be fully filled)
    end
    
    # Attempt to match order
    trades = match_order!(book, order, current_time)
    
    # Check if order still has remaining quantity
    remaining = order.quantity - order.filled
    
    # Handle IOC (Immediate-or-Cancel) orders
    if order.order_type == :ioc && remaining > 0
        # Cancel unfilled portion
        emit_market_data(book, OrderCancelledEvent(
            current_time, order.id, remaining
        ))
        return trades
    end
    
    # Handle market orders - should be fully filled or fail
    if order.order_type == :market && remaining > 0
        # Market order couldn't be fully filled (empty book)
        # In real markets this might be rejected; here we'll just leave it unfilled
        emit_market_data(book, OrderCancelledEvent(
            current_time, order.id, remaining
        ))
        return trades
    end
    
    # Add remaining quantity to book (for limit orders and iceberg orders)
    if remaining > 0 && order.order_type in [:limit, :iceberg, :post_only]
        # For iceberg orders, only show display quantity
        if order.order_type == :iceberg && order.iceberg_display_qty !== nothing
            # The order tracks both visible and hidden quantities
            add_order_to_book!(book, order)
        else
            add_order_to_book!(book, order)
        end
        
        # Emit book update
        emit_book_update(book, current_time)
    end
    
    # Check if any stop orders should trigger
    if !isempty(trades)
        stop_trades = check_stop_orders!(book, current_time)
        append!(trades, stop_trades)
    end
    
    return trades
end

# ============================================================================
# Order Cancellation
# ============================================================================

"""
Cancel an order by ID.

# Arguments
- `book`: The limit order book
- `order_id`: ID of the order to cancel
- `current_time`: Current simulation time

# Returns
- `Bool`: true if order was found and cancelled, false otherwise
"""
function cancel_order!(
    book::LimitOrderBook,
    order_id::Int64,
    current_time::Float64
)::Bool
    
    # Check if order exists in the book
    if haskey(book.orders, order_id)
        order = book.orders[order_id]
        remaining = order.quantity - order.filled
        
        # Remove from book
        remove_order_from_book!(book, order)
        
        # Emit cancellation event
        emit_market_data(book, OrderCancelledEvent(
            current_time, order_id, remaining
        ))
        
        # Emit book update
        emit_book_update(book, current_time)
        
        return true
    end
    
    # Check stop orders
    for (idx, stop_order) in enumerate(book.stop_orders)
        if stop_order.id == order_id
            deleteat!(book.stop_orders, idx)
            
            emit_market_data(book, OrderCancelledEvent(
                current_time, order_id, stop_order.quantity
            ))
            
            return true
        end
    end
    
    return false
end

end # module
