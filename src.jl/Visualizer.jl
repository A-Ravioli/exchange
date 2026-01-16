"""
Visualizer.jl - Order Book Visualization

Real-time visualization of the limit order book with ASCII art and optional plotting.
Displays bid/ask levels, volumes, spread, and trade flow.
"""
module Visualizer

using Printf

export visualize_book, visualize_book_simple, start_live_visualization
export print_trade_tape, print_book_stats

# Terminal colors
const COLOR_RESET = "\e[0m"
const COLOR_GREEN = "\e[32m"      # Bids
const COLOR_RED = "\e[31m"        # Asks
const COLOR_CYAN = "\e[36m"       # Headers
const COLOR_YELLOW = "\e[33m"     # Highlights
const COLOR_BOLD = "\e[1m"
const COLOR_DIM = "\e[2m"

"""
Simple text visualization of the order book.
Shows bid/ask levels with volumes in a compact format.
"""
function visualize_book_simple(book; levels::Int=5)
    println("\n" * "="^60)
    println("$(COLOR_CYAN)$(COLOR_BOLD)ORDER BOOK$(COLOR_RESET)")
    println("="^60)
    
    # Get best bid/ask
    best_bid_price, best_ask_price = get_best_bid_ask(book)
    spread = get_spread(book)
    mid = get_mid_price(book)
    
    # Print top info
    println("$(COLOR_YELLOW)Mid: \$$(round(mid, digits=2))  Spread: \$$(round(spread, digits=4))$(COLOR_RESET)")
    println()
    
    # Get book depth
    depth = get_book_depth(book, levels)
    
    # Print header
    println("$(COLOR_CYAN)       BIDS              |       ASKS$(COLOR_RESET)")
    println("$(COLOR_DIM)  Price    Volume         |   Price    Volume$(COLOR_RESET)")
    println("-"^60)
    
    # Find max length to align
    max_len = max(length(depth.bids), length(depth.asks))
    
    for i in 1:max_len
        # Bid side
        bid_str = if i <= length(depth.bids)
            price, vol = depth.bids[i]
            marker = i == 1 ? "►" : " "
            "$(COLOR_GREEN)$(marker) \$$(lpad(round(price, digits=2), 8)) $(lpad(vol, 8))$(COLOR_RESET)"
        else
            " " ^26
        end
        
        # Ask side
        ask_str = if i <= length(depth.asks)
            price, vol = depth.asks[i]
            marker = i == 1 ? "◄" : " "
            "$(COLOR_RED)$(marker) \$$(lpad(round(price, digits=2), 8)) $(lpad(vol, 8))$(COLOR_RESET)"
        else
            ""
        end
        
        println("$bid_str | $ask_str")
    end
    
    println("="^60)
end

"""
Detailed visualization with volume bars and statistics.
"""
function visualize_book(book; levels::Int=10, bar_width::Int=30)
    println("\n" * "="^80)
    println("$(COLOR_CYAN)$(COLOR_BOLD)        ▼▼▼ LIMIT ORDER BOOK DEPTH ▼▼▼$(COLOR_RESET)")
    println("="^80)
    
    # Get statistics
    best_bid_price, best_ask_price = get_best_bid_ask(book)
    spread = isnan(best_bid_price) || isnan(best_ask_price) ? NaN : best_ask_price - best_bid_price
    mid = get_mid_price(book)
    
    # Print stats header
    println()
    println("  $(COLOR_BOLD)Last Trade:$(COLOR_RESET) \$$(round(book.last_trade_price, digits=2))")
    println("  $(COLOR_BOLD)Mid Price:$(COLOR_RESET)  \$$(round(mid, digits=2))")
    println("  $(COLOR_BOLD)Spread:$(COLOR_RESET)     \$$(round(spread, digits=4)) ($(round(spread/mid * 10000, digits=2)) bps)")
    println()
    
    # Get depth
    depth = get_book_depth(book, levels)
    
    if isempty(depth.bids) && isempty(depth.asks)
        println("  $(COLOR_DIM)[ Empty book - no orders ]$(COLOR_RESET)")
        println("="^80)
        return
    end
    
    # Find max volume for scaling bars
    max_vol = 0
    for (_, vol) in depth.bids
        max_vol = max(max_vol, vol)
    end
    for (_, vol) in depth.asks
        max_vol = max(max_vol, vol)
    end
    
    # Print asks in reverse order (highest to lowest)
    println("  $(COLOR_RED)$(COLOR_BOLD)ASKS:$(COLOR_RESET)")
    for i in length(depth.asks):-1:1
        price, vol = depth.asks[i]
        bar_len = max(1, floor(Int, (vol / max_vol) * bar_width))
        bar = "█" ^ bar_len
        marker = i == 1 ? "◄ " : "  "
        
        price_str = @sprintf("\$%.2f", price)
        vol_str = @sprintf("%6d", vol)
        
        println("  $(COLOR_RED)$marker$(rpad(price_str, 9)) $(rpad(vol_str, 7)) $bar$(COLOR_RESET)")
    end
    
    # Print spread line
    spread_str = @sprintf("\$%.4f", spread)
    println()
    println("  $(COLOR_YELLOW)$(COLOR_BOLD)━━━━━━━━━━━━━━━  SPREAD: $spread_str  ━━━━━━━━━━━━━━━$(COLOR_RESET)")
    println()
    
    # Print bids (highest to lowest)
    println("  $(COLOR_GREEN)$(COLOR_BOLD)BIDS:$(COLOR_RESET)")
    for (price, vol) in depth.bids
        bar_len = max(1, floor(Int, (vol / max_vol) * bar_width))
        bar = "█" ^ bar_len
        marker = price == best_bid_price ? "► " : "  "
        
        price_str = @sprintf("\$%.2f", price)
        vol_str = @sprintf("%6d", vol)
        
        println("  $(COLOR_GREEN)$marker$(rpad(price_str, 9)) $(rpad(vol_str, 7)) $bar$(COLOR_RESET)")
    end
    
    println()
    println("="^80)
end

"""
Print recent trades in a tape format.
"""
function print_trade_tape(trades::Vector; limit::Int=10)
    println("\n$(COLOR_CYAN)$(COLOR_BOLD)═══════════════ TRADE TAPE ═══════════════$(COLOR_RESET)")
    println("$(COLOR_DIM)  Time        Price    Size    Side$(COLOR_RESET)")
    println("─"^45)
    
    recent_trades = trades[max(1, end-limit+1):end]
    
    for trade in recent_trades
        time_str = @sprintf("%.6f", trade.timestamp)
        price_str = @sprintf("\$%.2f", trade.price)
        size_str = @sprintf("%5d", trade.quantity)
        
        side_str = trade.aggressor_side == :buy ? 
            "$(COLOR_GREEN)BUY $(COLOR_RESET)" : 
            "$(COLOR_RED)SELL$(COLOR_RESET)"
        
        println("  $time_str  $price_str  $size_str  $side_str")
    end
    
    println("─"^45)
end

"""
Print book statistics (total volume, number of orders, etc.)
"""
function print_book_stats(book)
    println("\n$(COLOR_CYAN)$(COLOR_BOLD)═══════════════ BOOK STATISTICS ═══════════════$(COLOR_RESET)")
    
    # Count orders and volume
    num_bid_orders = length(book.orders)  # This counts all orders
    num_ask_orders = 0
    total_bid_vol = 0
    total_ask_vol = 0
    
    for (price, queue) in book.bids
        for order in queue
            total_bid_vol += (order.quantity - order.filled)
        end
    end
    
    for (price, queue) in book.asks
        for order in queue
            total_ask_vol += (order.quantity - order.filled)
        end
    end
    
    bid_levels = length(book.bids)
    ask_levels = length(book.asks)
    
    println("  $(COLOR_GREEN)Bid Levels:$(COLOR_RESET)    $bid_levels")
    println("  $(COLOR_GREEN)Bid Volume:$(COLOR_RESET)    $total_bid_vol")
    println()
    println("  $(COLOR_RED)Ask Levels:$(COLOR_RESET)    $ask_levels")
    println("  $(COLOR_RED)Ask Volume:$(COLOR_RESET)    $total_ask_vol")
    println()
    println("  $(COLOR_YELLOW)Total Orders:$(COLOR_RESET)  $(length(book.orders))")
    println("  $(COLOR_YELLOW)Stop Orders:$(COLOR_RESET)   $(length(book.stop_orders))")
    println()
    
    # Imbalance
    total_vol = total_bid_vol + total_ask_vol
    if total_vol > 0
        bid_pct = round(total_bid_vol / total_vol * 100, digits=1)
        ask_pct = round(total_ask_vol / total_vol * 100, digits=1)
        
        imbalance = bid_pct - ask_pct
        imbalance_str = imbalance > 0 ? 
            "$(COLOR_GREEN)+$(imbalance)% BID$(COLOR_RESET)" :
            "$(COLOR_RED)$(imbalance)% ASK$(COLOR_RESET)"
        
        println("  $(COLOR_BOLD)Imbalance:$(COLOR_RESET)     $imbalance_str")
        println("                 (Bid: $bid_pct% | Ask: $ask_pct%)")
    end
    
    println("═"^50)
end

"""
Live visualization that updates as orders are processed.
Returns a callback function that can be used as market_data_callback.
"""
function create_live_visualizer(; 
    levels::Int=8,
    update_interval::Float64=0.1,
    show_trades::Bool=true,
    show_stats::Bool=false
)
    last_update = Ref(0.0)
    trade_buffer = []
    
    return function(book, event, current_time)
        # Buffer trades
        if show_trades && event isa TradeEvent
            push!(trade_buffer, event)
            # Keep only last 20 trades
            if length(trade_buffer) > 20
                popfirst!(trade_buffer)
            end
        end
        
        # Update visualization at intervals
        if current_time - last_update[] >= update_interval
            # Clear screen
            print("\e[2J\e[H")  # ANSI escape codes to clear screen and move cursor to top
            
            # Print timestamp
            println("$(COLOR_CYAN)$(COLOR_BOLD)═══════════════════════════════════════════════════════════$(COLOR_RESET)")
            println("  Simulation Time: $(round(current_time, digits=6))s")
            println("$(COLOR_CYAN)$(COLOR_BOLD)═══════════════════════════════════════════════════════════$(COLOR_RESET)")
            
            # Visualize book
            visualize_book(book, levels=levels, bar_width=25)
            
            # Show recent trades
            if show_trades && !isempty(trade_buffer)
                print_trade_tape(trade_buffer, limit=10)
            end
            
            # Show stats
            if show_stats
                print_book_stats(book)
            end
            
            last_update[] = current_time
        end
    end
end

"""
Helper function to import required functions from Exchange module.
"""
function get_best_bid_ask(book)
    best_bid = isempty(book.bids) ? NaN : first(book.bids)[1]
    best_ask = isempty(book.asks) ? NaN : first(book.asks)[1]
    return (best_bid, best_ask)
end

function get_spread(book)
    best_bid, best_ask = get_best_bid_ask(book)
    if isnan(best_bid) || isnan(best_ask)
        return NaN
    end
    return best_ask - best_bid
end

function get_mid_price(book)
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

function get_book_depth(book, levels::Int)
    bids = []
    asks = []
    
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

# Define event types locally to avoid circular dependency
abstract type MarketDataEvent end

struct TradeEvent <: MarketDataEvent
    timestamp::Float64
    trade_id::Int64
    price::Float64
    quantity::Int64
    aggressor_side::Symbol
end

end # module
