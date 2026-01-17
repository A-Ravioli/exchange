use crate::exchange::{get_best_bid_ask, get_book_depth, get_mid_price, get_spread, LimitOrderBook, MarketDataEvent, Trade};

const COLOR_RESET: &str = "\x1b[0m";
const COLOR_GREEN: &str = "\x1b[32m";
const COLOR_RED: &str = "\x1b[31m";
const COLOR_CYAN: &str = "\x1b[36m";
const COLOR_YELLOW: &str = "\x1b[33m";
const COLOR_BOLD: &str = "\x1b[1m";
const COLOR_DIM: &str = "\x1b[2m";

pub fn visualize_book_simple(book: &LimitOrderBook, levels: usize) {
    println!("\n{}", "=".repeat(60));
    println!("{COLOR_CYAN}{COLOR_BOLD}ORDER BOOK{COLOR_RESET}");
    println!("{}", "=".repeat(60));

    let (_best_bid, _best_ask) = get_best_bid_ask(book);
    let spread = get_spread(book);
    let mid = get_mid_price(book);

    println!(
        "{COLOR_YELLOW}Mid: ${:.2}  Spread: ${:.4}{COLOR_RESET}",
        mid, spread
    );
    println!();

    let depth = get_book_depth(book, levels);
    println!("{COLOR_CYAN}       BIDS              |       ASKS{COLOR_RESET}");
    println!("{COLOR_DIM}  Price    Volume         |   Price    Volume{COLOR_RESET}");
    println!("{}", "-".repeat(60));

    let max_len = depth.bids.len().max(depth.asks.len());
    for i in 0..max_len {
        let bid_str = if i < depth.bids.len() {
            let (price, vol) = depth.bids[i];
            let marker = if i == 0 { "►" } else { " " };
            format!("{COLOR_GREEN}{marker} ${:8.2} {:8}{COLOR_RESET}", price, vol)
        } else {
            " ".repeat(26)
        };

        let ask_str = if i < depth.asks.len() {
            let (price, vol) = depth.asks[i];
            let marker = if i == 0 { "◄" } else { " " };
            format!("{COLOR_RED}{marker} ${:8.2} {:8}{COLOR_RESET}", price, vol)
        } else {
            String::new()
        };

        println!("{bid_str} | {ask_str}");
    }

    println!("{}", "=".repeat(60));
}

pub fn visualize_book(book: &LimitOrderBook, levels: usize, bar_width: usize) {
    println!("\n{}", "=".repeat(80));
    println!("{COLOR_CYAN}{COLOR_BOLD}        ▼▼▼ LIMIT ORDER BOOK DEPTH ▼▼▼{COLOR_RESET}");
    println!("{}", "=".repeat(80));

    let (best_bid, best_ask) = get_best_bid_ask(book);
    let spread = if best_bid.is_nan() || best_ask.is_nan() {
        f64::NAN
    } else {
        best_ask - best_bid
    };
    let mid = get_mid_price(book);

    println!();
    println!("  {COLOR_BOLD}Last Trade:{COLOR_RESET} ${:.2}", book.last_trade_price);
    println!("  {COLOR_BOLD}Mid Price:{COLOR_RESET}  ${:.2}", mid);
    if !spread.is_nan() && mid > 0.0 {
        let bps = spread / mid * 10000.0;
        println!("  {COLOR_BOLD}Spread:{COLOR_RESET}     ${:.4} ({:.2} bps)", spread, bps);
    } else {
        println!("  {COLOR_BOLD}Spread:{COLOR_RESET}     NaN");
    }
    println!();

    let depth = get_book_depth(book, levels);
    if depth.bids.is_empty() && depth.asks.is_empty() {
        println!("  {COLOR_DIM}[ Empty book - no orders ]{COLOR_RESET}");
        println!("{}", "=".repeat(80));
        return;
    }

    let max_vol = depth
        .bids
        .iter()
        .chain(depth.asks.iter())
        .map(|(_, v)| *v)
        .max()
        .unwrap_or(1)
        .max(1);

    println!("  {COLOR_RED}{COLOR_BOLD}ASKS:{COLOR_RESET}");
    for i in (0..depth.asks.len()).rev() {
        let (price, vol) = depth.asks[i];
        let bar_len = ((vol as f64 / max_vol as f64) * bar_width as f64).max(1.0) as usize;
        let bar = "█".repeat(bar_len);
        let marker = if i == 0 { "◄ " } else { "  " };
        println!("  {COLOR_RED}{marker}${:8.2} {:7} {bar}{COLOR_RESET}", price, vol);
    }

    let spread_str = if spread.is_nan() { "NaN".to_string() } else { format!("${:.4}", spread) };
    println!();
    println!(
        "  {COLOR_YELLOW}{COLOR_BOLD}━━━━━━━━━━━━━━━  SPREAD: {spread_str}  ━━━━━━━━━━━━━━━{COLOR_RESET}"
    );
    println!();

    println!("  {COLOR_GREEN}{COLOR_BOLD}BIDS:{COLOR_RESET}");
    for (price, vol) in depth.bids.iter() {
        let bar_len = ((*vol as f64 / max_vol as f64) * bar_width as f64).max(1.0) as usize;
        let bar = "█".repeat(bar_len);
        let marker = if *price == best_bid { "► " } else { "  " };
        println!("  {COLOR_GREEN}{marker}${:8.2} {:7} {bar}{COLOR_RESET}", price, vol);
    }

    println!();
    println!("{}", "=".repeat(80));
}

pub fn print_trade_tape(trades: &[Trade], limit: usize) {
    println!("\n{COLOR_CYAN}{COLOR_BOLD}═══════════════ TRADE TAPE ═══════════════{COLOR_RESET}");
    println!("{COLOR_DIM}  Time        Price    Size    Side{COLOR_RESET}");
    println!("{}", "─".repeat(45));

    let start = trades.len().saturating_sub(limit);
    for trade in &trades[start..] {
        let side_str = if trade.aggressor_side == crate::exchange::Side::Buy {
            format!("{COLOR_GREEN}BUY {COLOR_RESET}")
        } else {
            format!("{COLOR_RED}SELL{COLOR_RESET}")
        };
        println!(
            "  {:9.6}  ${:6.2}  {:5}  {}",
            trade.timestamp, trade.price, trade.quantity, side_str
        );
    }

    println!("{}", "─".repeat(45));
}

pub fn print_book_stats(book: &LimitOrderBook) {
    println!("\n{COLOR_CYAN}{COLOR_BOLD}═══════════════ BOOK STATISTICS ═══════════════{COLOR_RESET}");

    let total_bid_vol: i64 = book
        .bids
        .values()
        .flat_map(|q| q.iter())
        .map(|o| o.quantity - o.filled)
        .sum();
    let total_ask_vol: i64 = book
        .asks
        .values()
        .flat_map(|q| q.iter())
        .map(|o| o.quantity - o.filled)
        .sum();

    println!("  {COLOR_GREEN}Bid Levels:{COLOR_RESET}    {}", book.bids.len());
    println!("  {COLOR_GREEN}Bid Volume:{COLOR_RESET}    {}", total_bid_vol);
    println!();
    println!("  {COLOR_RED}Ask Levels:{COLOR_RESET}    {}", book.asks.len());
    println!("  {COLOR_RED}Ask Volume:{COLOR_RESET}    {}", total_ask_vol);
    println!();
    println!("  {COLOR_YELLOW}Total Orders:{COLOR_RESET}  {}", book.orders.len());
    println!("  {COLOR_YELLOW}Stop Orders:{COLOR_RESET}   {}", book.stop_orders.len());
    println!();

    let total_vol = total_bid_vol + total_ask_vol;
    if total_vol > 0 {
        let bid_pct = (total_bid_vol as f64 / total_vol as f64 * 100.0).round() / 10.0;
        let ask_pct = (total_ask_vol as f64 / total_vol as f64 * 100.0).round() / 10.0;
        let imbalance = bid_pct - ask_pct;
        let imbalance_str = if imbalance > 0.0 {
            format!("{COLOR_GREEN}+{}% BID{COLOR_RESET}", imbalance)
        } else {
            format!("{COLOR_RED}{}% ASK{COLOR_RESET}", imbalance)
        };
        println!("  {COLOR_BOLD}Imbalance:{COLOR_RESET}     {imbalance_str}");
        println!("                 (Bid: {bid_pct}% | Ask: {ask_pct}%)");
    }

    println!("{}", "═".repeat(50));
}

pub fn create_live_visualizer(
    levels: usize,
    update_interval: f64,
    show_trades: bool,
    show_stats: bool,
) -> impl FnMut(&LimitOrderBook, &MarketDataEvent, f64) {
    let mut last_update = 0.0;
    let mut trade_buffer: Vec<Trade> = Vec::new();

    move |book, event, current_time| {
        if show_trades {
            if let MarketDataEvent::Trade { timestamp, trade_id, price, quantity, aggressor_side } = event {
                trade_buffer.push(Trade {
                    trade_id: *trade_id,
                    timestamp: *timestamp,
                    price: *price,
                    quantity: *quantity,
                    buy_order_id: 0,
                    sell_order_id: 0,
                    aggressor_side: *aggressor_side,
                });
                if trade_buffer.len() > 20 {
                    trade_buffer.remove(0);
                }
            }
        }

        if current_time - last_update >= update_interval {
            print!("\x1b[2J\x1b[H");
            println!("{COLOR_CYAN}{COLOR_BOLD}═══════════════════════════════════════════════════════════{COLOR_RESET}");
            println!("  Simulation Time: {:.6}s", current_time);
            println!("{COLOR_CYAN}{COLOR_BOLD}═══════════════════════════════════════════════════════════{COLOR_RESET}");
            visualize_book(book, levels, 25);

            if show_trades && !trade_buffer.is_empty() {
                print_trade_tape(&trade_buffer, 10);
            }
            if show_stats {
                print_book_stats(book);
            }

            last_update = current_time;
        }
    }
}
