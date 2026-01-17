use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};
use ordered_float::OrderedFloat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    pub fn from_str(value: &str) -> Self {
        match value {
            "buy" | "Buy" | "BUY" => Side::Buy,
            _ => Side::Sell,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Side::Buy => "buy",
            Side::Sell => "sell",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderType {
    Limit,
    Market,
    Ioc,
    Fok,
    PostOnly,
    Iceberg,
    StopLimit,
    StopLoss,
}

impl OrderType {
    pub fn from_str(value: &str) -> Self {
        match value {
            "limit" => OrderType::Limit,
            "market" => OrderType::Market,
            "ioc" => OrderType::Ioc,
            "fok" => OrderType::Fok,
            "post_only" => OrderType::PostOnly,
            "iceberg" => OrderType::Iceberg,
            "stop_limit" => OrderType::StopLimit,
            "stop_loss" => OrderType::StopLoss,
            _ => OrderType::Limit,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            OrderType::Limit => "limit",
            OrderType::Market => "market",
            OrderType::Ioc => "ioc",
            OrderType::Fok => "fok",
            OrderType::PostOnly => "post_only",
            OrderType::Iceberg => "iceberg",
            OrderType::StopLimit => "stop_limit",
            OrderType::StopLoss => "stop_loss",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Order {
    pub id: i64,
    pub trader_id: i64,
    pub side: Side,
    pub order_type: OrderType,
    pub price: f64,
    pub quantity: i64,
    pub timestamp: f64,
    pub filled: i64,
    pub iceberg_display_qty: Option<i64>,
    pub iceberg_hidden_qty: i64,
    pub stop_price: Option<f64>,
    pub time_in_force: String,
    pub post_only: bool,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub trade_id: i64,
    pub timestamp: f64,
    pub price: f64,
    pub quantity: i64,
    pub buy_order_id: i64,
    pub sell_order_id: i64,
    pub aggressor_side: Side,
}

#[derive(Debug, Clone)]
pub enum MarketDataEvent {
    OrderAdded {
        timestamp: f64,
        order_id: i64,
        side: Side,
        price: f64,
        quantity: i64,
    },
    OrderExecuted {
        timestamp: f64,
        order_id: i64,
        exec_quantity: i64,
        exec_price: f64,
        trade_id: i64,
    },
    OrderCancelled {
        timestamp: f64,
        order_id: i64,
        remaining_qty: i64,
    },
    Trade {
        timestamp: f64,
        trade_id: i64,
        price: f64,
        quantity: i64,
        aggressor_side: Side,
    },
    BookUpdate {
        timestamp: f64,
        best_bid: f64,
        best_ask: f64,
        bid_volume: i64,
        ask_volume: i64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchingMode {
    Fifo,
    Prorata,
}

pub type PriceKey = OrderedFloat<f64>;

pub struct LimitOrderBook {
    pub bids: BTreeMap<PriceKey, VecDeque<Order>>,
    pub asks: BTreeMap<PriceKey, VecDeque<Order>>,
    pub orders: HashMap<i64, Order>,
    pub stop_orders: Vec<Order>,
    pub matching_mode: MatchingMode,
    pub tick_size: f64,
    pub last_trade_price: f64,
    pub next_trade_id: i64,
    pub prorata_top_order_pct: f64,
    pub enable_self_match_prevention: bool,
    pub market_data_callback:
        Option<Box<dyn FnMut(&LimitOrderBook, &MarketDataEvent, f64)>>,
}

pub struct BookDepth {
    pub bids: Vec<(f64, i64)>,
    pub asks: Vec<(f64, i64)>,
}

fn log_debug(hypothesis_id: &str, location: &str, message: &str, data: &str) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("/Users/aravioli/Desktop/Coding/exchange/.cursor/debug.log")
    {
        let line = format!(
            r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"{}","location":"{}","message":"{}","data":{},"timestamp":{}}}"#,
            hypothesis_id, location, message, data, ts
        );
        let _ = writeln!(file, "{}", line);
    }
}

pub fn init_exchange(
    tick_size: f64,
    matching_mode: String,
    initial_mid_price: f64,
    prorata_top_order_pct: f64,
    enable_self_match_prevention: bool,
) -> LimitOrderBook {
    let matching_mode = match matching_mode.as_str() {
        "prorata" => MatchingMode::Prorata,
        _ => MatchingMode::Fifo,
    };

    LimitOrderBook {
        bids: BTreeMap::new(),
        asks: BTreeMap::new(),
        orders: HashMap::new(),
        stop_orders: Vec::new(),
        matching_mode,
        tick_size,
        last_trade_price: initial_mid_price,
        next_trade_id: 1,
        prorata_top_order_pct,
        enable_self_match_prevention,
        market_data_callback: None,
    }
}

pub fn get_best_bid_ask(book: &LimitOrderBook) -> (f64, f64) {
    let best_bid = book
        .bids
        .iter()
        .next_back()
        .map(|(p, _)| p.0)
        .unwrap_or(f64::NAN);
    let best_ask = book
        .asks
        .iter()
        .next()
        .map(|(p, _)| p.0)
        .unwrap_or(f64::NAN);
    (best_bid, best_ask)
}

pub fn get_spread(book: &LimitOrderBook) -> f64 {
    let (best_bid, best_ask) = get_best_bid_ask(book);
    if best_bid.is_nan() || best_ask.is_nan() {
        f64::NAN
    } else {
        best_ask - best_bid
    }
}

pub fn get_mid_price(book: &LimitOrderBook) -> f64 {
    let (best_bid, best_ask) = get_best_bid_ask(book);
    if best_bid.is_nan() && best_ask.is_nan() {
        book.last_trade_price
    } else if best_bid.is_nan() {
        best_ask
    } else if best_ask.is_nan() {
        best_bid
    } else {
        (best_bid + best_ask) / 2.0
    }
}

pub fn get_book_depth(book: &LimitOrderBook, levels: usize) -> BookDepth {
    let mut bids = Vec::new();
    let mut asks = Vec::new();

    for (price, queue) in book.bids.iter().rev().take(levels) {
        let total_qty: i64 = queue.iter().map(|o| o.quantity - o.filled).sum();
        bids.push((price.0, total_qty));
    }

    for (price, queue) in book.asks.iter().take(levels) {
        let total_qty: i64 = queue.iter().map(|o| o.quantity - o.filled).sum();
        asks.push((price.0, total_qty));
    }

    BookDepth { bids, asks }
}

pub fn get_top_of_book_volume(book: &LimitOrderBook) -> (i64, i64) {
    let bid_vol = book
        .bids
        .iter()
        .next_back()
        .map(|(_, q)| q.iter().map(|o| o.quantity - o.filled).sum())
        .unwrap_or(0);
    let ask_vol = book
        .asks
        .iter()
        .next()
        .map(|(_, q)| q.iter().map(|o| o.quantity - o.filled).sum())
        .unwrap_or(0);
    (bid_vol, ask_vol)
}

fn emit_market_data(book: &mut LimitOrderBook, event: MarketDataEvent, current_time: f64) {
    let book_ptr: *const LimitOrderBook = book as *const _;
    if let Some(callback) = book.market_data_callback.as_mut() {
        unsafe { callback(&*book_ptr, &event, current_time) };
    }
}

fn emit_book_update(book: &mut LimitOrderBook, timestamp: f64) {
    let (mut best_bid, mut best_ask) = get_best_bid_ask(book);
    let (mut bid_vol, mut ask_vol) = get_top_of_book_volume(book);

    if best_bid.is_nan() {
        best_bid = 0.0;
        bid_vol = 0;
    }
    if best_ask.is_nan() {
        best_ask = 0.0;
        ask_vol = 0;
    }

    emit_market_data(
        book,
        MarketDataEvent::BookUpdate {
            timestamp,
            best_bid,
            best_ask,
            bid_volume: bid_vol,
            ask_volume: ask_vol,
        },
        timestamp,
    );
}

fn add_order_to_book(book: &mut LimitOrderBook, order: Order) {
    let side_book = match order.side {
        Side::Buy => &mut book.bids,
        Side::Sell => &mut book.asks,
    };
    let price = PriceKey::from(order.price);
    side_book
        .entry(price)
        .or_insert_with(VecDeque::new)
        .push_back(order.clone());
    book.orders.insert(order.id, order.clone());

    emit_market_data(
        book,
        MarketDataEvent::OrderAdded {
            timestamp: order.timestamp,
            order_id: order.id,
            side: order.side,
            price: order.price,
            quantity: order.quantity - order.filled,
        },
        order.timestamp,
    );
}

fn remove_order_from_book(book: &mut LimitOrderBook, order: &Order) {
    let side_book = match order.side {
        Side::Buy => &mut book.bids,
        Side::Sell => &mut book.asks,
    };
    let price = PriceKey::from(order.price);
    if let Some(queue) = side_book.get_mut(&price) {
        queue.retain(|o| o.id != order.id);
        if queue.is_empty() {
            side_book.remove(&price);
        }
    }
    book.orders.remove(&order.id);
}

fn would_cross_spread(book: &LimitOrderBook, order: &Order) -> bool {
    let (best_bid, best_ask) = get_best_bid_ask(book);
    match order.side {
        Side::Buy => !best_ask.is_nan() && order.price >= best_ask,
        Side::Sell => !best_bid.is_nan() && order.price <= best_bid,
    }
}

fn calculate_fillable_quantity(book: &LimitOrderBook, order: &Order) -> i64 {
    let contra_side = match order.side {
        Side::Buy => &book.asks,
        Side::Sell => &book.bids,
    };
    let remaining = order.quantity - order.filled;
    let mut fillable = 0;

    let iter: Box<dyn Iterator<Item = (&PriceKey, &VecDeque<Order>)>> = match order.side {
        Side::Buy => Box::new(contra_side.iter()),
        Side::Sell => Box::new(contra_side.iter().rev()),
    };

    for (price, queue) in iter {
        if order.order_type != OrderType::Market {
            match order.side {
                Side::Buy if price.0 > order.price => break,
                Side::Sell if price.0 < order.price => break,
                _ => {}
            }
        }

        for contra_order in queue.iter() {
            if book.enable_self_match_prevention && contra_order.trader_id == order.trader_id {
                continue;
            }
            let contra_remaining = contra_order.quantity - contra_order.filled;
            let add_qty = (remaining - fillable).min(contra_remaining);
            fillable += add_qty;
            if fillable >= remaining {
                return fillable;
            }
        }
    }

    fillable
}

fn match_fifo(book: &mut LimitOrderBook, order: &mut Order, current_time: f64) -> Vec<Trade> {
    let mut trades = Vec::new();
    let mut remaining = order.quantity - order.filled;

    while remaining > 0 {
        let best_price_key = match order.side {
            Side::Buy => book.asks.iter().next().map(|(p, _)| *p),
            Side::Sell => book.bids.iter().next_back().map(|(p, _)| *p),
        };
        let best_price_key = match best_price_key {
            Some(key) => key,
            None => break,
        };
        let best_price = best_price_key.0;

        if order.order_type != OrderType::Market {
            match order.side {
                Side::Buy if best_price > order.price => break,
                Side::Sell if best_price < order.price => break,
                _ => {}
            }
        }

        let mut queue = match order.side {
            Side::Buy => book.asks.remove(&best_price_key).unwrap(),
            Side::Sell => book.bids.remove(&best_price_key).unwrap(),
        };

        // #region agent log
        log_debug(
            "H3",
            "exchange.rs:match_fifo",
            "process_level",
            &format!(
                r#"{{"price":{},"queue_len":{},"remaining":{}}}"#,
                best_price,
                queue.len(),
                remaining
            ),
        );
        // #endregion

        let mut matched_any = false;
        let mut remove_indices: Vec<usize> = Vec::new();
        let mut events: Vec<MarketDataEvent> = Vec::new();

        let mut idx = 0usize;
        while idx < queue.len() && remaining > 0 {
            let self_match = book.enable_self_match_prevention
                && queue[idx].trader_id == order.trader_id;
            if self_match {
                idx += 1;
                continue;
            }

            matched_any = true;
            let fill_qty = {
                let contra_remaining = queue[idx].quantity - queue[idx].filled;
                remaining.min(contra_remaining)
            };

            let trade = Trade {
                trade_id: book.next_trade_id,
                timestamp: current_time,
                price: best_price,
                quantity: fill_qty,
                buy_order_id: if order.side == Side::Buy { order.id } else { queue[idx].id },
                sell_order_id: if order.side == Side::Sell { order.id } else { queue[idx].id },
                aggressor_side: order.side,
            };
            book.next_trade_id += 1;
            book.last_trade_price = best_price;
            trades.push(trade.clone());

            order.filled += fill_qty;
            queue[idx].filled += fill_qty;
            remaining -= fill_qty;

            events.push(MarketDataEvent::OrderExecuted {
                timestamp: current_time,
                order_id: order.id,
                exec_quantity: fill_qty,
                exec_price: best_price,
                trade_id: trade.trade_id,
            });
            events.push(MarketDataEvent::OrderExecuted {
                timestamp: current_time,
                order_id: queue[idx].id,
                exec_quantity: fill_qty,
                exec_price: best_price,
                trade_id: trade.trade_id,
            });
            events.push(MarketDataEvent::Trade {
                timestamp: current_time,
                trade_id: trade.trade_id,
                price: best_price,
                quantity: fill_qty,
                aggressor_side: order.side,
            });

            if queue[idx].order_type == OrderType::Iceberg
                && queue[idx].iceberg_display_qty.is_some()
                && queue[idx].filled < queue[idx].quantity
            {
                let display_qty = queue[idx].iceberg_display_qty.unwrap_or(0);
                let replenish_qty = (display_qty - (queue[idx].quantity - queue[idx].filled))
                    .min(queue[idx].iceberg_hidden_qty);
                if replenish_qty > 0 {
                    queue[idx].iceberg_hidden_qty -= replenish_qty;
                }
            }

            if queue[idx].filled >= queue[idx].quantity {
                remove_indices.push(idx);
            } else if let Some(reg) = book.orders.get_mut(&queue[idx].id) {
                reg.filled = queue[idx].filled;
                reg.iceberg_hidden_qty = queue[idx].iceberg_hidden_qty;
            }

            idx += 1;
        }

        for idx in remove_indices.iter().rev() {
            let order_id = queue[*idx].id;
            queue.remove(*idx);
            book.orders.remove(&order_id);
        }

        if !queue.is_empty() {
            match order.side {
                Side::Buy => {
                    book.asks.insert(best_price_key, queue);
                }
                Side::Sell => {
                    book.bids.insert(best_price_key, queue);
                }
            }
        }

        for event in events {
            emit_market_data(book, event, current_time);
        }

        if !matched_any {
            break;
        }
    }

    trades
}

fn match_prorata(book: &mut LimitOrderBook, order: &mut Order, current_time: f64) -> Vec<Trade> {
    let mut trades = Vec::new();
    let mut remaining = order.quantity - order.filled;

    while remaining > 0 {
        let best_price_key = match order.side {
            Side::Buy => book.asks.iter().next().map(|(p, _)| *p),
            Side::Sell => book.bids.iter().next_back().map(|(p, _)| *p),
        };
        let best_price_key = match best_price_key {
            Some(key) => key,
            None => break,
        };
        let best_price = best_price_key.0;

        if order.order_type != OrderType::Market {
            match order.side {
                Side::Buy if best_price > order.price => break,
                Side::Sell if best_price < order.price => break,
                _ => {}
            }
        }

        let mut queue = match order.side {
            Side::Buy => book.asks.remove(&best_price_key).unwrap(),
            Side::Sell => book.bids.remove(&best_price_key).unwrap(),
        };

        // #region agent log
        log_debug(
            "H4",
            "exchange.rs:match_prorata",
            "process_level",
            &format!(
                r#"{{"price":{},"queue_len":{},"remaining":{}}}"#,
                best_price,
                queue.len(),
                remaining
            ),
        );
        // #endregion

        let mut eligible_indices: Vec<usize> = Vec::new();
        let mut total_qty = 0;

        for (idx, contra) in queue.iter().enumerate() {
            if book.enable_self_match_prevention && contra.trader_id == order.trader_id {
                continue;
            }
            let contra_remaining = contra.quantity - contra.filled;
            if contra_remaining > 0 {
                eligible_indices.push(idx);
                total_qty += contra_remaining;
            }
        }

        if eligible_indices.is_empty() {
            continue;
        }

        let mut qty_to_allocate = remaining.min(total_qty);
        let mut events: Vec<MarketDataEvent> = Vec::new();

        let top_idx = eligible_indices[0];
        let top_remaining = queue[top_idx].quantity - queue[top_idx].filled;
        let top_allocation = (qty_to_allocate as f64 * book.prorata_top_order_pct).floor() as i64;
        let top_fill = top_allocation.min(top_remaining).min(remaining);

        if top_fill > 0 {
            let trade = Trade {
                trade_id: book.next_trade_id,
                timestamp: current_time,
                price: best_price,
                quantity: top_fill,
                buy_order_id: if order.side == Side::Buy { order.id } else { queue[top_idx].id },
                sell_order_id: if order.side == Side::Sell { order.id } else { queue[top_idx].id },
                aggressor_side: order.side,
            };
            book.next_trade_id += 1;
            book.last_trade_price = best_price;
            trades.push(trade.clone());

            order.filled += top_fill;
            queue[top_idx].filled += top_fill;
            remaining -= top_fill;
            qty_to_allocate -= top_fill;

            if let Some(reg) = book.orders.get_mut(&queue[top_idx].id) {
                reg.filled = queue[top_idx].filled;
            }

            events.push(MarketDataEvent::OrderExecuted {
                timestamp: current_time,
                order_id: order.id,
                exec_quantity: top_fill,
                exec_price: best_price,
                trade_id: trade.trade_id,
            });
            events.push(MarketDataEvent::OrderExecuted {
                timestamp: current_time,
                order_id: queue[top_idx].id,
                exec_quantity: top_fill,
                exec_price: best_price,
                trade_id: trade.trade_id,
            });
            events.push(MarketDataEvent::Trade {
                timestamp: current_time,
                trade_id: trade.trade_id,
                price: best_price,
                quantity: top_fill,
                aggressor_side: order.side,
            });
        }

        if qty_to_allocate > 0 && total_qty > 0 {
            for idx in eligible_indices.iter() {
                if remaining == 0 {
                    break;
                }
                let contra_remaining = queue[*idx].quantity - queue[*idx].filled;
                if contra_remaining == 0 {
                    continue;
                }
                let allocation = (qty_to_allocate as f64
                    * (contra_remaining as f64 / total_qty as f64))
                    .floor() as i64;
                let fill_qty = allocation.min(contra_remaining).min(remaining);
                if fill_qty <= 0 {
                    continue;
                }

                let trade = Trade {
                    trade_id: book.next_trade_id,
                    timestamp: current_time,
                    price: best_price,
                    quantity: fill_qty,
                    buy_order_id: if order.side == Side::Buy { order.id } else { queue[*idx].id },
                    sell_order_id: if order.side == Side::Sell { order.id } else { queue[*idx].id },
                    aggressor_side: order.side,
                };
                book.next_trade_id += 1;
                book.last_trade_price = best_price;
                trades.push(trade.clone());

                order.filled += fill_qty;
                queue[*idx].filled += fill_qty;
                remaining -= fill_qty;

                if let Some(reg) = book.orders.get_mut(&queue[*idx].id) {
                    reg.filled = queue[*idx].filled;
                }

                events.push(MarketDataEvent::OrderExecuted {
                    timestamp: current_time,
                    order_id: order.id,
                    exec_quantity: fill_qty,
                    exec_price: best_price,
                    trade_id: trade.trade_id,
                });
                events.push(MarketDataEvent::OrderExecuted {
                    timestamp: current_time,
                    order_id: queue[*idx].id,
                    exec_quantity: fill_qty,
                    exec_price: best_price,
                    trade_id: trade.trade_id,
                });
                events.push(MarketDataEvent::Trade {
                    timestamp: current_time,
                    trade_id: trade.trade_id,
                    price: best_price,
                    quantity: fill_qty,
                    aggressor_side: order.side,
                });
            }
        }

        let remove_ids: Vec<i64> = queue
            .iter()
            .filter(|o| o.filled >= o.quantity)
            .map(|o| o.id)
            .collect();
        queue.retain(|o| o.filled < o.quantity);
        for order_id in remove_ids {
            book.orders.remove(&order_id);
        }

        if !queue.is_empty() {
            match order.side {
                Side::Buy => {
                    book.asks.insert(best_price_key, queue);
                }
                Side::Sell => {
                    book.bids.insert(best_price_key, queue);
                }
            }
        }

        for event in events {
            emit_market_data(book, event, current_time);
        }
    }

    trades
}

fn match_order(book: &mut LimitOrderBook, order: &mut Order, current_time: f64) -> Vec<Trade> {
    match book.matching_mode {
        MatchingMode::Fifo => match_fifo(book, order, current_time),
        MatchingMode::Prorata => match_prorata(book, order, current_time),
    }
}

fn check_stop_orders(book: &mut LimitOrderBook, current_time: f64) -> Vec<Trade> {
    let mut all_trades = Vec::new();
    let mut triggered_ids = Vec::new();

    let stop_orders = book.stop_orders.clone();
    // #region agent log
    log_debug(
        "H2",
        "exchange.rs:check_stop_orders",
        "check_stop_orders",
        &format!(r#"{{"stop_count":{}}}"#, stop_orders.len()),
    );
    // #endregion
    for stop_order in stop_orders.iter() {
        let should_trigger = match stop_order.side {
            Side::Buy => stop_order.stop_price.map(|p| book.last_trade_price >= p).unwrap_or(false),
            Side::Sell => stop_order.stop_price.map(|p| book.last_trade_price <= p).unwrap_or(false),
        };
        if !should_trigger {
            continue;
        }
        triggered_ids.push(stop_order.id);

        let regular = if stop_order.order_type == OrderType::StopLoss {
            Order {
                id: stop_order.id,
                trader_id: stop_order.trader_id,
                side: stop_order.side,
                order_type: OrderType::Market,
                price: f64::NAN,
                quantity: stop_order.quantity,
                timestamp: current_time,
                filled: 0,
                iceberg_display_qty: None,
                iceberg_hidden_qty: 0,
                stop_price: None,
                time_in_force: "gtc".to_string(),
                post_only: false,
            }
        } else {
            Order {
                id: stop_order.id,
                trader_id: stop_order.trader_id,
                side: stop_order.side,
                order_type: OrderType::Limit,
                price: stop_order.price,
                quantity: stop_order.quantity,
                timestamp: current_time,
                filled: 0,
                iceberg_display_qty: None,
                iceberg_hidden_qty: 0,
                stop_price: None,
                time_in_force: "gtc".to_string(),
                post_only: false,
            }
        };
        all_trades.extend(submit_order(book, regular, current_time));
    }

    if !triggered_ids.is_empty() {
        book.stop_orders.retain(|o| !triggered_ids.contains(&o.id));
    }

    all_trades
}

pub fn submit_order(book: &mut LimitOrderBook, mut order: Order, current_time: f64) -> Vec<Trade> {
    let mut trades = Vec::new();

    // #region agent log
    log_debug(
        "H1",
        "exchange.rs:submit_order",
        "submit_order",
        &format!(
            r#"{{"order_id":{},"side":"{}","order_type":"{}","price":{},"qty":{}}}"#,
            order.id,
            order.side.as_str(),
            order.order_type.as_str(),
            order.price,
            order.quantity
        ),
    );
    // #endregion

    if order.order_type == OrderType::StopLoss || order.order_type == OrderType::StopLimit {
        book.stop_orders.push(order);
        return trades;
    }

    if order.post_only && would_cross_spread(book, &order) {
        emit_market_data(
            book,
            MarketDataEvent::OrderCancelled {
                timestamp: current_time,
                order_id: order.id,
                remaining_qty: order.quantity,
            },
            current_time,
        );
        return trades;
    }

    if order.order_type == OrderType::Fok {
        let fillable = calculate_fillable_quantity(book, &order);
        if fillable < order.quantity {
            emit_market_data(
                book,
                MarketDataEvent::OrderCancelled {
                    timestamp: current_time,
                    order_id: order.id,
                    remaining_qty: order.quantity,
                },
                current_time,
            );
            return trades;
        }
    }

    trades = match_order(book, &mut order, current_time);
    let remaining = order.quantity - order.filled;

    if order.order_type == OrderType::Ioc && remaining > 0 {
        emit_market_data(
            book,
            MarketDataEvent::OrderCancelled {
                timestamp: current_time,
                order_id: order.id,
                remaining_qty: remaining,
            },
            current_time,
        );
        return trades;
    }

    if order.order_type == OrderType::Market && remaining > 0 {
        emit_market_data(
            book,
            MarketDataEvent::OrderCancelled {
                timestamp: current_time,
                order_id: order.id,
                remaining_qty: remaining,
            },
            current_time,
        );
        return trades;
    }

    if remaining > 0
        && matches!(order.order_type, OrderType::Limit | OrderType::Iceberg | OrderType::PostOnly)
    {
        add_order_to_book(book, order);
        emit_book_update(book, current_time);
    }

    if !trades.is_empty() {
        trades.extend(check_stop_orders(book, current_time));
    }

    trades
}

pub fn cancel_order(book: &mut LimitOrderBook, order_id: i64, current_time: f64) -> bool {
    if let Some(order) = book.orders.get(&order_id).cloned() {
        let remaining = order.quantity - order.filled;
        remove_order_from_book(book, &order);
        emit_market_data(
            book,
            MarketDataEvent::OrderCancelled {
                timestamp: current_time,
                order_id,
                remaining_qty: remaining,
            },
            current_time,
        );
        emit_book_update(book, current_time);
        return true;
    }

    if let Some(pos) = book.stop_orders.iter().position(|o| o.id == order_id) {
        let order = book.stop_orders.remove(pos);
        emit_market_data(
            book,
            MarketDataEvent::OrderCancelled {
                timestamp: current_time,
                order_id,
                remaining_qty: order.quantity,
            },
            current_time,
        );
        return true;
    }

    false
}
