use std::collections::HashMap;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::exchange::{
    cancel_order, get_mid_price, submit_order, LimitOrderBook, MarketDataEvent, Order, OrderType,
    Side,
};

pub struct MarketDataBus {
    subscribers: Vec<Box<dyn FnMut(&MarketDataEvent)>>,
}

impl MarketDataBus {
    pub fn new() -> Self {
        Self { subscribers: Vec::new() }
    }

    pub fn subscribe<F>(&mut self, handler: F)
    where
        F: FnMut(&MarketDataEvent) + 'static,
    {
        self.subscribers.push(Box::new(handler));
    }

    pub fn emit(&mut self, event: &MarketDataEvent) {
        for handler in self.subscribers.iter_mut() {
            handler(event);
        }
    }
}

#[derive(Default)]
pub struct AlgoState {
    pub inventory: i64,
    pub cash: f64,
    pub trades: i64,
}

impl AlgoState {
    pub fn pnl(&self) -> f64 {
        self.cash
    }
}

pub struct AlgorithmBase {
    pub algo_id: i64,
    pub state: AlgoState,
    orders: HashMap<i64, Order>,
    next_order_id: i64,
}

impl AlgorithmBase {
    pub fn new(algo_id: i64) -> Self {
        Self {
            algo_id,
            state: AlgoState::default(),
            orders: HashMap::new(),
            next_order_id: algo_id * 1_000_000,
        }
    }

    pub fn next_order_id(&mut self) -> i64 {
        self.next_order_id += 1;
        self.next_order_id
    }

    pub fn on_market_data(&mut self, event: &MarketDataEvent) {
        match event {
            MarketDataEvent::OrderExecuted { order_id, exec_quantity, exec_price, .. } => {
                if let Some(order) = self.orders.get(order_id) {
                    match order.side {
                        Side::Buy => {
                            self.state.inventory += exec_quantity;
                            self.state.cash -= *exec_quantity as f64 * exec_price;
                        }
                        Side::Sell => {
                            self.state.inventory -= exec_quantity;
                            self.state.cash += *exec_quantity as f64 * exec_price;
                        }
                    }
                    self.state.trades += 1;
                }
            }
            _ => {}
        }
    }

    pub fn send_order(&mut self, book: &mut LimitOrderBook, order: Order, current_time: f64) {
        self.orders.insert(order.id, order.clone());
        submit_order(book, order, current_time);
    }

    pub fn cancel_order(&mut self, book: &mut LimitOrderBook, order_id: i64, current_time: f64) {
        cancel_order(book, order_id, current_time);
    }
}

pub struct RandomTrader {
    base: AlgorithmBase,
    rng: StdRng,
    max_qty: i64,
    price_band: f64,
    market_prob: f64,
}

impl RandomTrader {
    pub fn new(algo_id: i64, seed: u64) -> Self {
        Self {
            base: AlgorithmBase::new(algo_id),
            rng: StdRng::seed_from_u64(seed),
            max_qty: 5,
            price_band: 1.0,
            market_prob: 0.3,
        }
    }

    pub fn step(&mut self, book: &mut LimitOrderBook, current_time: f64) {
        let side = if self.rng.gen::<f64>() < 0.5 { Side::Buy } else { Side::Sell };
        let qty = self.rng.gen_range(1..=self.max_qty);
        let order_type = if self.rng.gen::<f64>() < self.market_prob {
            OrderType::Market
        } else {
            OrderType::Limit
        };
        let mut price = f64::NAN;
        if order_type == OrderType::Limit {
            let mid = get_mid_price(book);
            let offset = (self.rng.gen::<f64>() - 0.5) * 2.0 * self.price_band;
            price = (mid + offset).max(book.tick_size);
        }

        let order = Order {
            id: self.base.next_order_id(),
            trader_id: self.base.algo_id,
            side,
            order_type,
            price,
            quantity: qty,
            timestamp: current_time,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        self.base.send_order(book, order, current_time);
    }
}

pub struct MarketMaker {
    base: AlgorithmBase,
    spread: f64,
    size: i64,
    refresh_interval: f64,
    bid_id: Option<i64>,
    ask_id: Option<i64>,
}

impl MarketMaker {
    pub fn new(algo_id: i64) -> Self {
        Self {
            base: AlgorithmBase::new(algo_id),
            spread: 0.02,
            size: 5,
            refresh_interval: 0.2,
            bid_id: None,
            ask_id: None,
        }
    }

    pub fn step(&mut self, book: &mut LimitOrderBook, current_time: f64) {
        if let Some(id) = self.bid_id {
            self.base.cancel_order(book, id, current_time);
        }
        if let Some(id) = self.ask_id {
            self.base.cancel_order(book, id, current_time);
        }

        let mid = get_mid_price(book);
        let bid_price = (mid - self.spread / 2.0).max(book.tick_size);
        let ask_price = (mid + self.spread / 2.0).max(book.tick_size);

        let bid = Order {
            id: self.base.next_order_id(),
            trader_id: self.base.algo_id,
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: bid_price,
            quantity: self.size,
            timestamp: current_time,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: true,
        };
        let ask = Order {
            id: self.base.next_order_id(),
            trader_id: self.base.algo_id,
            side: Side::Sell,
            order_type: OrderType::Limit,
            price: ask_price,
            quantity: self.size,
            timestamp: current_time,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: true,
        };

        self.bid_id = Some(bid.id);
        self.ask_id = Some(ask.id);
        self.base.send_order(book, bid, current_time);
        self.base.send_order(book, ask, current_time);
    }
}
