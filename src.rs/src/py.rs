use pyo3::prelude::*;
use std::cell::RefCell;

use crate::exchange::{cancel_order, get_best_bid_ask, get_book_depth, init_exchange, submit_order};
use crate::exchange::{Order, OrderType, Side, Trade, LimitOrderBook};

#[pyclass]
#[derive(Clone)]
pub struct PyOrder {
    #[pyo3(get, set)]
    pub id: i64,
    #[pyo3(get, set)]
    pub trader_id: i64,
    #[pyo3(get, set)]
    pub side: String,
    #[pyo3(get, set)]
    pub order_type: String,
    #[pyo3(get, set)]
    pub price: f64,
    #[pyo3(get, set)]
    pub quantity: i64,
    #[pyo3(get, set)]
    pub timestamp: f64,
    #[pyo3(get, set)]
    pub filled: i64,
    #[pyo3(get, set)]
    pub iceberg_display_qty: Option<i64>,
    #[pyo3(get, set)]
    pub iceberg_hidden_qty: i64,
    #[pyo3(get, set)]
    pub stop_price: Option<f64>,
    #[pyo3(get, set)]
    pub time_in_force: String,
    #[pyo3(get, set)]
    pub post_only: bool,
}

impl From<PyOrder> for Order {
    fn from(o: PyOrder) -> Self {
        Order {
            id: o.id,
            trader_id: o.trader_id,
            side: Side::from_str(&o.side),
            order_type: OrderType::from_str(&o.order_type),
            price: o.price,
            quantity: o.quantity,
            timestamp: o.timestamp,
            filled: o.filled,
            iceberg_display_qty: o.iceberg_display_qty,
            iceberg_hidden_qty: o.iceberg_hidden_qty,
            stop_price: o.stop_price,
            time_in_force: o.time_in_force,
            post_only: o.post_only,
        }
    }
}

impl From<Order> for PyOrder {
    fn from(o: Order) -> Self {
        PyOrder {
            id: o.id,
            trader_id: o.trader_id,
            side: o.side.as_str().to_string(),
            order_type: o.order_type.as_str().to_string(),
            price: o.price,
            quantity: o.quantity,
            timestamp: o.timestamp,
            filled: o.filled,
            iceberg_display_qty: o.iceberg_display_qty,
            iceberg_hidden_qty: o.iceberg_hidden_qty,
            stop_price: o.stop_price,
            time_in_force: o.time_in_force,
            post_only: o.post_only,
        }
    }
}

#[pyclass]
pub struct PyTrade {
    #[pyo3(get)]
    pub trade_id: i64,
    #[pyo3(get)]
    pub timestamp: f64,
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub quantity: i64,
    #[pyo3(get)]
    pub buy_order_id: i64,
    #[pyo3(get)]
    pub sell_order_id: i64,
    #[pyo3(get)]
    pub aggressor_side: String,
}

impl From<Trade> for PyTrade {
    fn from(t: Trade) -> Self {
        PyTrade {
            trade_id: t.trade_id,
            timestamp: t.timestamp,
            price: t.price,
            quantity: t.quantity,
            buy_order_id: t.buy_order_id,
            sell_order_id: t.sell_order_id,
            aggressor_side: t.aggressor_side.as_str().to_string(),
        }
    }
}

#[pyclass(unsendable)]
pub struct PyBook {
    inner: RefCell<LimitOrderBook>,
}

#[pymethods]
impl PyBook {
    #[new]
    fn new(
        tick_size: Option<f64>,
        matching_mode: Option<String>,
        initial_mid_price: Option<f64>,
        prorata_top_order_pct: Option<f64>,
        enable_self_match_prevention: Option<bool>,
    ) -> Self {
        let book = init_exchange(
            tick_size.unwrap_or(0.01),
            matching_mode
                .as_deref()
                .unwrap_or("fifo")
                .to_string(),
            initial_mid_price.unwrap_or(100.0),
            prorata_top_order_pct.unwrap_or(0.4),
            enable_self_match_prevention.unwrap_or(true),
        );
        Self {
            inner: RefCell::new(book),
        }
    }

    fn submit_order(&self, order: PyOrder, current_time: f64) -> Vec<PyTrade> {
        let mut book = self.inner.borrow_mut();
        submit_order(&mut book, Order::from(order), current_time)
            .into_iter()
            .map(PyTrade::from)
            .collect()
    }

    fn cancel_order(&self, order_id: i64, current_time: f64) -> bool {
        let mut book = self.inner.borrow_mut();
        cancel_order(&mut book, order_id, current_time)
    }

    fn get_best_bid_ask(&self) -> (f64, f64) {
        let book = self.inner.borrow();
        get_best_bid_ask(&book)
    }

    fn get_book_depth(&self, levels: usize) -> Vec<(f64, i64, f64, i64)> {
        let book = self.inner.borrow();
        let depth = get_book_depth(&book, levels);
        let mut out = Vec::new();
        let max_len = depth.bids.len().max(depth.asks.len());
        for i in 0..max_len {
            let (bp, bv) = depth.bids.get(i).cloned().unwrap_or((0.0, 0));
            let (ap, av) = depth.asks.get(i).cloned().unwrap_or((0.0, 0));
            out.push((bp, bv, ap, av));
        }
        out
    }
}

#[pymodule]
fn exchange_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyOrder>()?;
    m.add_class::<PyTrade>()?;
    m.add_class::<PyBook>()?;
    Ok(())
}
