#[cfg(test)]
mod tests {
    use crate::exchange::{cancel_order, init_exchange, submit_order, Order, OrderType, Side};
    use crate::sim::init_sim;

    #[test]
    fn test_limit_match_fifo() {
        let mut book = init_exchange(0.01, "fifo".to_string(), 100.0, 0.4, true);
        let sell = Order {
            id: 1,
            trader_id: 10,
            side: Side::Sell,
            order_type: OrderType::Limit,
            price: 101.0,
            quantity: 5,
            timestamp: 0.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        let buy = Order {
            id: 2,
            trader_id: 11,
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: 102.0,
            quantity: 5,
            timestamp: 1.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        submit_order(&mut book, sell, 0.0);
        let trades = submit_order(&mut book, buy, 1.0);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, 5);
        assert_eq!(book.orders.len(), 0);
    }

    #[test]
    fn test_ioc_partial_cancel() {
        let mut book = init_exchange(0.01, "fifo".to_string(), 100.0, 0.4, true);
        let sell = Order {
            id: 1,
            trader_id: 10,
            side: Side::Sell,
            order_type: OrderType::Limit,
            price: 100.0,
            quantity: 5,
            timestamp: 0.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        submit_order(&mut book, sell, 0.0);

        let ioc_buy = Order {
            id: 2,
            trader_id: 11,
            side: Side::Buy,
            order_type: OrderType::Ioc,
            price: 101.0,
            quantity: 10,
            timestamp: 1.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        let trades = submit_order(&mut book, ioc_buy, 1.0);
        let total: i64 = trades.iter().map(|t| t.quantity).sum();
        assert_eq!(total, 5);
        assert!(!book.orders.contains_key(&2));
    }

    #[test]
    fn test_fok_reject() {
        let mut book = init_exchange(0.01, "fifo".to_string(), 100.0, 0.4, true);
        let sell = Order {
            id: 1,
            trader_id: 10,
            side: Side::Sell,
            order_type: OrderType::Limit,
            price: 100.0,
            quantity: 5,
            timestamp: 0.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        submit_order(&mut book, sell, 0.0);

        let fok_buy = Order {
            id: 2,
            trader_id: 11,
            side: Side::Buy,
            order_type: OrderType::Fok,
            price: 101.0,
            quantity: 10,
            timestamp: 1.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        let trades = submit_order(&mut book, fok_buy, 1.0);
        assert_eq!(trades.len(), 0);
        assert_eq!(book.orders.len(), 1);
    }

    #[test]
    fn test_stop_loss_trigger() {
        let mut book = init_exchange(0.01, "fifo".to_string(), 100.0, 0.4, true);
        let bid = Order {
            id: 1,
            trader_id: 10,
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: 98.0,
            quantity: 20,
            timestamp: 0.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        submit_order(&mut book, bid, 0.0);

        let stop_sell = Order {
            id: 2,
            trader_id: 11,
            side: Side::Sell,
            order_type: OrderType::StopLoss,
            price: f64::NAN,
            quantity: 10,
            timestamp: 0.1,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: Some(99.0),
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        submit_order(&mut book, stop_sell, 0.1);

        let market_sell = Order {
            id: 3,
            trader_id: 12,
            side: Side::Sell,
            order_type: OrderType::Market,
            price: f64::NAN,
            quantity: 5,
            timestamp: 1.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        let trades = submit_order(&mut book, market_sell, 1.0);
        assert!(!trades.is_empty());
        assert_eq!(book.stop_orders.len(), 0);
    }

    #[test]
    fn test_prorata_matching() {
        let mut book = init_exchange(0.01, "prorata".to_string(), 100.0, 0.5, true);
        let ask1 = Order {
            id: 1,
            trader_id: 10,
            side: Side::Sell,
            order_type: OrderType::Limit,
            price: 100.0,
            quantity: 6,
            timestamp: 0.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        let ask2 = Order {
            id: 2,
            trader_id: 11,
            side: Side::Sell,
            order_type: OrderType::Limit,
            price: 100.0,
            quantity: 4,
            timestamp: 0.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        submit_order(&mut book, ask1, 0.0);
        submit_order(&mut book, ask2, 0.0);

        let buy = Order {
            id: 3,
            trader_id: 12,
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: 101.0,
            quantity: 10,
            timestamp: 1.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        let trades = submit_order(&mut book, buy, 1.0);
        let mut fills: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
        for trade in trades.iter() {
            *fills.entry(trade.sell_order_id).or_insert(0) += trade.quantity;
        }
        assert_eq!(fills.get(&1).copied().unwrap_or(0), 6);
        assert_eq!(fills.get(&2).copied().unwrap_or(0), 4);
        assert!(!book.orders.contains_key(&1));
        assert!(!book.orders.contains_key(&2));
    }

    #[test]
    fn test_cancel_order() {
        let mut book = init_exchange(0.01, "fifo".to_string(), 100.0, 0.4, true);
        let buy = Order {
            id: 1,
            trader_id: 10,
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: 99.0,
            quantity: 5,
            timestamp: 0.0,
            filled: 0,
            iceberg_display_qty: None,
            iceberg_hidden_qty: 0,
            stop_price: None,
            time_in_force: "gtc".to_string(),
            post_only: false,
        };
        submit_order(&mut book, buy, 0.0);
        let cancelled = cancel_order(&mut book, 1, 1.0);
        assert!(cancelled);
        assert_eq!(book.orders.len(), 0);
    }

    #[test]
    fn test_event_ordering() {
        let mut sim = init_sim(f64::INFINITY);
        use std::cell::RefCell;
        use std::rc::Rc;

        let results: Rc<RefCell<Vec<&str>>> = Rc::new(RefCell::new(Vec::new()));
        let results_a = results.clone();
        let results_b = results.clone();

        sim.schedule_event(1.0, move || {
            results_b.borrow_mut().push("b");
        });
        sim.schedule_event(0.5, move || {
            results_a.borrow_mut().push("a");
        });
        sim.run_until(2.0);
        assert_eq!(results.borrow().as_slice(), ["a", "b"]);
    }
}
