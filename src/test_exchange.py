import math
import unittest

from src.exchange import (
    Order,
    init_exchange,
    submit_order,
    cancel_order,
)
from src.sim import init_sim


class ExchangeTests(unittest.TestCase):
    def test_limit_match_fifo(self) -> None:
        book = init_exchange()
        sell = Order(
            id=1,
            trader_id=10,
            side="sell",
            order_type="limit",
            price=101.0,
            quantity=5,
            timestamp=0.0,
        )
        buy = Order(
            id=2,
            trader_id=11,
            side="buy",
            order_type="limit",
            price=102.0,
            quantity=5,
            timestamp=1.0,
        )
        submit_order(book, sell, 0.0)
        trades = submit_order(book, buy, 1.0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].quantity, 5)
        self.assertEqual(len(book.orders), 0)

    def test_ioc_partial_cancel(self) -> None:
        book = init_exchange()
        sell = Order(
            id=1,
            trader_id=10,
            side="sell",
            order_type="limit",
            price=100.0,
            quantity=5,
            timestamp=0.0,
        )
        submit_order(book, sell, 0.0)

        ioc_buy = Order(
            id=2,
            trader_id=11,
            side="buy",
            order_type="ioc",
            price=101.0,
            quantity=10,
            timestamp=1.0,
        )
        trades = submit_order(book, ioc_buy, 1.0)
        self.assertEqual(sum(t.quantity for t in trades), 5)
        self.assertNotIn(2, book.orders)

    def test_fok_reject(self) -> None:
        book = init_exchange()
        sell = Order(
            id=1,
            trader_id=10,
            side="sell",
            order_type="limit",
            price=100.0,
            quantity=5,
            timestamp=0.0,
        )
        submit_order(book, sell, 0.0)

        fok_buy = Order(
            id=2,
            trader_id=11,
            side="buy",
            order_type="fok",
            price=101.0,
            quantity=10,
            timestamp=1.0,
        )
        trades = submit_order(book, fok_buy, 1.0)
        self.assertEqual(len(trades), 0)
        self.assertEqual(len(book.orders), 1)

    def test_stop_loss_trigger(self) -> None:
        book = init_exchange()
        bid = Order(
            id=1,
            trader_id=10,
            side="buy",
            order_type="limit",
            price=98.0,
            quantity=20,
            timestamp=0.0,
        )
        submit_order(book, bid, 0.0)

        stop_sell = Order(
            id=2,
            trader_id=11,
            side="sell",
            order_type="stop_loss",
            price=math.nan,
            quantity=10,
            timestamp=0.1,
            stop_price=99.0,
        )
        submit_order(book, stop_sell, 0.1)

        market_sell = Order(
            id=3,
            trader_id=12,
            side="sell",
            order_type="market",
            price=math.nan,
            quantity=5,
            timestamp=1.0,
        )
        trades = submit_order(book, market_sell, 1.0)
        self.assertGreaterEqual(len(trades), 1)
        self.assertEqual(len(book.stop_orders), 0)

    def test_prorata_matching(self) -> None:
        book = init_exchange(matching_mode="prorata", prorata_top_order_pct=0.5)
        ask1 = Order(
            id=1,
            trader_id=10,
            side="sell",
            order_type="limit",
            price=100.0,
            quantity=6,
            timestamp=0.0,
        )
        ask2 = Order(
            id=2,
            trader_id=11,
            side="sell",
            order_type="limit",
            price=100.0,
            quantity=4,
            timestamp=0.0,
        )
        submit_order(book, ask1, 0.0)
        submit_order(book, ask2, 0.0)

        buy = Order(
            id=3,
            trader_id=12,
            side="buy",
            order_type="limit",
            price=101.0,
            quantity=10,
            timestamp=1.0,
        )
        submit_order(book, buy, 1.0)
        self.assertEqual(ask1.filled, 6)
        self.assertEqual(ask2.filled, 4)

    def test_cancel_order(self) -> None:
        book = init_exchange()
        buy = Order(
            id=1,
            trader_id=10,
            side="buy",
            order_type="limit",
            price=99.0,
            quantity=5,
            timestamp=0.0,
        )
        submit_order(book, buy, 0.0)
        cancelled = cancel_order(book, 1, 1.0)
        self.assertTrue(cancelled)
        self.assertEqual(len(book.orders), 0)


class SimTests(unittest.TestCase):
    def test_event_ordering(self) -> None:
        sim = init_sim()
        results = []

        def a():
            results.append("a")

        def b():
            results.append("b")

        sim.schedule_event(1.0, b)
        sim.schedule_event(0.5, a)
        sim.run_until(2.0)
        self.assertEqual(results, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
