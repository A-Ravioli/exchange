from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import math
import random

from src.exchange import (
    BookUpdateEvent,
    Order,
    OrderExecutedEvent,
    TradeEvent,
    cancel_order,
    get_mid_price,
    submit_order,
)
from src.sim import Simulation

# interface fortrading algorithms that interact with the exchange

class MarketDataBus:
    # simple pub/sub for market events
    def __init__(self) -> None:
        self._subscribers: List[Callable] = []

    def subscribe(self, handler: Callable) -> None:
        self._subscribers.append(handler)

    def __call__(self, book, event, current_time) -> None:
        # broadcast event to all subscribers
        for handler in self._subscribers:
            handler(event)


@dataclass
class AlgoState:
    # tracks what the algo owns and how it's doing
    inventory: int = 0  # how many shares we have
    cash: float = 0.0  # money in/out from trades
    trades: int = 0  # number of trades executed

    @property
    def pnl(self) -> float:
        # profit/loss is just our cash (ignoring inventory value)
        return self.cash


class Algorithm:
    # base class for trading algorithms
    def __init__(self, algo_id: int, book, sim: Simulation) -> None:
        self.algo_id = algo_id
        self.book = book
        self.sim = sim
        self.state = AlgoState()
        self._orders: Dict[int, Order] = {}
        self._next_order_id = algo_id * 1_000_000  # unique id range per algo

    def next_order_id(self) -> int:
        # generate unique order ids
        self._next_order_id += 1
        return self._next_order_id

    def on_market_data(self, event) -> None:
        # handle market events and update our state
        if isinstance(event, OrderExecutedEvent):
            order = self._orders.get(event.order_id)
            if order is None:
                return
            # update inventory and cash based on trade
            if order.side == "buy":
                self.state.inventory += event.exec_quantity
                self.state.cash -= event.exec_quantity * event.exec_price
            else:
                self.state.inventory -= event.exec_quantity
                self.state.cash += event.exec_quantity * event.exec_price
            self.state.trades += 1
        if isinstance(event, TradeEvent):
            return

    def send_order(self, order: Order, current_time: float) -> None:
        # send order to the exchange
        self._orders[order.id] = order
        submit_order(self.book, order, current_time)

    def cancel_order(self, order_id: int, current_time: float) -> None:
        # cancel an order
        cancel_order(self.book, order_id, current_time)


class RandomTrader(Algorithm):
    # sends random orders at regular intervals
    def __init__(
        self,
        algo_id: int,
        book,
        sim: Simulation,
        seed: int = 7,
        interval: float = 0.1,  # time between orders
        max_qty: int = 5,
        price_band: float = 1.0,  # how far from mid to place limits
        market_prob: float = 0.3,  # chance of market order
    ) -> None:
        super().__init__(algo_id, book, sim)
        self.interval = interval
        self.max_qty = max_qty
        self.price_band = price_band
        self.market_prob = market_prob
        self.rng = random.Random(seed)
        self.sim.schedule_event(self.sim.current_time, self._step)

    def _step(self) -> None:
        # create and send a random order
        side = "buy" if self.rng.random() < 0.5 else "sell"
        qty = self.rng.randint(1, self.max_qty)
        order_type = "market" if self.rng.random() < self.market_prob else "limit"
        price = math.nan
        if order_type == "limit":
            # pick price near mid with some random offset
            mid = get_mid_price(self.book)
            offset = (self.rng.random() - 0.5) * 2 * self.price_band
            price = max(self.book.tick_size, mid + offset)

        order = Order(
            id=self.next_order_id(),
            trader_id=self.algo_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=qty,
            timestamp=self.sim.current_time,
        )
        self.send_order(order, self.sim.current_time)
        # schedule next order
        self.sim.schedule_event(self.sim.current_time + self.interval, self._step)


class MarketMaker(Algorithm):
    # posts bid/ask quotes on both sides of the book
    def __init__(
        self,
        algo_id: int,
        book,
        sim: Simulation,
        spread: float = 0.02,  # distance between our bid and ask
        size: int = 5,  # qty on each side
        refresh_interval: float = 0.2,  # how often to update quotes
    ) -> None:
        super().__init__(algo_id, book, sim)
        self.spread = spread
        self.size = size
        self.refresh_interval = refresh_interval
        self.bid_id: Optional[int] = None
        self.ask_id: Optional[int] = None
        self.sim.schedule_event(self.sim.current_time, self._refresh)

    def on_market_data(self, event) -> None:
        super().on_market_data(event)
        if isinstance(event, BookUpdateEvent):
            return

    def _refresh(self) -> None:
        # cancel old quotes and post new ones
        if self.bid_id is not None:
            self.cancel_order(self.bid_id, self.sim.current_time)
        if self.ask_id is not None:
            self.cancel_order(self.ask_id, self.sim.current_time)

        # calculate new bid/ask prices around mid
        mid = get_mid_price(self.book)
        bid_price = max(self.book.tick_size, mid - self.spread / 2)
        ask_price = max(self.book.tick_size, mid + self.spread / 2)

        # create bid and ask orders
        bid = Order(
            id=self.next_order_id(),
            trader_id=self.algo_id,
            side="buy",
            order_type="limit",
            price=bid_price,
            quantity=self.size,
            timestamp=self.sim.current_time,
            post_only=True,  # won't cross the spread
        )
        ask = Order(
            id=self.next_order_id(),
            trader_id=self.algo_id,
            side="sell",
            order_type="limit",
            price=ask_price,
            quantity=self.size,
            timestamp=self.sim.current_time,
            post_only=True,
        )

        self.bid_id = bid.id
        self.ask_id = ask.id
        self.send_order(bid, self.sim.current_time)
        self.send_order(ask, self.sim.current_time)
        # schedule next refresh
        self.sim.schedule_event(self.sim.current_time + self.refresh_interval, self._refresh)
