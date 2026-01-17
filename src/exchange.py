from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple
import math

from sortedcontainers import SortedDict

# limit order book matching engine with fifo and prorata matching

Side = str  # "buy" | "sell"
OrderType = str  # "limit" | "market" | "ioc" | "fok" | "post_only" | "iceberg" | "stop_limit" | "stop_loss"


@dataclass
class Order:
    # represents a single order in the book
    id: int
    trader_id: int
    side: Side  # buy or sell
    order_type: OrderType
    price: float  # limit price (nan for market orders)
    quantity: int  # total quantity
    timestamp: float
    filled: int = 0  # how much has been filled
    iceberg_display_qty: Optional[int] = None  # visible qty for iceberg orders
    iceberg_hidden_qty: int = 0  # hidden reserve
    stop_price: Optional[float] = None  # trigger price for stop orders
    time_in_force: str = "gtc"  # good til cancelled
    post_only: bool = False  # reject if would cross spread


@dataclass
class Trade:
    # a matched trade between two orders
    trade_id: int
    timestamp: float
    price: float
    quantity: int
    buy_order_id: int
    sell_order_id: int
    aggressor_side: Side  # who initiated the trade


class MarketDataEvent:
    # base class for market data events
    pass


@dataclass
class OrderAddedEvent(MarketDataEvent):
    # order was added to the book
    timestamp: float
    order_id: int
    side: Side
    price: float
    quantity: int


@dataclass
class OrderExecutedEvent(MarketDataEvent):
    # order got filled (partially or fully)
    timestamp: float
    order_id: int
    exec_quantity: int
    exec_price: float
    trade_id: int


@dataclass
class OrderCancelledEvent(MarketDataEvent):
    # order was cancelled
    timestamp: float
    order_id: int
    remaining_qty: int


@dataclass
class TradeEvent(MarketDataEvent):
    # a trade occurred
    timestamp: float
    trade_id: int
    price: float
    quantity: int
    aggressor_side: Side


@dataclass
class BookUpdateEvent(MarketDataEvent):
    # top of book changed
    timestamp: float
    best_bid: float
    best_ask: float
    bid_volume: int
    ask_volume: int


@dataclass
class LimitOrderBook:
    # the order book - holds all orders and state
    bids: SortedDict  # price -> deque of orders (sorted)
    asks: SortedDict  # price -> deque of orders (sorted)
    orders: Dict[int, Order]  # all active orders by id
    stop_orders: List[Order]  # pending stop orders
    matching_mode: str  # "fifo" or "prorata"
    tick_size: float  # minimum price increment
    last_trade_price: float  # last executed price
    next_trade_id: int  # counter for trade ids
    prorata_top_order_pct: float  # allocation % to first order in prorata
    enable_self_match_prevention: bool  # prevent same trader matching themselves
    market_data_callback: Optional[Callable]  # callback for events


def init_exchange(
    tick_size: float = 0.01,
    matching_mode: str = "fifo",
    initial_mid_price: float = 100.0,
    prorata_top_order_pct: float = 0.4,
    enable_self_match_prevention: bool = True,
    market_data_callback: Optional[Callable] = None,
) -> LimitOrderBook:
    # create a new order book
    if matching_mode not in {"fifo", "prorata"}:
        raise ValueError("matching_mode must be 'fifo' or 'prorata'")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    if not (0.0 <= prorata_top_order_pct <= 1.0):
        raise ValueError("prorata_top_order_pct must be in [0, 1]")

    bids = SortedDict()  # price sorted automatically
    asks = SortedDict()
    return LimitOrderBook(
        bids=bids,
        asks=asks,
        orders={},
        stop_orders=[],
        matching_mode=matching_mode,
        tick_size=tick_size,
        last_trade_price=initial_mid_price,
        next_trade_id=1,
        prorata_top_order_pct=prorata_top_order_pct,
        enable_self_match_prevention=enable_self_match_prevention,
        market_data_callback=market_data_callback,
    )


def _best_bid(book: LimitOrderBook) -> float:
    # highest bid price
    if not book.bids:
        return math.nan
    return book.bids.peekitem(-1)[0]  # last item (highest)


def _best_ask(book: LimitOrderBook) -> float:
    # lowest ask price
    if not book.asks:
        return math.nan
    return book.asks.peekitem(0)[0]  # first item (lowest)


def get_best_bid_ask(book: LimitOrderBook) -> Tuple[float, float]:
    # get top of book prices
    return _best_bid(book), _best_ask(book)


def get_spread(book: LimitOrderBook) -> float:
    # distance between best bid and ask
    best_bid, best_ask = get_best_bid_ask(book)
    if math.isnan(best_bid) or math.isnan(best_ask):
        return math.nan
    return best_ask - best_bid


def get_mid_price(book: LimitOrderBook) -> float:
    # midpoint between bid and ask (or last trade if book empty)
    best_bid, best_ask = get_best_bid_ask(book)
    if math.isnan(best_bid) and math.isnan(best_ask):
        return book.last_trade_price
    if math.isnan(best_bid):
        return best_ask
    if math.isnan(best_ask):
        return best_bid
    return (best_bid + best_ask) / 2.0


def get_book_depth(book: LimitOrderBook, levels: int = 5) -> Dict[str, List[Tuple[float, int]]]:
    # get top N price levels with aggregated quantities
    bids: List[Tuple[float, int]] = []
    asks: List[Tuple[float, int]] = []

    # bids from highest to lowest
    bid_levels = list(reversed(book.bids.items()))[:levels]
    for price, queue in bid_levels:
        total_qty = sum(o.quantity - o.filled for o in queue)
        bids.append((price, total_qty))

    # asks from lowest to highest
    for price, queue in list(book.asks.items())[:levels]:
        total_qty = sum(o.quantity - o.filled for o in queue)
        asks.append((price, total_qty))

    return {"bids": bids, "asks": asks}


def get_top_of_book_volume(book: LimitOrderBook) -> Tuple[int, int]:
    # total volume at best bid and ask
    bid_vol = 0
    ask_vol = 0

    if book.bids:
        _, queue = book.bids.peekitem(-1)
        bid_vol = sum(o.quantity - o.filled for o in queue)

    if book.asks:
        _, queue = book.asks.peekitem(0)
        ask_vol = sum(o.quantity - o.filled for o in queue)

    return bid_vol, ask_vol


def emit_market_data(book: LimitOrderBook, event: MarketDataEvent, current_time: float) -> None:
    # send market data event to callback (flexible signature handling)
    if book.market_data_callback is None:
        return
    callback = book.market_data_callback
    try:
        callback(book, event, current_time)
    except TypeError:
        try:
            callback(event)
        except TypeError:
            callback(book, event)


def emit_book_update(book: LimitOrderBook, timestamp: float) -> None:
    # send top of book snapshot
    best_bid, best_ask = get_best_bid_ask(book)
    bid_vol, ask_vol = get_top_of_book_volume(book)

    if math.isnan(best_bid):
        best_bid = 0.0
        bid_vol = 0
    if math.isnan(best_ask):
        best_ask = 0.0
        ask_vol = 0

    emit_market_data(
        book,
        BookUpdateEvent(timestamp, best_bid, best_ask, bid_vol, ask_vol),
        timestamp,
    )


def add_order_to_book(book: LimitOrderBook, order: Order) -> None:
    # place order in the book at its price level
    side_book = book.bids if order.side == "buy" else book.asks
    if order.price not in side_book:
        side_book[order.price] = deque()
    side_book[order.price].append(order)  # append to queue
    book.orders[order.id] = order

    emit_market_data(
        book,
        OrderAddedEvent(
            order.timestamp, order.id, order.side, order.price, order.quantity - order.filled
        ),
        order.timestamp,
    )


def remove_order_from_book(book: LimitOrderBook, order: Order) -> None:
    # remove order from book
    side_book = book.bids if order.side == "buy" else book.asks
    queue = side_book.get(order.price)
    if queue is not None:
        new_queue: Deque[Order] = deque(o for o in queue if o.id != order.id)
        if new_queue:
            side_book[order.price] = new_queue
        else:
            del side_book[order.price]  # remove price level if empty

    book.orders.pop(order.id, None)


def would_cross_spread(book: LimitOrderBook, order: Order) -> bool:
    # check if this order would immediately match
    best_bid, best_ask = get_best_bid_ask(book)
    if order.side == "buy":
        return not math.isnan(best_ask) and order.price >= best_ask
    return not math.isnan(best_bid) and order.price <= best_bid


def calculate_fillable_quantity(book: LimitOrderBook, order: Order) -> int:
    # how much of this order can be filled (used for fok validation)
    contra_side = book.asks if order.side == "buy" else book.bids
    remaining = order.quantity - order.filled
    fillable = 0

    items = contra_side.items()
    if contra_side is book.bids:
        items = reversed(list(items))  # bids from high to low

    for price, queue in items:
        # check if price is acceptable
        if order.order_type != "market":
            if order.side == "buy" and price > order.price:
                break
            if order.side == "sell" and price < order.price:
                break

        for contra in queue:
            if book.enable_self_match_prevention and contra.trader_id == order.trader_id:
                continue
            contra_remaining = contra.quantity - contra.filled
            fillable += min(remaining - fillable, contra_remaining)
            if fillable >= remaining:
                return fillable

    return fillable


def match_fifo(book: LimitOrderBook, order: Order, current_time: float) -> List[Trade]:
    # first in first out matching - orders match in time priority
    trades: List[Trade] = []
    contra_side = book.asks if order.side == "buy" else book.bids
    remaining = order.quantity - order.filled

    while remaining > 0 and contra_side:
        # get best price level
        if contra_side is book.bids:
            best_price, queue = contra_side.peekitem(-1)  # highest bid
        else:
            best_price, queue = contra_side.peekitem(0)  # lowest ask

        # check if price is acceptable
        if order.order_type != "market":
            if order.side == "buy" and best_price > order.price:
                break
            if order.side == "sell" and best_price < order.price:
                break

        orders_to_remove: List[Order] = []
        matched_any = False

        # match against orders in queue (fifo order)
        for contra in list(queue):
            if remaining == 0:
                break
            if book.enable_self_match_prevention and contra.trader_id == order.trader_id:
                continue

            matched_any = True
            contra_remaining = contra.quantity - contra.filled
            fill_qty = min(remaining, contra_remaining)

            # create trade
            trade = Trade(
                trade_id=book.next_trade_id,
                timestamp=current_time,
                price=best_price,
                quantity=fill_qty,
                buy_order_id=order.id if order.side == "buy" else contra.id,
                sell_order_id=order.id if order.side == "sell" else contra.id,
                aggressor_side=order.side,
            )
            trades.append(trade)
            book.next_trade_id += 1
            book.last_trade_price = best_price

            # update fill quantities
            order.filled += fill_qty
            contra.filled += fill_qty
            remaining -= fill_qty

            # emit events
            emit_market_data(
                book,
                OrderExecutedEvent(current_time, order.id, fill_qty, best_price, trade.trade_id),
                current_time,
            )
            emit_market_data(
                book,
                OrderExecutedEvent(current_time, contra.id, fill_qty, best_price, trade.trade_id),
                current_time,
            )
            emit_market_data(
                book,
                TradeEvent(current_time, trade.trade_id, best_price, fill_qty, order.side),
                current_time,
            )

            if contra.filled >= contra.quantity:
                orders_to_remove.append(contra)

            # handle iceberg order replenishment
            if (
                contra.order_type == "iceberg"
                and contra.iceberg_display_qty is not None
                and contra.filled < contra.quantity
            ):
                replenish_qty = min(
                    contra.iceberg_display_qty - (contra.quantity - contra.filled),
                    contra.iceberg_hidden_qty,
                )
                if replenish_qty > 0:
                    contra.iceberg_hidden_qty -= replenish_qty

        # clean up fully filled orders
        if orders_to_remove:
            ids_to_remove = {o.id for o in orders_to_remove}
            new_queue: Deque[Order] = deque(o for o in queue if o.id not in ids_to_remove)
            if new_queue:
                contra_side[best_price] = new_queue
            else:
                del contra_side[best_price]
            for removed in orders_to_remove:
                book.orders.pop(removed.id, None)

        if not matched_any:
            break

    return trades


def match_prorata(book: LimitOrderBook, order: Order, current_time: float) -> List[Trade]:
    # pro rata matching - allocate fills proportionally by order size
    trades: List[Trade] = []
    contra_side = book.asks if order.side == "buy" else book.bids
    remaining = order.quantity - order.filled

    while remaining > 0 and contra_side:
        # get best price level
        if contra_side is book.bids:
            best_price, queue = contra_side.peekitem(-1)
        else:
            best_price, queue = contra_side.peekitem(0)

        # check if price is acceptable
        if order.order_type != "market":
            if order.side == "buy" and best_price > order.price:
                break
            if order.side == "sell" and best_price < order.price:
                break

        # collect eligible orders and total qty
        eligible: List[Order] = []
        total_qty = 0
        for contra in queue:
            if book.enable_self_match_prevention and contra.trader_id == order.trader_id:
                continue
            contra_remaining = contra.quantity - contra.filled
            if contra_remaining > 0:
                eligible.append(contra)
                total_qty += contra_remaining

        if not eligible:
            del contra_side[best_price]
            continue

        qty_to_allocate = min(remaining, total_qty)

        # give top order (first in queue) preferential allocation
        top_order = eligible[0]
        top_allocation = int(math.floor(qty_to_allocate * book.prorata_top_order_pct))
        top_remaining = top_order.quantity - top_order.filled
        top_fill = min(top_allocation, top_remaining, remaining)

        if top_fill > 0:
            trade = Trade(
                trade_id=book.next_trade_id,
                timestamp=current_time,
                price=best_price,
                quantity=top_fill,
                buy_order_id=order.id if order.side == "buy" else top_order.id,
                sell_order_id=order.id if order.side == "sell" else top_order.id,
                aggressor_side=order.side,
            )
            trades.append(trade)
            book.next_trade_id += 1
            book.last_trade_price = best_price

            order.filled += top_fill
            top_order.filled += top_fill
            remaining -= top_fill
            qty_to_allocate -= top_fill

            emit_market_data(
                book,
                OrderExecutedEvent(current_time, order.id, top_fill, best_price, trade.trade_id),
                current_time,
            )
            emit_market_data(
                book,
                OrderExecutedEvent(current_time, top_order.id, top_fill, best_price, trade.trade_id),
                current_time,
            )
            emit_market_data(
                book,
                TradeEvent(current_time, trade.trade_id, best_price, top_fill, order.side),
                current_time,
            )

        # allocate remaining qty proportionally to all orders
        if qty_to_allocate > 0 and total_qty > 0:
            for contra in eligible:
                if remaining == 0:
                    break
                contra_remaining = contra.quantity - contra.filled
                if contra_remaining == 0:
                    continue
                # allocate based on proportion of total size
                allocation = int(math.floor(qty_to_allocate * (contra_remaining / total_qty)))
                fill_qty = min(allocation, contra_remaining, remaining)
                if fill_qty <= 0:
                    continue

                trade = Trade(
                    trade_id=book.next_trade_id,
                    timestamp=current_time,
                    price=best_price,
                    quantity=fill_qty,
                    buy_order_id=order.id if order.side == "buy" else contra.id,
                    sell_order_id=order.id if order.side == "sell" else contra.id,
                    aggressor_side=order.side,
                )
                trades.append(trade)
                book.next_trade_id += 1
                book.last_trade_price = best_price

                order.filled += fill_qty
                contra.filled += fill_qty
                remaining -= fill_qty

                emit_market_data(
                    book,
                    OrderExecutedEvent(
                        current_time, order.id, fill_qty, best_price, trade.trade_id
                    ),
                    current_time,
                )
                emit_market_data(
                    book,
                    OrderExecutedEvent(
                        current_time, contra.id, fill_qty, best_price, trade.trade_id
                    ),
                    current_time,
                )
                emit_market_data(
                    book,
                    TradeEvent(current_time, trade.trade_id, best_price, fill_qty, order.side),
                    current_time,
                )

        # clean up fully filled orders
        orders_to_remove: List[Order] = [o for o in queue if o.filled >= o.quantity]
        if orders_to_remove:
            ids_to_remove = {o.id for o in orders_to_remove}
            new_queue: Deque[Order] = deque(o for o in queue if o.id not in ids_to_remove)
            if new_queue:
                contra_side[best_price] = new_queue
            else:
                del contra_side[best_price]
            for removed in orders_to_remove:
                book.orders.pop(removed.id, None)

    return trades


def match_order(book: LimitOrderBook, order: Order, current_time: float) -> List[Trade]:
    # dispatch to appropriate matching algorithm
    if book.matching_mode == "fifo":
        return match_fifo(book, order, current_time)
    if book.matching_mode == "prorata":
        return match_prorata(book, order, current_time)
    raise ValueError(f"Unknown matching mode: {book.matching_mode}")


def check_stop_orders(book: LimitOrderBook, current_time: float) -> List[Trade]:
    # check if any stop orders should trigger based on last trade price
    all_trades: List[Trade] = []
    triggered: List[Order] = []

    for stop_order in book.stop_orders:
        should_trigger = False
        if stop_order.side == "buy":
            # buy stop triggers when price goes up
            should_trigger = book.last_trade_price >= (stop_order.stop_price or math.inf)
        else:
            # sell stop triggers when price goes down
            should_trigger = book.last_trade_price <= (stop_order.stop_price or -math.inf)

        if not should_trigger:
            continue

        triggered.append(stop_order)
        # convert stop order to regular order
        if stop_order.order_type == "stop_loss":
            regular = Order(
                id=stop_order.id,
                trader_id=stop_order.trader_id,
                side=stop_order.side,
                order_type="market",
                price=math.nan,
                quantity=stop_order.quantity,
                timestamp=current_time,
            )
        else:  # stop_limit
            regular = Order(
                id=stop_order.id,
                trader_id=stop_order.trader_id,
                side=stop_order.side,
                order_type="limit",
                price=stop_order.price,
                quantity=stop_order.quantity,
                timestamp=current_time,
            )
        all_trades.extend(submit_order(book, regular, current_time))

    # remove triggered stops
    if triggered:
        book.stop_orders = [o for o in book.stop_orders if o not in triggered]

    return all_trades


def submit_order(book: LimitOrderBook, order: Order, current_time: float) -> List[Trade]:
    # main entry point - submit an order to the exchange
    trades: List[Trade] = []

    # stop orders go into pending queue
    if order.order_type in {"stop_loss", "stop_limit"}:
        book.stop_orders.append(order)
        return trades

    # post only rejects if would cross
    if order.post_only and would_cross_spread(book, order):
        emit_market_data(
            book, OrderCancelledEvent(current_time, order.id, order.quantity), current_time
        )
        return trades

    # fill or kill checks if can be fully filled
    if order.order_type == "fok":
        fillable = calculate_fillable_quantity(book, order)
        if fillable < order.quantity:
            emit_market_data(
                book, OrderCancelledEvent(current_time, order.id, order.quantity), current_time
            )
            return trades

    # try to match the order
    trades = match_order(book, order, current_time)
    remaining = order.quantity - order.filled

    # immediate or cancel - cancel any unfilled qty
    if order.order_type == "ioc" and remaining > 0:
        emit_market_data(
            book, OrderCancelledEvent(current_time, order.id, remaining), current_time
        )
        return trades

    # market orders don't rest in book
    if order.order_type == "market" and remaining > 0:
        emit_market_data(
            book, OrderCancelledEvent(current_time, order.id, remaining), current_time
        )
        return trades

    # limit orders with remaining qty go in the book
    if remaining > 0 and order.order_type in {"limit", "iceberg", "post_only"}:
        add_order_to_book(book, order)
        emit_book_update(book, current_time)

    # check if any stop orders triggered
    if trades:
        trades.extend(check_stop_orders(book, current_time))

    return trades


def cancel_order(book: LimitOrderBook, order_id: int, current_time: float) -> bool:
    # cancel an order by id
    order = book.orders.get(order_id)
    if order is not None:
        # found in active orders
        remaining = order.quantity - order.filled
        remove_order_from_book(book, order)
        emit_market_data(
            book, OrderCancelledEvent(current_time, order_id, remaining), current_time
        )
        emit_book_update(book, current_time)
        return True

    # check stop orders
    for idx, stop_order in enumerate(list(book.stop_orders)):
        if stop_order.id == order_id:
            book.stop_orders.pop(idx)
            emit_market_data(
                book,
                OrderCancelledEvent(current_time, order_id, stop_order.quantity),
                current_time,
            )
            return True

    return False
