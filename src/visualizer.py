from __future__ import annotations

from typing import List
import math
import sys

from src.exchange import TradeEvent, get_best_bid_ask, get_book_depth, get_mid_price, get_spread

# terminal visualization for the order book

COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_CYAN = "\033[36m"
COLOR_YELLOW = "\033[33m"
COLOR_BOLD = "\033[1m"
COLOR_DIM = "\033[2m"


def visualize_book_simple(book, levels: int = 5) -> None:
    # basic table view of bids and asks
    print("\n" + "=" * 60)
    print(f"{COLOR_CYAN}{COLOR_BOLD}ORDER BOOK{COLOR_RESET}")
    print("=" * 60)

    best_bid_price, best_ask_price = get_best_bid_ask(book)
    spread = get_spread(book)
    mid = get_mid_price(book)

    print(
        f"{COLOR_YELLOW}Mid: ${mid:.2f}  Spread: ${spread:.4f}{COLOR_RESET}"
    )
    print()

    depth = get_book_depth(book, levels)
    print(f"{COLOR_CYAN}       BIDS              |       ASKS{COLOR_RESET}")
    print(f"{COLOR_DIM}  Price    Volume         |   Price    Volume{COLOR_RESET}")
    print("-" * 60)

    # print side by side
    max_len = max(len(depth["bids"]), len(depth["asks"]))
    for i in range(max_len):
        if i < len(depth["bids"]):
            price, vol = depth["bids"][i]
            marker = "►" if i == 0 else " "
            bid_str = f"{COLOR_GREEN}{marker} ${price:8.2f} {vol:8d}{COLOR_RESET}"
        else:
            bid_str = " " * 26

        if i < len(depth["asks"]):
            price, vol = depth["asks"][i]
            marker = "◄" if i == 0 else " "
            ask_str = f"{COLOR_RED}{marker} ${price:8.2f} {vol:8d}{COLOR_RESET}"
        else:
            ask_str = ""

        print(f"{bid_str} | {ask_str}")

    print("=" * 60)


def visualize_book(book, levels: int = 10, bar_width: int = 30) -> None:
    # fancy view with horizontal bars showing volume
    print("\n" + "=" * 80)
    print(f"{COLOR_CYAN}{COLOR_BOLD}        ▼▼▼ LIMIT ORDER BOOK DEPTH ▼▼▼{COLOR_RESET}")
    print("=" * 80)

    best_bid_price, best_ask_price = get_best_bid_ask(book)
    spread = (
        math.nan
        if math.isnan(best_bid_price) or math.isnan(best_ask_price)
        else best_ask_price - best_bid_price
    )
    mid = get_mid_price(book)

    print()
    print(f"  {COLOR_BOLD}Last Trade:{COLOR_RESET} ${book.last_trade_price:.2f}")
    print(f"  {COLOR_BOLD}Mid Price:{COLOR_RESET}  ${mid:.2f}")
    if not math.isnan(spread) and mid > 0:
        bps = spread / mid * 10000
        print(f"  {COLOR_BOLD}Spread:{COLOR_RESET}     ${spread:.4f} ({bps:.2f} bps)")
    else:
        print(f"  {COLOR_BOLD}Spread:{COLOR_RESET}     NaN")
    print()

    depth = get_book_depth(book, levels)
    if not depth["bids"] and not depth["asks"]:
        print(f"  {COLOR_DIM}[ Empty book - no orders ]{COLOR_RESET}")
        print("=" * 80)
        return

    # find max volume for bar scaling
    max_vol = 1
    for _, vol in depth["bids"]:
        max_vol = max(max_vol, vol)
    for _, vol in depth["asks"]:
        max_vol = max(max_vol, vol)

    # asks at top (reversed so best ask is closest to spread)
    print(f"  {COLOR_RED}{COLOR_BOLD}ASKS:{COLOR_RESET}")
    for i in range(len(depth["asks"]) - 1, -1, -1):
        price, vol = depth["asks"][i]
        bar_len = max(1, int((vol / max_vol) * bar_width))
        bar = "█" * bar_len
        marker = "◄ " if i == 0 else "  "
        print(f"  {COLOR_RED}{marker}${price:8.2f} {vol:7d} {bar}{COLOR_RESET}")

    spread_str = f"${spread:.4f}" if not math.isnan(spread) else "NaN"
    print()
    print(
        f"  {COLOR_YELLOW}{COLOR_BOLD}━━━━━━━━━━━━━━━  SPREAD: {spread_str}  ━━━━━━━━━━━━━━━{COLOR_RESET}"
    )
    print()

    # bids at bottom (best bid first)
    print(f"  {COLOR_GREEN}{COLOR_BOLD}BIDS:{COLOR_RESET}")
    for price, vol in depth["bids"]:
        bar_len = max(1, int((vol / max_vol) * bar_width))
        bar = "█" * bar_len
        marker = "► " if price == best_bid_price else "  "
        print(f"  {COLOR_GREEN}{marker}${price:8.2f} {vol:7d} {bar}{COLOR_RESET}")

    print()
    print("=" * 80)


def print_trade_tape(trades: List[TradeEvent], limit: int = 10) -> None:
    # show recent trades
    print(f"\n{COLOR_CYAN}{COLOR_BOLD}═══════════════ TRADE TAPE ═══════════════{COLOR_RESET}")
    print(f"{COLOR_DIM}  Time        Price    Size    Side{COLOR_RESET}")
    print("─" * 45)

    for trade in trades[-limit:]:
        side_str = (
            f"{COLOR_GREEN}BUY {COLOR_RESET}"
            if trade.aggressor_side == "buy"
            else f"{COLOR_RED}SELL{COLOR_RESET}"
        )
        print(f"  {trade.timestamp:9.6f}  ${trade.price:6.2f}  {trade.quantity:5d}  {side_str}")

    print("─" * 45)


def print_book_stats(book) -> None:
    # aggregate stats about the book
    print(f"\n{COLOR_CYAN}{COLOR_BOLD}═══════════════ BOOK STATISTICS ═══════════════{COLOR_RESET}")

    # sum up total volume on each side
    total_bid_vol = 0
    total_ask_vol = 0
    for _, queue in book.bids.items():
        for order in queue:
            total_bid_vol += order.quantity - order.filled
    for _, queue in book.asks.items():
        for order in queue:
            total_ask_vol += order.quantity - order.filled

    bid_levels = len(book.bids)
    ask_levels = len(book.asks)

    print(f"  {COLOR_GREEN}Bid Levels:{COLOR_RESET}    {bid_levels}")
    print(f"  {COLOR_GREEN}Bid Volume:{COLOR_RESET}    {total_bid_vol}")
    print()
    print(f"  {COLOR_RED}Ask Levels:{COLOR_RESET}    {ask_levels}")
    print(f"  {COLOR_RED}Ask Volume:{COLOR_RESET}    {total_ask_vol}")
    print()
    print(f"  {COLOR_YELLOW}Total Orders:{COLOR_RESET}  {len(book.orders)}")
    print(f"  {COLOR_YELLOW}Stop Orders:{COLOR_RESET}   {len(book.stop_orders)}")
    print()

    # calculate order imbalance
    total_vol = total_bid_vol + total_ask_vol
    if total_vol > 0:
        bid_pct = round(total_bid_vol / total_vol * 100, 1)
        ask_pct = round(total_ask_vol / total_vol * 100, 1)
        imbalance = bid_pct - ask_pct
        if imbalance > 0:
            imbalance_str = f"{COLOR_GREEN}+{imbalance}% BID{COLOR_RESET}"
        else:
            imbalance_str = f"{COLOR_RED}{imbalance}% ASK{COLOR_RESET}"
        print(f"  {COLOR_BOLD}Imbalance:{COLOR_RESET}     {imbalance_str}")
        print(f"                 (Bid: {bid_pct}% | Ask: {ask_pct}%)")

    print("═" * 50)


def create_live_visualizer(
    levels: int = 8,
    update_interval: float = 0.1,
    show_trades: bool = True,
    show_stats: bool = False,
):
    # creates a callback that updates the terminal in real time
    last_update = [0.0]
    trade_buffer: List[TradeEvent] = []

    def _callback(book, event, current_time: float) -> None:
        # buffer trades
        if show_trades and isinstance(event, TradeEvent):
            trade_buffer.append(event)
            if len(trade_buffer) > 20:
                trade_buffer.pop(0)

        # redraw screen at update interval
        if current_time - last_update[0] >= update_interval:
            sys.stdout.write("\033[2J\033[H")  # clear screen
            sys.stdout.flush()
            print(
                f"{COLOR_CYAN}{COLOR_BOLD}═══════════════════════════════════════════════════════════{COLOR_RESET}"
            )
            print(f"  Simulation Time: {current_time:.6f}s")
            print(
                f"{COLOR_CYAN}{COLOR_BOLD}═══════════════════════════════════════════════════════════{COLOR_RESET}"
            )
            visualize_book(book, levels=levels, bar_width=25)

            if show_trades and trade_buffer:
                print_trade_tape(trade_buffer, limit=10)
            if show_stats:
                print_book_stats(book)

            last_update[0] = current_time

    return _callback
