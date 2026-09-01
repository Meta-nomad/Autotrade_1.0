from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .models import FeatureSnapshot, LiquidationEvent, MinuteBar, Side, TradeEvent
from .orderbook import OrderBook


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (period + 1.0)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


@dataclass(slots=True)
class FeedStatus:
    name: str
    connected: bool = False
    last_event_at: float = 0.0
    messages: int = 0
    reconnects: int = 0
    last_error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SymbolState:
    symbol: str
    books: dict[str, OrderBook] = field(default_factory=dict)
    trades: dict[str, deque[TradeEvent]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=120_000))
    )
    price_history: dict[str, deque[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=20_000))
    )
    minute_bars: deque[MinuteBar] = field(default_factory=lambda: deque(maxlen=4_000))
    hour_bars: deque[MinuteBar] = field(default_factory=lambda: deque(maxlen=1_000))
    hour_closes: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=500))
    liquidations: deque[LiquidationEvent] = field(
        default_factory=lambda: deque(maxlen=20_000)
    )
    oi_history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=20_000))
    funding_rate: float = 0.0
    next_funding_at: float = 0.0
    contract_size: float = 1.0
    maintenance_margin_rate: float = 0.005
    api_allowed: bool = True
    last_feature: FeatureSnapshot | None = None
    _last_recorded_mid: dict[str, float] = field(default_factory=dict)

    def book(self, venue: str) -> OrderBook:
        if venue not in self.books:
            self.books[venue] = OrderBook(venue=venue, symbol=self.symbol)
        return self.books[venue]

    def add_trade(self, event: TradeEvent) -> None:
        bucket = self.trades[event.venue]
        bucket.append(event)
        self._prune_trades(bucket, event.ts)
        self.record_price(event.venue, event.price, event.ts)
        if event.venue == "mexc":
            self._update_minute_bar(event)

    @staticmethod
    def _prune_trades(bucket: deque[TradeEvent], now: float) -> None:
        cutoff = now - 3_600.0
        while bucket and bucket[0].ts < cutoff:
            bucket.popleft()

    def record_price(self, venue: str, price: float, ts: float | None = None) -> None:
        if price <= 0:
            return
        now = ts or time.time()
        last = self._last_recorded_mid.get(venue, 0.0)
        if now - last < 0.5:
            history = self.price_history[venue]
            if history:
                history[-1] = (now, price)
            return
        self.price_history[venue].append((now, price))
        self._last_recorded_mid[venue] = now
        cutoff = now - 24 * 3_600.0
        history = self.price_history[venue]
        while history and history[0][0] < cutoff:
            history.popleft()

    def record_book_mid(self, venue: str, ts: float | None = None) -> None:
        mid = self.book(venue).mid()
        if mid:
            self.record_price(venue, mid, ts)

    def _update_minute_bar(self, event: TradeEvent) -> None:
        minute = int(event.ts // 60) * 60
        if not self.minute_bars or self.minute_bars[-1].ts < minute:
            self.minute_bars.append(
                MinuteBar(
                    ts=minute,
                    open=event.price,
                    high=event.price,
                    low=event.price,
                    close=event.price,
                    volume_notional=event.notional,
                )
            )
        elif self.minute_bars[-1].ts == minute:
            self.minute_bars[-1].update(event.price, event.notional)

    def bootstrap_minutes(self, bars: Iterable[MinuteBar]) -> None:
        existing = {bar.ts: bar for bar in self.minute_bars}
        for bar in bars:
            existing[bar.ts] = bar
        ordered = sorted(existing.values(), key=lambda bar: bar.ts)[-4_000:]
        self.minute_bars = deque(ordered, maxlen=4_000)
        history = self.price_history["mexc"]
        live_points = [(ts, price) for ts, price in history if ts > ordered[-1].ts] if ordered else []
        seeded = [(float(bar.ts), bar.close) for bar in ordered]
        self.price_history["mexc"] = deque((seeded + live_points)[-20_000:], maxlen=20_000)

    def bootstrap_hours(self, closes: Iterable[tuple[int, float]]) -> None:
        merged = {ts: close for ts, close in self.hour_closes}
        for ts, close in closes:
            if close > 0:
                merged[int(ts)] = float(close)
        self.hour_closes = deque(sorted(merged.items())[-500:], maxlen=500)

    def bootstrap_hour_bars(self, bars: Iterable[MinuteBar]) -> None:
        merged = {bar.ts: bar for bar in self.hour_bars}
        for bar in bars:
            if bar.close > 0:
                merged[int(bar.ts)] = bar
        self.hour_bars = deque(
            [merged[key] for key in sorted(merged)][-1_000:], maxlen=1_000
        )

    def add_liquidation(self, event: LiquidationEvent) -> None:
        self.liquidations.append(event)
        cutoff = event.ts - 1_800.0
        while self.liquidations and self.liquidations[0].ts < cutoff:
            self.liquidations.popleft()

    def update_derivatives(
        self,
        *,
        ts: float,
        open_interest: float | None = None,
        funding_rate: float | None = None,
        next_funding_at: float | None = None,
    ) -> None:
        if open_interest is not None and open_interest >= 0:
            self.oi_history.append((ts, open_interest))
            cutoff = ts - 3_600.0
            while self.oi_history and self.oi_history[0][0] < cutoff:
                self.oi_history.popleft()
        if funding_rate is not None:
            self.funding_rate = funding_rate
        if next_funding_at:
            self.next_funding_at = next_funding_at

    def reference_bbo(self) -> tuple[float, float] | None:
        for venue in ("mexc", "bybit", "binance"):
            bbo = self.book(venue).best_bid_ask()
            if bbo:
                return bbo
        return None

    def fresh_reference_bbo(self, now: float, stale_after: float) -> tuple[float, float] | None:
        """Best available live BBO, preferring MEXC but allowing venue failover."""
        for venue in ("mexc", "bybit", "binance"):
            book = self.book(venue)
            if book.is_fresh(now, stale_after):
                bbo = book.best_bid_ask()
                if bbo:
                    return bbo
        return None

    def reference_price(self) -> float:
        bbo = self.reference_bbo()
        if bbo:
            return (bbo[0] + bbo[1]) / 2.0
        for venue in ("mexc", "bybit", "binance"):
            history = self.price_history.get(venue)
            if history:
                return history[-1][1]
        return 0.0

    def price_at(self, venue: str, seconds_ago: float, now: float) -> float | None:
        history = self.price_history.get(venue)
        if not history:
            return None
        target = now - seconds_ago
        for ts, price in reversed(history):
            if ts <= target:
                return price
        return history[0][1] if history else None

    def venue_return(self, venue: str, seconds: float, now: float) -> float:
        history = self.price_history.get(venue)
        if not history:
            return 0.0
        current = history[-1][1]
        prior = self.price_at(venue, seconds, now)
        return current / prior - 1.0 if prior and current > 0 else 0.0

    def window_flow(self, venue: str, seconds: float, now: float) -> tuple[float, int]:
        bucket = self.trades.get(venue)
        if not bucket:
            return 0.0, 0
        cutoff = now - seconds
        signed = 0.0
        total = 0.0
        count = 0
        for event in reversed(bucket):
            if event.ts < cutoff:
                break
            value = event.notional
            total += value
            signed += float(event.side) * value
            count += 1
        return (signed / total if total > 0 else 0.0), count

    def _realized_vol(self, venue: str, seconds: float, now: float) -> float:
        history = self.price_history.get(venue)
        if not history or len(history) < 3:
            return 0.0
        cutoff = now - seconds
        prices = [price for ts, price in history if ts >= cutoff and price > 0]
        if len(prices) < 3:
            prices = [price for _, price in list(history)[-10:] if price > 0]
        if len(prices) < 3:
            return 0.0
        returns = [math.log(b / a) for a, b in zip(prices, prices[1:]) if a > 0 and b > 0]
        if not returns:
            return 0.0
        return math.sqrt(sum(value * value for value in returns))

    def _price_position(self, venue: str, seconds: float, now: float) -> float:
        history = self.price_history.get(venue)
        if not history:
            return 0.5
        cutoff = now - seconds
        prices = [price for ts, price in history if ts >= cutoff]
        if len(prices) < 3:
            return 0.5
        low, high = min(prices), max(prices)
        if high <= low:
            return 0.5
        return _clamp((prices[-1] - low) / (high - low), 0.0, 1.0)

    def _trend_score(self, price: float) -> float:
        values = [close for _, close in self.hour_closes]
        if price > 0:
            values = (values + [price])[-300:]
        if len(values) < 200:
            return 0.0
        fast = _ema(values[-240:], 80)
        slow = _ema(values[-500:], 200)
        if slow <= 0:
            return 0.0
        return math.tanh((fast / slow - 1.0) * 80.0)

    def atr_pct(self, periods: int = 30) -> float:
        bars = list(self.minute_bars)[-max(periods + 1, 3) :]
        if len(bars) < 3:
            return 0.01
        ranges: list[float] = []
        previous = bars[0].close
        for bar in bars[1:]:
            true_range = max(
                bar.high - bar.low,
                abs(bar.high - previous),
                abs(bar.low - previous),
            )
            ranges.append(true_range)
            previous = bar.close
        price = bars[-1].close
        return sum(ranges) / len(ranges) / price if price > 0 else 0.01

    def fifteen_minute_bars(self, limit: int = 30) -> list[MinuteBar]:
        grouped: dict[int, MinuteBar] = {}
        for bar in self.minute_bars:
            bucket = int(bar.ts // 900) * 900
            existing = grouped.get(bucket)
            if existing is None:
                grouped[bucket] = MinuteBar(
                    ts=bucket,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume_notional=bar.volume_notional,
                )
            else:
                existing.high = max(existing.high, bar.high)
                existing.low = min(existing.low, bar.low)
                existing.close = bar.close
                existing.volume_notional += bar.volume_notional
        return [grouped[key] for key in sorted(grouped)][-limit:]

    def hourly_bars(self, limit: int = 120) -> list[MinuteBar]:
        grouped: dict[int, MinuteBar] = {
            bar.ts: MinuteBar(
                ts=bar.ts,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume_notional=bar.volume_notional,
            )
            for bar in self.hour_bars
        }
        live_grouped: dict[int, MinuteBar] = {}
        for bar in self.minute_bars:
            bucket = int(bar.ts // 3_600) * 3_600
            existing = live_grouped.get(bucket)
            if existing is None:
                live_grouped[bucket] = MinuteBar(
                    ts=bucket,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume_notional=bar.volume_notional,
                )
            else:
                existing.high = max(existing.high, bar.high)
                existing.low = min(existing.low, bar.low)
                existing.close = bar.close
                existing.volume_notional += bar.volume_notional
        for bucket, live in live_grouped.items():
            existing = grouped.get(bucket)
            if existing is None:
                grouped[bucket] = live
                continue
            # The REST bootstrap may contain the still-forming hour. Merge its
            # range with live trades, but do not count overlapping volume twice.
            existing.high = max(existing.high, live.high)
            existing.low = min(existing.low, live.low)
            existing.close = live.close
            existing.volume_notional = max(existing.volume_notional, live.volume_notional)
        return [grouped[key] for key in sorted(grouped)][-limit:]

    def four_hour_bars(self, limit: int = 120) -> list[MinuteBar]:
        grouped: dict[int, MinuteBar] = {}
        for bar in self.hourly_bars(limit=max(limit * 4 + 4, 120)):
            bucket = int(bar.ts // 14_400) * 14_400
            existing = grouped.get(bucket)
            if existing is None:
                grouped[bucket] = MinuteBar(
                    ts=bucket,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume_notional=bar.volume_notional,
                )
            else:
                existing.high = max(existing.high, bar.high)
                existing.low = min(existing.low, bar.low)
                existing.close = bar.close
                existing.volume_notional += bar.volume_notional
        return [grouped[key] for key in sorted(grouped)][-limit:]

    def features(self, now: float, stale_after: float) -> FeatureSnapshot:
        price = self.reference_price()
        fresh_books = [
            self.book(venue)
            for venue in ("mexc", "bybit", "binance")
            if self.book(venue).is_fresh(now, stale_after)
            and self.book(venue).best_bid_ask()
        ]
        spread = fresh_books[0].spread_bps() if fresh_books else 99999.0

        returns = {
            seconds: self.venue_return("mexc", seconds, now)
            for seconds in (60.0, 300.0, 1_800.0)
        }
        venue_flows: dict[str, float] = {}
        trade_counts: dict[str, int] = {}
        for venue in ("mexc", "bybit", "binance"):
            venue_flows[venue], trade_counts[venue] = self.window_flow(venue, 60.0, now)

        weighted_flow = 0.0
        total_weight = 0.0
        for venue, weight in (("mexc", 0.40), ("bybit", 0.35), ("binance", 0.25)):
            if trade_counts[venue] > 0:
                weighted_flow += venue_flows[venue] * weight
                total_weight += weight
        flow_fast = weighted_flow / total_weight if total_weight else 0.0

        slow_flow = 0.0
        slow_weight = 0.0
        for venue, weight in (("mexc", 0.40), ("bybit", 0.35), ("binance", 0.25)):
            value, count = self.window_flow(venue, 300.0, now)
            if count > 0:
                slow_flow += value * weight
                slow_weight += weight
        flow_slow = slow_flow / slow_weight if slow_weight else 0.0

        book_values: list[tuple[float, float]] = []
        micro_values: list[tuple[float, float]] = []
        for venue, weight in (("mexc", 0.45), ("bybit", 0.45), ("binance", 0.10)):
            book = self.book(venue)
            if book.best_bid_ask() and book.is_fresh(now, stale_after):
                book_values.append((book.imbalance(10), weight))
                micro_values.append((book.microprice_bps(), weight))
        book_imbalance = (
            sum(value * weight for value, weight in book_values)
            / sum(weight for _, weight in book_values)
            if book_values
            else 0.0
        )
        microprice = (
            sum(value * weight for value, weight in micro_values)
            / sum(weight for _, weight in micro_values)
            if micro_values
            else 0.0
        )

        consensus_values: list[float] = []
        for venue in ("mexc", "bybit", "binance"):
            history = self.price_history.get(venue)
            if history and len(history) >= 2:
                consensus_values.append(math.tanh(self.venue_return(venue, 30.0, now) * 500.0))
        cross_consensus = sum(consensus_values) / len(consensus_values) if consensus_values else 0.0

        oi_change = 0.0
        if self.oi_history:
            current_oi = self.oi_history[-1][1]
            target = now - 300.0
            prior_oi = self.oi_history[0][1]
            for ts, value in reversed(self.oi_history):
                if ts <= target:
                    prior_oi = value
                    break
            if prior_oi > 0:
                oi_change = current_oi / prior_oi - 1.0

        liq_signed = 0.0
        liq_total = 0.0
        for event in reversed(self.liquidations):
            if event.ts < now - 300.0:
                break
            liq_signed += float(event.pressure_side) * event.notional
            liq_total += event.notional
        liq_imbalance = liq_signed / liq_total if liq_total > 0 else 0.0

        vol_short = self._realized_vol("mexc", 60.0, now)
        vol_long = self._realized_vol("mexc", 300.0, now)
        vol_ratio = vol_short / vol_long * math.sqrt(5.0) if vol_long > 0 else 1.0

        stale_venues = tuple(
            venue
            for venue in ("mexc", "bybit", "binance")
            if venue != "binance" or venue in self.books
            if not self.book(venue).is_fresh(now, stale_after)
        )
        mexc_history = self.price_history.get("mexc")
        history_span = mexc_history[-1][0] - mexc_history[0][0] if mexc_history and len(mexc_history) > 1 else 0.0
        total_trades = sum(trade_counts.values())
        ready = bool(
            price > 0
            and len(fresh_books) >= 2
            and spread < 35.0
            and total_trades >= 20
            and history_span >= 300.0
            and len(self.hour_closes) >= 200
        )

        snapshot = FeatureSnapshot(
            symbol=self.symbol,
            ts=now,
            price=price,
            spread_bps=spread,
            data_ready=ready,
            ret_60s=returns[60.0],
            ret_300s=returns[300.0],
            ret_1800s=returns[1_800.0],
            trend_score=self._trend_score(price),
            vol_short=vol_short,
            vol_long=vol_long,
            vol_ratio=vol_ratio,
            price_position=self._price_position("mexc", 1_800.0, now),
            flow_fast=flow_fast,
            flow_slow=flow_slow,
            mexc_flow=venue_flows["mexc"],
            bybit_flow=venue_flows["bybit"],
            binance_flow=venue_flows["binance"],
            book_imbalance=book_imbalance,
            microprice_bps=microprice,
            cross_venue_consensus=cross_consensus,
            oi_change_300s=oi_change,
            funding_rate=self.funding_rate,
            liquidation_imbalance=liq_imbalance,
            atr_pct=self.atr_pct(),
            trade_count_60s=total_trades,
            stale_venues=stale_venues,
        )
        self.last_feature = snapshot
        return snapshot


class MarketState:
    def __init__(self, symbols: Iterable[str]) -> None:
        self.symbols: dict[str, SymbolState] = {
            symbol: SymbolState(symbol=symbol) for symbol in symbols
        }
        self.feeds: dict[str, FeedStatus] = {
            name: FeedStatus(name=name) for name in ("mexc", "bybit", "binance")
        }
        self.started_at = time.time()

    def symbol(self, symbol: str) -> SymbolState:
        normalised = symbol.upper().replace("/", "_")
        if normalised not in self.symbols:
            self.symbols[normalised] = SymbolState(symbol=normalised)
        return self.symbols[normalised]

    def feed_connected(self, name: str) -> None:
        status = self.feeds.setdefault(name, FeedStatus(name=name))
        if status.connected:
            status.reconnects += 1
        status.connected = True
        status.last_error = ""

    def feed_disconnected(self, name: str, error: str = "") -> None:
        status = self.feeds.setdefault(name, FeedStatus(name=name))
        status.connected = False
        status.last_error = error[:500]

    def feed_message(self, name: str, ts: float | None = None) -> None:
        status = self.feeds.setdefault(name, FeedStatus(name=name))
        status.last_event_at = ts or time.time()
        status.messages += 1

    def status(self) -> dict[str, Any]:
        now = time.time()
        return {
            "started_at": self.started_at,
            "uptime_seconds": now - self.started_at,
            "feeds": {name: status.as_dict() for name, status in self.feeds.items()},
            "symbols": {
                symbol: {
                    "price": state.reference_price(),
                    "funding_rate": state.funding_rate,
                    "next_funding_at": state.next_funding_at,
                    "feature": state.last_feature.as_dict() if state.last_feature else None,
                }
                for symbol, state in self.symbols.items()
            },
        }
