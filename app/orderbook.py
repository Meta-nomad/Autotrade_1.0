from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from .models import Side


@dataclass(slots=True)
class OrderBook:
    venue: str
    symbol: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    version: int | None = None
    updated_at: float = 0.0

    def apply_snapshot(
        self,
        bids: Iterable[Iterable[float | str]],
        asks: Iterable[Iterable[float | str]],
        *,
        version: int | None = None,
        qty_multiplier: float = 1.0,
        ts: float | None = None,
    ) -> None:
        self.bids = self._normalise(bids, qty_multiplier)
        self.asks = self._normalise(asks, qty_multiplier)
        self.version = version
        self.updated_at = ts or time.time()

    def apply_delta(
        self,
        bids: Iterable[Iterable[float | str]],
        asks: Iterable[Iterable[float | str]],
        *,
        version: int | None = None,
        qty_multiplier: float = 1.0,
        ts: float | None = None,
    ) -> bool:
        if version is not None and self.version is not None:
            if version <= self.version:
                return True
            if version != self.version + 1:
                return False
        self._apply_side(self.bids, bids, qty_multiplier)
        self._apply_side(self.asks, asks, qty_multiplier)
        if version is not None:
            self.version = version
        self.updated_at = ts or time.time()
        return True

    @staticmethod
    def _normalise(
        rows: Iterable[Iterable[float | str]], qty_multiplier: float
    ) -> dict[float, float]:
        result: dict[float, float] = {}
        for row in rows:
            values = list(row)
            if len(values) < 2:
                continue
            price = float(values[0])
            qty = float(values[1]) * qty_multiplier
            if price > 0 and qty > 0:
                result[price] = qty
        return result

    @staticmethod
    def _apply_side(
        target: dict[float, float],
        rows: Iterable[Iterable[float | str]],
        qty_multiplier: float,
    ) -> None:
        for row in rows:
            values = list(row)
            if len(values) < 2:
                continue
            price = float(values[0])
            qty = float(values[1]) * qty_multiplier
            if qty <= 0:
                target.pop(price, None)
            elif price > 0:
                target[price] = qty

    def top_bids(self, depth: int = 10) -> list[tuple[float, float]]:
        return sorted(self.bids.items(), reverse=True)[:depth]

    def top_asks(self, depth: int = 10) -> list[tuple[float, float]]:
        return sorted(self.asks.items())[:depth]

    def best_bid_ask(self) -> tuple[float, float] | None:
        if not self.bids or not self.asks:
            return None
        return max(self.bids), min(self.asks)

    def mid(self) -> float | None:
        bbo = self.best_bid_ask()
        if bbo is None:
            return None
        return (bbo[0] + bbo[1]) / 2.0

    def spread_bps(self) -> float:
        bbo = self.best_bid_ask()
        if bbo is None:
            return 99999.0
        bid, ask = bbo
        mid = (bid + ask) / 2.0
        return (ask - bid) / mid * 10_000.0 if mid > 0 else 99999.0

    def imbalance(self, depth: int = 10) -> float:
        bid_value = sum(price * qty for price, qty in self.top_bids(depth))
        ask_value = sum(price * qty for price, qty in self.top_asks(depth))
        total = bid_value + ask_value
        return (bid_value - ask_value) / total if total > 0 else 0.0

    def microprice_bps(self) -> float:
        bbo = self.best_bid_ask()
        if bbo is None:
            return 0.0
        bid, ask = bbo
        bid_qty = self.bids.get(bid, 0.0)
        ask_qty = self.asks.get(ask, 0.0)
        total = bid_qty + ask_qty
        if total <= 0:
            return 0.0
        microprice = (ask * bid_qty + bid * ask_qty) / total
        mid = (bid + ask) / 2.0
        return (microprice - mid) / mid * 10_000.0

    def impact_bps(self, side: Side, notional: float) -> float:
        if notional <= 0:
            return 0.0
        levels = self.top_asks(50) if side == Side.LONG else self.top_bids(50)
        bbo = self.best_bid_ask()
        if not levels or bbo is None:
            return 10.0
        reference = bbo[1] if side == Side.LONG else bbo[0]
        remaining = notional
        paid = 0.0
        base_filled = 0.0
        for price, qty in levels:
            level_notional = price * qty
            take = min(remaining, level_notional)
            paid += take
            base_filled += take / price
            remaining -= take
            if remaining <= 1e-9:
                break
        if remaining > 0 or base_filled <= 0:
            return 25.0
        average = paid / base_filled
        raw = (average - reference) / reference * 10_000.0 * float(side)
        return max(0.0, raw)

    def is_fresh(self, now: float, stale_after: float) -> bool:
        return bool(self.updated_at and now - self.updated_at <= stale_after)

