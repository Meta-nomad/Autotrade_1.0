from __future__ import annotations

import asyncio
import math
import random
import time

from ..config import Settings
from ..market import MarketState
from ..models import LiquidationEvent, MinuteBar, Side, TradeEvent


BASE_PRICES = {
    "BTC_USDT": 100_000.0,
    "ETH_USDT": 4_000.0,
    "SOL_USDT": 200.0,
    "XRP_USDT": 2.0,
    "BNB_USDT": 900.0,
    "DOGE_USDT": 0.25,
    "ADA_USDT": 0.8,
    "AVAX_USDT": 35.0,
    "LINK_USDT": 24.0,
    "LTC_USDT": 125.0,
}


class SyntheticFeed:
    """Deterministic high-activity feed for tests and local demonstrations."""

    name = "synthetic"

    def __init__(self, market: MarketState, settings: Settings) -> None:
        self.market = market
        self.settings = settings
        self.random = random.Random(20260827)
        self.prices = {symbol: BASE_PRICES.get(symbol, 100.0) for symbol in settings.symbols}
        self._stop = asyncio.Event()
        self._step = 0

    async def bootstrap(self) -> None:
        now = int(time.time())
        for index, symbol in enumerate(self.settings.symbols):
            state = self.market.symbol(symbol)
            base = self.prices[symbol]
            minute_bars: list[MinuteBar] = []
            for offset in range(420, 0, -1):
                ts = int((now - offset * 60) // 60) * 60
                phase = (420 - offset) / 35.0 + index
                close = base * (1.0 + 0.003 * math.sin(phase) + 0.00001 * (420 - offset))
                open_price = minute_bars[-1].close if minute_bars else close
                high = max(open_price, close) * 1.0008
                low = min(open_price, close) * 0.9992
                minute_bars.append(
                    MinuteBar(
                        ts=ts,
                        open=open_price,
                        high=high,
                        low=low,
                        close=close,
                        volume_notional=2_000_000.0 * (1.0 + 0.2 * math.sin(phase)),
                    )
                )
            state.bootstrap_minutes(minute_bars)
            hours = []
            for offset in range(300, 0, -1):
                ts = int((now - offset * 3_600) // 3_600) * 3_600
                close = base * (0.92 + (300 - offset) * 0.00028 + 0.01 * math.sin(offset / 17.0))
                hours.append((ts, close))
            state.bootstrap_hours(hours)
            state.contract_size = 0.001
            state.update_derivatives(
                ts=time.time(),
                open_interest=1_000_000.0 + index * 10_000,
                funding_rate=0.0001,
                next_funding_at=time.time() + 3_600,
            )
        for venue in ("mexc", "bybit", "binance"):
            self.market.feed_connected(venue)

    async def run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            self._step += 1
            for index, symbol in enumerate(self.settings.symbols):
                price = self.prices[symbol]
                cycle = (self._step + index * 11) % 600
                drift = 0.00001
                pressure = 0.10 * math.sin((self._step + index) / 25.0)
                if 60 <= cycle < 120:
                    drift = 0.00022
                    pressure = 0.72
                elif 240 <= cycle < 300:
                    drift = -0.00022
                    pressure = -0.72
                shock = self.random.gauss(0.0, 0.00007)
                price *= 1.0 + drift + shock
                self.prices[symbol] = price
                spread_bps = 1.2 + index * 0.08
                spread = price * spread_bps / 10_000.0
                for venue, venue_bias in (("mexc", 0.0), ("bybit", 0.15), ("binance", -0.10)):
                    venue_mid = price * (1.0 + venue_bias / 10_000.0)
                    bids = []
                    asks = []
                    levels = 20 if venue != "binance" else 1
                    for level in range(1, levels + 1):
                        qty = max(0.1, 10.0 * (1.0 + float(pressure) * (1 if venue != "mexc" else 0.8)))
                        bids.append([venue_mid - spread / 2.0 - level * spread, qty * (1.0 + pressure)])
                        asks.append([venue_mid + spread / 2.0 + level * spread, qty * (1.0 - pressure)])
                    self.market.symbol(symbol).book(venue).apply_snapshot(bids, asks, ts=now)
                    self.market.symbol(symbol).record_book_mid(venue, now)
                    self.market.feed_message(venue, now)

                for venue in ("mexc", "bybit", "binance"):
                    for _ in range(4):
                        buy_probability = _clamp_probability(0.5 + pressure * 0.35)
                        side = Side.LONG if self.random.random() < buy_probability else Side.SHORT
                        # During the scripted expansion regimes, large prints dominate
                        # the preceding neutral tape. This makes the offline demo exercise
                        # the same signal/open-position path as a real impulsive market.
                        activity = 8.0 if abs(pressure) >= 0.7 else 1.0
                        qty = max(
                            0.001,
                            3_000.0 / price * self.random.uniform(0.3, 2.0) * activity,
                        )
                        self.market.symbol(symbol).add_trade(
                            TradeEvent(
                                symbol=symbol,
                                venue=venue,
                                price=price * (1.0 + float(side) * spread_bps / 20_000.0),
                                base_qty=qty,
                                side=side,
                                ts=now,
                            )
                        )
                state = self.market.symbol(symbol)
                state.update_derivatives(
                    ts=now,
                    open_interest=1_000_000.0 * (1.0 + 0.001 * math.sin(self._step / 20.0)),
                    funding_rate=0.0001 * math.sin(self._step / 200.0),
                )
                if cycle in range(230, 235):
                    state.add_liquidation(
                        LiquidationEvent(
                            symbol=symbol,
                            venue="bybit",
                            price=price,
                            base_qty=200_000.0 / price,
                            pressure_side=Side.LONG,
                            ts=now,
                        )
                    )
                elif cycle in range(50, 55):
                    state.add_liquidation(
                        LiquidationEvent(
                            symbol=symbol,
                            venue="bybit",
                            price=price,
                            base_qty=200_000.0 / price,
                            pressure_side=Side.SHORT,
                            ts=now,
                        )
                    )
            await asyncio.sleep(0.2)

    async def stop(self) -> None:
        self._stop.set()


def _clamp_probability(value: float) -> float:
    return max(0.02, min(0.98, value))
