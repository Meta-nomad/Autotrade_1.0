from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Any

from .config import Settings, account_configs
from .feeds.binance import BinanceFeed
from .feeds.bybit import BybitFeed
from .feeds.mexc import MexcFeed
from .feeds.synthetic import SyntheticFeed
from .market import MarketState
from .models import FeatureSnapshot, Signal
from .paper import PaperBroker
from .storage import Storage
from .strategy import StrategyRouter


LOGGER = logging.getLogger(__name__)


class PaperTradingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.market = MarketState(settings.symbols)
        self.storage = Storage(settings.db_path)
        self.broker = PaperBroker(self.market, settings, account_configs(settings))
        self.router = StrategyRouter(settings)
        self.feeds: list[Any] = []
        self.tasks: list[asyncio.Task[Any]] = []
        self.started_at = time.time()
        self.last_engine_tick = 0.0
        self.last_engine_error = ""
        self._stopping = False
        self._last_signal: dict[tuple[str, str, str, int], float] = {}
        self._last_feature_persist = 0.0
        self._last_account_persist = 0.0
        self._last_equity_persist = 0.0

    async def start(self) -> None:
        await self.storage.initialise()
        restored = await self.storage.load_account_states()
        self.broker.restore(restored)
        await self.storage.event(time.time(), "INFO", "SERVICE_START", self.settings.data_mode)
        if self.settings.data_mode == "synthetic":
            feed = SyntheticFeed(self.market, self.settings)
            await feed.bootstrap()
            self.feeds = [feed]
            self.tasks.append(asyncio.create_task(feed.run(), name="synthetic-feed"))
        else:
            self.tasks.append(asyncio.create_task(self._start_live_feeds(), name="live-feeds-bootstrap"))
        self.tasks.append(asyncio.create_task(self._engine_loop(), name="paper-engine"))

    async def _start_live_feeds(self) -> None:
        mexc = MexcFeed(self.market, self.settings)
        bybit = BybitFeed(self.market, self.settings)
        feeds: list[Any] = [mexc, bybit]
        if self.settings.enable_binance:
            feeds.append(BinanceFeed(self.market, self.settings))
        self.feeds = feeds
        try:
            await mexc.bootstrap()
        except Exception as exc:
            LOGGER.exception("MEXC bootstrap failed: %s", exc)
            await self.storage.event(time.time(), "ERROR", "MEXC_BOOTSTRAP", str(exc))
        runners = [asyncio.create_task(feed.run(), name=f"feed-{feed.name}") for feed in feeds]
        self.tasks.extend(runners)
        await asyncio.gather(*runners)

    def _deduplicated(self, signal: Signal, now: float) -> bool:
        key = (signal.strategy, signal.symbol, signal.setup, int(signal.side))
        previous = self._last_signal.get(key, 0.0)
        if now - previous < 60.0:
            return False
        self._last_signal[key] = now
        return True

    async def _engine_loop(self) -> None:
        while not self._stopping:
            tick_started = time.time()
            try:
                features: dict[str, FeatureSnapshot] = {
                    symbol: state.features(tick_started, self.settings.stale_after_seconds)
                    for symbol, state in self.market.symbols.items()
                }
                closed = self.broker.evaluate_positions(features, tick_started)
                for trade in closed:
                    await self.storage.save_trade(trade)
                    LOGGER.info(
                        "PAPER CLOSE account=%s symbol=%s reason=%s pnl=%.2f R=%.2f",
                        trade.account,
                        trade.symbol,
                        trade.reason,
                        trade.net_pnl,
                        trade.r_multiple,
                    )

                for symbol, feature in features.items():
                    state = self.market.symbol(symbol)
                    control_signal, orderflow_signal = self.router.evaluate(state, feature, tick_started)
                    for signal in (control_signal, orderflow_signal):
                        if signal is None or not self._deduplicated(signal, tick_started):
                            continue
                        await self.storage.save_signal(signal)
                        opened = self.broker.handle_signal(signal, tick_started)
                        for position in opened:
                            LOGGER.info(
                                "PAPER OPEN account=%s symbol=%s side=%s setup=%s score=%.1f notional=%.2f",
                                position.account,
                                position.symbol,
                                position.side.label,
                                position.setup,
                                position.score,
                                position.notional,
                            )

                    if state.next_funding_at and tick_started >= state.next_funding_at:
                        settlement = state.next_funding_at
                        self.broker.apply_funding(symbol, state.funding_rate, settlement)
                        # Most contracts use an 8-hour cycle; a newer exchange update
                        # will replace this timestamp as soon as it arrives.
                        state.next_funding_at = settlement + 8 * 3_600.0

                if tick_started - self._last_feature_persist >= self.settings.feature_persist_seconds:
                    for feature in features.values():
                        await self.storage.save_feature(feature)
                    self._last_feature_persist = tick_started

                if tick_started - self._last_account_persist >= self.settings.account_persist_seconds:
                    await self.storage.save_account_states(tick_started, self.broker.snapshots())
                    self._last_account_persist = tick_started

                if tick_started - self._last_equity_persist >= 60.0:
                    await self.storage.save_equity(tick_started, self.broker.status())
                    self._last_equity_persist = tick_started

                self.last_engine_tick = tick_started
                self.last_engine_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_engine_error = str(exc)
                LOGGER.exception("Paper engine tick failed")
                with suppress(Exception):
                    await self.storage.event(time.time(), "ERROR", "ENGINE_TICK", str(exc))

            elapsed = time.time() - tick_started
            await asyncio.sleep(max(0.05, self.settings.evaluation_interval_seconds - elapsed))

    async def status(self) -> dict[str, Any]:
        now = time.time()
        return {
            "mode": "PAPER_ONLY",
            "data_mode": self.settings.data_mode,
            "live_trading_enabled": False,
            "started_at": self.started_at,
            "uptime_seconds": now - self.started_at,
            "last_engine_tick": self.last_engine_tick,
            "engine_lag_seconds": now - self.last_engine_tick if self.last_engine_tick else None,
            "last_engine_error": self.last_engine_error,
            "accounts": self.broker.status(),
            "market": self.market.status(),
        }

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        for feed in self.feeds:
            with suppress(Exception):
                await feed.stop()
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
        with suppress(Exception):
            await self.storage.save_account_states(time.time(), self.broker.snapshots())
        with suppress(Exception):
            await self.storage.event(time.time(), "INFO", "SERVICE_STOP", "")
        await self.storage.close()

