from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Any

from .config import Settings, account_configs
from . import __version__
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
        self.signal_count = 0
        self.open_count = 0
        self.close_count = 0

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
        self.tasks.append(asyncio.create_task(self._status_log_loop(), name="status-log"))
        LOGGER.info(
            "SERVICE READY version=%s mode=%s symbols=%d paper_balance=%.2f",
            __version__,
            self.settings.data_mode,
            len(self.settings.symbols),
            self.settings.paper_balance,
        )

    async def _start_live_feeds(self) -> None:
        mexc = MexcFeed(self.market, self.settings)
        bybit = BybitFeed(self.market, self.settings)
        feeds: list[Any] = [mexc, bybit]
        if self.settings.enable_binance:
            feeds.append(BinanceFeed(self.market, self.settings))
        self.feeds = feeds

        # Start public WebSockets immediately. MEXC REST warm-up is useful for
        # historical bars and contract metadata, but it must never hold Bybit
        # and Binance market data hostage when REST is slow or unavailable.
        runners = [asyncio.create_task(feed.run(), name=f"feed-{feed.name}") for feed in feeds]
        self.tasks.extend(runners)
        LOGGER.info("LIVE FEEDS STARTED venues=%s", ",".join(feed.name for feed in feeds))
        try:
            await mexc.bootstrap()
            LOGGER.info("MEXC BOOTSTRAP COMPLETE symbols=%d", len(self.settings.symbols))
        except Exception as exc:
            LOGGER.exception("MEXC bootstrap failed: %s", exc)
            await self.storage.event(time.time(), "ERROR", "MEXC_BOOTSTRAP", str(exc))
        await asyncio.gather(*runners)

    async def _status_log_loop(self) -> None:
        """Emit a low-volume operational heartbeat for hosting logs."""
        await asyncio.sleep(10.0)
        while not self._stopping:
            now = time.time()
            lag = now - self.last_engine_tick if self.last_engine_tick else -1.0
            feed_parts: list[str] = []
            errors: list[str] = []
            for name, status in self.market.feeds.items():
                state = "online" if status.connected else "offline"
                feed_parts.append(f"{name}:{state}/{status.messages}")
                if status.last_error:
                    errors.append(f"{name}={status.last_error[:120]}")
            diagnostics = self.router.diagnostics()
            state_counts: dict[str, int] = {}
            for item in diagnostics.values():
                decision_state = str(item.get("state") or "unknown")
                state_counts[decision_state] = state_counts.get(decision_state, 0) + 1
            ready_count = sum(bool(item.get("data_ready")) for item in diagnostics.values())
            scored = [
                item for item in diagnostics.values()
                if isinstance(item.get("best_score"), (int, float))
            ]
            best = max(scored, key=lambda item: float(item["best_score"])) if scored else None
            best_text = (
                f"{best['symbol']}/{best['best_setup']}:{float(best['best_score']):.1f}"
                if best
                else "none"
            )
            decision_text = ",".join(
                f"{name}:{count}" for name, count in sorted(state_counts.items())
            ) or "none"
            blocker_counts: dict[str, int] = {}
            for item in diagnostics.values():
                for blocker in item.get("blockers", []):
                    name = str(blocker)
                    blocker_counts[name] = blocker_counts.get(name, 0) + 1
            blocker_text = ",".join(
                f"{name}:{count}" for name, count in sorted(blocker_counts.items())
            ) or "none"
            accounts = self.broker.status()
            position_count = sum(int(item["positions_count"]) for item in accounts.values())
            LOGGER.info(
                "PAPER STATUS engine_lag=%.1fs feeds=%s ready=%d/%d blockers=%s decisions=%s "
                "best=%s regime=%s signals=%d opens=%d closes=%d positions=%d errors=%s",
                lag,
                ",".join(feed_parts),
                ready_count,
                len(self.settings.symbols),
                blocker_text,
                decision_text,
                best_text,
                self.router.last_regime.name,
                self.signal_count,
                self.open_count,
                self.close_count,
                position_count,
                " | ".join(errors) if errors else "none",
            )
            await asyncio.sleep(60.0)

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
                    self.close_count += 1
                    await self.storage.save_trade(trade)
                    LOGGER.info(
                        "PAPER CLOSE account=%s symbol=%s reason=%s pnl=%.2f R=%.2f",
                        trade.account,
                        trade.symbol,
                        trade.reason,
                        trade.net_pnl,
                        trade.r_multiple,
                    )

                signals = self.router.evaluate_all(self.market.symbols, features, tick_started)
                for signal in signals:
                    if not self._deduplicated(signal, tick_started):
                        continue
                    self.signal_count += 1
                    await self.storage.save_signal(signal)
                    opened = self.broker.handle_signal(signal, tick_started)
                    for position in opened:
                        self.open_count += 1
                        LOGGER.info(
                            "PAPER OPEN account=%s symbol=%s side=%s setup=%s score=%.1f notional=%.2f",
                            position.account,
                            position.symbol,
                            position.side.label,
                            position.setup,
                            position.score,
                            position.notional,
                        )

                for symbol, state in self.market.symbols.items():
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
            "diagnostics": await self.diagnostics(),
        }

    async def diagnostics(self) -> dict[str, Any]:
        return {
            "signal_count_since_start": self.signal_count,
            "open_count_since_start": self.open_count,
            "close_count_since_start": self.close_count,
            "market_regime": self.router.last_regime.name,
            "market_regime_direction": self.router.last_regime.direction,
            "market_breadth": self.router.last_regime.breadth,
            "market_stress": self.router.last_regime.stress,
            "symbols": self.router.diagnostics(),
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
