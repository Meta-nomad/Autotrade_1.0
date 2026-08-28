from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from typing import Any, Iterable

import websockets

from ..config import Settings
from ..market import MarketState
from ..models import LiquidationEvent, Side, TradeEvent


LOGGER = logging.getLogger(__name__)


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


class BybitFeed:
    name = "bybit"
    ws_url = "wss://stream.bybit.com/v5/public/linear"

    def __init__(self, market: MarketState, settings: Settings) -> None:
        self.market = market
        self.settings = settings
        self._stop = asyncio.Event()
        self._ticker_cache: dict[str, dict[str, Any]] = {}

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=None,
                    close_timeout=5,
                    max_size=16 * 1024 * 1024,
                ) as ws:
                    self.market.feed_connected(self.name)
                    backoff = 1.0
                    topics: list[str] = []
                    for mexc_symbol in self.settings.symbols:
                        symbol = mexc_symbol.replace("_", "")
                        topics.extend(
                            (
                                f"publicTrade.{symbol}",
                                f"orderbook.{self.settings.bybit_depth}.{symbol}",
                                f"tickers.{symbol}",
                                f"allLiquidation.{symbol}",
                            )
                        )
                    for group in _chunks(topics, 10):
                        await ws.send(json.dumps({"op": "subscribe", "args": group}))
                    heartbeat = asyncio.create_task(self._heartbeat(ws))
                    try:
                        async for raw in ws:
                            await self._handle(json.loads(raw))
                    finally:
                        heartbeat.cancel()
                        with suppress(asyncio.CancelledError):
                            await heartbeat
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.market.feed_disconnected(self.name, str(exc))
                LOGGER.warning("Bybit websocket reconnect in %.1fs: %s", backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _heartbeat(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(20.0)
            await ws.send('{"op":"ping"}')

    @staticmethod
    def _mexc_symbol(symbol: str) -> str:
        upper = symbol.upper()
        if upper.endswith("USDT"):
            return f"{upper[:-4]}_USDT"
        return upper

    async def _handle(self, payload: dict[str, Any]) -> None:
        topic = str(payload.get("topic") or "")
        if not topic:
            return
        ts = float(payload.get("ts") or time.time() * 1000) / 1000.0
        self.market.feed_message(self.name, ts)

        if topic.startswith("publicTrade."):
            rows = payload.get("data") or []
            for row in rows:
                symbol = self._mexc_symbol(str(row.get("s") or topic.split(".")[-1]))
                if symbol not in self.market.symbols:
                    continue
                self.market.symbol(symbol).add_trade(
                    TradeEvent(
                        symbol=symbol,
                        venue="bybit",
                        price=float(row.get("p") or 0.0),
                        base_qty=float(row.get("v") or 0.0),
                        side=Side.LONG if str(row.get("S")).lower() == "buy" else Side.SHORT,
                        ts=float(row.get("T") or ts * 1000) / 1000.0,
                    )
                )
        elif topic.startswith("orderbook."):
            data = payload.get("data") or {}
            symbol = self._mexc_symbol(str(data.get("s") or topic.split(".")[-1]))
            if symbol not in self.market.symbols:
                return
            book = self.market.symbol(symbol).book("bybit")
            event_ts = float(payload.get("cts") or payload.get("ts") or ts * 1000) / 1000.0
            if payload.get("type") == "snapshot":
                book.apply_snapshot(data.get("b", []), data.get("a", []), version=int(data.get("u") or 0), ts=event_ts)
            else:
                # Bybit sequence can jump by more than one; `u` is a monotonic update id,
                # not a guaranteed local +1 sequence.
                book._apply_side(book.bids, data.get("b", []), 1.0)
                book._apply_side(book.asks, data.get("a", []), 1.0)
                book.version = int(data.get("u") or book.version or 0)
                book.updated_at = event_ts
            self.market.symbol(symbol).record_book_mid("bybit", event_ts)
        elif topic.startswith("tickers."):
            data = payload.get("data") or {}
            if isinstance(data, list):
                data = data[0] if data else {}
            bybit_symbol = str(data.get("symbol") or topic.split(".")[-1])
            symbol = self._mexc_symbol(bybit_symbol)
            if symbol not in self.market.symbols:
                return
            cached = self._ticker_cache.setdefault(symbol, {})
            cached.update(data)
            state = self.market.symbol(symbol)
            if cached.get("lastPrice"):
                state.record_price("bybit", float(cached["lastPrice"]), ts)
            state.update_derivatives(
                ts=ts,
                open_interest=float(cached["openInterest"]) if cached.get("openInterest") else None,
                funding_rate=float(cached["fundingRate"]) if cached.get("fundingRate") else None,
                next_funding_at=(
                    float(cached["nextFundingTime"]) / 1000.0
                    if cached.get("nextFundingTime")
                    else None
                ),
            )
        elif topic.startswith("allLiquidation."):
            rows = payload.get("data") or []
            if isinstance(rows, dict):
                rows = [rows]
            for row in rows:
                symbol = self._mexc_symbol(str(row.get("s") or topic.split(".")[-1]))
                if symbol not in self.market.symbols:
                    continue
                # Bybit documents S=Buy as a liquidated long; that creates sell pressure.
                pressure = Side.SHORT if str(row.get("S")).lower() == "buy" else Side.LONG
                self.market.symbol(symbol).add_liquidation(
                    LiquidationEvent(
                        symbol=symbol,
                        venue="bybit",
                        price=float(row.get("p") or 0.0),
                        base_qty=float(row.get("v") or 0.0),
                        pressure_side=pressure,
                        ts=float(row.get("T") or ts * 1000) / 1000.0,
                    )
                )

    async def stop(self) -> None:
        self._stop.set()

