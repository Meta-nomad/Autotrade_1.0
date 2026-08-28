from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import websockets

from ..config import Settings
from ..market import MarketState
from ..models import Side, TradeEvent


LOGGER = logging.getLogger(__name__)


class BinanceFeed:
    name = "binance"
    market_ws_url = "wss://fstream.binance.com/market/stream"
    public_ws_url = "wss://fstream.binance.com/public/stream"

    def __init__(self, market: MarketState, settings: Settings) -> None:
        self.market = market
        self.settings = settings
        self._stop = asyncio.Event()

    async def run(self) -> None:
        await asyncio.gather(
            self._run_channel("trades", self.market_ws_url),
            self._run_channel("book", self.public_ws_url),
        )

    async def _run_channel(self, kind: str, url: str) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_size=8 * 1024 * 1024,
                ) as ws:
                    self.market.feed_connected(self.name)
                    backoff = 1.0
                    suffix = "aggTrade" if kind == "trades" else "bookTicker"
                    params = [f"{symbol.replace('_', '').lower()}@{suffix}" for symbol in self.settings.symbols]
                    await ws.send(json.dumps({"method": "SUBSCRIBE", "params": params, "id": kind}))
                    async for raw in ws:
                        payload = json.loads(raw)
                        data = payload.get("data", payload)
                        if isinstance(data, dict):
                            await self._handle(kind, data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.market.feed_disconnected(self.name, str(exc))
                LOGGER.warning("Binance %s websocket reconnect in %.1fs: %s", kind, backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    @staticmethod
    def _mexc_symbol(symbol: str) -> str:
        upper = symbol.upper()
        return f"{upper[:-4]}_USDT" if upper.endswith("USDT") else upper

    async def _handle(self, kind: str, data: dict[str, Any]) -> None:
        # After Binance's UM/CM stream merge, st=1 means USD-M.
        if data.get("st") not in (None, 1):
            return
        symbol = self._mexc_symbol(str(data.get("s") or data.get("ps") or ""))
        if symbol not in self.market.symbols:
            return
        ts = float(data.get("T") or data.get("E") or time.time() * 1000) / 1000.0
        self.market.feed_message(self.name, ts)
        state = self.market.symbol(symbol)
        if kind == "trades" and data.get("e") == "aggTrade":
            state.add_trade(
                TradeEvent(
                    symbol=symbol,
                    venue="binance",
                    price=float(data.get("p") or 0.0),
                    base_qty=float(data.get("nq") or data.get("q") or 0.0),
                    # m=true: buyer was maker, therefore seller was aggressor.
                    side=Side.SHORT if bool(data.get("m")) else Side.LONG,
                    ts=ts,
                )
            )
        elif kind == "book" and data.get("b") and data.get("a"):
            state.book("binance").apply_snapshot(
                [[data["b"], data.get("B") or 0.0]],
                [[data["a"], data.get("A") or 0.0]],
                version=int(data.get("u") or 0),
                ts=ts,
            )
            state.record_book_mid("binance", ts)

    async def stop(self) -> None:
        self._stop.set()

