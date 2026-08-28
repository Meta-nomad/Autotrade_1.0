from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
from contextlib import suppress
from typing import Any

import httpx
import websockets

from ..config import Settings
from ..market import MarketState
from ..models import MinuteBar, Side, TradeEvent


LOGGER = logging.getLogger(__name__)


class MexcFeed:
    name = "mexc"
    rest_url = "https://api.mexc.com"
    ws_url = "wss://contract.mexc.com/edge"

    def __init__(self, market: MarketState, settings: Settings) -> None:
        self.market = market
        self.settings = settings
        self._stop = asyncio.Event()
        self._http: httpx.AsyncClient | None = None
        self._refreshing: set[str] = set()

    async def bootstrap(self) -> None:
        self._http = httpx.AsyncClient(timeout=15.0, headers={"Language": "en-US"})
        semaphore = asyncio.Semaphore(4)

        async def load(symbol: str) -> None:
            async with semaphore:
                try:
                    await self._load_contract(symbol)
                    await self._refresh_depth(symbol)
                    await self._load_klines(symbol)
                    await self._load_funding(symbol)
                except Exception as exc:  # feed retries live even if bootstrap is partial
                    LOGGER.warning("MEXC bootstrap failed for %s: %s", symbol, exc)

        await asyncio.gather(*(load(symbol) for symbol in self.settings.symbols))

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=15.0, headers={"Language": "en-US"})
        response = await self._http.get(f"{self.rest_url}{path}", params=params)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success", False):
            raise RuntimeError(payload.get("message") or f"MEXC error {payload.get('code')}")
        return payload.get("data")

    async def _load_contract(self, symbol: str) -> None:
        data = await self._get("/api/v1/contract/detail/country", {"symbol": symbol})
        if isinstance(data, list):
            data = next((item for item in data if item.get("symbol") == symbol), {})
        if not isinstance(data, dict):
            return
        state = self.market.symbol(symbol)
        state.contract_size = float(data.get("contractSize") or 1.0)
        state.maintenance_margin_rate = float(data.get("maintenanceMarginRate") or 0.005)
        state.api_allowed = bool(data.get("apiAllowed", True))

    async def _refresh_depth(self, symbol: str) -> None:
        if symbol in self._refreshing:
            return
        self._refreshing.add(symbol)
        try:
            data = await self._get(f"/api/v1/contract/depth/{symbol}", {"limit": 200})
            if not isinstance(data, dict):
                return
            state = self.market.symbol(symbol)
            state.book("mexc").apply_snapshot(
                data.get("bids", []),
                data.get("asks", []),
                version=int(data["version"]) if data.get("version") is not None else None,
                qty_multiplier=state.contract_size,
                ts=float(data.get("timestamp", time.time() * 1000)) / 1000.0,
            )
            state.record_book_mid("mexc")
        finally:
            self._refreshing.discard(symbol)

    async def _load_klines(self, symbol: str) -> None:
        now = int(time.time())
        minute_data, hour_data = await asyncio.gather(
            self._get(
                f"/api/v1/contract/kline/{symbol}",
                {"interval": "Min1", "start": now - 420 * 60, "end": now},
            ),
            self._get(
                f"/api/v1/contract/kline/{symbol}",
                {"interval": "Min60", "start": now - 320 * 3_600, "end": now},
            ),
        )
        if isinstance(minute_data, dict):
            times = minute_data.get("time", [])
            opens = minute_data.get("open", [])
            highs = minute_data.get("high", [])
            lows = minute_data.get("low", [])
            closes = minute_data.get("close", [])
            amounts = minute_data.get("amount", minute_data.get("vol", []))
            count = min(map(len, (times, opens, highs, lows, closes, amounts))) if times else 0
            bars = [
                MinuteBar(
                    ts=int(times[index]),
                    open=float(opens[index]),
                    high=float(highs[index]),
                    low=float(lows[index]),
                    close=float(closes[index]),
                    volume_notional=float(amounts[index]),
                )
                for index in range(count)
            ]
            self.market.symbol(symbol).bootstrap_minutes(bars)
        if isinstance(hour_data, dict):
            closes = hour_data.get("close", [])
            times = hour_data.get("time", [])
            self.market.symbol(symbol).bootstrap_hours(
                (int(ts), float(close)) for ts, close in zip(times, closes)
            )

    async def _load_funding(self, symbol: str) -> None:
        data = await self._get(f"/api/v1/contract/funding_rate/{symbol}")
        if not isinstance(data, dict):
            return
        next_settle = float(data.get("nextSettleTime") or 0.0)
        if next_settle > 10_000_000_000:
            next_settle /= 1000.0
        self.market.symbol(symbol).update_derivatives(
            ts=time.time(),
            funding_rate=float(data.get("fundingRate") or data.get("rate") or 0.0),
            next_funding_at=next_settle,
        )

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=None,
                    close_timeout=5,
                    max_size=8 * 1024 * 1024,
                ) as ws:
                    self.market.feed_connected(self.name)
                    backoff = 1.0
                    for symbol in self.settings.symbols:
                        await ws.send(json.dumps({"method": "sub.deal", "param": {"symbol": symbol}}))
                        await ws.send(json.dumps({"method": "sub.depth", "param": {"symbol": symbol}}))
                        await ws.send(
                            json.dumps(
                                {
                                    "method": "sub.funding.rate",
                                    "param": {"symbol": symbol},
                                    "gzip": False,
                                }
                            )
                        )
                    heartbeat = asyncio.create_task(self._heartbeat(ws))
                    try:
                        async for raw in ws:
                            await self._handle(raw)
                    finally:
                        heartbeat.cancel()
                        with suppress(asyncio.CancelledError):
                            await heartbeat
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.market.feed_disconnected(self.name, str(exc))
                LOGGER.warning("MEXC websocket reconnect in %.1fs: %s", backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _heartbeat(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(15.0)
            await ws.send('{"method":"ping"}')

    @staticmethod
    def _decode(raw: str | bytes) -> dict[str, Any]:
        if isinstance(raw, bytes):
            with suppress(OSError):
                raw = gzip.decompress(raw)
            raw = raw.decode("utf-8")
        return json.loads(raw)

    async def _handle(self, raw: str | bytes) -> None:
        payload = self._decode(raw)
        channel = payload.get("channel", "")
        if channel == "pong" or not channel:
            return
        symbol = str(payload.get("symbol") or payload.get("data", {}).get("symbol") or "")
        if not symbol:
            return
        symbol = symbol.upper().replace("/", "_")
        if symbol not in self.market.symbols:
            return
        self.market.feed_message(self.name)
        state = self.market.symbol(symbol)
        ts = float(payload.get("ts") or time.time() * 1000) / 1000.0

        if channel == "push.deal":
            rows = payload.get("data", [])
            if isinstance(rows, dict):
                rows = [rows]
            for row in rows:
                price = float(row.get("p") or 0.0)
                contracts = float(row.get("v") or 0.0)
                event_ts = float(row.get("cts") or row.get("t") or ts * 1000) / 1000.0
                state.add_trade(
                    TradeEvent(
                        symbol=symbol,
                        venue="mexc",
                        price=price,
                        base_qty=contracts * state.contract_size,
                        side=Side.LONG if int(row.get("T") or 1) == 1 else Side.SHORT,
                        ts=event_ts,
                        open_close=int(row["O"]) if row.get("O") is not None else None,
                    )
                )
        elif channel == "push.depth":
            data = payload.get("data") or {}
            version = int(data["version"]) if data.get("version") is not None else None
            applied = state.book("mexc").apply_delta(
                data.get("bids", []),
                data.get("asks", []),
                version=version,
                qty_multiplier=state.contract_size,
                ts=float(data.get("cts") or ts * 1000) / 1000.0,
            )
            if applied:
                state.record_book_mid("mexc", ts)
            else:
                asyncio.create_task(self._refresh_depth(symbol))
        elif channel == "push.funding.rate":
            data = payload.get("data") or {}
            next_settle = float(data.get("nextSettleTime") or 0.0)
            if next_settle > 10_000_000_000:
                next_settle /= 1000.0
            state.update_derivatives(
                ts=ts,
                funding_rate=float(data.get("fundingRate") or data.get("rate") or 0.0),
                next_funding_at=next_settle or None,
            )

    async def stop(self) -> None:
        self._stop.set()
        if self._http is not None:
            await self._http.aclose()

