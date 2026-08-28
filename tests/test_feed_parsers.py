import asyncio
import json

from app.feeds.binance import BinanceFeed
from app.feeds.bybit import BybitFeed
from app.feeds.mexc import MexcFeed
from app.market import MarketState
from app.models import Side

from conftest import make_settings


def test_public_feed_message_parsers(tmp_path) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path / "feeds.db")
        market = MarketState(settings.symbols)
        state = market.symbol("BTC_USDT")
        state.contract_size = 0.001
        state.book("mexc").apply_snapshot(
            [[100, 10]], [[101, 10]], version=10, qty_multiplier=0.001, ts=999
        )

        mexc = MexcFeed(market, settings)
        subscriptions = mexc.subscription_messages()
        assert {item["method"] for item in subscriptions} == {
            "sub.deal",
            "sub.depth.full",
            "sub.funding.rate",
        }
        assert not any(item["method"] == "sub.depth" for item in subscriptions)
        # MEXC acknowledges subscriptions with a string data payload. This
        # message must be ignored instead of tearing down the websocket.
        await mexc._handle(
            json.dumps({"channel": "rs.sub.deal", "data": "success"})
        )
        await mexc._handle(
            json.dumps(
                {
                    "channel": "push.deal",
                    "symbol": "BTC_USDT",
                    "ts": 1_000_000,
                    "data": [{"p": "100.5", "v": "20", "T": 1, "cts": 1_000_000}],
                }
            )
        )
        await mexc._handle(
            json.dumps(
                {
                    "channel": "push.depth",
                    "symbol": "BTC_USDT",
                    "ts": 1_001_000,
                    "data": {
                        # Full-depth snapshots do not require a contiguous
                        # version and replace the previous local book.
                        "version": 50,
                        "cts": 1_001_000,
                        "bids": [["100", "0"], ["99", "30"]],
                        "asks": [["101", "15"]],
                    },
                }
            )
        )
        assert state.trades["mexc"][-1].side == Side.LONG
        assert state.trades["mexc"][-1].base_qty == 0.02
        assert state.book("mexc").version == 50
        assert state.book("mexc").best_bid_ask() == (99.0, 101.0)

        bybit = BybitFeed(market, settings)
        await bybit._handle(
            {
                "topic": "orderbook.50.BTCUSDT",
                "type": "snapshot",
                "ts": 1_002_000,
                "data": {"s": "BTCUSDT", "u": 1, "b": [["100", "5"]], "a": [["101", "5"]]},
            }
        )
        await bybit._handle(
            {
                "topic": "publicTrade.BTCUSDT",
                "ts": 1_003_000,
                "data": [{"s": "BTCUSDT", "p": "100.7", "v": "0.4", "S": "Sell", "T": 1_003_000}],
            }
        )
        assert state.trades["bybit"][-1].side == Side.SHORT
        assert state.book("bybit").best_bid_ask() == (100.0, 101.0)

        binance = BinanceFeed(market, settings)
        await binance._handle(
            "book",
            {"e": "bookTicker", "s": "BTCUSDT", "st": 1, "b": "100", "B": "4", "a": "101", "A": "4", "u": 2, "E": 1_004_000},
        )
        await binance._handle(
            "trades",
            {"e": "aggTrade", "s": "BTCUSDT", "st": 1, "p": "100.8", "q": "0.3", "m": False, "T": 1_005_000},
        )
        assert state.book("binance").best_bid_ask() == (100.0, 101.0)
        assert state.trades["binance"][-1].side == Side.LONG

    asyncio.run(scenario())
