from app.market import SymbolState
from app.models import MinuteBar, Side, TradeEvent
from app.orderbook import OrderBook


def test_orderbook_metrics_and_sequence_gap() -> None:
    book = OrderBook("mexc", "BTC_USDT")
    book.apply_snapshot(
        bids=[[100, 10], [99, 5]],
        asks=[[101, 2], [102, 3]],
        version=10,
        ts=1000,
    )

    assert book.best_bid_ask() == (100.0, 101.0)
    assert book.imbalance(2) > 0
    assert book.microprice_bps() > 0
    assert book.impact_bps(Side.LONG, 100) >= 0

    assert book.apply_delta([[100, 0]], [[101, 4]], version=11, ts=1001)
    assert book.best_bid_ask() == (99.0, 101.0)
    assert not book.apply_delta([], [], version=13, ts=1002)
    assert book.version == 11


def test_features_are_ready_with_two_fresh_non_mexc_books() -> None:
    now = 10_000.0
    state = SymbolState("BTC_USDT")
    state.bootstrap_minutes(
        MinuteBar(
            ts=int(now - 420 + index * 60),
            open=100,
            high=101,
            low=99,
            close=100 + index * 0.01,
            volume_notional=1_000,
        )
        for index in range(7)
    )
    state.bootstrap_hours((int(now - (200 - index) * 3_600), 80 + index * 0.1) for index in range(200))
    for index in range(20):
        state.add_trade(
            TradeEvent(
                symbol="BTC_USDT",
                venue="mexc",
                price=100 + index * 0.001,
                base_qty=1,
                side=Side.LONG,
                ts=now - 10 + index * 0.1,
            )
        )
    for venue in ("bybit", "binance"):
        state.book(venue).apply_snapshot(
            bids=[[99.99, 10]], asks=[[100.01, 10]], ts=now
        )

    feature = state.features(now, stale_after=8)

    assert feature.data_ready is True
    assert feature.spread_bps < 35
    assert feature.stale_venues == ("mexc",)
