from app.models import Side
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

