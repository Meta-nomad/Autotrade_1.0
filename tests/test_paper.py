from app.config import account_configs
from app.market import MarketState
from app.models import FeatureSnapshot, Side, Signal
from app.paper import PaperBroker

from conftest import make_settings


def _seed_books(market: MarketState, bid: float, ask: float, ts: float) -> None:
    state = market.symbol("BTC_USDT")
    for venue in ("mexc", "bybit"):
        state.book(venue).apply_snapshot(
            bids=[[bid, 100]], asks=[[ask, 100]], ts=ts
        )


def test_orderflow_signal_opens_two_virtual_accounts_and_closes_at_target(tmp_path) -> None:
    settings = make_settings(tmp_path / "paper.db")
    market = MarketState(settings.symbols)
    _seed_books(market, 99.9, 100.1, 1_000)
    broker = PaperBroker(market, settings, account_configs(settings))
    signal = Signal(
        symbol="BTC_USDT",
        strategy="order_flow",
        setup="EXPANSION_BREAKOUT",
        side=Side.LONG,
        score=95,
        stop_pct=0.01,
        target_r=2.5,
        ts=1_000,
    )

    opened = broker.handle_signal(signal, now=1_000)

    assert {position.account for position in opened} == {
        "ORDER_FLOW",
        "ORDER_FLOW_TURBO",
    }
    assert not broker.accounts["CONTROL_20"].positions
    assert all(position.notional > 0 for position in opened)

    _seed_books(market, 103.0, 103.1, 1_100)
    feature = FeatureSnapshot(
        symbol="BTC_USDT", ts=1_100, price=103, spread_bps=2, data_ready=True
    )
    closed = broker.evaluate_positions({"BTC_USDT": feature}, now=1_100)

    assert len(closed) == 2
    assert all(trade.reason == "TARGET" for trade in closed)
    assert all(trade.net_pnl > 0 for trade in closed)

