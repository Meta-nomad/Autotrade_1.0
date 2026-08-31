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


def test_baseline_signal_opens_virtual_account_and_closes_at_target(tmp_path) -> None:
    settings = make_settings(tmp_path / "paper.db")
    market = MarketState(settings.symbols)
    _seed_books(market, 99.9, 100.1, 1_000)
    broker = PaperBroker(market, settings, account_configs(settings))
    signal = Signal(
        symbol="BTC_USDT",
        strategy="baseline",
        setup="EXPANSION_BREAKOUT",
        side=Side.LONG,
        score=95,
        stop_pct=0.01,
        target_r=2.5,
        ts=1_000,
    )

    opened = broker.handle_signal(signal, now=1_000)

    assert {position.account for position in opened} == {"BASELINE_016"}
    assert all(position.notional > 0 for position in opened)

    _seed_books(market, 103.0, 103.1, 1_100)
    feature = FeatureSnapshot(
        symbol="BTC_USDT", ts=1_100, price=103, spread_bps=2, data_ready=True
    )
    closed = broker.evaluate_positions({"BTC_USDT": feature}, now=1_100)

    assert len(closed) == 1
    assert all(trade.reason == "TARGET" for trade in closed)
    assert all(trade.net_pnl > 0 for trade in closed)


def test_paper_execution_fails_over_when_mexc_book_is_missing(tmp_path) -> None:
    settings = make_settings(tmp_path / "paper-failover.db")
    market = MarketState(settings.symbols)
    state = market.symbol("BTC_USDT")
    for venue in ("bybit", "binance"):
        state.book(venue).apply_snapshot(
            bids=[[99.9, 100]], asks=[[100.1, 100]], ts=1_000
        )
    broker = PaperBroker(market, settings, account_configs(settings))
    signal = Signal(
        symbol="BTC_USDT",
        strategy="trend_orderflow",
        setup="TREND_PULLBACK",
        side=Side.LONG,
        score=90,
        stop_pct=0.01,
        target_r=2.8,
        ts=1_000,
    )

    opened = broker.handle_signal(signal, now=1_000)

    assert len(opened) == 1
    assert all(position.entry_price > 100 for position in opened)


def test_flow_reversal_needs_orderflow_account_age_and_confirmation(tmp_path) -> None:
    settings = make_settings(
        tmp_path / "paper-exit.db",
        min_flow_exit_minutes=15,
        flow_exit_confirm_seconds=90,
        stale_after_seconds=5_000,
    )
    market = MarketState(settings.symbols)
    _seed_books(market, 99.9, 100.1, 1_000)
    broker = PaperBroker(market, settings, account_configs(settings))
    signal = Signal(
        symbol="BTC_USDT",
        strategy="baseline",
        setup="TREND_PULLBACK",
        side=Side.LONG,
        score=95,
        stop_pct=0.02,
        target_r=2.8,
        ts=1_000,
    )
    assert len(broker.handle_signal(signal, now=1_000)) == 1
    adverse = FeatureSnapshot(
        symbol="BTC_USDT",
        ts=2_000,
        price=99.9,
        spread_bps=2,
        data_ready=True,
        flow_fast=-0.5,
        flow_slow=-0.3,
        book_imbalance=-0.2,
        cross_venue_consensus=-0.2,
    )

    assert broker.evaluate_positions({"BTC_USDT": adverse}, now=1_899) == []
    assert broker.evaluate_positions({"BTC_USDT": adverse}, now=1_900) == []
    assert broker.evaluate_positions({"BTC_USDT": adverse}, now=1_989) == []
    closed = broker.evaluate_positions({"BTC_USDT": adverse}, now=1_990)

    assert len(closed) == 1
    assert all(trade.reason == "ORDER_FLOW_REVERSAL" for trade in closed)


def test_trend_account_ignores_baseline_orderflow_reversal(tmp_path) -> None:
    settings = make_settings(
        tmp_path / "paper-control-exit.db",
        min_flow_exit_minutes=0,
        flow_exit_confirm_seconds=0,
        stale_after_seconds=5_000,
    )
    market = MarketState(settings.symbols)
    _seed_books(market, 99.9, 100.1, 1_000)
    broker = PaperBroker(market, settings, account_configs(settings))
    signal = Signal(
        symbol="BTC_USDT",
        strategy="trend_orderflow",
        setup="REGIME_TREND",
        side=Side.LONG,
        score=90,
        stop_pct=0.02,
        target_r=3.0,
        ts=1_000,
        exit_mode="trend",
        max_holding_minutes=1_440,
    )
    opened = broker.handle_signal(signal, now=1_000)
    assert len(opened) == 1
    adverse = FeatureSnapshot(
        symbol="BTC_USDT",
        ts=2_000,
        price=99.9,
        spread_bps=2,
        data_ready=True,
        flow_fast=-0.5,
        flow_slow=-0.3,
        book_imbalance=-0.2,
        cross_venue_consensus=-0.2,
    )

    assert broker.evaluate_positions({"BTC_USDT": adverse}, now=2_000) == []
    assert "BTC_USDT" in broker.accounts["TREND_ORDERFLOW"].positions
