from app.market import SymbolState
from app.models import FeatureSnapshot, MinuteBar, Side
from app.strategy import ControlStrategy, OrderFlowStrategy

from conftest import make_settings


def test_order_flow_expansion_signal_scores_high(tmp_path) -> None:
    settings = make_settings(tmp_path / "test.db")
    state = SymbolState("BTC_USDT")
    feature = FeatureSnapshot(
        symbol="BTC_USDT",
        ts=1_000,
        price=100,
        spread_bps=2,
        data_ready=True,
        trend_score=0.9,
        vol_ratio=1.8,
        price_position=0.98,
        flow_fast=0.7,
        flow_slow=0.6,
        book_imbalance=0.6,
        microprice_bps=3,
        cross_venue_consensus=0.6,
        oi_change_300s=0.02,
        atr_pct=0.006,
    )

    strategy = OrderFlowStrategy(settings)
    signal = strategy.evaluate(state, feature, now=1_000)

    assert signal is not None
    assert signal.setup == "EXPANSION_BREAKOUT"
    assert signal.side == Side.LONG
    assert signal.score >= 90
    assert strategy.last_diagnostics["BTC_USDT"]["state"] == "signal"
    assert strategy.last_diagnostics["BTC_USDT"]["best_score"] == signal.score


def test_order_flow_diagnostics_explain_not_ready(tmp_path) -> None:
    settings = make_settings(tmp_path / "not-ready.db")
    state = SymbolState("BTC_USDT")
    feature = FeatureSnapshot(
        symbol="BTC_USDT",
        ts=1_000,
        price=100,
        spread_bps=2,
        data_ready=False,
        trade_count_60s=18,
        stale_venues=("mexc",),
    )
    strategy = OrderFlowStrategy(settings)

    assert strategy.evaluate(state, feature, now=1_000) is None
    diagnostic = strategy.last_diagnostics["BTC_USDT"]
    assert diagnostic["state"] == "not_ready"
    assert diagnostic["stale_venues"] == ["mexc"]
    assert diagnostic["trade_count_60s"] == 18
    assert "mexc_book_stale" in diagnostic["blockers"]
    assert "too_few_trades" in diagnostic["blockers"]


def test_control_is_exactly_previous_twenty_completed_bars() -> None:
    state = SymbolState("BTC_USDT")
    base_ts = 1_000_000
    base_ts = (base_ts // 900) * 900
    for index in range(21):
        price = 100 + index * 0.02
        if index == 20:
            price = 103
        state.minute_bars.append(
            MinuteBar(
                ts=base_ts + index * 900,
                open=price - 0.05,
                high=price + 0.05,
                low=price - 0.05,
                close=price,
                volume_notional=2_000 if index == 20 else 1_000,
            )
        )
    state.bootstrap_hours((base_ts - (200 - i) * 3600, 80 + i * 0.1) for i in range(200))
    feature = FeatureSnapshot(
        symbol="BTC_USDT", ts=0, price=103, spread_bps=2, data_ready=True, atr_pct=0.006
    )
    now = base_ts + 21 * 900 + 10

    signal = ControlStrategy().evaluate(state, feature, now=now)

    assert signal is not None
    assert signal.setup == "DONCHIAN_20"
    assert signal.side == Side.LONG
