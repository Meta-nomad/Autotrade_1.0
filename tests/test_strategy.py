import math

from app.market import SymbolState
from app.models import FeatureSnapshot, MinuteBar, Side
from app.strategy import (
    ControlStrategy,
    CrossSectionalMomentumStrategy,
    MarketRegime,
    RegimeDetector,
    OrderFlowStrategy,
    LiquidationReversalStrategy,
    StrategyRouter,
    TrendOrderFlowStrategy,
)

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
        stale_venues=("mexc", "bybit", "binance"),
    )
    strategy = OrderFlowStrategy(settings)

    assert strategy.evaluate(state, feature, now=1_000) is None
    diagnostic = strategy.last_diagnostics["BTC_USDT"]
    assert diagnostic["state"] == "not_ready"
    assert diagnostic["stale_venues"] == ["mexc", "bybit", "binance"]
    assert diagnostic["trade_count_60s"] == 18
    assert "insufficient_fresh_books" in diagnostic["blockers"]
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


def test_regime_detector_uses_breadth_and_btc_direction() -> None:
    features = {
        symbol: FeatureSnapshot(
            symbol=symbol,
            ts=1_000,
            price=100,
            spread_bps=2,
            data_ready=True,
            trend_score=0.6,
            vol_ratio=1.1,
        )
        for symbol in ("BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT")
    }

    regime = RegimeDetector.detect(features)

    assert regime.name == "TREND_UP"
    assert regime.direction == 1
    assert regime.breadth == 1.0


def test_trend_orderflow_requires_regime_and_flow_confirmation(tmp_path) -> None:
    settings = make_settings(tmp_path / "trend.db")
    feature = FeatureSnapshot(
        symbol="BTC_USDT",
        ts=1_000,
        price=100,
        spread_bps=2,
        data_ready=True,
        trend_score=0.8,
        price_position=0.9,
        flow_fast=0.5,
        flow_slow=0.4,
        book_imbalance=0.2,
        cross_venue_consensus=0.4,
        ret_300s=0.002,
        atr_pct=0.006,
    )
    strategy = TrendOrderFlowStrategy(settings)

    signal = strategy.evaluate(
        SymbolState("BTC_USDT"), feature, MarketRegime("TREND_UP", 1, 0.8, 0.1), 1_000
    )

    assert signal is not None
    assert signal.strategy == "trend_orderflow"
    assert signal.exit_mode == "trend"
    assert signal.max_holding_minutes == 1_440


def test_cross_sectional_momentum_selects_extremes_once_per_bar(tmp_path) -> None:
    settings = make_settings(tmp_path / "cross.db")
    symbols = tuple(f"C{i}_USDT" for i in range(8))
    states = {symbol: SymbolState(symbol) for symbol in symbols}
    features = {}
    for index, symbol in enumerate(symbols):
        strength = (index - 3.5) / 4.0
        features[symbol] = FeatureSnapshot(
            symbol=symbol,
            ts=1_000,
            price=100,
            spread_bps=2,
            data_ready=True,
            trend_score=strength,
            ret_1800s=strength * 0.01,
            price_position=(strength + 1.0) / 2.0,
            flow_slow=strength * 0.2,
            cross_venue_consensus=strength * 0.4,
            atr_pct=0.006,
        )
    strategy = CrossSectionalMomentumStrategy(settings)
    regime = MarketRegime("RANGE", 0, 0.5, 0.1)

    signals = strategy.evaluate_all(states, features, regime, 1_000)

    assert len(signals) == 4
    assert {signal.side for signal in signals} == {Side.LONG, Side.SHORT}
    assert strategy.evaluate_all(states, features, regime, 1_001) == []


def test_liquidation_reversal_is_separate_sparse_module(tmp_path) -> None:
    settings = make_settings(tmp_path / "reversal.db")
    feature = FeatureSnapshot(
        symbol="BTC_USDT",
        ts=1_000,
        price=100,
        spread_bps=2,
        data_ready=True,
        trend_score=-0.5,
        price_position=0.04,
        flow_fast=0.7,
        flow_slow=0.5,
        book_imbalance=0.4,
        microprice_bps=2,
        cross_venue_consensus=0.3,
        liquidation_imbalance=-0.8,
        atr_pct=0.006,
    )

    signal = LiquidationReversalStrategy(settings).evaluate(
        SymbolState("BTC_USDT"), feature, MarketRegime("STRESS", -1, 0.2, 1.0), 1_000
    )

    assert signal is not None
    assert signal.strategy == "liquidation_reversal"
    assert signal.target_r == 1.5
    assert signal.max_holding_minutes == 180


def test_router_emits_only_single_composite_portfolio_signal(tmp_path) -> None:
    symbols = tuple(f"C{i}_USDT" for i in range(6))
    settings = make_settings(
        tmp_path / "router.db",
        symbols=symbols,
        startup_warmup_seconds=0,
        regime_confirm_seconds=0,
        min_ready_ratio=0.8,
    )
    states = {symbol: SymbolState(symbol) for symbol in symbols}
    features = {}
    base = 14_400 * 1_000
    for symbol_index, symbol in enumerate(symbols):
        for index in range(45):
            price = 100 + index * 0.42 + math.sin(index * math.pi / 3) * 2.2
            if index == 44:
                price += 7.0
            states[symbol].hour_bars.append(
                MinuteBar(
                    ts=base + index * 14_400,
                    open=price - 0.2,
                    high=price + 0.4,
                    low=price - 0.4,
                    close=price,
                    volume_notional=2_000 if index == 44 else 1_000,
                )
            )
        strength = 0.45 + symbol_index * 0.08
        features[symbol] = FeatureSnapshot(
            symbol=symbol,
            ts=base,
            price=states[symbol].hour_bars[-1].close,
            spread_bps=2,
            data_ready=True,
            trend_score=strength,
            flow_fast=0.5,
            flow_slow=0.4,
            book_imbalance=0.3,
            cross_venue_consensus=0.4,
            ret_1800s=0.01,
            atr_pct=0.012,
        )
    now = base + 45 * 14_400 + 1
    signals = StrategyRouter(settings).evaluate_all(states, features, now=now)

    assert signals
    assert all(signal.strategy == "composite" for signal in signals)
    assert len(signals) <= settings.max_open_positions
