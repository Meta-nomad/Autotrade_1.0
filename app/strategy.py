from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

from .config import Settings
from .market import SymbolState
from .models import FeatureSnapshot, Signal, Side


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low), 0.0, 1.0)


def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (period + 1.0)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


@dataclass(slots=True)
class Candidate:
    setup: str
    side: Side
    score: float
    target_r: float
    reasons: list[str]


class ControlStrategy:
    """The deliberately simple 20 x 15-minute Donchian control strategy."""

    def __init__(self) -> None:
        self._last_evaluated_bar: dict[str, int] = {}

    def evaluate(
        self,
        state: SymbolState,
        features: FeatureSnapshot,
        now: float | None = None,
    ) -> Signal | None:
        current_time = now or time.time()
        bars = state.fifteen_minute_bars(limit=30)
        current_bucket = int(current_time // 900) * 900
        completed = [bar for bar in bars if bar.ts < current_bucket]
        if len(completed) < 21:
            return None
        latest = completed[-1]
        if self._last_evaluated_bar.get(state.symbol) == latest.ts:
            return None
        self._last_evaluated_bar[state.symbol] = latest.ts

        previous = completed[-21:-1]
        high = max(bar.high for bar in previous)
        low = min(bar.low for bar in previous)
        average_volume = sum(bar.volume_notional for bar in previous) / len(previous)
        volume_ratio = latest.volume_notional / average_volume if average_volume > 0 else 0.0

        hour_values = [close for _, close in state.hour_closes]
        if len(hour_values) < 200:
            return None
        ema200 = _ema(hour_values[-300:], 200)

        side: Side | None = None
        if latest.close > high and latest.close > ema200 and volume_ratio >= 1.5:
            side = Side.LONG
        elif latest.close < low and latest.close < ema200 and volume_ratio >= 1.5:
            side = Side.SHORT
        if side is None:
            return None

        stop_pct = _clamp(max(features.atr_pct * 1.8, 0.006), 0.006, 0.025)
        score = _clamp(75.0 + min((volume_ratio - 1.5) * 8.0, 15.0), 75.0, 90.0)
        return Signal(
            symbol=state.symbol,
            strategy="control",
            setup="DONCHIAN_20",
            side=side,
            score=score,
            stop_pct=stop_pct,
            target_r=3.0,
            ts=current_time,
            reasons=[
                f"15m close broke the previous 20-bar range ({side.label})",
                f"15m volume ratio {volume_ratio:.2f}x",
                "price is on the matching side of hourly EMA200",
            ],
            feature_data=features.as_dict(),
        )


class OrderFlowStrategy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.last_diagnostics: dict[str, dict[str, object]] = {}

    def evaluate(
        self,
        state: SymbolState,
        features: FeatureSnapshot,
        now: float | None = None,
    ) -> Signal | None:
        current_time = now or time.time()
        mexc_history = state.price_history.get("mexc")
        history_span = (
            mexc_history[-1][0] - mexc_history[0][0]
            if mexc_history and len(mexc_history) > 1
            else 0.0
        )
        blockers: list[str] = []
        if features.price <= 0:
            blockers.append("no_price")
        fresh_book_count = 3 - len(features.stale_venues)
        if fresh_book_count < 2:
            blockers.append("insufficient_fresh_books")
        if features.spread_bps >= 35.0:
            blockers.append("spread_too_wide")
        if features.trade_count_60s < 20:
            blockers.append("too_few_trades")
        if history_span < 300.0:
            blockers.append("short_price_history")
        if len(state.hour_closes) < 50:
            blockers.append("short_hour_history")
        diagnostic: dict[str, object] = {
            "symbol": state.symbol,
            "ts": current_time,
            "data_ready": features.data_ready,
            "state": "not_ready",
            "candidate_count": 0,
            "best_setup": None,
            "best_score": None,
            "threshold": self.settings.signal_threshold,
            "stale_venues": list(features.stale_venues),
            "spread_bps": features.spread_bps,
            "trade_count_60s": features.trade_count_60s,
            "history_span_seconds": history_span,
            "hour_closes": len(state.hour_closes),
            "fresh_book_count": fresh_book_count,
            "blockers": blockers,
        }
        self.last_diagnostics[state.symbol] = diagnostic
        if not features.data_ready:
            return None
        candidates = [
            candidate
            for candidate in (
                self._trend_pullback(features),
                self._expansion_breakout(features),
                self._liquidation_reversal(features),
            )
            if candidate is not None
        ]
        diagnostic["candidate_count"] = len(candidates)
        if not candidates:
            diagnostic["state"] = "no_setup"
            return None
        candidate = max(candidates, key=lambda item: item.score)
        diagnostic["best_setup"] = candidate.setup
        diagnostic["best_score"] = candidate.score
        if candidate.score < self.settings.signal_threshold:
            diagnostic["state"] = "below_threshold"
            return None
        diagnostic["state"] = "signal"

        stop_multiplier = {
            "TREND_PULLBACK": 1.9,
            "EXPANSION_BREAKOUT": 2.2,
            "LIQUIDATION_REVERSAL": 1.6,
        }[candidate.setup]
        stop_pct = _clamp(
            max(
                features.atr_pct * stop_multiplier,
                features.spread_bps * 5.0 / 10_000.0,
                0.006,
            ),
            0.006,
            0.03,
        )
        return Signal(
            symbol=state.symbol,
            strategy="baseline",
            setup=candidate.setup,
            side=candidate.side,
            score=candidate.score,
            stop_pct=stop_pct,
            target_r=candidate.target_r,
            ts=current_time,
            reasons=candidate.reasons,
            feature_data=features.as_dict(),
            exit_mode="baseline",
            max_holding_minutes=120,
        )

    @staticmethod
    def _directional_components(features: FeatureSnapshot, side: Side) -> dict[str, float]:
        direction = float(side)
        flow = direction * (0.65 * features.flow_fast + 0.35 * features.flow_slow)
        micro_normalised = math.tanh(features.microprice_bps / 2.0)
        book = direction * (0.8 * features.book_imbalance + 0.2 * micro_normalised)
        trend = direction * features.trend_score
        cross = direction * features.cross_venue_consensus
        return {"flow": flow, "book": book, "trend": trend, "cross": cross}

    def _common_score(
        self,
        features: FeatureSnapshot,
        side: Side,
        *,
        location_score: float,
        regime_override: float | None = None,
        derivative_score: float = 0.5,
    ) -> tuple[float, dict[str, float]]:
        values = self._directional_components(features, side)
        regime = values["trend"] if regime_override is None else regime_override
        score = 0.0
        score += 25.0 * _scale(regime, 0.05, 0.75)
        score += 20.0 * _clamp(location_score, 0.0, 1.0)
        score += 25.0 * _scale(values["flow"], 0.04, 0.32)
        score += 15.0 * _scale(values["book"], 0.01, 0.28)
        score += 10.0 * _clamp(derivative_score, 0.0, 1.0)
        score += 5.0 * _scale(values["cross"], -0.05, 0.55)

        # Avoid paying MEXC API costs for weak moves and penalise crowded funding.
        if features.spread_bps > 8.0:
            score -= min((features.spread_bps - 8.0) * 0.8, 12.0)
        if float(side) * features.funding_rate > 0.0005:
            score -= 5.0
        return _clamp(score, 0.0, 100.0), values

    def _trend_pullback(self, features: FeatureSnapshot) -> Candidate | None:
        if abs(features.trend_score) < 0.18:
            return None
        side = Side.LONG if features.trend_score > 0 else Side.SHORT
        values = self._directional_components(features, side)
        if values["flow"] < 0.07 or values["book"] < 0.01:
            return None
        directional_return = float(side) * features.ret_300s
        if directional_return > 0.006 or directional_return < -0.015:
            return None

        # Best pullbacks are neither at the old extreme nor in the wrong half entirely.
        position = features.price_position if side == Side.LONG else 1.0 - features.price_position
        location = 1.0 - min(abs(position - 0.55) / 0.45, 1.0)
        derivative = 0.45 + 0.35 * _scale(float(side) * features.oi_change_300s, -0.005, 0.015)
        score, values = self._common_score(
            features,
            side,
            location_score=location,
            derivative_score=derivative,
        )
        return Candidate(
            setup="TREND_PULLBACK",
            side=side,
            score=score,
            target_r=2.8,
            reasons=[
                f"hourly trend alignment {values['trend']:+.2f}",
                f"executed-flow alignment {values['flow']:+.2f}",
                f"multi-book alignment {values['book']:+.2f}",
                f"pullback location quality {location:.2f}",
            ],
        )

    def _expansion_breakout(self, features: FeatureSnapshot) -> Candidate | None:
        if features.price_position >= 0.84:
            side = Side.LONG
            location = _scale(features.price_position, 0.82, 1.0)
        elif features.price_position <= 0.16:
            side = Side.SHORT
            location = _scale(1.0 - features.price_position, 0.82, 1.0)
        else:
            return None
        values = self._directional_components(features, side)
        if features.vol_ratio < 1.12 or values["flow"] < 0.10 or values["cross"] < -0.05:
            return None
        regime = 0.55 * values["trend"] + 0.45 * _scale(features.vol_ratio, 1.0, 2.0)
        oi_alignment = float(side) * features.oi_change_300s
        derivative = 0.45 + 0.4 * _scale(oi_alignment, -0.003, 0.02)
        score, values = self._common_score(
            features,
            side,
            location_score=location,
            regime_override=regime,
            derivative_score=derivative,
        )
        return Candidate(
            setup="EXPANSION_BREAKOUT",
            side=side,
            score=score,
            target_r=2.5,
            reasons=[
                f"30m range location {features.price_position:.2f}",
                f"volatility expansion ratio {features.vol_ratio:.2f}",
                f"executed-flow alignment {values['flow']:+.2f}",
                f"cross-venue consensus {values['cross']:+.2f}",
            ],
        )

    def _liquidation_reversal(self, features: FeatureSnapshot) -> Candidate | None:
        if features.price_position <= 0.10:
            side = Side.LONG
            location = _scale(0.12 - features.price_position, 0.02, 0.12)
        elif features.price_position >= 0.90:
            side = Side.SHORT
            location = _scale(features.price_position - 0.88, 0.02, 0.12)
        else:
            return None
        values = self._directional_components(features, side)
        adverse_liquidation = -float(side) * features.liquidation_imbalance
        if adverse_liquidation < 0.20 or values["flow"] < 0.06 or values["book"] < 0.0:
            return None
        # Reversal ignores the old trend direction but needs a strong sweep and flow flip.
        regime = 0.35 + 0.65 * _scale(adverse_liquidation, 0.2, 0.9)
        derivative = _scale(adverse_liquidation, 0.15, 0.8)
        score, values = self._common_score(
            features,
            side,
            location_score=location,
            regime_override=regime,
            derivative_score=derivative,
        )
        return Candidate(
            setup="LIQUIDATION_REVERSAL",
            side=side,
            score=score,
            target_r=2.1,
            reasons=[
                f"liquidation sweep strength {adverse_liquidation:.2f}",
                f"extreme 30m location {features.price_position:.2f}",
                f"post-sweep flow flip {values['flow']:+.2f}",
                f"order-book flip {values['book']:+.2f}",
            ],
        )


@dataclass(slots=True)
class MarketRegime:
    name: str
    direction: int
    breadth: float
    stress: float


class RegimeDetector:
    @staticmethod
    def detect(features: dict[str, FeatureSnapshot]) -> MarketRegime:
        ready = [item for item in features.values() if item.data_ready]
        if not ready:
            return MarketRegime("WARMUP", 0, 0.5, 0.0)
        breadth = sum(item.trend_score > 0.15 for item in ready) / len(ready)
        negative_breadth = sum(item.trend_score < -0.15 for item in ready) / len(ready)
        stress = sum(max(0.0, item.vol_ratio - 1.0) for item in ready) / len(ready)
        btc = features.get("BTC_USDT")
        btc_trend = btc.trend_score if btc and btc.data_ready else 0.0
        if stress >= 0.65 or (btc and abs(btc.ret_300s) >= 0.012):
            return MarketRegime("STRESS", 1 if btc_trend > 0 else -1, breadth, stress)
        if btc_trend >= 0.20 and breadth >= 0.55:
            return MarketRegime("TREND_UP", 1, breadth, stress)
        if btc_trend <= -0.20 and negative_breadth >= 0.55:
            return MarketRegime("TREND_DOWN", -1, breadth, stress)
        return MarketRegime("RANGE", 0, breadth, stress)


class TrendOrderFlowStrategy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(
        self,
        state: SymbolState,
        features: FeatureSnapshot,
        regime: MarketRegime,
        now: float,
    ) -> Signal | None:
        if not features.data_ready or abs(features.trend_score) < 0.24:
            return None
        side = Side.LONG if features.trend_score > 0 else Side.SHORT
        if regime.direction and int(side) != regime.direction:
            return None
        position = features.price_position if side == Side.LONG else 1.0 - features.price_position
        if position < 0.68:
            return None
        direction = float(side)
        flow = direction * (0.55 * features.flow_fast + 0.45 * features.flow_slow)
        book = direction * features.book_imbalance
        cross = direction * features.cross_venue_consensus
        pullback = direction * features.ret_300s
        if flow < 0.035 or book < -0.025 or cross < -0.08:
            return None
        if pullback < -0.006 or pullback > 0.015:
            return None
        score = 0.0
        score += 30.0 * _scale(direction * features.trend_score, 0.20, 0.80)
        score += 20.0 * _scale(position, 0.65, 1.0)
        score += 22.0 * _scale(flow, 0.02, 0.28)
        score += 10.0 * _scale(book, -0.03, 0.22)
        score += 10.0 * _scale(cross, -0.08, 0.55)
        score += 8.0 if regime.direction == int(side) else 3.0
        score = _clamp(score, 0.0, 100.0)
        if score < self.settings.signal_threshold:
            return None
        stop_pct = _clamp(max(features.atr_pct * 2.2, 0.007), 0.007, 0.035)
        return Signal(
            symbol=state.symbol,
            strategy="trend_orderflow",
            setup="REGIME_TREND",
            side=side,
            score=score,
            stop_pct=stop_pct,
            target_r=5.0,
            ts=now,
            reasons=[
                f"market regime {regime.name}",
                f"hourly trend {direction * features.trend_score:+.2f}",
                f"multi-venue flow {flow:+.2f}",
                f"30m range position {position:.2f}",
            ],
            feature_data=features.as_dict(),
            exit_mode="trend",
            max_holding_minutes=1_440,
        )


class CrossSectionalMomentumStrategy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._last_bucket = -1

    def evaluate_all(
        self,
        states: dict[str, SymbolState],
        features: dict[str, FeatureSnapshot],
        regime: MarketRegime,
        now: float,
    ) -> list[Signal]:
        bucket = int(now // 900)
        if bucket == self._last_bucket:
            return []
        ranked: list[tuple[float, str, FeatureSnapshot]] = []
        for symbol, item in features.items():
            if not item.data_ready:
                continue
            flow = 0.4 * item.flow_slow + 0.6 * item.cross_venue_consensus
            momentum = (
                0.45 * item.trend_score
                + 0.25 * math.tanh(item.ret_1800s * 120.0)
                + 0.20 * (2.0 * item.price_position - 1.0)
                + 0.10 * flow
            )
            ranked.append((momentum, symbol, item))
        if len(ranked) < 6:
            return []
        self._last_bucket = bucket
        ranked.sort()
        selected = ranked[:2] + ranked[-2:]
        signals: list[Signal] = []
        for momentum, symbol, item in selected:
            if abs(momentum) < 0.22:
                continue
            side = Side.LONG if momentum > 0 else Side.SHORT
            if regime.direction and int(side) != regime.direction and regime.name != "RANGE":
                continue
            score = _clamp(74.0 + abs(momentum) * 28.0, 0.0, 96.0)
            stop_pct = _clamp(max(item.atr_pct * 2.5, 0.008), 0.008, 0.04)
            signals.append(
                Signal(
                    symbol=symbol,
                    strategy="cross_momentum",
                    setup="CROSS_MOMENTUM",
                    side=side,
                    score=score,
                    stop_pct=stop_pct,
                    target_r=4.0,
                    ts=now,
                    reasons=[
                        f"cross-sectional rank extreme {momentum:+.2f}",
                        f"hourly trend {item.trend_score:+.2f}",
                        f"30m return {item.ret_1800s:+.2%}",
                        f"market regime {regime.name}",
                    ],
                    feature_data=item.as_dict(),
                    exit_mode="trend",
                    max_holding_minutes=720,
                )
            )
        return signals


class LiquidationReversalStrategy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._scorer = OrderFlowStrategy(settings)

    def evaluate(
        self,
        state: SymbolState,
        features: FeatureSnapshot,
        regime: MarketRegime,
        now: float,
    ) -> Signal | None:
        if not features.data_ready or regime.name not in {"STRESS", "RANGE"}:
            return None
        candidate = self._scorer._liquidation_reversal(features)
        if candidate is None or abs(features.liquidation_imbalance) < 0.35:
            return None
        score = _clamp(candidate.score + 5.0, 0.0, 100.0)
        if score < max(self.settings.signal_threshold, 80.0):
            return None
        stop_pct = _clamp(max(features.atr_pct * 1.6, 0.006), 0.006, 0.025)
        return Signal(
            symbol=state.symbol,
            strategy="liquidation_reversal",
            setup="LIQUIDATION_EXHAUSTION",
            side=candidate.side,
            score=score,
            stop_pct=stop_pct,
            target_r=1.5,
            ts=now,
            reasons=[f"market regime {regime.name}", *candidate.reasons],
            feature_data=features.as_dict(),
            exit_mode="fixed",
            max_holding_minutes=180,
        )


class StrategyRouter:
    def __init__(self, settings: Settings) -> None:
        self.baseline = OrderFlowStrategy(settings)
        self.trend = TrendOrderFlowStrategy(settings)
        self.cross = CrossSectionalMomentumStrategy(settings)
        self.reversal = LiquidationReversalStrategy(settings)
        self.last_regime = MarketRegime("WARMUP", 0, 0.5, 0.0)

    def evaluate_all(
        self,
        states: dict[str, SymbolState],
        features: dict[str, FeatureSnapshot],
        now: float | None = None,
    ) -> list[Signal]:
        current_time = now or time.time()
        regime = RegimeDetector.detect(features)
        self.last_regime = regime
        signals: list[Signal] = []
        ensemble_candidates: dict[str, Signal] = {}
        for symbol, state in states.items():
            item = features[symbol]
            baseline = self.baseline.evaluate(state, item, current_time)
            trend = self.trend.evaluate(state, item, regime, current_time)
            reversal = self.reversal.evaluate(state, item, regime, current_time)
            for signal in (baseline, trend, reversal):
                if signal is not None:
                    signals.append(signal)
            for signal in (trend, reversal):
                if signal is not None and signal.score >= 82.0:
                    prior = ensemble_candidates.get(symbol)
                    if prior is None or signal.score > prior.score:
                        ensemble_candidates[symbol] = signal
        cross_signals = self.cross.evaluate_all(states, features, regime, current_time)
        signals.extend(cross_signals)
        for signal in cross_signals:
            if signal.score >= 82.0:
                prior = ensemble_candidates.get(signal.symbol)
                if prior is None or signal.score > prior.score:
                    ensemble_candidates[signal.symbol] = signal
        signals.extend(
            replace(
                signal,
                strategy="ensemble",
                setup=f"ENSEMBLE_{signal.setup}",
                reasons=[f"ensemble selected {signal.strategy}", *signal.reasons],
            )
            for signal in ensemble_candidates.values()
        )
        return signals

    def diagnostics(self) -> dict[str, dict[str, object]]:
        diagnostics = dict(self.baseline.last_diagnostics)
        for item in diagnostics.values():
            item["regime"] = self.last_regime.name
            item["regime_direction"] = self.last_regime.direction
            item["market_breadth"] = self.last_regime.breadth
            item["market_stress"] = self.last_regime.stress
        return diagnostics
