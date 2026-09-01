from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .config import Settings
from .market import SymbolState
from .models import FeatureSnapshot, Signal, Side
from .structure import StructureContext, analyse_structure


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
        if len(state.hour_closes) < 200:
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


class CompositeFlowStrategy:
    """One portfolio strategy; modules provide evidence, never duplicate trades."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._last_bar: dict[str, int] = {}
        self._last_reversal_evaluation: dict[str, float] = {}
        self._reversal_scorer = OrderFlowStrategy(settings)
        self.last_diagnostics: dict[str, dict[str, object]] = {}

    @staticmethod
    def _rank_scores(features: dict[str, FeatureSnapshot]) -> dict[str, float]:
        values: list[tuple[float, str]] = []
        for symbol, item in features.items():
            if not item.data_ready:
                continue
            flow = 0.45 * item.flow_slow + 0.55 * item.cross_venue_consensus
            momentum = (
                0.55 * item.trend_score
                + 0.25 * math.tanh(item.ret_1800s * 100.0)
                + 0.20 * flow
            )
            values.append((momentum, symbol))
        values.sort()
        if len(values) < 2:
            return {symbol: 0.5 for _, symbol in values}
        return {
            symbol: index / (len(values) - 1)
            for index, (_, symbol) in enumerate(values)
        }

    def _risk_pct(self, score: float, regime: MarketRegime) -> float:
        if score >= 92.0:
            risk = self.settings.composite_high_risk_pct
        elif score >= 85.0:
            risk = self.settings.composite_base_risk_pct
        else:
            risk = self.settings.composite_low_risk_pct
        if regime.name == "STRESS":
            risk *= 0.5
        return risk

    @staticmethod
    def _flow_components(item: FeatureSnapshot, side: Side) -> tuple[float, float, float]:
        direction = float(side)
        flow = direction * (0.60 * item.flow_fast + 0.40 * item.flow_slow)
        book = direction * item.book_imbalance
        cross = direction * item.cross_venue_consensus
        return flow, book, cross

    def _trend_candidate(
        self,
        state: SymbolState,
        item: FeatureSnapshot,
        regime: MarketRegime,
        rank: float,
        bars: list[object],
        now: float,
    ) -> Signal | None:
        if regime.name == "STRESS":
            return None
        side = Side.LONG if item.trend_score > 0 else Side.SHORT
        direction = float(side)
        directional_trend = direction * item.trend_score
        rank_alignment = rank if side == Side.LONG else 1.0 - rank
        if (
            directional_trend < 0.20
            or rank_alignment < 0.55
            or (regime.direction and regime.direction != int(side))
        ):
            return None
        structure: StructureContext = analyse_structure(bars, side)
        if len(bars) < 40 or structure.atr_pct <= 0:
            return None
        flow, book, cross = self._flow_components(item, side)
        if flow < -0.04 or book < -0.12 or cross < -0.15:
            return None
        current = bars[-1]
        prior = bars[-21:-1]
        breakout = (
            current.close > max(bar.high for bar in prior)
            if side == Side.LONG
            else current.close < min(bar.low for bar in prior)
        )
        average_volume = sum(bar.volume_notional for bar in prior) / len(prior)
        volume_ratio = current.volume_notional / average_volume if average_volume > 0 else 0.0
        if not breakout or volume_ratio < 0.90:
            return None
        score = (
            45.0
            + 20.0 * _scale(directional_trend, 0.20, 0.75)
            + 15.0 * rank_alignment
            + 10.0 * _scale(flow, -0.04, 0.25)
            + 5.0 * _scale(book, -0.12, 0.24)
            + 5.0 * _scale(volume_ratio, 0.90, 2.50)
        )
        score = _clamp(score, 0.0, 100.0)
        if score < self.settings.signal_threshold:
            return None
        stop_pct = _clamp(max(structure.atr_pct * 2.2, 0.012), 0.012, 0.060)
        return Signal(
            symbol=state.symbol,
            strategy="composite",
            setup="SYSTEMATIC_BREAKOUT_4H",
            side=side,
            score=score,
            stop_pct=stop_pct,
            target_r=5.0,
            ts=now,
            risk_pct=self._risk_pct(score, regime),
            reasons=[
                f"stable market regime {regime.name}",
                "confirmed 4h 20-bar breakout",
                f"structure diagnostic ready={structure.ready} impulse={structure.impulse_atr:.1f}ATR",
                f"fibonacci diagnostic={structure.retracement:.3f} (not an entry trigger)",
                f"trendline diagnostic={structure.trendline_distance_atr:.2f}ATR (not a filter)",
                f"order-flow/book/cross={flow:+.2f}/{book:+.2f}/{cross:+.2f}",
                f"cross-sectional rank={rank:.2f}",
            ],
            feature_data={
                **item.as_dict(),
                "regime": regime.name,
                "rank": rank,
                "structure": {
                    "retracement": structure.retracement,
                    "fib_quality": structure.fib_quality,
                    "impulse_atr": structure.impulse_atr,
                    "trendline_distance_atr": structure.trendline_distance_atr,
                    "volume_ratio": volume_ratio,
                },
            },
            exit_mode="trend",
            max_holding_minutes=10_080,
        )

    def _reversal_candidate(
        self,
        state: SymbolState,
        item: FeatureSnapshot,
        regime: MarketRegime,
        now: float,
    ) -> Signal | None:
        if regime.name not in {"RANGE", "STRESS"}:
            return None
        previous = self._last_reversal_evaluation.get(state.symbol, 0.0)
        if now - previous < 60.0:
            return None
        self._last_reversal_evaluation[state.symbol] = now
        candidate = self._reversal_scorer._liquidation_reversal(item)
        if candidate is None or abs(item.liquidation_imbalance) < 0.45:
            return None
        score = _clamp(candidate.score + (8.0 if regime.name == "STRESS" else 4.0), 0.0, 100.0)
        if score < max(82.0, self.settings.signal_threshold):
            return None
        stop_pct = _clamp(max(item.atr_pct * 1.8, 0.006), 0.006, 0.025)
        return Signal(
            symbol=state.symbol,
            strategy="composite",
            setup="LIQUIDATION_EXHAUSTION",
            side=candidate.side,
            score=score,
            stop_pct=stop_pct,
            target_r=1.8,
            ts=now,
            risk_pct=self._risk_pct(score, regime),
            reasons=[f"stable market regime {regime.name}", *candidate.reasons],
            feature_data={**item.as_dict(), "regime": regime.name},
            exit_mode="fixed",
            max_holding_minutes=180,
        )

    def evaluate_all(
        self,
        states: dict[str, SymbolState],
        features: dict[str, FeatureSnapshot],
        regime: MarketRegime,
        now: float,
    ) -> list[Signal]:
        ranks = self._rank_scores(features)
        signals: list[Signal] = []
        for symbol, state in states.items():
            item = features[symbol]
            diagnostic: dict[str, object] = {
                "symbol": symbol,
                "ts": now,
                "data_ready": item.data_ready,
                "state": "no_setup",
                "best_setup": None,
                "best_score": None,
                "threshold": self.settings.signal_threshold,
                "spread_bps": item.spread_bps,
                "trade_count_60s": item.trade_count_60s,
                "rank": ranks.get(symbol, 0.5),
                "blockers": [],
            }
            self.last_diagnostics[symbol] = diagnostic
            if not item.data_ready:
                diagnostic["state"] = "not_ready"
                diagnostic["blockers"] = ["data_not_ready"]
                continue
            if item.spread_bps > self.settings.max_entry_spread_bps:
                diagnostic["state"] = "blocked"
                diagnostic["blockers"] = ["spread_too_wide"]
                continue

            bars = state.four_hour_bars(limit=120)
            current_bucket = int(now // 14_400) * 14_400
            completed = [bar for bar in bars if bar.ts < current_bucket]
            trend_signal: Signal | None = None
            if completed:
                latest_ts = completed[-1].ts
                if self._last_bar.get(symbol) != latest_ts:
                    self._last_bar[symbol] = latest_ts
                    trend_signal = self._trend_candidate(
                        state, item, regime, ranks.get(symbol, 0.5), completed, now
                    )
            reversal_signal = self._reversal_candidate(state, item, regime, now)
            candidate = max(
                (value for value in (trend_signal, reversal_signal) if value is not None),
                key=lambda value: value.score,
                default=None,
            )
            if candidate is None:
                diagnostic["blockers"] = [
                    "no_qualified_setup" if completed else "no_completed_4h"
                ]
                continue
            diagnostic["state"] = "signal"
            diagnostic["best_setup"] = candidate.setup
            diagnostic["best_score"] = candidate.score
            signals.append(candidate)
        # Portfolio arbitration: at most the two highest-confidence ideas are
        # sent to the broker. Correlation/risk checks can reduce this further.
        signals.sort(key=lambda value: value.score, reverse=True)
        return signals[: self.settings.max_open_positions]


class StrategyRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.composite = CompositeFlowStrategy(settings)
        self.last_regime = MarketRegime("WARMUP", 0, 0.5, 0.0)
        self.raw_regime = self.last_regime
        self._started_at: float | None = None
        self._candidate_key: tuple[str, int] | None = None
        self._candidate_since = 0.0

    def evaluate_all(
        self,
        states: dict[str, SymbolState],
        features: dict[str, FeatureSnapshot],
        now: float | None = None,
    ) -> list[Signal]:
        current_time = now or time.time()
        if self._started_at is None:
            self._started_at = current_time
        raw = RegimeDetector.detect(features)
        self.raw_regime = raw
        key = (raw.name, raw.direction)
        if key != self._candidate_key:
            self._candidate_key = key
            self._candidate_since = current_time
        ready_count = sum(item.data_ready for item in features.values())
        ready_ratio = ready_count / len(features) if features else 0.0
        warmup_elapsed = current_time - self._started_at
        regime_elapsed = current_time - self._candidate_since
        stable = bool(
            raw.name != "WARMUP"
            and ready_ratio >= self.settings.min_ready_ratio
            and warmup_elapsed >= self.settings.startup_warmup_seconds
            and regime_elapsed >= self.settings.regime_confirm_seconds
        )
        self.last_regime = raw if stable else MarketRegime("WARMUP", 0, raw.breadth, raw.stress)
        if not stable:
            for symbol, item in features.items():
                self.composite.last_diagnostics[symbol] = {
                    "symbol": symbol,
                    "ts": current_time,
                    "data_ready": item.data_ready,
                    "state": "warmup",
                    "best_setup": None,
                    "best_score": None,
                    "threshold": self.settings.signal_threshold,
                    "spread_bps": item.spread_bps,
                    "trade_count_60s": item.trade_count_60s,
                    "blockers": [
                        f"startup={warmup_elapsed:.0f}/{self.settings.startup_warmup_seconds}s",
                        f"regime={regime_elapsed:.0f}/{self.settings.regime_confirm_seconds}s",
                        f"ready={ready_ratio:.0%}",
                    ],
                }
            return []
        return self.composite.evaluate_all(states, features, raw, current_time)

    def diagnostics(self) -> dict[str, dict[str, object]]:
        diagnostics = dict(self.composite.last_diagnostics)
        for item in diagnostics.values():
            item["regime"] = self.last_regime.name
            item["raw_regime"] = self.raw_regime.name
            item["regime_direction"] = self.last_regime.direction
            item["market_breadth"] = self.last_regime.breadth
            item["market_stress"] = self.last_regime.stress
        return diagnostics
