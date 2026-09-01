from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .models import Side


class BarLike(Protocol):
    open: float
    high: float
    low: float
    close: float
    volume_notional: float


@dataclass(frozen=True, slots=True)
class StructureContext:
    side: Side
    ready: bool
    atr: float
    atr_pct: float
    swing_start: float
    swing_end: float
    impulse_atr: float
    retracement: float
    fib_quality: float
    trendline: float
    trendline_distance_atr: float
    trendline_holds: bool
    trendline_slope_atr: float
    breakout: bool
    volume_ratio: float
    invalidation_price: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def true_range_average(bars: Sequence[BarLike], period: int = 14) -> float:
    if len(bars) < 3:
        return 0.0
    ranges: list[float] = []
    previous = bars[0].close
    for bar in bars[1:]:
        ranges.append(max(bar.high - bar.low, abs(bar.high - previous), abs(bar.low - previous)))
        previous = bar.close
    sample = ranges[-period:]
    return sum(sample) / len(sample) if sample else 0.0


def confirmed_pivots(
    bars: Sequence[BarLike], left: int = 2, right: int = 2
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return causal pivots; the final ``right`` bars can never be pivots."""
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    if len(bars) < left + right + 1:
        return highs, lows
    for index in range(left, len(bars) - right):
        window = bars[index - left : index + right + 1]
        high = bars[index].high
        low = bars[index].low
        if high == max(item.high for item in window) and high > max(
            item.high for offset, item in enumerate(window) if offset != left
        ):
            highs.append((index, high))
        if low == min(item.low for item in window) and low < min(
            item.low for offset, item in enumerate(window) if offset != left
        ):
            lows.append((index, low))
    return highs, lows


def _linear_projection(points: Sequence[tuple[int, float]], target: int) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    xs = [float(item[0]) for item in points]
    ys = [item[1] for item in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator <= 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    return intercept + slope * target, slope


def _empty(side: Side, close: float, atr: float) -> StructureContext:
    return StructureContext(
        side=side,
        ready=False,
        atr=atr,
        atr_pct=atr / close if close > 0 else 0.0,
        swing_start=0.0,
        swing_end=0.0,
        impulse_atr=0.0,
        retracement=0.0,
        fib_quality=0.0,
        trendline=0.0,
        trendline_distance_atr=99.0,
        trendline_holds=False,
        trendline_slope_atr=0.0,
        breakout=False,
        volume_ratio=0.0,
        invalidation_price=0.0,
    )


def analyse_structure(bars: Sequence[BarLike], side: Side) -> StructureContext:
    """Analyse trendline, Fibonacci location and breakout without look-ahead."""
    if len(bars) < 40:
        close = bars[-1].close if bars else 0.0
        return _empty(side, close, 0.0)
    current = bars[-1]
    atr = true_range_average(bars, 14)
    if atr <= 0 or current.close <= 0:
        return _empty(side, current.close, atr)
    highs, lows = confirmed_pivots(bars)
    if len(highs) < 2 or len(lows) < 2:
        return _empty(side, current.close, atr)

    if side == Side.LONG:
        end_index, swing_end = highs[-1]
        starts = [(index, value) for index, value in lows if index < end_index]
        if not starts:
            return _empty(side, current.close, atr)
        start_index, swing_start = starts[-1]
        line_points = lows[-3:]
        retracement = (swing_end - current.close) / max(swing_end - swing_start, atr * 0.1)
        prior_boundary = max(item.high for item in bars[-21:-1])
        breakout = current.close > prior_boundary
    else:
        end_index, swing_end = lows[-1]
        starts = [(index, value) for index, value in highs if index < end_index]
        if not starts:
            return _empty(side, current.close, atr)
        start_index, swing_start = starts[-1]
        line_points = highs[-3:]
        retracement = (current.close - swing_end) / max(swing_start - swing_end, atr * 0.1)
        prior_boundary = min(item.low for item in bars[-21:-1])
        breakout = current.close < prior_boundary

    impulse = abs(swing_end - swing_start)
    impulse_atr = impulse / atr
    projection, slope = _linear_projection(line_points, len(bars) - 1)
    directional_distance = float(side) * (current.close - projection) / atr
    trendline_holds = directional_distance >= -0.45
    trendline_distance = abs(current.close - projection) / atr
    slope_atr = float(side) * slope / atr

    # A broad 0.382-0.705 band is used. Quality peaks around the golden-ratio
    # region rather than treating an exact decimal as magic.
    fib_quality = 1.0 - abs(retracement - 0.55) / 0.22
    fib_quality = _clamp(fib_quality, 0.0, 1.0)
    if retracement < 0.32 or retracement > 0.79:
        fib_quality = 0.0

    average_volume = sum(item.volume_notional for item in bars[-21:-1]) / 20.0
    volume_ratio = current.volume_notional / average_volume if average_volume > 0 else 0.0
    if side == Side.LONG:
        fib_786 = swing_end - 0.786 * impulse
        invalidation = min(current.close - atr * 0.8, fib_786 - atr * 0.20)
    else:
        fib_786 = swing_end + 0.786 * impulse
        invalidation = max(current.close + atr * 0.8, fib_786 + atr * 0.20)

    return StructureContext(
        side=side,
        ready=end_index > start_index and impulse_atr >= 2.5,
        atr=atr,
        atr_pct=atr / current.close,
        swing_start=swing_start,
        swing_end=swing_end,
        impulse_atr=impulse_atr,
        retracement=retracement,
        fib_quality=fib_quality,
        trendline=projection,
        trendline_distance_atr=trendline_distance,
        trendline_holds=trendline_holds,
        trendline_slope_atr=slope_atr,
        breakout=breakout,
        volume_ratio=volume_ratio,
        invalidation_price=invalidation,
    )
