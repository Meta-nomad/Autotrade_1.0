#!/usr/bin/env python3
"""Causal four-hour portfolio backtest for COMPOSITE_FLOW.

Historical Binance USD-M klines contain taker-buy volume, allowing an executed
order-flow proxy. Historical order-book snapshots and liquidation events are
not inferred from candles, so the live liquidation-exhaustion module is not
included in the historical result.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BAR_HOURS = 4
BAR_MINUTES = BAR_HOURS * 60
CLUSTERS = {
    "BTCUSDT": "majors",
    "ETHUSDT": "majors",
    "SOLUSDT": "majors",
    "BNBUSDT": "majors",
    "XRPUSDT": "high_beta",
    "ADAUSDT": "high_beta",
    "DOGEUSDT": "high_beta",
    "LINKUSDT": "infrastructure",
    "AVAXUSDT": "infrastructure",
    "LTCUSDT": "payments",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def scale(value: float, low: float, high: float) -> float:
    if not math.isfinite(value) or high <= low:
        return 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0)


@dataclass
class Position:
    symbol: str
    side: int
    setup: str
    score: float
    opened_i: int
    entry: float
    qty: float
    notional: float
    margin: float
    risk_usdt: float
    initial_stop: float
    stop: float
    target: float
    entry_fee: float
    max_bars: int
    best: float
    worst: float


@dataclass
class Trade:
    symbol: str
    setup: str
    side: str
    score: float
    opened_at: str
    closed_at: str
    entry: float
    exit: float
    net_pnl: float
    r_multiple: float
    fees: float
    funding_cost: float
    reason: str


def load_symbol(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    numeric = [
        "open", "high", "low", "close", "volume", "quote_volume",
        "taker_buy_volume", "taker_buy_quote_volume",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")]
    # The live strategy uses completed four-hour structure and faster real-time
    # order flow as its trigger. Historical taker flow is aggregated per bar.
    return frame.resample(f"{BAR_HOURS}h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "taker_buy_volume": "sum",
        "taker_buy_quote_volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]].copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    previous = close.shift(1)
    tr = pd.concat([(high - low), (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    result["atr"] = tr.ewm(span=14, adjust=False).mean()
    result["atr_pct"] = result["atr"] / close
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    result["trend"] = np.tanh((ema20 / ema50 - 1.0) * 55.0)
    result["ret8"] = close.pct_change(8)
    returns = close.pct_change()
    short_vol = returns.rolling(4, min_periods=4).std()
    long_vol = returns.rolling(32, min_periods=16).std()
    result["vol_ratio"] = short_vol / long_vol

    quote = result["quote_volume"].replace(0, np.nan)
    imbalance = (2.0 * result["taker_buy_quote_volume"] / quote - 1.0).clip(-1.0, 1.0)
    result["flow_fast"] = imbalance.ewm(span=4, adjust=False).mean()
    result["flow_slow"] = imbalance.ewm(span=20, adjust=False).mean()
    result["volume_ratio"] = quote / quote.rolling(20, min_periods=10).mean().shift(1)
    rolling_high = high.rolling(32, min_periods=20).max().shift(1)
    rolling_low = low.rolling(32, min_periods=20).min().shift(1)
    result["range_position"] = ((close - rolling_low) / (rolling_high - rolling_low)).clip(0.0, 1.0)
    result["breakout_long"] = close > high.rolling(20, min_periods=20).max().shift(1)
    result["breakout_short"] = close < low.rolling(20, min_periods=20).min().shift(1)

    # A pivot at t is only available at t+2. Shifting the sparse pivot event by
    # two bars is the crucial no-look-ahead step.
    pivot_high_raw = high.where(high == high.rolling(5, center=True, min_periods=5).max())
    pivot_low_raw = low.where(low == low.rolling(5, center=True, min_periods=5).min())
    pivot_high = pivot_high_raw.shift(2)
    pivot_low = pivot_low_raw.shift(2)
    index_values = pd.Series(np.arange(len(result), dtype=float), index=result.index)
    pivot_high_i = index_values.where(pivot_high_raw.notna()).shift(2)
    pivot_low_i = index_values.where(pivot_low_raw.notna()).shift(2)

    last_high = pivot_high.ffill()
    last_high_i = pivot_high_i.ffill()
    previous_high_at_event = pivot_high.ffill().shift(1).where(pivot_high.notna()).ffill()
    previous_high_i_at_event = pivot_high_i.ffill().shift(1).where(pivot_high.notna()).ffill()
    last_low = pivot_low.ffill()
    last_low_i = pivot_low_i.ffill()
    previous_low_at_event = pivot_low.ffill().shift(1).where(pivot_low.notna()).ffill()
    previous_low_i_at_event = pivot_low_i.ffill().shift(1).where(pivot_low.notna()).ffill()

    low_before_high = last_low.shift(1).where(pivot_high.notna()).ffill()
    low_i_before_high = last_low_i.shift(1).where(pivot_high.notna()).ffill()
    high_before_low = last_high.shift(1).where(pivot_low.notna()).ffill()
    high_i_before_low = last_high_i.shift(1).where(pivot_low.notna()).ffill()
    long_end = pivot_high.ffill()
    long_end_i = pivot_high_i.ffill()
    short_end = pivot_low.ffill()
    short_end_i = pivot_low_i.ffill()

    long_impulse = long_end - low_before_high
    short_impulse = high_before_low - short_end
    result["long_retrace"] = (long_end - close) / long_impulse
    result["short_retrace"] = (close - short_end) / short_impulse
    result["long_impulse_atr"] = long_impulse / result["atr"]
    result["short_impulse_atr"] = short_impulse / result["atr"]
    result["long_structure_order"] = long_end_i > low_i_before_high
    result["short_structure_order"] = short_end_i > high_i_before_low

    long_slope = (last_low - previous_low_at_event) / (last_low_i - previous_low_i_at_event).replace(0, np.nan)
    short_slope = (last_high - previous_high_at_event) / (last_high_i - previous_high_i_at_event).replace(0, np.nan)
    long_line = last_low + long_slope * (index_values - last_low_i)
    short_line = last_high + short_slope * (index_values - last_high_i)
    result["long_line_distance"] = (close - long_line).abs() / result["atr"]
    result["short_line_distance"] = (close - short_line).abs() / result["atr"]
    result["long_line_holds"] = close >= long_line - 0.45 * result["atr"]
    result["short_line_holds"] = close <= short_line + 0.45 * result["atr"]
    result["long_slope_atr"] = long_slope / result["atr"]
    result["short_slope_atr"] = -short_slope / result["atr"]
    result["long_invalidation"] = np.minimum(close - 0.8 * result["atr"], long_end - 0.786 * long_impulse - 0.2 * result["atr"])
    result["short_invalidation"] = np.maximum(close + 0.8 * result["atr"], short_end + 0.786 * short_impulse + 0.2 * result["atr"])
    result["flow_prior"] = result["flow_slow"].shift(2)
    return result.replace([np.inf, -np.inf], np.nan)


def build_dataset(data_dir: Path) -> tuple[pd.DatetimeIndex, dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("*USDT-15m.csv.gz")):
        symbol = path.name.split("-", 1)[0]
        frames[symbol] = feature_frame(load_symbol(path))
    if len(frames) < 6:
        raise RuntimeError("at least six aligned symbols are required")
    common = None
    for frame in frames.values():
        common = frame.index if common is None else common.intersection(frame.index)
    assert common is not None
    common = common.sort_values()
    data: dict[str, dict[str, np.ndarray]] = {
        symbol: {column: frame.loc[common, column].to_numpy() for column in frame.columns}
        for symbol, frame in frames.items()
    }
    symbols = list(data)
    trend = np.column_stack([data[symbol]["trend"] for symbol in symbols])
    momentum = np.column_stack([
        0.55 * data[symbol]["trend"]
        + 0.25 * np.tanh(np.nan_to_num(data[symbol]["ret8"]) * 100.0)
        + 0.20 * (0.45 * np.nan_to_num(data[symbol]["flow_slow"]))
        for symbol in symbols
    ])
    order = np.argsort(np.argsort(momentum, axis=1), axis=1)
    ranks = order / max(1, len(symbols) - 1)
    btc_i = symbols.index("BTCUSDT")
    breadth = np.nanmean(trend > 0.15, axis=1)
    negative_breadth = np.nanmean(trend < -0.15, axis=1)
    vol_matrix = np.column_stack([data[s]["vol_ratio"] for s in symbols])
    valid_vol = np.isfinite(vol_matrix).sum(axis=1)
    stress = np.divide(
        np.nansum(vol_matrix, axis=1),
        valid_vol,
        out=np.ones(len(common), dtype=float),
        where=valid_vol > 0,
    )
    btc_ret = pd.Series(data["BTCUSDT"]["close"]).pct_change(4).to_numpy()
    raw_regime = np.zeros(len(common), dtype=np.int8)  # 0 range, 1 up, -1 down, 2 stress
    raw_regime[(stress >= 1.70) | (np.abs(btc_ret) >= 0.02)] = 2
    raw_regime[(raw_regime == 0) & (trend[:, btc_i] >= 0.20) & (breadth >= 0.55)] = 1
    raw_regime[(raw_regime == 0) & (trend[:, btc_i] <= -0.20) & (negative_breadth >= 0.55)] = -1
    stable_regime = raw_regime.copy()
    for offset in range(1, 4):
        stable_regime[offset:][raw_regime[offset:] != raw_regime[:-offset]] = 9
    stable_regime[:220] = 9
    meta = {
        "symbols": np.array(symbols, dtype=object),
        "ranks": ranks,
        "raw_regime": raw_regime,
        "regime": stable_regime,
    }
    return common, data, meta


def candidate_at(
    i: int,
    symbol: str,
    row: dict[str, np.ndarray],
    rank: float,
    regime: int,
    threshold: float,
    fib_low: float,
    fib_high: float,
    flow_min: float,
    mode: str,
) -> dict[str, float | str | int] | None:
    if regime == 9 or not math.isfinite(row["trend"][i]):
        return None
    # Stress is reserved for the live liquidation-exhaustion module because
    # candle archives cannot reconstruct historical liquidation events.
    if regime in (2, 9):
        return None
    side = 1 if row["trend"][i] > 0 else -1
    directional_trend = side * row["trend"][i]
    rank_alignment = rank if side == 1 else 1.0 - rank
    flow = side * (0.60 * row["flow_fast"][i] + 0.40 * row["flow_slow"][i])
    breakout = bool(row["breakout_long"][i] if side == 1 else row["breakout_short"][i])
    retracement = row["long_retrace"][i] if side == 1 else row["short_retrace"][i]
    impulse_atr = row["long_impulse_atr"][i] if side == 1 else row["short_impulse_atr"][i]
    structure_order = bool(row["long_structure_order"][i] if side == 1 else row["short_structure_order"][i])
    line_holds = bool(row["long_line_holds"][i] if side == 1 else row["short_line_holds"][i])
    slope_atr = row["long_slope_atr"][i] if side == 1 else row["short_slope_atr"][i]
    volume_ratio = row["volume_ratio"][i]
    if (
        directional_trend < 0.20
        or rank_alignment < 0.55
        or regime == -side
    ):
        return None
    if mode == "trend_only":
        if not breakout or volume_ratio < 0.90:
            return None
    elif mode in {"breakout", "breakout_structure", "breakout_trendline"}:
        if flow < flow_min or not breakout or volume_ratio < 0.90:
            return None
        if mode in {"breakout_structure", "breakout_trendline"} and (
            not structure_order or not math.isfinite(impulse_atr) or impulse_atr < 2.5
        ):
            return None
        if mode == "breakout_trendline" and (
            not line_holds or not math.isfinite(slope_atr) or slope_atr < -0.05
        ):
            return None
    elif mode == "fib_pullback":
        if (
            flow < flow_min
            or not structure_order
            or not math.isfinite(impulse_atr)
            or impulse_atr < 2.5
            or not math.isfinite(retracement)
            or retracement < fib_low
            or retracement > fib_high
            or not line_holds
            or not math.isfinite(slope_atr)
            or slope_atr < -0.05
            or volume_ratio < 0.80
        ):
            return None
    else:
        raise ValueError(f"unknown candidate mode: {mode}")
    score = (
        45.0
        + 20.0 * scale(directional_trend, 0.20, 0.75)
        + 15.0 * rank_alignment
        + 10.0 * scale(flow, flow_min, 0.25)
        + 10.0 * scale(volume_ratio, 0.90, 2.50)
    )
    if mode == "fib_pullback":
        score += 8.0 * scale(impulse_atr, 2.5, 8.0)
        score += 7.0 * (1.0 - clamp(abs(retracement - 0.55) / 0.22, 0.0, 1.0))
    if score < threshold:
        return None
    setup = {
        "trend_only": "TREND_BREAKOUT_NO_FLOW",
        "breakout": "SYSTEMATIC_BREAKOUT_4H",
        "breakout_structure": "BREAKOUT_STRUCTURE_FILTER",
        "breakout_trendline": "BREAKOUT_TRENDLINE_FILTER",
        "fib_pullback": "FIBONACCI_PULLBACK_DIRECT",
    }[mode]
    return {
        "symbol": symbol,
        "side": side,
        "setup": setup,
        "score": score,
        "stop_pct": clamp(max(row["atr_pct"][i] * 2.2, 0.012), 0.012, 0.060),
        "target_r": 5.0,
        "max_bars": 42,
    }


def simulate(
    index: pd.DatetimeIndex,
    data: dict[str, dict[str, np.ndarray]],
    meta: dict[str, np.ndarray],
    *,
    threshold: float = 86.0,
    fib_low: float = 0.382,
    fib_high: float = 0.705,
    flow_min: float = 0.02,
    fee_rate: float = 0.0008,
    slippage_bps: float = 1.0,
    mode: str = "breakout",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = [str(item) for item in meta["symbols"]]
    balance = 1000.0
    peak = balance
    positions: dict[str, Position] = {}
    cooldowns: dict[str, int] = {}
    trades: list[Trade] = []
    equity_rows: list[tuple[pd.Timestamp, float, float, int]] = []
    day_key = ""
    month_key = ""
    day_start = balance
    month_start = balance
    stop_losses_today = 0
    halted_day = False
    halted_month = False
    slip = slippage_bps / 10_000.0

    def marked_equity(i: int) -> float:
        value = balance
        for symbol, position in positions.items():
            value += position.side * position.qty * (data[symbol]["close"][i] - position.entry)
        return value

    for i in range(221, len(index)):
        timestamp = index[i]
        current_day = timestamp.strftime("%Y-%m-%d")
        current_month = timestamp.strftime("%Y-%m")
        if current_day != day_key:
            day_key = current_day
            day_start = marked_equity(i - 1)
            stop_losses_today = 0
            halted_day = False
        if current_month != month_key:
            month_key = current_month
            month_start = marked_equity(i - 1)
            halted_month = False

        # Existing positions are evaluated before new signals. If both stop
        # and target occur in one candle, stop wins (conservative ordering).
        for symbol, position in list(positions.items()):
            row = data[symbol]
            high = row["high"][i]
            low = row["low"][i]
            close = row["close"][i]
            position.best = max(position.best, high) if position.side == 1 else min(position.best, low)
            position.worst = min(position.worst, low) if position.side == 1 else max(position.worst, high)
            stop_hit = low <= position.stop if position.side == 1 else high >= position.stop
            target_hit = high >= position.target if position.side == 1 else low <= position.target
            reason = ""
            raw_exit = close
            if stop_hit:
                reason, raw_exit = "STOP", position.stop
            elif target_hit:
                reason, raw_exit = "TARGET", position.target
            age = i - position.opened_i
            current_r = position.side * position.qty * (close - position.entry) / position.risk_usdt
            if not reason and age >= position.max_bars and current_r < 0.75:
                reason, raw_exit = "TIME_STOP", close
            if reason:
                exit_price = raw_exit * (1.0 - position.side * slip)
                gross = position.side * position.qty * (exit_price - position.entry)
                exit_notional = abs(position.qty * exit_price)
                exit_fee = exit_notional * fee_rate
                funding_intervals = max(0, int((age * BAR_MINUTES) // 480))
                funding_cost = position.notional * 0.0001 * funding_intervals
                net = gross - position.entry_fee - exit_fee - funding_cost
                balance += gross - exit_fee - funding_cost
                trade = Trade(
                    symbol=symbol, setup=position.setup, side="LONG" if position.side == 1 else "SHORT",
                    score=position.score, opened_at=index[position.opened_i].isoformat(),
                    closed_at=timestamp.isoformat(), entry=position.entry, exit=exit_price,
                    net_pnl=net, r_multiple=net / position.risk_usdt, fees=position.entry_fee + exit_fee,
                    funding_cost=funding_cost, reason=reason,
                )
                trades.append(trade)
                if reason == "STOP" and net < 0:
                    stop_losses_today += 1
                    if stop_losses_today >= 3:
                        halted_day = True
                del positions[symbol]
                continue

            # Bar-close trailing update becomes active for the next candle.
            if current_r >= 1.20:
                cost_buffer = 2 * fee_rate + 2 * slip
                break_even = position.entry * (1.0 + position.side * cost_buffer)
                position.stop = max(position.stop, break_even) if position.side == 1 else min(position.stop, break_even)
            if position.setup != "FIBONACCI_PULLBACK_DIRECT" and current_r >= 1.25:
                risk_distance = abs(position.entry - position.initial_stop)
                trailing = position.best - position.side * risk_distance * 1.35
                position.stop = max(position.stop, trailing) if position.side == 1 else min(position.stop, trailing)

        equity = marked_equity(i)
        peak = max(peak, equity)
        if day_start > 0 and equity / day_start - 1 <= -0.02:
            halted_day = True
        if month_start > 0 and equity / month_start - 1 <= -0.08:
            halted_month = True

        if not halted_day and not halted_month:
            signal_i = i - 1
            regime = int(meta["regime"][signal_i])
            candidates: list[dict[str, float | str | int]] = []
            for column, symbol in enumerate(symbols):
                if symbol in positions or cooldowns.get(symbol, -1) > i:
                    continue
                candidate = candidate_at(
                    signal_i, symbol, data[symbol], float(meta["ranks"][signal_i, column]),
                    regime, threshold, fib_low, fib_high, flow_min, mode,
                )
                if candidate:
                    candidates.append(candidate)
            candidates.sort(key=lambda item: float(item["score"]), reverse=True)
            for candidate in candidates:
                if len(positions) >= 2:
                    break
                symbol = str(candidate["symbol"])
                cluster = CLUSTERS.get(symbol, symbol)
                if any(CLUSTERS.get(open_symbol, open_symbol) == cluster for open_symbol in positions):
                    continue
                side = int(candidate["side"])
                score = float(candidate["score"])
                risk_pct = 0.60 if score >= 92 else 0.50 if score >= 85 else 0.35
                if regime == 2:
                    risk_pct *= 0.5
                risk_budget = equity * risk_pct / 100.0
                stop_pct = float(candidate["stop_pct"])
                risk_fraction = stop_pct + 2 * fee_rate + 2 * slip
                desired_notional = risk_budget / risk_fraction
                used_margin = sum(position.margin for position in positions.values())
                available_margin = max(0.0, equity * 0.75 - used_margin)
                notional = min(desired_notional, available_margin * 5.0)
                open_risk = sum(position.risk_usdt for position in positions.values())
                if notional < 10 or open_risk + risk_budget > equity * 0.012:
                    continue
                raw_open = data[symbol]["open"][i]
                entry = raw_open * (1.0 + side * slip)
                qty = notional / entry
                entry_fee = notional * fee_rate
                balance -= entry_fee
                actual_risk = notional * risk_fraction
                stop = entry * (1.0 - side * stop_pct)
                target_gross = float(candidate["target_r"]) * actual_risk + entry_fee + notional * fee_rate
                target = entry * (1.0 + side * target_gross / notional)
                positions[symbol] = Position(
                    symbol=symbol, side=side, setup=str(candidate["setup"]), score=score,
                    opened_i=i, entry=entry, qty=qty, notional=notional, margin=notional / 5.0,
                    risk_usdt=actual_risk, initial_stop=stop, stop=stop, target=target,
                    entry_fee=entry_fee, max_bars=int(candidate["max_bars"]), best=entry, worst=entry,
                )
                cooldowns[symbol] = i + 4

        equity = marked_equity(i)
        equity_rows.append((timestamp, equity, balance, len(positions)))

    equity_frame = pd.DataFrame(equity_rows, columns=["timestamp", "equity", "balance", "positions"]).set_index("timestamp")
    trade_frame = pd.DataFrame([asdict(item) for item in trades])
    return equity_frame, trade_frame


def metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | str]:
    if equity.empty:
        return {}
    daily = equity["equity"].resample("1D").last().dropna()
    returns = daily.pct_change().dropna()
    years = max((daily.index[-1] - daily.index[0]).total_seconds() / (365.25 * 86_400), 1 / 365.25)
    total_return = daily.iloc[-1] / daily.iloc[0] - 1.0
    cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1 / years) - 1.0 if daily.iloc[0] > 0 else -1.0
    volatility = returns.std() * math.sqrt(365.25) if len(returns) > 1 else 0.0
    sharpe = returns.mean() / returns.std() * math.sqrt(365.25) if returns.std() > 0 else 0.0
    drawdown = daily / daily.cummax() - 1.0
    monthly = daily.resample("ME").last().pct_change().dropna()
    wins = trades[trades["net_pnl"] > 0] if not trades.empty else trades
    losses = trades[trades["net_pnl"] < 0] if not trades.empty else trades
    gross_profit = wins["net_pnl"].sum() if not wins.empty else 0.0
    gross_loss = -losses["net_pnl"].sum() if not losses.empty else 0.0
    return {
        "start": daily.index[0].isoformat(),
        "end": daily.index[-1].isoformat(),
        "ending_equity": round(float(daily.iloc[-1]), 2),
        "total_return_pct": round(float(total_return * 100), 2),
        "cagr_pct": round(float(cagr * 100), 2),
        "volatility_pct": round(float(volatility * 100), 2),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 2),
        "worst_month_pct": round(float(monthly.min() * 100), 2) if len(monthly) else 0.0,
        "trades": int(len(trades)),
        "win_rate_pct": round(float((trades["net_pnl"] > 0).mean() * 100), 2) if len(trades) else 0.0,
        "profit_factor": round(float(gross_profit / gross_loss), 2) if gross_loss > 0 else 0.0,
        "average_r": round(float(trades["r_multiple"].mean()), 3) if len(trades) else 0.0,
        "net_pnl": round(float(trades["net_pnl"].sum()), 2) if len(trades) else 0.0,
    }


def slice_metrics(equity: pd.DataFrame, trades: pd.DataFrame, start: str, end: str | None = None) -> dict[str, object]:
    section = equity.loc[start:end].copy()
    if not section.empty:
        section["equity"] = section["equity"] / section["equity"].iloc[0] * 1000.0
    if trades.empty:
        selected = trades
    else:
        closed = pd.to_datetime(trades["closed_at"], utc=True)
        mask = closed >= pd.Timestamp(start, tz="UTC")
        if end:
            mask &= closed < pd.Timestamp(end, tz="UTC")
        selected = trades.loc[mask]
    return metrics(section, selected)


def annual_returns(equity: pd.DataFrame) -> dict[str, float]:
    daily = equity["equity"].resample("1D").last().dropna()
    if daily.empty:
        return {}
    by_year: dict[str, float] = {}
    for year, values in daily.groupby(daily.index.year):
        prior = daily[daily.index < values.index[0]]
        start_value = prior.iloc[-1] if not prior.empty else values.iloc[0]
        by_year[str(year)] = round(float((values.iloc[-1] / start_value - 1.0) * 100), 2)
    return by_year


def btc_benchmark(index: pd.DatetimeIndex, data: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    closes = data["BTCUSDT"]["close"][221:]
    benchmark = 1_000.0 * closes / closes[0]
    return pd.DataFrame(
        {"equity": benchmark, "balance": benchmark, "positions": np.ones(len(benchmark), dtype=int)},
        index=index[221:],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("research/data"))
    parser.add_argument("--output", type=Path, default=Path("research/output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("loading and preparing 15m dataset", flush=True)
    index, data, meta = build_dataset(args.data)
    print(f"aligned bars={len(index)} symbols={len(meta['symbols'])}", flush=True)

    # Threshold 86 maximised Sharpe on the pre-declared 2022-2023 development
    # set. It is frozen before the 2024-2026 out-of-sample evaluation.
    base_params = {"threshold": 86.0, "fib_low": 0.382, "fib_high": 0.705, "flow_min": -0.04, "fee_rate": 0.0008, "slippage_bps": 1.0}
    equity, trades = simulate(index, data, meta, **base_params)
    equity.to_csv(args.output / "equity_curve.csv")
    trades.to_csv(args.output / "trades.csv", index=False)
    summary = {
        "method": "causal 4h trend/rank/order-flow/volume breakout aggregated from Binance USD-M 15m klines; next-bar entry; stop-first ambiguous bars",
        "data": {
            "symbols": [str(item) for item in meta["symbols"]],
            "aligned_bars": len(index),
            "start": index[0].isoformat(),
            "end": index[-1].isoformat(),
            "orderflow_proxy": "Binance taker-buy quote volume imbalance",
            "fibonacci": "calculated causally; direct entries disabled because development result was flat and out-of-sample result failed",
            "not_available": ["historical multi-venue order book", "historical liquidations", "exact funding per position"],
        },
        "parameters": base_params,
        "full": metrics(equity, trades),
        "development_2022_2023": slice_metrics(equity, trades, "2022-01-01", "2024-01-01"),
        "out_of_sample_2024_2026": slice_metrics(equity, trades, "2024-01-01"),
        "annual_returns_pct": annual_returns(equity),
        "btc_buy_and_hold": metrics(btc_benchmark(index, data), pd.DataFrame()),
        "setups": trades.groupby("setup").agg(
            trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
            win_rate=("net_pnl", lambda values: float((values > 0).mean())),
            average_r=("r_multiple", "mean"),
        ).round(4).reset_index().to_dict(orient="records") if not trades.empty else [],
    }

    sensitivity_specs = [
        (78.0, 0.382, 0.705, -0.04, 0.0008, 1.0),
        (80.0, 0.382, 0.705, -0.04, 0.0008, 1.0),
        (82.0, 0.382, 0.705, -0.04, 0.0008, 1.0),
        (84.0, 0.382, 0.705, -0.04, 0.0008, 1.0),
        (86.0, 0.382, 0.705, -0.04, 0.0008, 1.0),
        (88.0, 0.382, 0.705, -0.04, 0.0008, 1.0),
        (86.0, 0.382, 0.705, -0.04, 0.0010, 2.0),
    ]
    sensitivity: list[dict[str, object]] = []
    for number, spec in enumerate(sensitivity_specs, 1):
        threshold, fib_low, fib_high, flow_min, fee_rate, slippage = spec
        print(f"sensitivity {number}/{len(sensitivity_specs)}", flush=True)
        test_equity, test_trades = simulate(
            index, data, meta, threshold=threshold, fib_low=fib_low, fib_high=fib_high,
            flow_min=flow_min, fee_rate=fee_rate, slippage_bps=slippage,
        )
        row = {
            "threshold": threshold, "fib_low": fib_low, "fib_high": fib_high,
            "flow_min": flow_min, "fee_rate": fee_rate, "slippage_bps": slippage,
            **metrics(test_equity, test_trades),
        }
        development = slice_metrics(test_equity, test_trades, "2022-01-01", "2024-01-01")
        out_of_sample = slice_metrics(test_equity, test_trades, "2024-01-01")
        row.update(
            dev_cagr_pct=development.get("cagr_pct", 0.0),
            dev_sharpe=development.get("sharpe", 0.0),
            dev_profit_factor=development.get("profit_factor", 0.0),
            oos_cagr_pct=out_of_sample.get("cagr_pct", 0.0),
            oos_sharpe=out_of_sample.get("sharpe", 0.0),
            oos_profit_factor=out_of_sample.get("profit_factor", 0.0),
        )
        sensitivity.append(row)
    pd.DataFrame(sensitivity).to_csv(args.output / "parameter_sensitivity.csv", index=False)
    summary["sensitivity"] = sensitivity

    variant_rows: list[dict[str, object]] = []
    variant_details: dict[str, dict[str, object]] = {}
    for mode in ("trend_only", "breakout", "breakout_structure", "breakout_trendline", "fib_pullback"):
        print(f"variant {mode}", flush=True)
        if mode == "breakout":
            variant_equity, variant_trades = equity, trades
        else:
            variant_equity, variant_trades = simulate(index, data, meta, **base_params, mode=mode)
        full_metrics = metrics(variant_equity, variant_trades)
        development = slice_metrics(variant_equity, variant_trades, "2022-01-01", "2024-01-01")
        out_of_sample = slice_metrics(variant_equity, variant_trades, "2024-01-01")
        variant_details[mode] = {
            "full": full_metrics,
            "development_2022_2023": development,
            "out_of_sample_2024_2026": out_of_sample,
        }
        variant_rows.append({
            "mode": mode,
            "full_cagr_pct": full_metrics.get("cagr_pct", 0.0),
            "full_sharpe": full_metrics.get("sharpe", 0.0),
            "full_max_drawdown_pct": full_metrics.get("max_drawdown_pct", 0.0),
            "full_profit_factor": full_metrics.get("profit_factor", 0.0),
            "dev_cagr_pct": development.get("cagr_pct", 0.0),
            "dev_sharpe": development.get("sharpe", 0.0),
            "dev_profit_factor": development.get("profit_factor", 0.0),
            "oos_cagr_pct": out_of_sample.get("cagr_pct", 0.0),
            "oos_sharpe": out_of_sample.get("sharpe", 0.0),
            "oos_profit_factor": out_of_sample.get("profit_factor", 0.0),
        })
    pd.DataFrame(variant_rows).to_csv(args.output / "strategy_variants.csv", index=False)
    summary["strategy_variants"] = variant_details
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["full"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
