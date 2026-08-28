from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any


class Side(IntEnum):
    SHORT = -1
    LONG = 1

    @property
    def label(self) -> str:
        return "LONG" if self == Side.LONG else "SHORT"


@dataclass(slots=True)
class TradeEvent:
    symbol: str
    venue: str
    price: float
    base_qty: float
    side: Side
    ts: float
    open_close: int | None = None

    @property
    def notional(self) -> float:
        return abs(self.price * self.base_qty)


@dataclass(slots=True)
class LiquidationEvent:
    symbol: str
    venue: str
    price: float
    base_qty: float
    pressure_side: Side
    ts: float

    @property
    def notional(self) -> float:
        return abs(self.price * self.base_qty)


@dataclass(slots=True)
class MinuteBar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume_notional: float = 0.0

    def update(self, price: float, volume_notional: float = 0.0) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume_notional += max(0.0, volume_notional)


@dataclass(slots=True)
class FeatureSnapshot:
    symbol: str
    ts: float
    price: float
    spread_bps: float
    data_ready: bool
    ret_60s: float = 0.0
    ret_300s: float = 0.0
    ret_1800s: float = 0.0
    trend_score: float = 0.0
    vol_short: float = 0.0
    vol_long: float = 0.0
    vol_ratio: float = 1.0
    price_position: float = 0.5
    flow_fast: float = 0.0
    flow_slow: float = 0.0
    mexc_flow: float = 0.0
    bybit_flow: float = 0.0
    binance_flow: float = 0.0
    book_imbalance: float = 0.0
    microprice_bps: float = 0.0
    cross_venue_consensus: float = 0.0
    oi_change_300s: float = 0.0
    funding_rate: float = 0.0
    liquidation_imbalance: float = 0.0
    atr_pct: float = 0.01
    trade_count_60s: int = 0
    stale_venues: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Signal:
    symbol: str
    strategy: str
    setup: str
    side: Side
    score: float
    stop_pct: float
    target_r: float
    ts: float
    reasons: list[str] = field(default_factory=list)
    feature_data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["side"] = self.side.label
        return payload


@dataclass(slots=True)
class Position:
    id: str
    account: str
    symbol: str
    setup: str
    side: Side
    score: float
    opened_at: float
    entry_price: float
    qty: float
    notional: float
    leverage: float
    margin: float
    initial_risk_usdt: float
    initial_stop_price: float
    stop_price: float
    target_price: float
    target_r: float
    entry_fee: float
    accrued_funding: float = 0.0
    last_funding_at: float = 0.0
    best_price: float = 0.0
    worst_price: float = 0.0

    def __post_init__(self) -> None:
        if not self.best_price:
            self.best_price = self.entry_price
        if not self.worst_price:
            self.worst_price = self.entry_price

    def unrealized_pnl(self, mark_price: float) -> float:
        return float(self.side) * self.qty * (mark_price - self.entry_price)

    def current_r(self, mark_price: float) -> float:
        if self.initial_risk_usdt <= 0:
            return 0.0
        return self.unrealized_pnl(mark_price) / self.initial_risk_usdt

    def as_dict(self, mark_price: float | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["side"] = self.side.label
        if mark_price is not None:
            payload["mark_price"] = mark_price
            payload["unrealized_pnl"] = self.unrealized_pnl(mark_price)
            payload["current_r"] = self.current_r(mark_price)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Position":
        data = dict(payload)
        raw_side = data.get("side")
        if isinstance(raw_side, str):
            data["side"] = Side.LONG if raw_side.upper() == "LONG" else Side.SHORT
        else:
            data["side"] = Side(int(raw_side))
        data.pop("mark_price", None)
        data.pop("unrealized_pnl", None)
        data.pop("current_r", None)
        return cls(**data)


@dataclass(slots=True)
class ClosedTrade:
    id: str
    account: str
    symbol: str
    setup: str
    side: Side
    score: float
    opened_at: float
    closed_at: float
    entry_price: float
    exit_price: float
    qty: float
    notional: float
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float
    r_multiple: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["side"] = self.side.label
        return payload


@dataclass(slots=True)
class AccountState:
    name: str
    strategy: str
    starting_balance: float
    balance: float
    risk_pct: float
    max_leverage: float
    peak_equity: float
    day_start_equity: float
    month_start_equity: float
    day_key: str
    month_key: str
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    total_funding: float = 0.0
    wins: int = 0
    losses: int = 0
    positions: dict[str, Position] = field(default_factory=dict)
    cooldowns: dict[str, float] = field(default_factory=dict)
    halted_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["positions"] = {
            symbol: position.as_dict() for symbol, position in self.positions.items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AccountState":
        data = dict(payload)
        data["positions"] = {
            symbol: Position.from_dict(position)
            for symbol, position in data.get("positions", {}).items()
        }
        return cls(**data)
