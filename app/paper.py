from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable

from .config import AccountConfig, Settings
from .market import MarketState
from .models import AccountState, ClosedTrade, FeatureSnapshot, Position, Side, Signal


def _utc_keys(ts: float) -> tuple[str, str]:
    current = datetime.fromtimestamp(ts, tz=UTC)
    return current.strftime("%Y-%m-%d"), current.strftime("%Y-%m")


@dataclass(slots=True)
class Fill:
    price: float
    slippage_bps: float


class PaperBroker:
    """Virtual execution engine. It has no exchange credentials or order client."""

    def __init__(
        self,
        market: MarketState,
        settings: Settings,
        configs: Iterable[AccountConfig],
    ) -> None:
        self.market = market
        self.settings = settings
        now = time.time()
        day_key, month_key = _utc_keys(now)
        self.accounts: dict[str, AccountState] = {
            config.name: AccountState(
                name=config.name,
                strategy=config.strategy,
                starting_balance=config.starting_balance,
                balance=config.starting_balance,
                risk_pct=config.risk_pct,
                max_leverage=config.max_leverage,
                peak_equity=config.starting_balance,
                day_start_equity=config.starting_balance,
                month_start_equity=config.starting_balance,
                day_key=day_key,
                month_key=month_key,
            )
            for config in configs
        }

    def restore(self, payloads: dict[str, dict[str, Any]]) -> None:
        for name, payload in payloads.items():
            if name in self.accounts:
                self.accounts[name] = AccountState.from_dict(payload)

    def mark_price(self, symbol: str) -> float:
        return self.market.symbol(symbol).reference_price()

    def equity(self, account: AccountState) -> float:
        unrealized = sum(
            position.unrealized_pnl(self.mark_price(symbol))
            for symbol, position in account.positions.items()
            if self.mark_price(symbol) > 0
        )
        return account.balance + unrealized

    @staticmethod
    def used_margin(account: AccountState) -> float:
        return sum(position.margin for position in account.positions.values())

    def _roll_risk_periods(self, account: AccountState, now: float) -> None:
        day_key, month_key = _utc_keys(now)
        equity = self.equity(account)
        if account.day_key != day_key:
            account.day_key = day_key
            account.day_start_equity = equity
            if account.halted_reason.startswith("daily"):
                account.halted_reason = ""
        if account.month_key != month_key:
            account.month_key = month_key
            account.month_start_equity = equity
            if account.halted_reason.startswith("monthly"):
                account.halted_reason = ""
        account.peak_equity = max(account.peak_equity, equity)

        daily_return = (
            equity / account.day_start_equity - 1.0 if account.day_start_equity > 0 else 0.0
        )
        monthly_return = (
            equity / account.month_start_equity - 1.0 if account.month_start_equity > 0 else 0.0
        )
        if monthly_return <= -self.settings.monthly_stop_pct / 100.0:
            account.halted_reason = f"monthly stop {monthly_return:.1%}"
        elif daily_return <= -self.settings.daily_stop_pct / 100.0:
            account.halted_reason = f"daily stop {daily_return:.1%}"

    def _fill(self, symbol: str, execution_side: Side, notional: float, *, stress: float = 1.0) -> Fill | None:
        state = self.market.symbol(symbol)
        bbo = state.book("mexc").best_bid_ask()
        if not bbo:
            return None
        bid, ask = bbo
        reference = ask if execution_side == Side.LONG else bid
        impact = state.book("bybit").impact_bps(execution_side, notional)
        impact = min(max(impact, 0.0), self.settings.impact_slippage_bps)
        slippage = max(self.settings.min_slippage_bps, impact) * stress
        price = reference * (1.0 + float(execution_side) * slippage / 10_000.0)
        return Fill(price=price, slippage_bps=slippage)

    def _effective_risk_pct(self, account: AccountState, signal: Signal) -> float:
        if signal.score < 84.0:
            conviction = 0.65
        elif signal.score < self.settings.high_conviction_threshold:
            conviction = 0.85
        else:
            conviction = 1.0
        drawdown = 1.0 - self.equity(account) / account.peak_equity if account.peak_equity > 0 else 0.0
        if drawdown >= self.settings.risk_reduction_drawdown_pct / 100.0:
            conviction *= 0.5
        return account.risk_pct * conviction

    def can_open(self, account: AccountState, signal: Signal, now: float) -> bool:
        self._roll_risk_periods(account, now)
        if account.halted_reason:
            return False
        if signal.symbol in account.positions:
            return False
        if len(account.positions) >= self.settings.max_open_positions:
            return False
        if account.cooldowns.get(signal.symbol, 0.0) > now:
            return False
        if signal.score < self.settings.signal_threshold:
            return False
        return True

    def open_from_signal(self, account: AccountState, signal: Signal, now: float) -> Position | None:
        if not self.can_open(account, signal, now):
            return None
        equity = self.equity(account)
        if equity <= 0:
            account.halted_reason = "account depleted"
            return None

        risk_pct = self._effective_risk_pct(account, signal)
        risk_budget = equity * risk_pct / 100.0
        estimated_cost_pct = 2.0 * self.settings.taker_fee_rate + (
            2.0 * self.settings.min_slippage_bps / 10_000.0
        )
        risk_fraction = signal.stop_pct + estimated_cost_pct
        desired_notional = risk_budget / risk_fraction if risk_fraction > 0 else 0.0

        available_margin = max(
            0.0,
            equity * self.settings.max_margin_utilization - self.used_margin(account),
        )
        notional = min(desired_notional, available_margin * account.max_leverage)
        if notional < 10.0:
            return None
        fill = self._fill(signal.symbol, signal.side, notional)
        if fill is None:
            return None
        qty = notional / fill.price
        entry_fee = notional * self.settings.taker_fee_rate
        account.balance -= entry_fee
        account.total_fees += entry_fee

        stop_price = fill.price * (1.0 - float(signal.side) * signal.stop_pct)
        target_price = fill.price * (1.0 + float(signal.side) * signal.stop_pct * signal.target_r)
        actual_risk = notional * (
            signal.stop_pct
            + 2.0 * self.settings.taker_fee_rate
            + 2.0 * fill.slippage_bps / 10_000.0
        )
        position = Position(
            id=uuid.uuid4().hex,
            account=account.name,
            symbol=signal.symbol,
            setup=signal.setup,
            side=signal.side,
            score=signal.score,
            opened_at=now,
            entry_price=fill.price,
            qty=qty,
            notional=notional,
            leverage=account.max_leverage,
            margin=notional / account.max_leverage,
            initial_risk_usdt=actual_risk,
            initial_stop_price=stop_price,
            stop_price=stop_price,
            target_price=target_price,
            target_r=signal.target_r,
            entry_fee=entry_fee,
        )
        account.positions[signal.symbol] = position
        account.cooldowns[signal.symbol] = now + self.settings.cooldown_seconds
        return position

    def handle_signal(self, signal: Signal, now: float | None = None) -> list[Position]:
        current_time = now or time.time()
        opened: list[Position] = []
        for account in self.accounts.values():
            if account.strategy != signal.strategy:
                continue
            position = self.open_from_signal(account, signal, current_time)
            if position:
                opened.append(position)
        return opened

    def close_position(
        self,
        account: AccountState,
        position: Position,
        reason: str,
        now: float,
        *,
        stress: float = 1.0,
        liquidation_fee_rate: float = 0.0,
    ) -> ClosedTrade | None:
        exit_side = Side.SHORT if position.side == Side.LONG else Side.LONG
        fill = self._fill(position.symbol, exit_side, position.notional, stress=stress)
        if fill is None:
            return None
        gross = position.unrealized_pnl(fill.price)
        exit_notional = abs(position.qty * fill.price)
        exit_fee = exit_notional * self.settings.taker_fee_rate
        liquidation_fee = exit_notional * liquidation_fee_rate
        account.balance += gross - exit_fee - liquidation_fee
        account.total_fees += exit_fee + liquidation_fee
        net = gross - position.entry_fee - exit_fee - liquidation_fee + position.accrued_funding
        account.realized_pnl += gross - position.entry_fee - exit_fee - liquidation_fee
        if net >= 0:
            account.wins += 1
        else:
            account.losses += 1
        trade = ClosedTrade(
            id=position.id,
            account=account.name,
            symbol=position.symbol,
            setup=position.setup,
            side=position.side,
            score=position.score,
            opened_at=position.opened_at,
            closed_at=now,
            entry_price=position.entry_price,
            exit_price=fill.price,
            qty=position.qty,
            notional=position.notional,
            gross_pnl=gross,
            fees=position.entry_fee + exit_fee + liquidation_fee,
            funding=position.accrued_funding,
            net_pnl=net,
            r_multiple=net / position.initial_risk_usdt if position.initial_risk_usdt > 0 else 0.0,
            reason=reason,
        )
        account.positions.pop(position.symbol, None)
        self._roll_risk_periods(account, now)
        return trade

    def evaluate_positions(
        self,
        features: dict[str, FeatureSnapshot],
        now: float | None = None,
    ) -> list[ClosedTrade]:
        current_time = now or time.time()
        closed: list[ClosedTrade] = []
        for account in self.accounts.values():
            self._roll_risk_periods(account, current_time)
            for symbol, position in list(account.positions.items()):
                state = self.market.symbol(symbol)
                bbo = state.book("mexc").best_bid_ask()
                if not bbo:
                    continue
                bid, ask = bbo
                mark = bid if position.side == Side.LONG else ask
                position.best_price = (
                    max(position.best_price, mark)
                    if position.side == Side.LONG
                    else min(position.best_price, mark)
                )
                position.worst_price = (
                    min(position.worst_price, mark)
                    if position.side == Side.LONG
                    else max(position.worst_price, mark)
                )

                leverage_distance = max(0.001, 1.0 / position.leverage - state.maintenance_margin_rate)
                liquidation_price = position.entry_price * (
                    1.0 - float(position.side) * leverage_distance
                )
                liquidated = (
                    mark <= liquidation_price
                    if position.side == Side.LONG
                    else mark >= liquidation_price
                )
                stop_hit = mark <= position.stop_price if position.side == Side.LONG else mark >= position.stop_price
                target_hit = mark >= position.target_price if position.side == Side.LONG else mark <= position.target_price

                current_r = position.current_r(mark)
                if current_r >= 1.5:
                    break_even_buffer = 2.0 * self.settings.taker_fee_rate + (
                        2.0 * self.settings.min_slippage_bps / 10_000.0
                    )
                    break_even = position.entry_price * (
                        1.0 + float(position.side) * break_even_buffer
                    )
                    if position.side == Side.LONG:
                        position.stop_price = max(position.stop_price, break_even)
                    else:
                        position.stop_price = min(position.stop_price, break_even)

                age = current_time - position.opened_at
                feature = features.get(symbol)
                flow_reversal = bool(
                    age >= 300.0
                    and feature
                    and current_r < 1.0
                    and float(position.side) * feature.flow_fast < -0.25
                    and float(position.side) * feature.book_imbalance < -0.08
                )
                timed_out = age >= self.settings.max_holding_minutes * 60 and current_r < 0.75

                reason = ""
                stress = 1.0
                liquidation_fee = 0.0
                if liquidated:
                    reason = "LIQUIDATION"
                    stress = 3.0
                    liquidation_fee = 0.0004
                elif stop_hit:
                    reason = "STOP"
                    stress = 2.0
                elif target_hit:
                    reason = "TARGET"
                elif flow_reversal:
                    reason = "ORDER_FLOW_REVERSAL"
                elif timed_out:
                    reason = "TIME_STOP"

                if reason:
                    trade = self.close_position(
                        account,
                        position,
                        reason,
                        current_time,
                        stress=stress,
                        liquidation_fee_rate=liquidation_fee,
                    )
                    if trade:
                        closed.append(trade)
        return closed

    def apply_funding(self, symbol: str, rate: float, settlement_ts: float) -> None:
        if not rate or not settlement_ts:
            return
        for account in self.accounts.values():
            position = account.positions.get(symbol)
            if not position or settlement_ts <= position.last_funding_at:
                continue
            cashflow = -position.notional * rate * float(position.side)
            account.balance += cashflow
            account.total_funding += cashflow
            position.accrued_funding += cashflow
            position.last_funding_at = settlement_ts

    def account_summary(self, account: AccountState) -> dict[str, Any]:
        equity = self.equity(account)
        drawdown = equity / account.peak_equity - 1.0 if account.peak_equity > 0 else 0.0
        total_return = equity / account.starting_balance - 1.0 if account.starting_balance > 0 else 0.0
        trades = account.wins + account.losses
        return {
            "name": account.name,
            "strategy": account.strategy,
            "starting_balance": account.starting_balance,
            "balance": account.balance,
            "equity": equity,
            "return_pct": total_return * 100.0,
            "drawdown_pct": drawdown * 100.0,
            "peak_equity": account.peak_equity,
            "risk_pct": account.risk_pct,
            "max_leverage": account.max_leverage,
            "used_margin": self.used_margin(account),
            "positions_count": len(account.positions),
            "wins": account.wins,
            "losses": account.losses,
            "win_rate_pct": account.wins / trades * 100.0 if trades else 0.0,
            "realized_pnl": account.realized_pnl,
            "total_fees": account.total_fees,
            "total_funding": account.total_funding,
            "halted_reason": account.halted_reason,
            "positions": [
                position.as_dict(self.mark_price(position.symbol))
                for position in account.positions.values()
            ],
        }

    def status(self) -> dict[str, Any]:
        return {
            name: self.account_summary(account) for name, account in self.accounts.items()
        }

    def snapshots(self) -> dict[str, dict[str, Any]]:
        return {name: account.as_dict() for name, account in self.accounts.items()}

