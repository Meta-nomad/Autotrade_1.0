from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


DEFAULT_SYMBOLS = (
    "BTC_USDT",
    "ETH_USDT",
    "SOL_USDT",
    "XRP_USDT",
    "BNB_USDT",
    "DOGE_USDT",
    "ADA_USDT",
    "AVAX_USDT",
    "LINK_USDT",
    "LTC_USDT",
)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw in (None, "") else float(raw)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw in (None, "") else int(raw)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    symbols: tuple[str, ...]
    data_mode: str
    db_path: Path
    paper_balance: float
    maker_fee_rate: float
    taker_fee_rate: float
    min_slippage_bps: float
    impact_slippage_bps: float
    base_risk_pct: float
    turbo_risk_pct: float
    control_risk_pct: float
    baseline_risk_pct: float
    module_risk_pct: float
    reversal_risk_pct: float
    ensemble_risk_pct: float
    base_max_leverage: float
    turbo_max_leverage: float
    max_margin_utilization: float
    max_portfolio_risk_pct: float
    max_open_positions: int
    daily_stop_pct: float
    monthly_stop_pct: float
    risk_reduction_drawdown_pct: float
    signal_threshold: float
    high_conviction_threshold: float
    evaluation_interval_seconds: float
    feature_persist_seconds: int
    account_persist_seconds: int
    cooldown_seconds: int
    max_holding_minutes: int
    min_flow_exit_minutes: int
    flow_exit_confirm_seconds: int
    stale_after_seconds: int
    bybit_depth: int
    enable_binance: bool
    dashboard_token: str
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        symbols_raw = os.getenv("SYMBOLS", ",".join(DEFAULT_SYMBOLS))
        symbols = tuple(
            item.strip().upper().replace("/", "_")
            for item in symbols_raw.split(",")
            if item.strip()
        )
        if not symbols:
            raise ValueError("SYMBOLS must contain at least one contract")

        data_mode = os.getenv("DATA_MODE", "live").strip().lower()
        if data_mode not in {"live", "synthetic"}:
            raise ValueError("DATA_MODE must be 'live' or 'synthetic'")

        db_path = Path(os.getenv("DB_PATH", "data/paper_trader.db")).expanduser()
        return cls(
            symbols=symbols,
            data_mode=data_mode,
            db_path=db_path,
            paper_balance=_float("PAPER_BALANCE", 1000.0),
            # MEXC standard API Futures rates announced for 2026-06-01.
            # Both remain configurable because actual account rates may differ.
            maker_fee_rate=_float("MAKER_FEE_RATE", 0.0006),
            taker_fee_rate=_float("TAKER_FEE_RATE", 0.0008),
            min_slippage_bps=_float("MIN_SLIPPAGE_BPS", 0.5),
            impact_slippage_bps=_float("IMPACT_SLIPPAGE_BPS", 8.0),
            base_risk_pct=_float("BASE_RISK_PCT", 1.5),
            turbo_risk_pct=_float("TURBO_RISK_PCT", 3.0),
            control_risk_pct=_float("CONTROL_RISK_PCT", 2.0),
            baseline_risk_pct=_float("BASELINE_RISK_PCT", 0.25),
            module_risk_pct=_float("MODULE_RISK_PCT", 0.50),
            reversal_risk_pct=_float("REVERSAL_RISK_PCT", 0.35),
            ensemble_risk_pct=_float("ENSEMBLE_RISK_PCT", 0.50),
            base_max_leverage=_float("BASE_MAX_LEVERAGE", 5.0),
            turbo_max_leverage=_float("TURBO_MAX_LEVERAGE", 8.0),
            max_margin_utilization=_float("MAX_MARGIN_UTILIZATION", 0.75),
            max_portfolio_risk_pct=_float("MAX_PORTFOLIO_RISK_PCT", 1.50),
            max_open_positions=_int("MAX_OPEN_POSITIONS", 2),
            daily_stop_pct=_float("DAILY_STOP_PCT", 4.0),
            monthly_stop_pct=_float("MONTHLY_STOP_PCT", 15.0),
            risk_reduction_drawdown_pct=_float("RISK_REDUCTION_DRAWDOWN_PCT", 10.0),
            signal_threshold=_float("SIGNAL_THRESHOLD", 75.0),
            high_conviction_threshold=_float("HIGH_CONVICTION_THRESHOLD", 90.0),
            evaluation_interval_seconds=_float("EVALUATION_INTERVAL_SECONDS", 2.0),
            feature_persist_seconds=_int("FEATURE_PERSIST_SECONDS", 10),
            account_persist_seconds=_int("ACCOUNT_PERSIST_SECONDS", 10),
            cooldown_seconds=_int("COOLDOWN_SECONDS", 900),
            max_holding_minutes=_int("MAX_HOLDING_MINUTES", 120),
            min_flow_exit_minutes=_int("MIN_FLOW_EXIT_MINUTES", 15),
            flow_exit_confirm_seconds=_int("FLOW_EXIT_CONFIRM_SECONDS", 90),
            stale_after_seconds=_int("STALE_AFTER_SECONDS", 8),
            bybit_depth=_int("BYBIT_DEPTH", 50),
            enable_binance=_bool("ENABLE_BINANCE", True),
            dashboard_token=os.getenv("DASHBOARD_TOKEN", "").strip(),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )


@dataclass(frozen=True, slots=True)
class AccountConfig:
    name: str
    strategy: str
    starting_balance: float
    risk_pct: float
    max_leverage: float


def account_configs(settings: Settings) -> tuple[AccountConfig, ...]:
    return (
        AccountConfig(
            name="BASELINE_016",
            strategy="baseline",
            starting_balance=settings.paper_balance,
            risk_pct=settings.baseline_risk_pct,
            max_leverage=settings.base_max_leverage,
        ),
        AccountConfig(
            name="TREND_ORDERFLOW",
            strategy="trend_orderflow",
            starting_balance=settings.paper_balance,
            risk_pct=settings.module_risk_pct,
            max_leverage=settings.base_max_leverage,
        ),
        AccountConfig(
            name="CROSS_MOMENTUM",
            strategy="cross_momentum",
            starting_balance=settings.paper_balance,
            risk_pct=settings.module_risk_pct,
            max_leverage=settings.base_max_leverage,
        ),
        AccountConfig(
            name="LIQUIDATION_REVERSAL",
            strategy="liquidation_reversal",
            starting_balance=settings.paper_balance,
            risk_pct=settings.reversal_risk_pct,
            max_leverage=settings.base_max_leverage,
        ),
        AccountConfig(
            name="ENSEMBLE",
            strategy="ensemble",
            starting_balance=settings.paper_balance,
            risk_pct=settings.ensemble_risk_pct,
            max_leverage=settings.base_max_leverage,
        ),
    )
