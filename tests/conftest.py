from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.config import Settings


def make_settings(path: Path, **overrides: object) -> Settings:
    base = Settings.from_env()
    values = {
        "symbols": ("BTC_USDT",),
        "data_mode": "synthetic",
        "db_path": path,
        "evaluation_interval_seconds": 0.05,
        "feature_persist_seconds": 1,
        "account_persist_seconds": 1,
        "cooldown_seconds": 0,
    }
    values.update(overrides)
    return replace(base, **values)

