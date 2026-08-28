from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import ClosedTrade, FeatureSnapshot, Signal


class Storage:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection: sqlite3.Connection | None = None
        self.lock = asyncio.Lock()

    async def initialise(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                setup TEXT NOT NULL,
                side TEXT NOT NULL,
                score REAL NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts DESC);

            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                account TEXT NOT NULL,
                symbol TEXT NOT NULL,
                setup TEXT NOT NULL,
                side TEXT NOT NULL,
                opened_at REAL NOT NULL,
                closed_at REAL NOT NULL,
                net_pnl REAL NOT NULL,
                r_multiple REAL NOT NULL,
                reason TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trades_closed ON trades(closed_at DESC);
            CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account, closed_at DESC);

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                account TEXT NOT NULL,
                equity REAL NOT NULL,
                balance REAL NOT NULL,
                drawdown_pct REAL NOT NULL,
                return_pct REAL NOT NULL,
                positions_count INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_equity_account_ts
                ON equity_snapshots(account, ts DESC);

            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                data_ready INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_features_symbol_ts ON features(symbol, ts DESC);

            CREATE TABLE IF NOT EXISTS account_states (
                name TEXT PRIMARY KEY,
                updated_at REAL NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS service_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                level TEXT NOT NULL,
                event TEXT NOT NULL,
                detail TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def _conn(self) -> sqlite3.Connection:
        if self.connection is None:
            raise RuntimeError("Storage is not initialised")
        return self.connection

    async def save_signal(self, signal: Signal) -> None:
        payload = signal.as_dict()
        async with self.lock:
            self._conn().execute(
                """INSERT INTO signals(ts,symbol,strategy,setup,side,score,payload)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    signal.ts,
                    signal.symbol,
                    signal.strategy,
                    signal.setup,
                    signal.side.label,
                    signal.score,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            self._conn().commit()

    async def save_trade(self, trade: ClosedTrade) -> None:
        payload = trade.as_dict()
        async with self.lock:
            self._conn().execute(
                """INSERT OR REPLACE INTO trades(
                       id,account,symbol,setup,side,opened_at,closed_at,
                       net_pnl,r_multiple,reason,payload
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade.id,
                    trade.account,
                    trade.symbol,
                    trade.setup,
                    trade.side.label,
                    trade.opened_at,
                    trade.closed_at,
                    trade.net_pnl,
                    trade.r_multiple,
                    trade.reason,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            self._conn().commit()

    async def save_feature(self, feature: FeatureSnapshot) -> None:
        async with self.lock:
            self._conn().execute(
                "INSERT INTO features(ts,symbol,price,data_ready,payload) VALUES(?,?,?,?,?)",
                (
                    feature.ts,
                    feature.symbol,
                    feature.price,
                    int(feature.data_ready),
                    json.dumps(feature.as_dict(), ensure_ascii=False, separators=(",", ":")),
                ),
            )
            self._conn().commit()

    async def save_account_states(self, ts: float, payloads: dict[str, dict[str, Any]]) -> None:
        rows = [
            (name, ts, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            for name, payload in payloads.items()
        ]
        async with self.lock:
            self._conn().executemany(
                """INSERT INTO account_states(name,updated_at,payload) VALUES(?,?,?)
                   ON CONFLICT(name) DO UPDATE SET
                       updated_at=excluded.updated_at,
                       payload=excluded.payload""",
                rows,
            )
            self._conn().commit()

    async def load_account_states(self) -> dict[str, dict[str, Any]]:
        async with self.lock:
            rows = self._conn().execute("SELECT name,payload FROM account_states").fetchall()
        return {row["name"]: json.loads(row["payload"]) for row in rows}

    async def save_equity(self, ts: float, accounts: dict[str, dict[str, Any]]) -> None:
        rows = [
            (
                ts,
                name,
                float(data["equity"]),
                float(data["balance"]),
                float(data["drawdown_pct"]),
                float(data["return_pct"]),
                int(data["positions_count"]),
            )
            for name, data in accounts.items()
        ]
        async with self.lock:
            self._conn().executemany(
                """INSERT INTO equity_snapshots(
                       ts,account,equity,balance,drawdown_pct,return_pct,positions_count
                   ) VALUES(?,?,?,?,?,?,?)""",
                rows,
            )
            self._conn().commit()

    async def event(self, ts: float, level: str, event: str, detail: str = "") -> None:
        async with self.lock:
            self._conn().execute(
                "INSERT INTO service_events(ts,level,event,detail) VALUES(?,?,?,?)",
                (ts, level, event, detail[:2_000]),
            )
            self._conn().commit()

    async def recent_trades(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self.lock:
            rows = self._conn().execute(
                "SELECT payload FROM trades ORDER BY closed_at DESC LIMIT ?", (max(1, min(limit, 1_000)),)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    async def recent_signals(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self.lock:
            rows = self._conn().execute(
                "SELECT payload FROM signals ORDER BY ts DESC LIMIT ?", (max(1, min(limit, 1_000)),)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    async def equity_history(self, account: str, limit: int = 1_000) -> list[dict[str, Any]]:
        async with self.lock:
            rows = self._conn().execute(
                """SELECT ts,equity,balance,drawdown_pct,return_pct,positions_count
                   FROM equity_snapshots WHERE account=? ORDER BY ts DESC LIMIT ?""",
                (account, max(1, min(limit, 10_000))),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    async def close(self) -> None:
        async with self.lock:
            if self.connection is not None:
                self.connection.commit()
                self.connection.close()
                self.connection = None

