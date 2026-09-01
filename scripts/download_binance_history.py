#!/usr/bin/env python3
"""Download and normalize public Binance USD-M futures klines.

The source is Binance Vision. No API key is required. Monthly ZIP files are
downloaded concurrently, validated by ``zipfile``, and merged into one gzipped
CSV per symbol. Existing normalized files are replaced atomically.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path


BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
HEADER = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


@dataclass(frozen=True)
class Download:
    symbol: str
    month: str
    rows: tuple[tuple[str, ...], ...]
    status: str


def months_between(start: str, end: str) -> list[str]:
    start_y, start_m = map(int, start.split("-"))
    end_y, end_m = map(int, end.split("-"))
    cursor = date(start_y, start_m, 1)
    finish = date(end_y, end_m, 1)
    result: list[str] = []
    while cursor <= finish:
        result.append(cursor.strftime("%Y-%m"))
        cursor = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
    return result


def normalize_timestamp(raw: str) -> str:
    value = int(raw)
    # Binance spot archives switched to microseconds in 2025; futures files
    # currently use milliseconds. Supporting both makes the loader robust.
    if value > 10_000_000_000_000:
        value //= 1_000
    return str(value)


def fetch_month(symbol: str, interval: str, month: str, retries: int) -> Download:
    filename = f"{symbol}-{interval}-{month}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{filename}"
    last_error = ""
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CompositeFlowResearch/0.3"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if len(members) != 1:
                    raise ValueError(f"unexpected archive members: {archive.namelist()}")
                text = io.TextIOWrapper(archive.open(members[0]), encoding="utf-8", newline="")
                rows: list[tuple[str, ...]] = []
                for row in csv.reader(text):
                    if not row or not row[0].isdigit() or len(row) < 12:
                        continue
                    row[0] = normalize_timestamp(row[0])
                    row[6] = normalize_timestamp(row[6])
                    rows.append(tuple(row[:12]))
            return Download(symbol, month, tuple(rows), "ok")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return Download(symbol, month, (), "missing")
            last_error = f"HTTP {exc.code}"
        except Exception as exc:  # network/ZIP validation retry
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(1.0 + attempt * 1.5)
    return Download(symbol, month, (), f"error:{last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--start", default="2022-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, default=Path("research/data"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    symbols = [item.upper().replace("_", "") for item in args.symbols]
    months = months_between(args.start, args.end)
    jobs = [(symbol, month) for symbol in symbols for month in months]
    args.output.mkdir(parents=True, exist_ok=True)
    by_symbol: dict[str, list[Download]] = {symbol: [] for symbol in symbols}
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(fetch_month, symbol, args.interval, month, args.retries): (symbol, month)
            for symbol, month in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            by_symbol[result.symbol].append(result)
            completed += 1
            if completed % 20 == 0 or completed == len(jobs):
                print(f"downloaded {completed}/{len(jobs)}", flush=True)

    failures: list[str] = []
    for symbol, results in by_symbol.items():
        errors = [item for item in results if item.status.startswith("error:")]
        failures.extend(f"{item.symbol}/{item.month} {item.status}" for item in errors)
        ordered_rows: list[tuple[str, ...]] = []
        for item in sorted(results, key=lambda value: value.month):
            ordered_rows.extend(item.rows)
        ordered_rows.sort(key=lambda row: int(row[0]))
        deduplicated: list[tuple[str, ...]] = []
        last_ts = -1
        for row in ordered_rows:
            ts = int(row[0])
            if ts <= last_ts:
                continue
            deduplicated.append(row)
            last_ts = ts
        target = args.output / f"{symbol}-{args.interval}.csv.gz"
        fd, temporary_name = tempfile.mkstemp(prefix=target.name, dir=args.output)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(HEADER)
                writer.writerows(deduplicated)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"{symbol}: rows={len(deduplicated)} file={target}")

    if failures:
        print("FAILED DOWNLOADS:")
        print("\n".join(failures))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
