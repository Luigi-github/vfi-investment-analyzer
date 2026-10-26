"""Refresh the free local cache of Colombian equity daily OHLCV data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.market_data_store import ColombiaMarketDataStore
from app.research.universes import get_universe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols; defaults to Colombia universe, excluding US ADR CIB.")
    parser.add_argument("--data-dir", default="data/market_cache/colombia")
    parser.add_argument("--max-stale-days", type=int, default=7)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = (
        [item.strip() for item in args.symbols.split(",") if item.strip()]
        if args.symbols
        else [symbol for symbol in get_universe("COLOMBIA") if symbol != "CIB"]
    )
    store = ColombiaMarketDataStore(args.data_dir)
    audit, _quotes = store.refresh_many(
        symbols, max_stale_calendar_days=args.max_stale_days,
        sleep_seconds=args.sleep_seconds,
    )
    valid_fresh = (audit["Valid"] == True) & (audit["Fresh"] == True)
    problems = (~valid_fresh) | (audit["Error"] != "")
    print(f"Solicitados: {len(audit)}")
    print(f"Válidos y recientes: {int(valid_fresh.sum())}")
    print(f"Con problemas: {int(problems.sum())}")
    print(f"Auditoría: {store.audit_path}")
    print(f"Últimas cotizaciones: {store.quotes_path}")
    print(audit[["Symbol", "Last_Date", "Status", "Age_Calendar_Days", "Issues", "Error"]].to_string(index=False))
    return 1 if problems.any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
