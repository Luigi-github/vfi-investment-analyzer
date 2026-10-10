from __future__ import annotations
"""Resumable, per-symbol Binance MTF research runner for a free cloud VM."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.data.binance_market_data import BinanceMarketDataService

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)

def discover_top_symbols(service: BinanceMarketDataService, limit: int, min_volume: float) -> list[str]:
    spot = service.discover_spot_usdt(min_volume)
    perp = service.discover_usdt_perpetuals(min_volume)
    spot_rank = {symbol: i for i, symbol in enumerate(spot)}
    perp_rank = {symbol: i for i, symbol in enumerate(perp)}
    common = set(spot_rank) & set(perp_rank)
    return sorted(common, key=lambda s: (spot_rank[s] + perp_rank[s], max(spot_rank[s], perp_rank[s]), s))[:limit]

def run_stage(args: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(args), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace")
        log.write(proc.stdout)
        log.flush()
        print(proc.stdout[-12000:], end="" if proc.stdout.endswith("\n") else "\n", flush=True)
        if proc.returncode:
            raise RuntimeError(f"Command failed ({proc.returncode}); see {log_path}")

def read_csvs(paths: list[Path], output: Path) -> None:
    frames = []
    for path in paths:
        if path.exists() and path.stat().st_size:
            try:
                frame = pd.read_csv(path)
                if not frame.empty:
                    frames.append(frame)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(output, index=False)

def csv_has_rows(path: Path) -> bool:
    try:
        return not pd.read_csv(path).empty
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False

def aggregate_outputs(root: Path) -> None:
    per_symbol = root / "per_symbol"
    patterns = {
        "crypto_mtf_ranking.csv": "*/mtf/crypto_mtf_ranking.csv",
        "derivatives_snapshot.csv": "*/mtf/derivatives_snapshot.csv",
        "strategy_pnl_comparison.csv": "*/backtest/strategy_pnl_comparison.csv",
        "walk_forward_oos.csv": "*/backtest/walk_forward_oos.csv",
        "oos_trades.csv": "*/backtest/oos_trades.csv",
        "walk_forward_summary_by_symbol.csv": "*/backtest/walk_forward_summary.csv",
    }
    for name, pattern in patterns.items():
        read_csvs(sorted(per_symbol.glob(pattern)), root / name)

def main() -> int:
    p = argparse.ArgumentParser(description="Resumable top-liquidity Binance MTF backtests.")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--min-volume", type=float, default=0,
                   help="Minimum 24h quote volume in USD for both spot and perpetual markets.")
    p.add_argument("--output-dir", default="reports/free_cloud_backtest")
    p.add_argument("--max-symbols", type=int, default=0, help="Optional smoke test; 0 means full universe.")
    p.add_argument("--pause", type=float, default=1.0)
    args = p.parse_args()
    output = (ROOT / args.output_dir).resolve()
    manifest_path = output / "checkpoint.json"
    output.mkdir(parents=True, exist_ok=True)
    service = BinanceMarketDataService()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        symbols = manifest["symbols"]
        print(f"Resuming saved universe: {len(symbols)} symbols; completed={len(manifest.get('completed', []))}", flush=True)
    else:
        symbols = discover_top_symbols(service, args.limit, args.min_volume)
        if args.max_symbols > 0:
            symbols = symbols[:args.max_symbols]
        manifest = {
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "selection_method": "intersection of Binance spot USDT and USD-M perpetual markets; sorted by combined 24h quote-volume ranks",
            "min_quote_volume_usd_each_market": args.min_volume,
            "requested_limit": args.limit, "symbols": symbols, "completed": [],
            "failed_symbol": None, "status": "running",
        }
        atomic_json(manifest_path, manifest)
        (output / "universe.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")
        print(f"Saved universe with {len(symbols)} symbols to {output / 'universe.txt'}", flush=True)
    completed = set(manifest.get("completed", []))
    for index, symbol in enumerate(symbols, start=1):
        if symbol in completed:
            print(f"[{index}/{len(symbols)}] SKIP {symbol}: already checkpointed", flush=True)
            continue
        print(f"\n[{index}/{len(symbols)}] START {symbol}", flush=True)
        symbol_root = output / "per_symbol" / symbol
        mtf_dir, backtest_dir = symbol_root / "mtf", symbol_root / "backtest"
        try:
            run_stage([sys.executable, "-u", "scripts/run_crypto_mtf_analysis.py",
                       "--symbols", symbol, "--min-volume", str(args.min_volume),
                       "--top", "1", "--output-dir", str(mtf_dir.relative_to(ROOT))],
                      symbol_root / "analysis.log")
            ranking_file = mtf_dir / "crypto_mtf_ranking.csv"
            if not csv_has_rows(ranking_file):
                raise RuntimeError(f"Ranking output is missing or empty for {symbol}")
            run_stage([sys.executable, "-u", "scripts/run_crypto_mtf_real_backtest.py",
                       "--symbols", symbol, "--output-dir", str(backtest_dir.relative_to(ROOT))],
                      symbol_root / "backtest.log")
            pnl_file = backtest_dir / "strategy_pnl_comparison.csv"
            if not csv_has_rows(pnl_file):
                raise RuntimeError(f"Backtest output is missing or empty for {symbol}")
            completed.add(symbol)
            manifest["completed"] = [s for s in symbols if s in completed]
            manifest["failed_symbol"] = None
            manifest["last_completed_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
            manifest["status"] = "running"
            atomic_json(manifest_path, manifest)
            aggregate_outputs(output)
            print(f"CHECKPOINT SAVED: {symbol} ({len(completed)}/{len(symbols)})", flush=True)
            time.sleep(max(0, args.pause))
        except Exception as exc:
            manifest["completed"] = [s for s in symbols if s in completed]
            manifest["failed_symbol"] = symbol
            manifest["last_error"] = repr(exc)
            manifest["status"] = "interrupted"
            atomic_json(manifest_path, manifest)
            aggregate_outputs(output)
            print(f"STOPPED at {symbol}; progress saved. Rerun the same command to resume. Error: {exc}", flush=True)
            return 1
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["failed_symbol"] = None
    atomic_json(manifest_path, manifest)
    aggregate_outputs(output)
    print(f"ALL DONE: {len(completed)}/{len(symbols)} symbols", flush=True)
    print(f"Aggregated reports: {output}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
