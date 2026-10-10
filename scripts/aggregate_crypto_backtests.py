from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_resumable_crypto_backtests import aggregate_outputs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/free_cloud_backtest")
    args = parser.parse_args()
    root = (ROOT / args.output_dir).resolve()
    manifest = root / "checkpoint.json"
    aggregate_outputs(root)
    if manifest.exists():
        import json
        state = json.loads(manifest.read_text(encoding="utf-8"))
        symbols = state.get("symbols", [])
        status_file = root / "backtest_status_by_symbol.csv"
        status_map = {}
        if status_file.exists() and status_file.stat().st_size:
            statuses = pd.read_csv(status_file)
            if {"Symbol", "Status"}.issubset(statuses.columns):
                status_map = dict(zip(statuses["Symbol"].astype(str), statuses["Status"].astype(str)))
        state["completed"] = sorted(s for s in symbols if status_map.get(s) == "completed")
        state["no_oos_trades"] = sorted(s for s in symbols if status_map.get(s) == "no_oos_trades")
        state["insufficient_history"] = sorted(s for s in symbols if status_map.get(s) == "insufficient_history")
        state["failed_symbols"] = sorted(s for s in symbols if status_map.get(s) == "error")
        accepted = {"completed", "no_oos_trades"}
        state["failed_or_missing"] = sorted(s for s in symbols if status_map.get(s) not in accepted)
        if state["failed_or_missing"]:
            state["status"] = "partial"
        elif state["no_oos_trades"]:
            state["status"] = "completed_with_warnings"
        else:
            state["status"] = "completed"
        manifest.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Aggregated report files in {root}")

if __name__ == "__main__":
    main()
