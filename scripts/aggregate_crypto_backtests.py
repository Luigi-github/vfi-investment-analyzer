from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_resumable_crypto_backtests import aggregate_outputs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/free_cloud_backtest")
    args = parser.parse_args()
    root = (ROOT / args.output_dir).resolve()
    manifest = root / "checkpoint.json"
    if manifest.exists():
        import json
        state = json.loads(manifest.read_text(encoding="utf-8"))
        symbols = state.get("symbols", [])
        state["completed"] = sorted({
            p.name for p in (root / "per_symbol").iterdir()
            if p.is_dir() and (p / "backtest" / "strategy_pnl_comparison.csv").exists()
        }) if (root / "per_symbol").exists() else []
        state["status"] = "completed" if len(state["completed"]) == len(symbols) else "partial"
        manifest.write_text(json.dumps(state, indent=2), encoding="utf-8")
    aggregate_outputs(root)
    print(f"Aggregated report files in {root}")

if __name__ == "__main__":
    main()
