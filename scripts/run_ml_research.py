from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.vfi_analyzer import VFIAnalyzer
from app.analysis.vfi_cycle import VFICycleAnalyzer
from app.backtest.multi_asset_recovery_analysis import MultiAssetRecoveryAnalyzer
from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.research.ml_event_analyzer import VFIEventMLAnalyzer
from app.scoring.vfi_score import VFIScore
from app.research.universes import MARKET_UNIVERSES, all_symbols, get_universe


def parse_args(argv=None):
    parser = ArgumentParser(description="Run OOS ML research on VFI recovery events.")
    parser.add_argument(
        "--universe",
        choices=[*MARKET_UNIVERSES.keys(), "ALL"],
        default="USA_STOCKS",
    )
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--initial-train-weeks", type=int, default=260)
    parser.add_argument("--test-weeks", type=int, default=52)
    parser.add_argument("--step-weeks", type=int, default=52)
    parser.add_argument("--minimum-train-events", type=int, default=50)
    return parser.parse_args(argv)


def resolve_symbols(universe, symbols):
    if symbols:
        return list(dict.fromkeys(
            item.strip() for item in symbols.split(",") if item.strip()
        ))
    return all_symbols() if universe == "ALL" else get_universe(universe)


def main(argv=None):
    args = parse_args(argv)
    symbols = resolve_symbols(args.universe, args.symbols)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "reports" / "ml_research" / args.universe.lower()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    market = MarketDataService()
    indicator = VFIIndicator()
    analyzer = VFIAnalyzer()
    scorer = VFIScore()
    cycle_stage = VFICycleAnalyzer()
    recovery = MultiAssetRecoveryAnalyzer(
        horizons=(4, 8, 12, 26, 52),
        primary_horizon=26,
    )

    event_frames = []
    for symbol in symbols:
        print("=" * 80)
        print(f"ML RESEARCH {symbol}")
        print("=" * 80)
        try:
            daily = market.get_daily_data(symbol, period="max")
            weekly = market.convert_to_weekly(daily)
            data = indicator.calculate(weekly)
            data = analyzer.analyze(data)
            data = scorer.calculate(data)
            data = cycle_stage.analyze(data)
            events = recovery.analyze_asset_causal(symbol, data)
            if not events.empty:
                events = events.copy()
                events["Symbol"] = symbol
                event_frames.append(events)
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")

    if not event_frames:
        raise RuntimeError("No causal recovery events were generated.")

    events = pd.concat(event_frames).sort_index()
    model = VFIEventMLAnalyzer(
        primary_horizon=26,
        minimum_train_events=args.minimum_train_events,
    )
    summary, predictions = model.walk_forward(
        events,
        initial_train_weeks=args.initial_train_weeks,
        test_weeks=args.test_weeks,
        step_weeks=args.step_weeks,
    )

    summary.to_csv(output_dir / "ml_walk_forward.csv", index=False)
    predictions.to_csv(output_dir / "ml_predictions.csv", index=False)
    model.summarize(summary).to_csv(
        output_dir / "ml_summary.csv",
        index=False,
    )

    print("=" * 80)
    print("ML WALK-FORWARD SUMMARY")
    print("=" * 80)
    print(model.summarize(summary).to_string(index=False))


if __name__ == "__main__":
    main()
