from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.market_data import MarketDataService
from app.research.parameter_sensitivity import VFIParameterSensitivityAnalyzer
from app.research.universes import MARKET_UNIVERSES, all_symbols, get_universe


DEFAULT_GRID = {
    "length": [104, 130, 156],
    "coef": [0.15, 0.2, 0.25],
    "vcoef": [2.0, 2.5, 3.0],
    "signal_length": [5],
}


def parse_args(argv=None):
    parser = ArgumentParser(description="Walk-forward VFI parameter sensitivity.")
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
    parser.add_argument("--minimum-events", type=int, default=10)
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
        else ROOT / "reports" / "parameter_sensitivity" / args.universe.lower()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    market_data = MarketDataService()
    analyzer = VFIParameterSensitivityAnalyzer(
        horizons=(4, 8, 12, 26, 52),
        primary_horizon=26,
    )

    rows = []
    for symbol in symbols:
        print("=" * 80)
        print(f"PARAMETER SENSITIVITY {symbol}")
        print("=" * 80)
        try:
            daily = market_data.get_daily_data(symbol, period="max")
            weekly = market_data.convert_to_weekly(daily)
            result = analyzer.walk_forward(
                weekly,
                DEFAULT_GRID,
                initial_train_weeks=args.initial_train_weeks,
                test_weeks=args.test_weeks,
                step_weeks=args.step_weeks,
                minimum_events=args.minimum_events,
            )
            if not result.empty:
                result["Symbol"] = symbol
                rows.append(result)
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")

    if not rows:
        raise RuntimeError("No parameter sensitivity results were generated.")

    report = pd.concat(rows, ignore_index=True)
    report.to_csv(output_dir / "parameter_walk_forward.csv", index=False)

    ranking = (
        report.dropna(subset=["Test_Average_Return"])
        .groupby(["length", "coef", "vcoef", "signal_length"], as_index=False)
        .agg(
            Windows=("Window", "count"),
            Average_Test_Return=("Test_Average_Return", "mean"),
            Median_Test_Return=("Test_Average_Return", "median"),
            Positive_Window_Rate=(
                "Test_Average_Return",
                lambda values: (values > 0).mean() * 100.0,
            ),
        )
        .sort_values(
            ["Average_Test_Return", "Positive_Window_Rate", "Windows"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    ranking.to_csv(output_dir / "parameter_oos_ranking.csv", index=False)

    print("=" * 80)
    print("PARAMETER OOS RANKING")
    print("=" * 80)
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
