from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.vfi_analyzer import VFIAnalyzer
from app.analysis.vfi_cycle import VFICycleAnalyzer
from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.scoring.vfi_score import VFIScore
from app.backtest.multi_asset_cycle_analysis import MultiAssetCycleAnalyzer
from app.backtest.multi_asset_recovery_analysis import MultiAssetRecoveryAnalyzer
from app.backtest.regime_analysis import RegimeAnalyzer
from app.scoring.research_ranking import ResearchRankingAnalyzer
from app.research.universes import MARKET_UNIVERSES, all_symbols, get_universe
from app.features.vfi_normalization import add_vfi_normalized_features
from app.research.reproducibility import build_run_metadata, write_run_metadata, RESEARCH_VERSION, VFI_FORMULA_VERSION, CYCLE_DEFINITION_VERSION, STAGE_DEFINITION_VERSION


DEFAULT_UNIVERSE = "ALL"
OUTPUT_ROOT = ROOT / "reports" / "cycle_analysis"


def parse_args(argv=None):
    parser = ArgumentParser(description="Run causal VFI cycle research.")
    parser.add_argument(
        "--universe",
        choices=[*MARKET_UNIVERSES.keys(), "ALL"],
        default=DEFAULT_UNIVERSE,
        help="Configured research universe to analyze.",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated yfinance symbols. Overrides --universe.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional report directory. Defaults to reports/cycle_analysis/<universe>.",
    )
    return parser.parse_args(argv)


def resolve_symbols(universe=DEFAULT_UNIVERSE, symbols=None):
    if symbols:
        resolved = [item.strip() for item in symbols.split(",") if item.strip()]
        if not resolved:
            raise ValueError("--symbols must contain at least one symbol.")
        return list(dict.fromkeys(resolved))

    if universe == "ALL":
        return all_symbols()

    return get_universe(universe)


def build_output_dir(universe, output_dir=None):
    if output_dir:
        return Path(output_dir)
    return OUTPUT_ROOT / universe.lower()


def prepare_asset(
    symbol,
    market_data_service,
    vfi_indicator,
    vfi_analyzer,
    scorer,
    cycle_analyzer,
):
    daily = market_data_service.get_daily_data(symbol, period="max")
    weekly = market_data_service.convert_to_weekly(daily)
    vfi = vfi_indicator.calculate(weekly)
    analyzed = vfi_analyzer.analyze(vfi)
    scored = scorer.calculate(analyzed)
    normalized = add_vfi_normalized_features(scored)
    return cycle_analyzer.analyze(normalized), len(daily)


def build_asset_event_summary(events, primary_horizon):
    if events.empty:
        return pd.DataFrame()

    column = f"Return_{primary_horizon}W"
    if column not in events.columns:
        return pd.DataFrame()

    valid = events.dropna(subset=[column]).copy()
    if valid.empty:
        return pd.DataFrame()

    return (
        valid.groupby("Symbol")
        .agg(
            Events=(column, "count"),
            Average_Return=(column, "mean"),
            Median_Return=(column, "median"),
            Positive_Rate=(column, lambda values: values.gt(0).mean() * 100),
            Worst_Return=(column, "min"),
            Best_Return=(column, "max"),
        )
        .reset_index()
        .sort_values(
            ["Average_Return", "Positive_Rate", "Events"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def build_cost_sensitivity(asset_data, cost_bps_levels=(0, 10, 25, 50, 100)):
    """Evaluate OOS condition discovery under a range of round-trip cost assumptions."""
    recovery = MultiAssetRecoveryAnalyzer(
        horizons=(4, 8, 12, 26, 52),
        primary_horizon=26,
    )
    causal_events = {}
    for symbol, data in asset_data.items():
        events = recovery.analyze_asset_causal(symbol, data)
        if not events.empty:
            causal_events[symbol] = events

    rows = []
    for cost_bps in cost_bps_levels:
        discovery = recovery.analyze_condition_discovery_walk_forward(
            causal_events,
            initial_train_weeks=104,
            test_weeks=52,
            step_weeks=52,
            minimum_events=10,
            top_n=1,
            ranking_metric="Average_Net_Return",
            fee_bps=cost_bps / 4.0,
            slippage_bps=cost_bps / 4.0,
        )
        summary = discovery["summary"]
        if summary.empty:
            continue

        valid = summary["Test_Average_Net_Return"].dropna()
        if valid.empty:
            continue

        rows.append({
            "Round_Trip_Cost_Bps": cost_bps,
            "Assets": summary["Symbol"].nunique(),
            "Valid_Windows": int(valid.size),
            "Average_Test_Net_Return": float(valid.mean()),
            "Median_Test_Net_Return": float(valid.median()),
            "Positive_Window_Rate": float((valid > 0).mean() * 100.0),
            "Worst_Window_Return": float(valid.min()),
            "Best_Window_Return": float(valid.max()),
        })

    return pd.DataFrame(rows)


def build_oos_research_outputs(asset_data):
    """Build causal recovery walk-forward summaries and transparent OOS ranking."""
    recovery = MultiAssetRecoveryAnalyzer(
        horizons=(4, 8, 12, 26, 52),
        primary_horizon=26,
    )
    causal_events = {}
    for symbol, data in asset_data.items():
        events = recovery.analyze_asset_causal(symbol, data)
        if not events.empty:
            causal_events[symbol] = events

    if not causal_events:
        return {
            "walk_forward": pd.DataFrame(),
            "ranking": pd.DataFrame(),
            "condition_discovery": pd.DataFrame(),
        }

    walk_forward = recovery.analyze_assets_walk_forward(
        causal_events,
        initial_train_weeks=104,
        test_weeks=52,
        step_weeks=52,
    )
    ranking_input = recovery.summarize_walk_forward_assets(walk_forward)
    ranking = ResearchRankingAnalyzer().rank(ranking_input)

    discovery = recovery.analyze_condition_discovery_walk_forward(
        causal_events,
        initial_train_weeks=104,
        test_weeks=52,
        step_weeks=52,
        minimum_events=10,
        top_n=1,
        ranking_metric="Average_Return",
    )

    condition_discovery = discovery["summary"]
    return {
        "walk_forward": walk_forward,
        "ranking": ranking,
        "condition_discovery": condition_discovery,
    }


def main(argv=None):
    args = parse_args(argv)
    symbols = resolve_symbols(args.universe, args.symbols)
    output_dir = build_output_dir(args.universe, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Universe: {args.universe}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Output: {output_dir}")

    market_data_service = MarketDataService()
    vfi_indicator = VFIIndicator()
    vfi_analyzer = VFIAnalyzer()
    scorer = VFIScore()
    cycle_stage_analyzer = VFICycleAnalyzer()
    cycle_analyzer = MultiAssetCycleAnalyzer(
        horizons=(4, 8, 12, 26, 52),
        primary_horizon=26,
    )

    asset_data = {}
    metadata = []

    for symbol in symbols:
        print("=" * 80)
        print(f"ANALYZING {symbol}")
        print("=" * 80)
        try:
            data, daily_rows = prepare_asset(
                symbol,
                market_data_service,
                vfi_indicator,
                vfi_analyzer,
                scorer,
                cycle_stage_analyzer,
            )
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")
            metadata.append(
                {
                    "Symbol": symbol,
                    "Status": "ERROR",
                    "Instrument_Type": "ADR" if symbol == "CIB" else "EQUITY",
                    "Error": str(exc),
                }
            )
            continue

        asset_data[symbol] = data
        valid = data[data["VFI"].notna()]
        run_metadata = build_run_metadata(
            symbol=symbol,
            provider=market_data_service.PROVIDER,
            retrieved_at=market_data_service.get_last_metadata().get("Retrieved_At_UTC"),
            data=data,
            vfi_parameters={
                "length": vfi_indicator.length,
                "coef": vfi_indicator.coef,
                "vcoef": vfi_indicator.vcoef,
                "signal_length": vfi_indicator.signal_length,
                "smooth_vfi": vfi_indicator.smooth_vfi,
                "context_window": vfi_indicator.context_window,
            },
            extra={
                "Research_Version": RESEARCH_VERSION,
                "VFI_Formula_Version": VFI_FORMULA_VERSION,
                "Cycle_Definition_Version": CYCLE_DEFINITION_VERSION,
                "Stage_Definition_Version": STAGE_DEFINITION_VERSION,
                "Daily_Rows": daily_rows,
                "Weekly_Rows": len(data),
            },
        )
        metadata_path = output_dir / "metadata"
        metadata_path.mkdir(parents=True, exist_ok=True)
        write_run_metadata(
            metadata_path / f"{symbol.replace('-', '_')}.json",
            run_metadata,
        )
        metadata.append(
            {
                "Symbol": symbol,
                "Status": "OK",
                "Instrument_Type": "ADR" if symbol == "CIB" else "EQUITY",
                "Daily_Rows": daily_rows,
                "Weekly_Rows": len(data),
                "Valid_VFI_Rows": len(valid),
                "Latest_Date": valid.index[-1] if not valid.empty else None,
                "Latest_VFI": valid.iloc[-1]["VFI"] if not valid.empty else None,
                "Latest_Cycle_Stage": (
                    valid.iloc[-1]["VFI_Cycle_Stage"] if not valid.empty else None
                ),
            }
        )

    if not asset_data:
        raise RuntimeError("No asset could be analyzed successfully.")

    result = cycle_analyzer.analyze(asset_data)

    benchmark_daily = market_data_service.get_daily_data("SPY", period="max")
    benchmark_weekly = market_data_service.convert_to_weekly(benchmark_daily)
    regime_analyzer = RegimeAnalyzer(lookback_weeks=26)
    result["events"] = regime_analyzer.label_events(
        result["events"], benchmark_weekly
    )
    result["regime_summary"] = regime_analyzer.summarize(
        result["events"], f"Return_{cycle_analyzer.primary_horizon}W"
    )
    result["asset_event_summary"] = build_asset_event_summary(
        result["events"], cycle_analyzer.primary_horizon
    )

    result["cycles"].to_csv(output_dir / "cycles.csv", index=False)
    result["events"].to_csv(output_dir / "stage_entries.csv")
    result["quality_summary"].to_csv(output_dir / "quality_summary.csv", index=False)
    result["stage_summary"].to_csv(output_dir / "stage_summary.csv", index=False)
    result["stage_statistical_summary"].to_csv(
        output_dir / "stage_statistical_summary.csv", index=False
    )
    result["asset_summary"].to_csv(output_dir / "asset_summary.csv", index=False)
    result["asset_event_summary"].to_csv(
        output_dir / "asset_event_summary.csv", index=False
    )

    oos = build_oos_research_outputs(asset_data)
    cost_sensitivity = build_cost_sensitivity(asset_data)
    oos["walk_forward"].to_csv(
        output_dir / "asset_walk_forward.csv", index=False
    )
    oos["ranking"].to_csv(
        output_dir / "asset_oos_ranking.csv", index=False
    )
    oos["condition_discovery"].to_csv(
        output_dir / "condition_discovery_walk_forward.csv", index=False
    )
    cost_sensitivity.to_csv(
        output_dir / "cost_sensitivity.csv", index=False
    )

    pd.DataFrame(metadata).to_csv(output_dir / "asset_metadata.csv", index=False)

    print("=" * 80)
    print("ASSET EVENT SUMMARY")
    print("=" * 80)
    print(result["asset_event_summary"].to_string(index=False))

    print("=" * 80)
    print("OUT-OF-SAMPLE ASSET RANKING")
    print("=" * 80)
    print(oos["ranking"].to_string(index=False))

    print("=" * 80)
    print("CONDITION DISCOVERY WALK-FORWARD")
    print("=" * 80)
    print(oos["condition_discovery"].to_string(index=False))
    print("=" * 80)
    print("COST SENSITIVITY")
    print("=" * 80)
    print(cost_sensitivity.to_string(index=False))

    print("=" * 80)
    print("STAGE STATISTICAL SUMMARY")
    print("=" * 80)
    print(result["stage_statistical_summary"].to_string(index=False))

    print("=" * 80)
    print("REGIME SUMMARY")
    print("=" * 80)
    print(result["regime_summary"].to_string(index=False))

    print("Reports written to", output_dir)


if __name__ == "__main__":
    main()
