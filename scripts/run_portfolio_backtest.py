from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backtest.portfolio_backtest import PortfolioBacktester
from app.data.market_data import MarketDataService
from app.analysis.vfi_analyzer import VFIAnalyzer
from app.analysis.vfi_cycle import VFICycleAnalyzer
from app.indicators.vfi import VFIIndicator
from app.scoring.vfi_score import VFIScore
from app.backtest.multi_asset_recovery_analysis import MultiAssetRecoveryAnalyzer
from app.research.universes import MARKET_UNIVERSES, all_symbols, get_universe


def parse_args(argv=None):
    parser = ArgumentParser(description="Run causal VFI portfolio backtests.")
    parser.add_argument(
        "--universe",
        choices=[*MARKET_UNIVERSES.keys(), "ALL"],
        default="ALL",
    )
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--horizon-weeks", type=int, default=26)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--max-position-weight", type=float, default=None)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    return parser.parse_args(argv)


def resolve_symbols(universe, symbols):
    if symbols:
        return list(dict.fromkeys(
            item.strip() for item in symbols.split(",") if item.strip()
        ))
    return all_symbols() if universe == "ALL" else get_universe(universe)


def summarize_equity_curve(curve):
    if curve.empty:
        return {
            "Periods": 0,
            "Total_Return": np.nan,
            "CAGR": np.nan,
            "Annualized_Volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Max_Drawdown": np.nan,
            "Positive_Period_Rate": np.nan,
            "Average_Turnover": np.nan,
        }
    return PortfolioBacktester._summary(
        curve["Net_Return"] / 100.0,
        curve,
    )


def main(argv=None):
    args = parse_args(argv)
    symbols = resolve_symbols(args.universe, args.symbols)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "reports" / "portfolio_backtest" / args.universe.lower()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    market_data = MarketDataService()
    vfi_indicator = VFIIndicator()
    vfi_analyzer = VFIAnalyzer()
    scorer = VFIScore()
    cycle_analyzer = VFICycleAnalyzer()
    recovery = MultiAssetRecoveryAnalyzer(
        horizons=(4, 8, 12, 26, 52),
        primary_horizon=args.horizon_weeks,
    )

    asset_data = {}
    prices = {}
    signal_frames = []

    for symbol in symbols:
        print("=" * 80)
        print(f"ANALYZING {symbol}")
        print("=" * 80)
        try:
            daily = market_data.get_daily_data(symbol, period="max")
            weekly = market_data.convert_to_weekly(daily)
            vfi = vfi_indicator.calculate(weekly)
            analyzed = vfi_analyzer.analyze(vfi)
            scored = scorer.calculate(analyzed)
            data = cycle_analyzer.analyze(scored)

            events = recovery.analyze_asset_causal(symbol, data)
            if not events.empty:
                signal_frames.append(events[["Symbol"]])

            asset_data[symbol] = data
            prices[symbol] = data[["Close"]].copy()
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")

    if not signal_frames:
        raise RuntimeError("No causal recovery signals were generated.")

    signals = pd.concat(signal_frames).sort_index()

    all_summaries = []
    for weighting in ("equal", "inverse_vol", "risk_parity"):
        backtester = PortfolioBacktester(
            horizon_weeks=args.horizon_weeks,
            max_positions=args.max_positions,
            weighting=weighting,
            max_position_weight=args.max_position_weight,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )
        result = backtester.run(prices, signals)

        curve = result["equity_curve"].copy()
        trades = result["trades"].copy()
        summary = result["summary"].copy()
        summary["Weighting"] = weighting
        summary["Universe"] = args.universe
        summary["Horizon_Weeks"] = args.horizon_weeks
        summary["Max_Positions"] = args.max_positions
        summary["Fee_Bps"] = args.fee_bps
        summary["Slippage_Bps"] = args.slippage_bps
        all_summaries.append(summary)

        curve.to_csv(output_dir / f"equity_curve_{weighting}.csv", index=False)
        trades.to_csv(output_dir / f"trades_{weighting}.csv", index=False)

    summary_frame = pd.DataFrame(all_summaries)
    summary_frame.to_csv(output_dir / "portfolio_summary.csv", index=False)

    signals.reset_index(names="Date").to_csv(
        output_dir / "portfolio_signals.csv", index=False
    )

    print("=" * 80)
    print("PORTFOLIO BACKTEST SUMMARY")
    print("=" * 80)
    print(summary_frame.to_string(index=False))
    print("=" * 80)
    print(f"Causal signals: {len(signals)}")
    print(f"Reports written to {output_dir}")


if __name__ == "__main__":
    main()
