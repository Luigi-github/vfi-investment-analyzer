from pathlib import Path

import pandas as pd

from app.analysis.vfi_analyzer import VFIAnalyzer
from app.analysis.vfi_cycle import VFICycleAnalyzer
from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.scoring.vfi_score import VFIScore
from app.backtest.multi_asset_cycle_analysis import MultiAssetCycleAnalyzer


SYMBOLS = ["DDD", "INTC", "NVDA", "BTC-USD", "SQM"]
OUTPUT_DIR = Path("reports/cycle_analysis")


def prepare_asset(symbol, market_data_service, vfi_indicator, vfi_analyzer, scorer, cycle_analyzer):
    daily = market_data_service.get_daily_data(symbol, period="max")
    weekly = market_data_service.convert_to_weekly(daily)
    vfi = vfi_indicator.calculate(weekly)
    analyzed = vfi_analyzer.analyze(vfi)
    scored = scorer.calculate(analyzed)
    return cycle_analyzer.analyze(scored)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    for symbol in SYMBOLS:
        print("=" * 80)
        print(f"ANALYZING {symbol}")
        print("=" * 80)
        data = prepare_asset(
            symbol,
            market_data_service,
            vfi_indicator,
            vfi_analyzer,
            scorer,
            cycle_stage_analyzer,
        )
        asset_data[symbol] = data
        valid = data[data["VFI"].notna()]
        metadata.append(
            {
                "Symbol": symbol,
                "Daily_Rows": len(
                    market_data_service.get_daily_data(symbol, period="max")
                ),
                "Weekly_Rows": len(data),
                "Valid_VFI_Rows": len(valid),
                "Latest_Date": valid.index[-1] if not valid.empty else None,
                "Latest_VFI": valid.iloc[-1]["VFI"] if not valid.empty else None,
                "Latest_Cycle_Stage": (
                    valid.iloc[-1]["VFI_Cycle_Stage"] if not valid.empty else None
                ),
            }
        )

    result = cycle_analyzer.analyze(asset_data)

    result["cycles"].to_csv(OUTPUT_DIR / "cycles.csv", index=False)
    result["events"].to_csv(OUTPUT_DIR / "stage_entries.csv")
    result["quality_summary"].to_csv(OUTPUT_DIR / "quality_summary.csv", index=False)
    result["stage_summary"].to_csv(OUTPUT_DIR / "stage_summary.csv", index=False)
    result["asset_summary"].to_csv(OUTPUT_DIR / "asset_summary.csv", index=False)
    pd.DataFrame(metadata).to_csv(OUTPUT_DIR / "asset_metadata.csv", index=False)

    print("\n" + "=" * 80)
    print("STAGE SUMMARY")
    print("=" * 80)
    print(result["stage_summary"].to_string(index=False))

    print("\n" + "=" * 80)
    print("ASSET SUMMARY")
    print("=" * 80)
    print(result["asset_summary"].to_string(index=False))

    print("\n" + "=" * 80)
    print("QUALITY SUMMARY")
    print("=" * 80)
    print(result["quality_summary"].to_string(index=False))

    print("\nReports written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
