"""Rank early accumulation using weekly VFI and monthly Volume Divergence.
EMA 5/20/80/200 are weighted toward slower periods; monthly bars are completed only.
The score is a research ranking, not a probability of profit or a buy signal.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM
from scripts.study_early_vfi_accumulation import (
    MIN_WEEKLY_BASELINE,
    evaluate_combo,
    build_asset_summary,
)

DEFAULT_SYMBOLS = (
    "DDD,INTC,NVDA,BTC-USD,EC,CIB,MSFT,EEM,SQM,LTC-USD,AU,SPY,AMZN,PBR,NU,QQQ,"
    "ISA.CL,GEB.CL,MINEROS.CL,PFAVAL.CL,PFDAVVNDA.CL,TERPEL.CL"
).split(",")
EMA_SHORT, EMA_MEDIUM, EMA_LONG = 5, 20, 80
EMA_VERY_LONG = 200
MIN_MONTHLY_BASELINE = 36
VFI_LENGTH = 130
OUT = ROOT / "reports" / "early_vfi_accumulation"


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args(argv)
    symbols = list(dict.fromkeys(s.strip() for s in args.symbols.split(",") if s.strip()))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    market = MarketDataService()
    rows, errors = [], []

    for symbol in symbols:
        print("=" * 72)
        print(f"EARLY ACCUMULATION SCAN (EMA {EMA_SHORT}/{EMA_MEDIUM}/{EMA_LONG}): {symbol}", flush=True)
        try:
            daily = market.get_daily_data(symbol, period="max").sort_index()
            if daily.empty:
                raise ValueError("Market data provider returned no rows.")
            # Exclude today's potentially incomplete daily candle.
            daily = daily[daily.index.normalize() < pd.Timestamp.now().normalize()]
            if daily.empty:
                raise ValueError("No completed daily candles remain.")
            weekly_ohlcv = market.convert_to_weekly(daily).sort_index()
            weekly = VFIIndicator(length=VFI_LENGTH).calculate(weekly_ohlcv)
            weekly = weekly.replace([np.inf, -np.inf], np.nan).dropna(subset=["VFI"])
            monthly_ohlcv = daily.resample("ME").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum",
            }).dropna(subset=["Open", "High", "Low", "Close"])
            current_month = pd.Timestamp.now().to_period("M")
            monthly_ohlcv = monthly_ohlcv[monthly_ohlcv.index.to_period("M") < current_month]
            monthly_div = VolumeDivergenceMM().calculate(monthly_ohlcv)
            monthly_div = monthly_div.replace([np.inf, -np.inf], np.nan).dropna(subset=["Volume_Divergence_MM"])
            if weekly.empty or monthly_div.empty:
                raise ValueError("No valid data after VFI/monthly-divergence warm-up.")
            minimum_valid_monthly = MIN_MONTHLY_BASELINE  # EMA 200 is optional when the asset has shorter history
            if len(monthly_div) < minimum_valid_monthly:
                raise ValueError(f"Insufficient monthly divergence baseline: {len(monthly_div)} bars; need {minimum_valid_monthly}. EMA 200 will be treated as unavailable when history is shorter than 200 months.")
            minimum_valid_weekly = MIN_WEEKLY_BASELINE + EMA_LONG + 1
            if len(weekly) < minimum_valid_weekly:
                raise ValueError(
                    f"Insufficient valid weekly VFI history: {len(weekly)} bars; "
                    f"need at least {minimum_valid_weekly} after VFI warm-up."
                )
            rows.append(evaluate_combo(
                symbol, weekly, monthly_div, EMA_SHORT, EMA_MEDIUM, EMA_LONG
            ))
            print(
                f"Weekly VFI bars: {len(weekly)}; completed monthly divergence bars: {len(monthly_div)}; "
                f"monthly data through {monthly_div.index[-1].date()}", flush=True
            )
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc)})
            print(f"SKIPPED {symbol}: {exc}", flush=True)

    if rows:
        all_combos = pd.DataFrame(rows)
        all_combos.to_csv(output / "ema_combo_scan.csv", index=False)
        summary = build_asset_summary(all_combos)
        summary.to_csv(output / "early_vfi_accumulation_ranking.csv", index=False)
        summary[summary["Eligible"]].to_csv(output / "qualified_accumulation_candidates.csv", index=False)
        summary[~summary["Eligible"]].to_csv(output / "diagnostic_candidates_not_qualified.csv", index=False)
    else:
        summary = pd.DataFrame()
        for filename in (
            "ema_combo_scan.csv",
            "early_vfi_accumulation_ranking.csv",
            "qualified_accumulation_candidates.csv",
            "diagnostic_candidates_not_qualified.csv",
        ):
            pd.DataFrame().to_csv(output / filename, index=False)

    pd.DataFrame(errors, columns=["Symbol", "Error"]).to_csv(
        output / "early_vfi_accumulation_errors.csv", index=False
    )
    print(f"Reports written to: {output}")
    print(f"Indicator EMA configuration: {EMA_SHORT}/{EMA_MEDIUM}/{EMA_LONG}/{EMA_VERY_LONG}; weights 5%/15%/30%/50%")
    print("Divergence timeframe: completed monthly candles; current month excluded.")
    print(f"Assets ranked: {len(rows)}; qualified candidates: {int(summary['Eligible'].sum()) if rows else 0}")
    print("All assets are ranked by score; pass/fail columns are diagnostics only.")
    print("Score is comparative, not a probability of profit.")


if __name__ == "__main__":
    main()
