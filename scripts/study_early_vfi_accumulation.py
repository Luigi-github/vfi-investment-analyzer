"""Scan early VFI accumulation: weekly EMAs on VFI(130), then daily Volume Divergence by MM.

The output ranks assets by the current weekly setup, the uninterrupted run of
negative VFI with positive weekly slope, and a recent confirmed daily bullish
volume divergence. EMA spans are scanned over a fixed candidate grid; this is
a screening study, not a return-maximizing strategy or a forecast.
"""
from itertools import product
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

DEFAULT_SYMBOLS = tuple(
    "DDD,INTC,NVDA,BTC-USD,EC,CIB,MSFT,EEM,SQM,LTC-USD,AU,SPY,AMZN,PBR,NU,QQQ,ISA.CL,GEB.CL,MINEROS.CL,PFAVAL.CL,PFDAVVNDA.CL,TERPEL.CL".split(",")
)
SHORTS = (3, 5, 7, 9)
MEDIUMS = (15, 20, 25, 30)
LONGS = (40, 50, 60, 80)
VFI_LENGTH = 130
NEAR_ZERO = 5.0
DIVERGENCE_FRESH_DAYS = 5
OUT = ROOT / "reports" / "early_vfi_accumulation"


def ema_combinations():
    return [(s, m, l) for s, m, l in product(SHORTS, MEDIUMS, LONGS) if s < m < l]


def positive_negative_streak(vfi: pd.Series) -> int:
    """Count consecutive closed weekly bars ending now with VFI<0 and 1-bar slope>0."""
    slope = vfi.diff()
    ok = (vfi < 0) & (slope > 0)
    count = 0
    for flag in ok.iloc[::-1]:
        if not bool(flag):
            break
        count += 1
    return count


def weekly_features(weekly: pd.DataFrame) -> pd.DataFrame:
    result = VFIIndicator(length=VFI_LENGTH).calculate(weekly.sort_index())
    result["VFI_Slope_1W"] = result["VFI"].diff()
    return result


def evaluate_combo(symbol, weekly, daily, short, medium, long):
    w = weekly.copy()
    v = w["VFI"]
    w["EMA_short"] = v.ewm(span=short, adjust=False, min_periods=short).mean()
    w["EMA_medium"] = v.ewm(span=medium, adjust=False, min_periods=medium).mean()
    w["EMA_long"] = v.ewm(span=long, adjust=False, min_periods=long).mean()
    w["VFI_Slope_4W"] = (v - v.shift(4)) / 4.0

    latest = w.dropna(subset=["VFI", "EMA_short", "EMA_medium", "EMA_long"]).iloc[-1]
    # EMA lines are calculated from VFI, not from the asset price.
    ema_structure = (
        latest["EMA_short"] < 0
        and latest["EMA_medium"] < 0
        and latest["EMA_short"] > latest["EMA_long"]
        and latest["EMA_medium"] > latest["EMA_long"]
    )
    vfi_negative = latest["VFI"] < 0
    vfi_positive_slope = latest["VFI_Slope_1W"] > 0
    near_zero = abs(float(latest["VFI"])) <= NEAR_ZERO
    weekly_pass = bool(ema_structure and vfi_negative and vfi_positive_slope and near_zero)

    streak = positive_negative_streak(w["VFI"])
    recent = daily.tail(DIVERGENCE_FRESH_DAYS)
    bullish = bool(recent["Volume_Div_Regular_Bull"].fillna(False).astype(bool).any())
    bearish = bool(recent["Volume_Div_Regular_Bear"].fillna(False).astype(bool).any())
    last_bull_date = ""
    if bullish:
        last_bull_date = str(recent.index[recent["Volume_Div_Regular_Bull"].fillna(False).astype(bool)][-1].date())

    return {
        "Symbol": symbol,
        "EMA_short": short,
        "EMA_medium": medium,
        "EMA_long": long,
        "Weekly_Date": str(w.index[-1].date()),
        "VFI_130": float(latest["VFI"]),
        "VFI_Slope_1W": float(latest["VFI_Slope_1W"]),
        "VFI_Slope_4W": float(latest["VFI_Slope_4W"]) if pd.notna(latest["VFI_Slope_4W"]) else np.nan,
        "EMA_short_VFI": float(latest["EMA_short"]),
        "EMA_medium_VFI": float(latest["EMA_medium"]),
        "EMA_long_VFI": float(latest["EMA_long"]),
        "EMA_Structure_Pass": bool(ema_structure),
        "VFI_Negative": bool(vfi_negative),
        "VFI_Positive_Slope": bool(vfi_positive_slope),
        "VFI_Near_Zero": bool(near_zero),
        "Negative_Positive_Slope_Streak_Weeks": int(streak),
        "Weekly_Filter_Pass": weekly_pass,
        "Daily_Bullish_Divergence_Recent": bullish,
        "Daily_Bearish_Divergence_Recent": bearish,
        "Last_Daily_Bullish_Divergence": last_bull_date,
        "Final_Filter_Pass": bool(weekly_pass and bullish and not bearish),
    }


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
        print(f"EARLY VFI ACCUMULATION: {symbol}")
        try:
            daily = market.get_daily_data(symbol, period="max")
            weekly_ohlcv = market.convert_to_weekly(daily)
            weekly = weekly_features(weekly_ohlcv).dropna(subset=["VFI"])
            daily_div = VolumeDivergenceMM().calculate(daily.sort_index())
            daily_div = daily_div.replace([np.inf, -np.inf], np.nan).dropna(subset=["Volume_Divergence_MM"])
            if weekly.empty or daily_div.empty:
                raise ValueError("No valid data after indicator warm-up.")
            for short, medium, long in ema_combinations():
                rows.append(evaluate_combo(symbol, weekly, daily_div, short, medium, long))
            print(f"Weekly bars after VFI warm-up: {len(weekly)}; daily bars for divergence: {len(daily_div)}; as of {weekly.index[-1].date()}")
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc)})
            print(f"SKIPPED {symbol}: {exc}")

    if rows:
        all_combos = pd.DataFrame(rows)
        all_combos.to_csv(output / "ema_combo_scan.csv", index=False)
        # Per-asset summary includes how robust the weekly filter is across candidate EMA spans.
        summaries = []
        for symbol, group in all_combos.groupby("Symbol", sort=False):
            qualified = group[group["Weekly_Filter_Pass"]]
            final = group[group["Final_Filter_Pass"]]
            # Select the longest negative/positive-slope history among currently qualifying
            # combinations. If none qualifies, retain the longest streak for diagnostics.
            pool = qualified if not qualified.empty else group
            best = pool.sort_values(
                ["Negative_Positive_Slope_Streak_Weeks", "VFI_130"],
                ascending=[False, False],
            ).iloc[0]
            summaries.append({
                "Symbol": symbol,
                "Weekly_Combos_Passing": int(len(qualified)),
                "Final_Combos_Passing": int(len(final)),
                "Best_EMA_short": int(best["EMA_short"]),
                "Best_EMA_medium": int(best["EMA_medium"]),
                "Best_EMA_long": int(best["EMA_long"]),
                "VFI_130": float(best["VFI_130"]),
                "VFI_Slope_1W": float(best["VFI_Slope_1W"]),
                "EMA_short_VFI": float(best["EMA_short_VFI"]),
                "EMA_medium_VFI": float(best["EMA_medium_VFI"]),
                "EMA_long_VFI": float(best["EMA_long_VFI"]),
                "Negative_Positive_Slope_Streak_Weeks": int(best["Negative_Positive_Slope_Streak_Weeks"]),
                "Daily_Bullish_Divergence_Recent": bool(best["Daily_Bullish_Divergence_Recent"]),
                "Daily_Bearish_Divergence_Recent": bool(best["Daily_Bearish_Divergence_Recent"]),
                "Last_Daily_Bullish_Divergence": best["Last_Daily_Bullish_Divergence"],
                "Weekly_Filter_Pass": bool(len(qualified) > 0),
                "Final_Filter_Pass": bool(len(final) > 0),
                "Weekly_Date": best["Weekly_Date"],
            })
        summary = pd.DataFrame(summaries).sort_values(
            ["Final_Filter_Pass", "Weekly_Filter_Pass", "Negative_Positive_Slope_Streak_Weeks", "VFI_130"],
            ascending=[False, False, False, False],
        )
        summary.to_csv(output / "early_vfi_accumulation_ranking.csv", index=False)
    pd.DataFrame(errors, columns=["Symbol", "Error"]).to_csv(output / "early_vfi_accumulation_errors.csv", index=False)
    print(f"Reports written to: {output}")
    print(f"Assets scanned: {len(set(r['Symbol'] for r in rows))}; combinations per asset: {len(ema_combinations())}")
    print("Ranking prioritizes weekly+daily candidates, then consecutive weeks with VFI<0 and positive slope.")
    print("EMA lines are computed from VFI(130), not the asset's price.")


if __name__ == "__main__":
    main()
