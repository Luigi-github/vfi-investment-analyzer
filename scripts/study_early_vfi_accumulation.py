"""Scan early accumulation using matching VFI and Volume Divergence EMA rules.

Weekly gate: VFI(130) with negative short/medium EMAs above the long EMA,
negative VFI near zero, and positive slope.
Daily confirmation: apply the same EMA/near-zero/positive-slope structure to
Volume Divergence by MM itself (not to price), using the same EMA spans.
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
OUT = ROOT / "reports" / "early_vfi_accumulation"


def ema_combinations():
    return [(s, m, l) for s, m, l in product(SHORTS, MEDIUMS, LONGS) if s < m < l]


def positive_negative_streak(indicator: pd.Series) -> int:
    """Consecutive bars ending now with indicator<0 and positive one-bar slope."""
    ok = (indicator < 0) & (indicator.diff() > 0)
    count = 0
    for flag in ok.iloc[::-1]:
        if not bool(flag):
            break
        count += 1
    return count


def add_indicator_emas(frame: pd.DataFrame, value_column: str,
                       short: int, medium: int, long: int) -> pd.DataFrame:
    result = frame.copy()
    values = result[value_column]
    result["EMA_short"] = values.ewm(span=short, adjust=False, min_periods=short).mean()
    result["EMA_medium"] = values.ewm(span=medium, adjust=False, min_periods=medium).mean()
    result["EMA_long"] = values.ewm(span=long, adjust=False, min_periods=long).mean()
    result["Indicator_Slope_1Bar"] = values.diff()
    return result


def evaluate_combo(symbol, weekly, daily_div, short, medium, long):
    w = add_indicator_emas(weekly, "VFI", short, medium, long)
    d = add_indicator_emas(daily_div, "Volume_Divergence_MM", short, medium, long)

    w_valid = w.dropna(subset=["VFI", "EMA_short", "EMA_medium", "EMA_long", "Indicator_Slope_1Bar"])
    d_valid = d.dropna(subset=["Volume_Divergence_MM", "EMA_short", "EMA_medium", "EMA_long", "Indicator_Slope_1Bar"])
    if w_valid.empty or d_valid.empty:
        raise ValueError(f"{symbol}: insufficient history for EMA combination {short}/{medium}/{long}.")

    wl = w_valid.iloc[-1]
    dl = d_valid.iloc[-1]

    weekly_ema_structure = (
        wl["EMA_short"] < 0 and wl["EMA_medium"] < 0
        and wl["EMA_short"] > wl["EMA_long"]
        and wl["EMA_medium"] > wl["EMA_long"]
    )
    weekly_value_ok = wl["VFI"] < 0 and abs(float(wl["VFI"])) <= NEAR_ZERO
    weekly_slope_ok = wl["Indicator_Slope_1Bar"] > 0
    weekly_pass = bool(weekly_ema_structure and weekly_value_ok and weekly_slope_ok)

    daily_ema_structure = (
        dl["EMA_short"] < 0 and dl["EMA_medium"] < 0
        and dl["EMA_short"] > dl["EMA_long"]
        and dl["EMA_medium"] > dl["EMA_long"]
    )
    daily_value_ok = (
        dl["Volume_Divergence_MM"] < 0
        and abs(float(dl["Volume_Divergence_MM"])) <= NEAR_ZERO
    )
    daily_slope_ok = dl["Indicator_Slope_1Bar"] > 0
    daily_pass = bool(daily_ema_structure and daily_value_ok and daily_slope_ok)

    vfi_streak = positive_negative_streak(w["VFI"])
    divergence_streak = positive_negative_streak(d["Volume_Divergence_MM"])

    return {
        "Symbol": symbol,
        "EMA_short": short,
        "EMA_medium": medium,
        "EMA_long": long,
        "Weekly_Date": str(w.index[-1].date()),
        "Daily_Date": str(d.index[-1].date()),
        "VFI_130": float(wl["VFI"]),
        "VFI_Slope_1W": float(wl["Indicator_Slope_1Bar"]),
        "EMA_short_VFI": float(wl["EMA_short"]),
        "EMA_medium_VFI": float(wl["EMA_medium"]),
        "EMA_long_VFI": float(wl["EMA_long"]),
        "Weekly_EMA_Structure_Pass": bool(weekly_ema_structure),
        "Weekly_VFI_Negative_Near_Zero": bool(weekly_value_ok),
        "Weekly_VFI_Positive_Slope": bool(weekly_slope_ok),
        "Weekly_Filter_Pass": weekly_pass,
        "Volume_Divergence_MM": float(dl["Volume_Divergence_MM"]),
        "Volume_Div_Slope_1D": float(dl["Indicator_Slope_1Bar"]),
        "EMA_short_Volume_Div": float(dl["EMA_short"]),
        "EMA_medium_Volume_Div": float(dl["EMA_medium"]),
        "EMA_long_Volume_Div": float(dl["EMA_long"]),
        "Daily_EMA_Structure_Pass": bool(daily_ema_structure),
        "Daily_Divergence_Negative_Near_Zero": bool(daily_value_ok),
        "Daily_Divergence_Positive_Slope": bool(daily_slope_ok),
        "Daily_Filter_Pass": daily_pass,
        "VFI_Negative_Positive_Slope_Streak_Weeks": int(vfi_streak),
        "Divergence_Negative_Positive_Slope_Streak_Days": int(divergence_streak),
        "Final_Filter_Pass": bool(weekly_pass and daily_pass),
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
        print(f"EARLY ACCUMULATION SCAN: {symbol}", flush=True)
        try:
            daily = market.get_daily_data(symbol, period="max").sort_index()
            # Drop the potentially incomplete current daily candle.
            if not daily.empty:
                daily = daily[daily.index.normalize() < pd.Timestamp.now().normalize()]
            weekly_ohlcv = market.convert_to_weekly(daily)
            weekly = VFIIndicator(length=VFI_LENGTH).calculate(weekly_ohlcv.sort_index())
            weekly = weekly.replace([np.inf, -np.inf], np.nan).dropna(subset=["VFI"])
            daily_div = VolumeDivergenceMM().calculate(daily)
            daily_div = daily_div.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["Volume_Divergence_MM"]
            )
            if weekly.empty or daily_div.empty:
                raise ValueError("No valid data after indicator warm-up.")
            for short, medium, long in ema_combinations():
                rows.append(evaluate_combo(symbol, weekly, daily_div, short, medium, long))
            print(
                f"Weekly bars: {len(weekly)}; daily divergence bars: {len(daily_div)}; "
                f"data through {daily_div.index[-1].date()}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc)})
            print(f"SKIPPED {symbol}: {exc}", flush=True)

    if rows:
        all_combos = pd.DataFrame(rows)
        all_combos.to_csv(output / "ema_combo_scan.csv", index=False)
        summaries = []
        for symbol, group in all_combos.groupby("Symbol", sort=False):
            weekly_ok = group[group["Weekly_Filter_Pass"]]
            final_ok = group[group["Final_Filter_Pass"]]
            # Diagnostic fallback is explicitly flagged and must not be treated as a pass.
            pool = final_ok if not final_ok.empty else (weekly_ok if not weekly_ok.empty else group)
            best = pool.sort_values(
                ["VFI_Negative_Positive_Slope_Streak_Weeks",
                 "Divergence_Negative_Positive_Slope_Streak_Days"],
                ascending=[False, False],
            ).iloc[0]
            summaries.append({
                "Symbol": symbol,
                "Weekly_Combos_Passing": int(len(weekly_ok)),
                "Daily_Combos_Passing": int(group["Daily_Filter_Pass"].sum()),
                "Final_Combos_Passing": int(len(final_ok)),
                "Best_EMA_short": int(best["EMA_short"]),
                "Best_EMA_medium": int(best["EMA_medium"]),
                "Best_EMA_long": int(best["EMA_long"]),
                "VFI_130": float(best["VFI_130"]),
                "VFI_Slope_1W": float(best["VFI_Slope_1W"]),
                "Volume_Divergence_MM": float(best["Volume_Divergence_MM"]),
                "Volume_Div_Slope_1D": float(best["Volume_Div_Slope_1D"]),
                "VFI_Negative_Positive_Slope_Streak_Weeks": int(best["VFI_Negative_Positive_Slope_Streak_Weeks"]),
                "Divergence_Negative_Positive_Slope_Streak_Days": int(best["Divergence_Negative_Positive_Slope_Streak_Days"]),
                "Weekly_Filter_Pass": bool(len(weekly_ok) > 0),
                "Daily_Filter_Pass": bool(group["Daily_Filter_Pass"].any()),
                "Final_Filter_Pass": bool(len(final_ok) > 0),
                "Diagnostic_Fallback_Used": bool(final_ok.empty),
                "Weekly_Date": best["Weekly_Date"],
                "Daily_Date": best["Daily_Date"],
            })
        summary = pd.DataFrame(summaries).sort_values(
            ["Final_Filter_Pass", "Weekly_Filter_Pass", "VFI_Negative_Positive_Slope_Streak_Weeks",
             "Divergence_Negative_Positive_Slope_Streak_Days"],
            ascending=[False, False, False, False],
        )
        summary.to_csv(output / "early_vfi_accumulation_ranking.csv", index=False)
    pd.DataFrame(errors, columns=["Symbol", "Error"]).to_csv(
        output / "early_vfi_accumulation_errors.csv", index=False
    )
    print(f"Reports written to: {output}")
    print(f"Assets scanned: {len(set(r['Symbol'] for r in rows))}; EMA combinations per asset: {len(ema_combinations())}")
    print("Both timeframes use EMAs calculated on the indicator values, never on price.")
    print("A diagnostic fallback row is not a qualified signal; use Final_Filter_Pass=True only.")


if __name__ == "__main__":
    main()
