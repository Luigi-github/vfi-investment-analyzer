from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.analysis.vfi_analyzer import VFIAnalyzer
from app.scoring.vfi_score import VFIScore
from app.analysis.vfi_cycle import VFICycleAnalyzer

ASSETS = {
    "PFCIBEST": "PFCIBEST.CL",
    "PFDAVVNDA": "PFDAVVNDA.CL",
    "PFAVAL": "PFAVAL.CL",
    "GEB": "GEB.CL",
    "ISA": "ISA.CL",
    "ECOPETROL": "ECOPETROL.CL",
    "NU": "NU",
    "AMZN": "AMZN",
    "SQM": "SQM",
}

def slope(series: pd.Series, periods: int) -> float:
    s = series.dropna().tail(periods)
    if len(s) < 2:
        return float("nan")
    x = pd.Series(range(len(s)), index=s.index, dtype=float)
    return float(s.cov(x) / x.var()) if x.var() else 0.0

def current_row(symbol: str, ticker: str, service, indicator, analyzer, scorer, cycle):
    daily = service.get_daily_data(ticker, period="max")
    weekly = service.convert_to_weekly(daily)
    vfi = indicator.calculate(weekly)
    analyzed = analyzer.analyze(vfi)
    scored = scorer.calculate(analyzed)
    data = cycle.analyze(scored)

    valid = data[data["VFI"].notna()].copy()
    if valid.empty:
        raise ValueError(
            f"No valid VFI rows; daily={len(daily)}, weekly={len(weekly)}, "
            f"volume_nonzero={int((weekly["Volume"] > 0).sum())}, "
            f"volume_nan={int(weekly["Volume"].isna().sum())}, "
            f"vfi_nonnull={int(vfi["VFI"].notna().sum())}"
        )

    latest = valid.iloc[-1]
    recent = valid["VFI"].dropna()

    # Existing project score (0-100) is deliberately retained as-is.
    recovery_events = int(
        ((valid["VFI"] < 0) & (valid["VFI_Slope"] > 0)).sum()
    )
    accumulation_events = int(valid["Accumulation_Confirmed"].fillna(False).sum())

    # A separate CURRENT OPPORTUNITY score. It is causal/current-state only.
    # 40 direction/trend, 20 location/zero, 20 state, 20 penalty/bonus.
    s1 = float(latest["VFI_Slope"])
    s4 = float(latest["VFI_Slope_4W"])
    s8 = float(latest["VFI_Slope_8W"])
    vfi_value = float(latest["VFI"])
    dist0 = float(latest["VFI_Distance_Zero"])

    direction = 10 if s1 > 0 else 0
    trend = 20 if s4 > 0 and s8 > 0 else (10 if s4 > 0 or s8 > 0 else 0)

    location = 10 if dist0 <= 0.50 else (5 if dist0 <= 1.50 else 0)

    state_points = {
        "EARLY_ACCUMULATION": 20,
        "SUSTAINED_RECOVERY": 18,
        "RECOVERY": 15,
        "ACCUMULATION": 18,
        "BULLISH_FLOW": 12,
        "NEUTRAL": 5,
        "WEAKENING": 0,
        "SUSTAINED_WEAKENING": 0,
        "DISTRIBUTION": 0,
        "BEARISH_FLOW": 0,
    }
    state = str(latest["VFI_State"])
    state_score = state_points.get(state, 0)

    # Prefer early recovery: negative VFI + positive slope.
    setup = 20 if vfi_value < 0 and s1 > 0 else (
        15 if vfi_value >= 0 and s1 > 0 else 0
    )

    current_score = float(direction + trend + location + state_score + setup)

    return {
        "Asset": symbol,
        "Yahoo_Ticker": ticker,
        "Date": valid.index[-1].strftime("%Y-%m-%d"),
        "Price": float(latest["Close"]),
        "VFI": vfi_value,
        "Signal": float(latest["VFI_Signal"]),
        "VFI_minus_Signal": vfi_value - float(latest["VFI_Signal"]),
        "Slope_1W": s1,
        "Slope_4W": s4,
        "Slope_8W": s8,
        "Distance_Zero": dist0,
        "Range_Position": float(latest["VFI_Range_Position"]),
        "Recovery_From_Min_Pct": float(latest["VFI_Recovery_From_Min_Pct"]),
        "VFI_State": state,
        "Cycle_Stage": str(latest["VFI_Cycle_Stage"]),
        "Recovery_Quality": str(latest["Recovery_Quality"]),
        "Recovery_Score": float(latest["VFI_Recovery_Score"]),
        "Accumulation_Score": float(latest["Accumulation_Score"]),
        "Accumulation_Confirmed": bool(latest["Accumulation_Confirmed"]),
        "Recovery_Events_Historical": recovery_events,
        "Accumulation_Events_Historical": accumulation_events,
        "Weeks_Data": int(len(valid)),
        "Current_Opportunity_Score": current_score,
    }

def main():
    service = MarketDataService()
    indicator = VFIIndicator()
    analyzer = VFIAnalyzer()
    scorer = VFIScore()
    cycle = VFICycleAnalyzer()

    rows = []
    errors = []

    for asset, ticker in ASSETS.items():
        print(f"Calculating {asset} ({ticker})...")
        try:
            rows.append(current_row(asset, ticker, service, indicator, analyzer, scorer, cycle))
        except Exception as exc:
            errors.append({"Asset": asset, "Ticker": ticker, "Error": str(exc)})
            print(f"ERROR {asset}: {exc}")

    if not rows:
        raise RuntimeError("No asset could be calculated.")

    df = pd.DataFrame(rows)

    # Final ranking: 70% current opportunity + 30% existing recovery score.
    df["Final_Score"] = (
        df["Current_Opportunity_Score"] * 0.70
        + df["Recovery_Score"] * 0.30
    )
    df["Rank"] = (
        df["Final_Score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    df = df.sort_values(["Final_Score", "Current_Opportunity_Score"], ascending=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    cols = [
        "Rank", "Asset", "Date", "Price", "VFI", "Signal", "VFI_minus_Signal",
        "Slope_1W", "Slope_4W", "Slope_8W", "Distance_Zero",
        "Range_Position", "Recovery_From_Min_Pct", "VFI_State", "Cycle_Stage",
        "Recovery_Quality", "Recovery_Score", "Accumulation_Score",
        "Accumulation_Confirmed", "Recovery_Events_Historical",
        "Weeks_Data", "Current_Opportunity_Score", "Final_Score"
    ]

    print("\n" + "=" * 220)
    print("VFI CURRENT OPPORTUNITY RANKING")
    print("=" * 220)
    print(df[cols].to_string(index=False))

    print("\nSCORING FORMULA")
    print("Current Opportunity = Direction(10) + Trend(20) + Location(10) + State(20) + Setup(20)")
    print("Final Score = 70% Current Opportunity + 30% existing VFI Recovery Score")

    df.to_csv("vfi_9_assets_ranking.csv", index=False)
    df.to_json("vfi_9_assets_ranking.json", orient="records", indent=2)

    if errors:
        pd.DataFrame(errors).to_csv("vfi_9_assets_errors.csv", index=False)
        print("\nERRORS")
        print(pd.DataFrame(errors).to_string(index=False))

if __name__ == "__main__":
    main()
