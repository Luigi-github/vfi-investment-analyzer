from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.vfi_analyzer import VFIAnalyzer
from app.analysis.vfi_cycle import VFICycleAnalyzer
from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM
from app.scoring.vfi_score import VFIScore

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

HORIZONS = (4, 8, 12, 26)


def build_asset(service, vfi_indicator, div_indicator, analyzer, scorer, cycle, ticker):
    daily = service.get_daily_data(ticker, period="max")
    weekly = service.convert_to_weekly(daily)
    vfi = vfi_indicator.calculate(weekly)
    analyzed = analyzer.analyze(vfi)
    scored = scorer.calculate(analyzed)
    data = cycle.analyze(scored)
    div = div_indicator.calculate(weekly)
    data = data.join(
        div[
            [
                "Volume_Div_Regular_Bull",
                "Volume_Div_Hidden_Bull",
                "Volume_Div_Regular_Bear",
                "Volume_Div_Hidden_Bear",
                "Volume_Div_Bullish",
                "Volume_Div_Bearish",
                "Volume_Div_Signal",
            ]
        ],
        how="left",
    )
    return data.dropna(subset=["Close"])


def event_rows(asset, data):
    rows = []
    for i in range(len(data)):
        if not bool(data["Volume_Div_Bullish"].iat[i]) and not bool(data["Volume_Div_Bearish"].iat[i]):
            continue

        close = float(data["Close"].iat[i])
        signal = str(data["Volume_Div_Signal"].iat[i])
        bullish = bool(data["Volume_Div_Bullish"].iat[i])

        row = {
            "Asset": asset,
            "Signal_Date": data.index[i].strftime("%Y-%m-%d"),
            "Signal": signal,
            "Direction": "BULLISH" if bullish else "BEARISH",
            "Price": close,
            "VFI": float(data["VFI"].iat[i]) if pd.notna(data["VFI"].iat[i]) else np.nan,
            "VFI_Slope_1W": float(data["VFI_Slope"].iat[i]) if pd.notna(data["VFI_Slope"].iat[i]) else np.nan,
            "VFI_Slope_4W": float(data["VFI_Slope_4W"].iat[i]) if pd.notna(data["VFI_Slope_4W"].iat[i]) else np.nan,
            "VFI_Slope_8W": float(data["VFI_Slope_8W"].iat[i]) if pd.notna(data["VFI_Slope_8W"].iat[i]) else np.nan,
            "VFI_Below_Zero": bool(pd.notna(data["VFI"].iat[i]) and data["VFI"].iat[i] < 0),
        }

        for h in HORIZONS:
            j = i + h
            if j < len(data):
                future = float(data["Close"].iat[j])
                ret = future / close - 1.0
                row[f"Return_{h}W"] = ret
                row[f"Excess_Return_{h}W"] = ret
            else:
                row[f"Return_{h}W"] = np.nan
                row[f"Excess_Return_{h}W"] = np.nan

        row["VFI_Confluence"] = (
            bullish
            and row["VFI_Below_Zero"]
            and row["VFI_Slope_1W"] > 0
        ) or (
            (not bullish)
            and (not row["VFI_Below_Zero"])
            and row["VFI_Slope_1W"] < 0
        )
        rows.append(row)
    return rows


def summarize(events):
    if events.empty:
        return pd.DataFrame()

    groups = []
    for (direction, signal), g in events.groupby(["Direction", "Signal"], dropna=False):
        base = {
            "Direction": direction,
            "Signal": signal,
            "Events": len(g),
            "Confluence_Events": int(g["VFI_Confluence"].sum()),
        }
        for h in HORIZONS:
            r = g[f"Return_{h}W"].dropna()
            if len(r):
                base[f"Mean_{h}W"] = r.mean()
                base[f"Median_{h}W"] = r.median()
                base[f"WinRate_{h}W"] = (r > 0).mean()
            else:
                base[f"Mean_{h}W"] = np.nan
                base[f"Median_{h}W"] = np.nan
                base[f"WinRate_{h}W"] = np.nan
        groups.append(base)

    # Direct comparison: all bullish divergences vs bullish + VFI confluence.
    for label, mask in (
        ("ALL_BULLISH", events["Direction"] == "BULLISH"),
        ("BULLISH_VFI_CONFLUENCE", (events["Direction"] == "BULLISH") & events["VFI_Confluence"]),
        ("ALL_BEARISH", events["Direction"] == "BEARISH"),
        ("BEARISH_VFI_CONFLUENCE", (events["Direction"] == "BEARISH") & events["VFI_Confluence"]),
    ):
        g = events.loc[mask]
        base = {"Direction": label, "Signal": "AGGREGATE", "Events": len(g), "Confluence_Events": int(g["VFI_Confluence"].sum())}
        for h in HORIZONS:
            r = g[f"Return_{h}W"].dropna()
            if len(r):
                if "BEARISH" in label:
                    r = -r
                base[f"Mean_{h}W"] = r.mean()
                base[f"Median_{h}W"] = r.median()
                base[f"WinRate_{h}W"] = (r > 0).mean()
            else:
                base[f"Mean_{h}W"] = np.nan
                base[f"Median_{h}W"] = np.nan
                base[f"WinRate_{h}W"] = np.nan
        groups.append(base)

    return pd.DataFrame(groups)


def main():
    service = MarketDataService()
    vfi_indicator = VFIIndicator()
    div_indicator = VolumeDivergenceMM()
    analyzer = VFIAnalyzer()
    scorer = VFIScore()
    cycle = VFICycleAnalyzer()

    all_events = []
    errors = []

    for asset, ticker in ASSETS.items():
        print(f"Studying {asset} ({ticker})...")
        try:
            data = build_asset(service, vfi_indicator, div_indicator, analyzer, scorer, cycle, ticker)
            all_events.extend(event_rows(asset, data))
        except Exception as exc:
            errors.append({"Asset": asset, "Ticker": ticker, "Error": str(exc)})
            print(f"ERROR {asset}: {exc}")

    events = pd.DataFrame(all_events)
    if events.empty:
        raise RuntimeError("No divergence events found.")

    summary = summarize(events)
    events.to_csv("vfi_divergence_events.csv", index=False)
    summary.to_csv("vfi_divergence_summary.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n" + "=" * 180)
    print("VFI + VOLUME DIVERGENCE BY MM EVENT STUDY")
    print("=" * 180)
    print(summary.to_string(index=False))
    print("\nInterpretation: this is an observational event study, not a validated trading strategy.")
    print("Signals are timestamped when the 5-right-bar pivot becomes confirmed; no pivot is backdated.")

    if errors:
        pd.DataFrame(errors).to_csv("vfi_divergence_errors.csv", index=False)
        print("\nERRORS")
        print(pd.DataFrame(errors).to_string(index=False))


if __name__ == "__main__":
    main()
