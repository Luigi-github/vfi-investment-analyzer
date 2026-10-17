"""Walk-forward optimization of short/medium/long weekly EMAs for early VFI recovery.

The default study uses DDD, INTC, NVDA and BTC-USD. Candidate periods are selected
on the earlier 60% of each asset's weekly history and evaluated on the later 40%.
Signals are computed at weekly close and returns are applied from the next bar.
This is research, not a guarantee of future performance.
"""
from argparse import ArgumentParser
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

DEFAULT_SYMBOLS = ("DDD", "INTC", "NVDA", "BTC-USD")
SHORT_CANDIDATES = (3, 5, 7, 9)
MEDIUM_CANDIDATES = (15, 20, 25, 30)
LONG_CANDIDATES = (40, 50, 60, 80)
DEFAULT_OUTPUT = ROOT / "reports" / "ema_optimization"


def candidate_periods(shorts=SHORT_CANDIDATES, mediums=MEDIUM_CANDIDATES, longs=LONG_CANDIDATES):
    """Return valid short < medium < long triples in stable order."""
    return [
        (short, medium, long)
        for short, medium, long in product(shorts, mediums, longs)
        if short < medium < long
    ]


def add_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add VFI and close-based EMA features to weekly OHLCV data."""
    if not isinstance(weekly.index, pd.DatetimeIndex):
        raise ValueError("Weekly data must use a DatetimeIndex.")
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required.difference(weekly.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = VFIIndicator().calculate(weekly.sort_index())
    return result


def simulate_strategy(data: pd.DataFrame, short: int, medium: int, long: int, near_zero: float = 5.0):
    """Simulate a long-only early-recovery rule with next-bar returns.

    Entry: EMA-short > EMA-medium, close above EMA-long, VFI rising and VFI <=
    near_zero. Exit: EMA-short <= EMA-medium or VFI no longer rising.
    Position is lagged one full bar before applying close-to-close returns.
    """
    if not (0 < short < medium < long):
        raise ValueError("EMA periods must satisfy 0 < short < medium < long.")
    frame = data.copy().sort_index()
    frame["EMA_short"] = frame["Close"].ewm(span=short, adjust=False, min_periods=short).mean()
    frame["EMA_medium"] = frame["Close"].ewm(span=medium, adjust=False, min_periods=medium).mean()
    frame["EMA_long"] = frame["Close"].ewm(span=long, adjust=False, min_periods=long).mean()
    vfi_rising = frame["VFI"] > frame["VFI"].shift(1)
    entry = (
        (frame["EMA_short"] > frame["EMA_medium"])
        & (frame["Close"] > frame["EMA_long"])
        & vfi_rising
        & (frame["VFI"] <= near_zero)
    ).fillna(False)
    exit_ = (
        (frame["EMA_short"] <= frame["EMA_medium"])
        | ~vfi_rising
    ).fillna(False)

    active = False
    positions = []
    entry_events = []
    for enter, leave in zip(entry.to_numpy(), exit_.to_numpy()):
        opened = False
        if active and leave:
            active = False
        elif not active and enter:
            active = True
            opened = True
        positions.append(1.0 if active else 0.0)
        entry_events.append(1 if opened else 0)

    frame["Signal"] = pd.Series(entry_events, index=frame.index, dtype=int)
    frame["Position"] = pd.Series(positions, index=frame.index).shift(1).fillna(0.0)
    frame["Asset_Return"] = frame["Close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["Strategy_Return"] = frame["Position"] * frame["Asset_Return"]
    frame["Equity"] = (1.0 + frame["Strategy_Return"]).cumprod()
    return frame


def summarize_returns(frame: pd.DataFrame, start: int = 0, end=None) -> dict:
    """Summarize a slice of returns and risk; excludes the first undefined return."""
    sample = frame.iloc[start:end].copy()
    if sample.empty:
        return {"Bars": 0, "Total_Return_Pct": np.nan, "CAGR_Pct": np.nan,
                "Max_Drawdown_Pct": np.nan, "Exposure_Pct": np.nan, "Entries": 0,
                "BuyHold_Total_Return_Pct": np.nan, "BuyHold_CAGR_Pct": np.nan,
                "BuyHold_Max_Drawdown_Pct": np.nan}
    returns = sample["Strategy_Return"].fillna(0.0)
    equity = (1.0 + returns).cumprod()
    total = float((equity.iloc[-1] - 1.0) * 100.0)
    years = max(len(sample) / 52.0, 1.0 / 52.0)
    final_equity = max(float(equity.iloc[-1]), 1e-12)
    cagr = (final_equity ** (1.0 / years) - 1.0) * 100.0
    drawdown = equity / equity.cummax() - 1.0
    buy_hold_equity = sample["Close"].astype(float) / float(sample["Close"].iloc[0])
    buy_hold_total = float((buy_hold_equity.iloc[-1] - 1.0) * 100.0)
    buy_hold_cagr = (max(float(buy_hold_equity.iloc[-1]), 1e-12) ** (1.0 / years) - 1.0) * 100.0
    buy_hold_drawdown = buy_hold_equity / buy_hold_equity.cummax() - 1.0
    return {
        "Bars": int(len(sample)),
        "Total_Return_Pct": total,
        "CAGR_Pct": float(cagr),
        "Max_Drawdown_Pct": float(drawdown.min() * 100.0),
        "Exposure_Pct": float(sample["Position"].mean() * 100.0),
        "Entries": int(sample["Signal"].sum()),
        "BuyHold_Total_Return_Pct": buy_hold_total,
        "BuyHold_CAGR_Pct": float(buy_hold_cagr),
        "BuyHold_Max_Drawdown_Pct": float(buy_hold_drawdown.min() * 100.0),
    }


def training_score(metrics: dict) -> float:
    """Risk-penalized selection score calculated on training data only."""
    if metrics["Bars"] == 0 or metrics["Entries"] < 3 or not np.isfinite(metrics["CAGR_Pct"]):
        return -np.inf
    return metrics["CAGR_Pct"] - 0.5 * abs(metrics["Max_Drawdown_Pct"])


def optimize_asset(symbol: str, weekly: pd.DataFrame, near_zero: float = 5.0):
    """Select EMA periods on train, then evaluate that fixed triple on test."""
    data = add_features(weekly).dropna(subset=["VFI", "Close"])
    if len(data) < 260:
        raise ValueError(f"{symbol}: needs at least 260 weekly bars after VFI warm-up; got {len(data)}.")
    split = int(len(data) * 0.60)
    if split < 104 or len(data) - split < 52:
        raise ValueError(f"{symbol}: insufficient train/test history after split.")

    trials = []
    for short, medium, long in candidate_periods():
        simulated = simulate_strategy(data, short, medium, long, near_zero)
        train = summarize_returns(simulated, 0, split)
        test = summarize_returns(simulated, split, None)
        trials.append({
            "Symbol": symbol, "EMA_short": short, "EMA_medium": medium, "EMA_long": long,
            "Train_Score": training_score(train), "Train_CAGR_Pct": train["CAGR_Pct"],
            "Train_Max_Drawdown_Pct": train["Max_Drawdown_Pct"], "Train_Entries": train["Entries"],
            "Test_Total_Return_Pct": test["Total_Return_Pct"], "Test_CAGR_Pct": test["CAGR_Pct"],
            "Test_Max_Drawdown_Pct": test["Max_Drawdown_Pct"], "Test_Exposure_Pct": test["Exposure_Pct"],
            "Test_Entries": test["Entries"],
            "Test_BuyHold_Total_Return_Pct": test["BuyHold_Total_Return_Pct"],
            "Test_BuyHold_CAGR_Pct": test["BuyHold_CAGR_Pct"],
            "Test_BuyHold_Max_Drawdown_Pct": test["BuyHold_Max_Drawdown_Pct"],
            "Train_Bars": train["Bars"], "Test_Bars": test["Bars"],
            "Train_Start": str(data.index[0].date()), "Test_Start": str(data.index[split].date()),
            "Data_End": str(data.index[-1].date()),
        })
    results = pd.DataFrame(trials).sort_values(
        ["Train_Score", "EMA_short", "EMA_medium", "EMA_long"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    baseline = results[
        (results["EMA_short"] == 5)
        & (results["EMA_medium"] == 20)
        & (results["EMA_long"] == 50)
    ]
    if baseline.empty:
        raise RuntimeError("The 5/20/50 baseline must be present in the candidate grid.")
    eligible = results[results["Train_Entries"] >= 3]
    if eligible.empty:
        raise ValueError(f"{symbol}: no EMA combination generated at least three training entries.")
    best = eligible.iloc[0]
    results["Selected_On_Train"] = False
    results.loc[best.name, "Selected_On_Train"] = True
    results["Is_5_20_50_Baseline"] = (
        (results["EMA_short"] == 5) & (results["EMA_medium"] == 20) & (results["EMA_long"] == 50)
    )
    return results, best.to_dict(), baseline.iloc[0].to_dict()


def parse_args(argv=None):
    parser = ArgumentParser(description="Optimize weekly short/medium/long EMA periods with a chronological holdout.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated yfinance symbols.")
    parser.add_argument("--near-zero", type=float, default=5.0, help="Maximum VFI value for early-recovery entries (default: 5).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directory for CSV reports.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    symbols = list(dict.fromkeys(s.strip() for s in args.symbols.split(",") if s.strip()))
    if not symbols:
        raise ValueError("At least one symbol is required.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    market = MarketDataService()
    all_trials, selected, baselines, errors = [], [], [], []

    for symbol in symbols:
        print("=" * 72)
        print(f"EMA OPTIMIZATION: {symbol}")
        try:
            daily = market.get_daily_data(symbol, period="max")
            weekly = market.convert_to_weekly(daily)
            trials, best, baseline = optimize_asset(symbol, weekly, args.near_zero)
            all_trials.append(trials)
            selected.append(best)
            baselines.append(baseline)
            print(f"Bars: {len(weekly)} | selected EMA: {int(best['EMA_short'])}/{int(best['EMA_medium'])}/{int(best['EMA_long'])}")
            print(f"Holdout strategy CAGR: {best['Test_CAGR_Pct']:.2f}% | buy-and-hold CAGR: {best['Test_BuyHold_CAGR_Pct']:.2f}%")
            print(f"Holdout strategy max drawdown: {best['Test_Max_Drawdown_Pct']:.2f}% | buy-and-hold: {best['Test_BuyHold_Max_Drawdown_Pct']:.2f}% | entries: {int(best['Test_Entries'])}")
            print(f"5/20/50 baseline strategy CAGR: {baseline['Test_CAGR_Pct']:.2f}% | buy-and-hold CAGR: {baseline['Test_BuyHold_CAGR_Pct']:.2f}%")
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc)})
            print(f"SKIPPED {symbol}: {exc}")

    if all_trials:
        pd.concat(all_trials, ignore_index=True).to_csv(output / "ema_parameter_trials.csv", index=False)
        pd.DataFrame(selected).to_csv(output / "ema_selected_periods.csv", index=False)
        pd.DataFrame(baselines).to_csv(output / "ema_5_20_50_baseline.csv", index=False)
    pd.DataFrame(errors, columns=["Symbol", "Error"]).to_csv(output / "ema_optimization_errors.csv", index=False)
    print(f"Reports written to: {output}")
    print("Selection uses the first 60% of weekly history; the final 40% is held out.")
    print("The holdout is not used to choose periods. Review the per-asset CSV before treating any period as robust.")


if __name__ == "__main__":
    main()
