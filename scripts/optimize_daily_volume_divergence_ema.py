"""Walk-forward daily EMA optimization using confirmed Volume Divergence by MM."""
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.market_data import MarketDataService
from app.indicators.volume_divergence_mm import VolumeDivergenceMM
from scripts.optimize_ema_periods import (
    DEFAULT_SYMBOLS, SHORT_CANDIDATES, MEDIUM_CANDIDATES, LONG_CANDIDATES,
    candidate_periods, training_score,
)

DEFAULT_OUTPUT = ROOT / "reports" / "daily_volume_divergence_ema"
BARS_PER_YEAR = 252
MIN_DAILY_BARS = 260
DIVERGENCE_FRESH_BARS = 5


def simulate_strategy(data: pd.DataFrame, short: int, medium: int, long: int,
                      divergence_fresh_bars: int = DIVERGENCE_FRESH_BARS) -> pd.DataFrame:
    """Enter on fresh confirmed regular bullish divergence plus EMA alignment."""
    if not (0 < short < medium < long):
        raise ValueError("EMA periods must satisfy 0 < short < medium < long.")
    if divergence_fresh_bars < 1:
        raise ValueError("divergence_fresh_bars must be >= 1.")
    frame = data.copy().sort_index()
    frame["EMA_short"] = frame["Close"].ewm(span=short, adjust=False, min_periods=short).mean()
    frame["EMA_medium"] = frame["Close"].ewm(span=medium, adjust=False, min_periods=medium).mean()
    frame["EMA_long"] = frame["Close"].ewm(span=long, adjust=False, min_periods=long).mean()
    bullish = frame["Volume_Div_Regular_Bull"].fillna(False).astype(bool)
    bearish = frame["Volume_Div_Regular_Bear"].fillna(False).astype(bool)
    bullish_recent = bullish.rolling(divergence_fresh_bars, min_periods=1).max().astype(bool)
    bearish_recent = bearish.rolling(divergence_fresh_bars, min_periods=1).max().astype(bool)
    entry = (
        bullish_recent
        & (frame["EMA_short"] > frame["EMA_medium"])
        & (frame["Close"] > frame["EMA_long"])
        & (frame["EMA_short"] > frame["EMA_short"].shift(1))
    ).fillna(False)
    exit_ = ((frame["EMA_short"] <= frame["EMA_medium"]) | bearish_recent).fillna(False)
    active, positions, entry_events = False, [], []
    for enter, leave in zip(entry.to_numpy(), exit_.to_numpy()):
        opened = False
        if active and leave:
            active = False
        elif not active and enter:
            active, opened = True, True
        positions.append(1.0 if active else 0.0)
        entry_events.append(1 if opened else 0)
    frame["Signal"] = pd.Series(entry_events, index=frame.index, dtype=int)
    frame["Position"] = pd.Series(positions, index=frame.index, dtype=float).shift(1).fillna(0.0)
    frame["Asset_Return"] = frame["Close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["Strategy_Return"] = frame["Position"] * frame["Asset_Return"]
    return frame


def summarize_returns(frame: pd.DataFrame, start: int = 0, end=None) -> dict:
    sample = frame.iloc[start:end].copy()
    if sample.empty:
        return {
            "Bars": 0, "Total_Return_Pct": np.nan, "CAGR_Pct": np.nan,
            "Max_Drawdown_Pct": np.nan, "Exposure_Pct": np.nan, "Entries": 0,
            "BuyHold_Total_Return_Pct": np.nan, "BuyHold_CAGR_Pct": np.nan,
            "BuyHold_Max_Drawdown_Pct": np.nan,
        }
    returns = sample["Strategy_Return"].fillna(0.0)
    equity = (1.0 + returns).cumprod()
    years = max(len(sample) / BARS_PER_YEAR, 1.0 / BARS_PER_YEAR)
    final_equity = max(float(equity.iloc[-1]), 1e-12)
    drawdown = equity / equity.cummax() - 1.0
    buy_hold = sample["Close"].astype(float) / float(sample["Close"].iloc[0])
    buy_hold_dd = buy_hold / buy_hold.cummax() - 1.0
    return {
        "Bars": int(len(sample)),
        "Total_Return_Pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "CAGR_Pct": float((final_equity ** (1.0 / years) - 1.0) * 100.0),
        "Max_Drawdown_Pct": float(drawdown.min() * 100.0),
        "Exposure_Pct": float(sample["Position"].mean() * 100.0),
        "Entries": int(sample["Signal"].sum()),
        "BuyHold_Total_Return_Pct": float((buy_hold.iloc[-1] - 1.0) * 100.0),
        "BuyHold_CAGR_Pct": float((max(float(buy_hold.iloc[-1]), 1e-12) ** (1.0 / years) - 1.0) * 100.0),
        "BuyHold_Max_Drawdown_Pct": float(buy_hold_dd.min() * 100.0),
    }


def optimize_asset(symbol: str, daily: pd.DataFrame,
                   divergence_fresh_bars: int = DIVERGENCE_FRESH_BARS):
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = VolumeDivergenceMM().calculate(daily.sort_index())
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close", "Volume_Divergence_MM"])
    if len(data) < MIN_DAILY_BARS:
        raise ValueError(f"{symbol}: needs at least {MIN_DAILY_BARS} valid daily bars; got {len(data)}.")
    split = int(len(data) * 0.60)
    if split < 156 or len(data) - split < 104:
        raise ValueError(f"{symbol}: insufficient train/test history after split.")
    trials = []
    for short, medium, long in candidate_periods():
        simulated = simulate_strategy(data, short, medium, long, divergence_fresh_bars)
        train, test = summarize_returns(simulated, 0, split), summarize_returns(simulated, split, None)
        trials.append({
            "Symbol": symbol, "EMA_short": short, "EMA_medium": medium, "EMA_long": long,
            "Train_Score": training_score(train), "Train_CAGR_Pct": train["CAGR_Pct"],
            "Train_Max_Drawdown_Pct": train["Max_Drawdown_Pct"], "Train_Entries": train["Entries"],
            "Test_Total_Return_Pct": test["Total_Return_Pct"], "Test_CAGR_Pct": test["CAGR_Pct"],
            "Test_Max_Drawdown_Pct": test["Max_Drawdown_Pct"], "Test_Exposure_Pct": test["Exposure_Pct"],
            "Test_Entries": test["Entries"], "Test_BuyHold_Total_Return_Pct": test["BuyHold_Total_Return_Pct"],
            "Test_BuyHold_CAGR_Pct": test["BuyHold_CAGR_Pct"],
            "Test_BuyHold_Max_Drawdown_Pct": test["BuyHold_Max_Drawdown_Pct"],
            "Train_Bars": train["Bars"], "Test_Bars": test["Bars"],
            "Train_Start": str(data.index[0].date()), "Test_Start": str(data.index[split].date()),
            "Data_End": str(data.index[-1].date()), "Divergence_Fresh_Bars": divergence_fresh_bars,
        })
    results = pd.DataFrame(trials).sort_values(
        ["Train_Score", "EMA_short", "EMA_medium", "EMA_long"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    baseline = results[
        (results["EMA_short"] == 5) & (results["EMA_medium"] == 20) & (results["EMA_long"] == 50)
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
    parser = ArgumentParser(description="Optimize daily EMAs using confirmed Volume Divergence by MM.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated Yahoo Finance symbols.")
    parser.add_argument("--divergence-fresh-bars", type=int, default=DIVERGENCE_FRESH_BARS,
                        help="Daily bars after a confirmed regular bullish divergence (default: 5).")
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
        print(f"DAILY VOLUME DIVERGENCE + EMA OPTIMIZATION: {symbol}", flush=True)
        try:
            daily = market.get_daily_data(symbol, period="max")
            # Exclude today's potentially incomplete candle.
            if not daily.empty:
                daily = daily[daily.index.normalize() < pd.Timestamp.now().normalize()]
            trials, best, baseline = optimize_asset(symbol, daily, args.divergence_fresh_bars)
            all_trials.append(trials)
            selected.append(best)
            baselines.append(baseline)
            print(
                f"Selected EMA: {int(best['EMA_short'])}/{int(best['EMA_medium'])}/{int(best['EMA_long'])} | "
                f"holdout CAGR {best['Test_CAGR_Pct']:.2f}% vs buy-and-hold "
                f"{best['Test_BuyHold_CAGR_Pct']:.2f}% | entries {int(best['Test_Entries'])}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc)})
            print(f"SKIPPED {symbol}: {exc}", flush=True)
    if all_trials:
        pd.concat(all_trials, ignore_index=True).to_csv(output / "daily_ema_parameter_trials.csv", index=False)
        pd.DataFrame(selected).to_csv(output / "daily_ema_selected_periods.csv", index=False)
        pd.DataFrame(baselines).to_csv(output / "daily_ema_5_20_50_baseline.csv", index=False)
    pd.DataFrame(errors, columns=["Symbol", "Error"]).to_csv(output / "daily_ema_optimization_errors.csv", index=False)
    print(f"Reports written to: {output}")
    print("Selection uses the first 60% of daily history; the final 40% is chronological holdout.")
    print("Divergence pivots are confirmed causally; positions execute no earlier than the next daily bar.")


if __name__ == "__main__":
    main()
