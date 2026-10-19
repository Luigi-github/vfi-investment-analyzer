"""Historical tuning of early-accumulation score weights.

Features are computed causally from completed bars. Weights are selected on the
first 70% of each asset's history; the final 30% is reserved as a holdout.
This is a research ranking, not a trading recommendation.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.indicators.volume_divergence_mm import VolumeDivergenceMM

ASSETS = {
    "DDD": "DDD", "INTC": "INTC", "NVDA": "NVDA", "BTC-USD": "BTC-USD",
    "EC": "EC", "CIB": "CIB", "MSFT": "MSFT", "EEM": "EEM", "SQM": "SQM",
    "LTC-USD": "LTC-USD", "AU": "AU", "SPY": "SPY", "AMZN": "AMZN",
    "PBR": "PBR", "NU": "NU", "QQQ": "QQQ", "ISA.CL": "ISA.CL",
    "GEB.CL": "GEB.CL", "MINEROS.CL": "MINEROS.CL", "PFAVAL.CL": "PFAVAL.CL",
    "PFDAVVNDA.CL": "PFDAVVNDA.CL", "TERPEL.CL": "TERPEL.CL",
}
FEATURES = ["vfi_early", "vfi_slope", "div_early", "div_slope",
            "ema_structure", "persistence", "extension_penalty", "bearish_penalty"]
HORIZON = 40
TRAIN_FRACTION = 0.70
COST_PCT = 0.25
MIN_ROWS_ASSET = 300
OUT_DEFAULT = "reports/score_weight_tuning"


def completed_daily(frame):
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.sort_index().copy()
    idx = pd.DatetimeIndex(frame.index)
    today = pd.Timestamp.now(tz=idx.tz).normalize() if idx.tz is not None else pd.Timestamp.now().normalize()
    frame = frame.loc[idx.normalize() < today]
    return frame.loc[~frame.index.duplicated(keep="last")].dropna(
        subset=["Open", "High", "Low", "Close", "Volume"])


def _unit(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def _features(series, slope_bars, window, near_q=0.30):
    """Causal features; the current observation is excluded from rolling thresholds."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    min_periods = max(30, window // 3)
    prior_abs = s.abs().shift(1).rolling(window, min_periods=min_periods)
    near_threshold = prior_abs.quantile(near_q)
    scale = s.shift(1).rolling(window, min_periods=min_periods).std()
    proximity = _unit(1 - (s.abs() / near_threshold.replace(0, np.nan)))
    early = np.where(s < 0, proximity, 0.0)
    slope = (s - s.shift(slope_bars)) / slope_bars
    slope_scale = scale / max(1, np.sqrt(slope_bars))
    # Zero slope earns zero points; positive slopes add points, while negative
    # slopes are deducted separately by the bearish-penalty feature.
    slope_strength = _unit(slope / slope_scale.replace(0, np.nan))
    slope_feature = np.where(s < 0, slope_strength, 0.0)
    extension = _unit(s.clip(lower=0) / (scale.replace(0, np.nan) * 2.0))
    e5 = s.ewm(span=5, adjust=False, min_periods=5).mean()
    e20 = s.ewm(span=20, adjust=False, min_periods=20).mean()
    e50 = s.ewm(span=50, adjust=False, min_periods=50).mean()
    structure = 0.5 * (e5 > e20).astype(float) + 0.5 * (e20 > e50).astype(float)
    good = (s < 0) & (s.diff() > 0)
    streak = good.groupby((~good).cumsum()).cumcount() + 1
    persistence = _unit(streak.where(good, 0) / 10.0)
    return pd.DataFrame({
        "early": early, "slope": slope_feature, "ema": structure,
        "persistence": persistence, "extension": extension, "raw_slope": slope,
    }, index=s.index)


def build_asset_rows(symbol, daily):
    weekly_ohlcv = daily.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"}).dropna()
    weekly = VFIIndicator(length=130).calculate(weekly_ohlcv)["VFI"].dropna()
    divergence = VolumeDivergenceMM().calculate(daily)["Volume_Divergence_MM"].dropna()
    if len(weekly) < 90 or len(divergence) < MIN_ROWS_ASSET:
        raise ValueError(f"insufficient history: weekly VFI={len(weekly)}, daily divergence={len(divergence)}")
    # Calculate weekly features on weekly bars first, then carry the latest
    # completed weekly value into daily snapshots. This preserves a true weekly slope.
    wf_weekly = _features(weekly, slope_bars=4, window=156)
    weekly_daily = weekly.reindex(daily.index, method="ffill")
    wf = wf_weekly.reindex(daily.index, method="ffill")
    df = _features(divergence, slope_bars=5, window=252).reindex(daily.index, method="ffill")
    idx = daily.index
    feat = pd.DataFrame(index=idx)
    feat["vfi_early"] = wf["early"]
    feat["vfi_slope"] = wf["slope"]
    feat["div_early"] = df["early"]
    feat["div_slope"] = df["slope"]
    feat["ema_structure"] = 0.5 * wf["ema"] + 0.5 * df["ema"]
    feat["persistence"] = 0.5 * wf["persistence"] + 0.5 * df["persistence"]
    feat["extension_penalty"] = wf["extension"]
    feat["bearish_penalty"] = 0.5 * (
        (wf["raw_slope"] < 0).astype(float)
        + (df["raw_slope"] < 0).astype(float))
    feat["VFI_130"] = weekly_daily
    feat["VFI_Slope_1W"] = wf["raw_slope"]
    feat["Volume_Divergence_MM"] = divergence.reindex(idx)
    feat["Volume_Div_Slope_5D"] = df["raw_slope"].reindex(idx)
    close = daily["Close"].astype(float)
    feat["Forward_Return_40D_Pct"] = (close.shift(-HORIZON) / close - 1.0) * 100.0 - COST_PCT
    feat["Date"] = idx
    feat["Symbol"] = symbol
    # Keep the most recent feature snapshots even though their 40-day forward
    # return is not observable yet. They are used for the current ranking only.
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    return feat.reset_index(drop=True)


def score_features(frame, weights):
    positive = sum(frame[k] * weights[k] for k in FEATURES if k not in ("extension_penalty", "bearish_penalty"))
    negative = sum(frame[k] * weights[k] for k in ("extension_penalty", "bearish_penalty"))
    return positive - negative


def metrics(frame, score, split):
    x = frame.copy()
    x["Score"] = np.asarray(score)
    if x.empty:
        return {"Split": split, "Rows": 0}
    cutoff = x["Score"].quantile(0.80)
    top = x.loc[x["Score"] >= cutoff]
    by_asset = top.groupby("Symbol")["Forward_Return_40D_Pct"].mean()
    return {
        "Split": split, "Rows": int(len(x)), "Top20_Rows": int(len(top)),
        "Top20_Mean_Return_40D_Pct": float(top["Forward_Return_40D_Pct"].mean()),
        "Top20_Median_Return_40D_Pct": float(top["Forward_Return_40D_Pct"].median()),
        "Top20_Win_Rate_40D_Pct": float((top["Forward_Return_40D_Pct"] > 0).mean() * 100),
        "Top20_Asset_Balanced_Mean_40D_Pct": float(by_asset.mean()) if len(by_asset) else np.nan,
        "Spearman_Score_Return": float(x[["Score", "Forward_Return_40D_Pct"]].corr(method="spearman").iloc[0, 1]),
    }


def candidate_weights():
    # Positive contributions total exactly 100; penalties are additional deductions.
    for vfi, vslope, div, dslope, ema, persistence in itertools.product(
        (15, 20, 25, 30), (5, 10, 15, 20), (10, 15, 20, 25),
        (5, 10, 15, 20), (5, 10, 15), (5, 10, 15)):
        vals = {"vfi_early": vfi, "vfi_slope": vslope, "div_early": div,
                "div_slope": dslope, "ema_structure": ema, "persistence": persistence}
        if sum(vals.values()) != 100:
            continue
        # Preserve the strategy's intended behavior: positive VFI extension and
        # negative slopes must always carry a meaningful deduction.
        for extension, bearish in itertools.product((10, 15, 20, 25), (5, 10, 15, 20)):
            yield {**vals, "extension_penalty": extension, "bearish_penalty": bearish}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(ASSETS))
    parser.add_argument("--output-dir", default=OUT_DEFAULT)
    parser.add_argument("--max-candidates", type=int, default=600)
    args = parser.parse_args()
    requested = [s.strip() for s in args.symbols.split(",") if s.strip()]
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    market = MarketDataService()
    all_rows, errors = [], []
    for symbol in requested:
        try:
            print(f"[{symbol}] downloading history", flush=True)
            daily = completed_daily(market.get_daily_data(symbol, period="max"))
            if daily.empty:
                raise ValueError("no completed daily OHLCV")
            rows = build_asset_rows(symbol, daily)
            split_pos = int(len(rows) * TRAIN_FRACTION)
            rows["Split"] = np.where(np.arange(len(rows)) < split_pos, "train", "holdout")
            if split_pos > HORIZON:
                rows.loc[max(0, split_pos - HORIZON):split_pos - 1, "Split"] = "purged"
            all_rows.append(rows)
            print(f"[{symbol}] usable snapshots={len(rows)}", flush=True)
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": repr(exc)})
            print(f"[{symbol}] ERROR: {exc!r}", flush=True)
    if not all_rows:
        raise RuntimeError("No assets produced usable historical snapshots.")
    data = pd.concat(all_rows, ignore_index=True)
    labeled = data[data["Forward_Return_40D_Pct"].notna()].copy()
    train = labeled[labeled["Split"] == "train"].copy()
    holdout = labeled[labeled["Split"] == "holdout"].copy()
    if train.empty or holdout.empty:
        raise RuntimeError("Training or holdout split is empty.")
    baseline = {"vfi_early": 20, "vfi_slope": 15, "div_early": 20, "div_slope": 15,
                "ema_structure": 15, "persistence": 15, "extension_penalty": 15, "bearish_penalty": 10}
    baseline_train = metrics(train, score_features(train, baseline), "train")
    baseline_test = metrics(holdout, score_features(holdout, baseline), "holdout")
    candidates = list(candidate_weights())
    if len(candidates) > args.max_candidates:
        take = np.linspace(0, len(candidates) - 1, args.max_candidates, dtype=int)
        candidates = [candidates[i] for i in take]
    results = []
    for weights in candidates:
        m = metrics(train, score_features(train, weights), "train")
        results.append({**weights, **m})
    ranked = pd.DataFrame(results).sort_values(
        ["Top20_Asset_Balanced_Mean_40D_Pct", "Top20_Win_Rate_40D_Pct", "Spearman_Score_Return"],
        ascending=False, na_position="last").reset_index(drop=True)
    ranked.insert(0, "Train_Rank", np.arange(1, len(ranked) + 1))
    best = {k: float(ranked.iloc[0][k]) for k in FEATURES}
    best_train = metrics(train, score_features(train, best), "train")
    best_test = metrics(holdout, score_features(holdout, best), "holdout")
    ranked.to_csv(out / "weight_search_training_rank.csv", index=False)
    comparison_rows = []
    for label, weights, train_metrics, holdout_metrics in (
        ("BASELINE", baseline, baseline_train, baseline_test),
        ("OPTIMIZED_TRAIN_ONLY", best, best_train, best_test),
    ):
        comparison_rows.append({
            "Configuration": label, **weights,
            **{f"Train_{k}": v for k, v in train_metrics.items() if k != "Split"},
            **{f"Holdout_{k}": v for k, v in holdout_metrics.items() if k != "Split"},
        })
    pd.DataFrame(comparison_rows).to_csv(out / "baseline_vs_optimized.csv", index=False)
    holdout_scored = holdout.copy()
    holdout_scored["Baseline_Score"] = score_features(holdout_scored, baseline)
    holdout_scored["Optimized_Score"] = score_features(holdout_scored, best)
    holdout_scored.to_csv(out / "holdout_asset_snapshots.csv", index=False)
    latest = data.sort_values("Date").groupby("Symbol", as_index=False).tail(1).copy()
    latest["Baseline_Score"] = score_features(latest, baseline)
    latest["Optimized_Score"] = score_features(latest, best)
    latest = latest.sort_values("Optimized_Score", ascending=False)
    latest.insert(0, "Optimized_Rank", np.arange(1, len(latest) + 1))
    latest.to_csv(out / "latest_optimized_ranking.csv", index=False)
    summary = {
        "assets_requested": requested, "assets_with_data": sorted(data["Symbol"].unique().tolist()),
        "errors": errors, "horizon_trading_days": HORIZON, "round_trip_cost_assumption_pct": COST_PCT,
        "split": "first 70% of usable snapshots for training; final 30% holdout; 40 bars around boundary purged from training",
        "baseline_weights": baseline, "optimized_weights": best,
        "baseline_train": baseline_train, "baseline_holdout": baseline_test,
        "optimized_train": best_train, "optimized_holdout": best_test,
        "selection_objective": "maximize training top-20%-score asset-balanced mean 40-trading-day forward return; tie-break on win rate and score/return rank correlation",
        "caveat": "Weights can overfit the finite example universe. Holdout is a historical test, not a guarantee. Score ranks all assets without excluding them.",
    }
    (out / "metadata.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nBASELINE TRAIN/HOLDOUT:", json.dumps({"train": baseline_train, "holdout": baseline_test}, indent=2), flush=True)
    print("\nOPTIMIZED WEIGHTS:", json.dumps(best, indent=2), flush=True)
    print("\nOPTIMIZED TRAIN/HOLDOUT:", json.dumps({"train": best_train, "holdout": best_test}, indent=2), flush=True)
    print("\nLATEST OPTIMIZED RANKING:")
    print(latest[["Optimized_Rank", "Symbol", "Optimized_Score", "Baseline_Score", "VFI_130", "VFI_Slope_1W", "Volume_Divergence_MM", "Volume_Div_Slope_5D"]].to_string(index=False), flush=True)
    print(f"Reports written to {out}", flush=True)


if __name__ == "__main__":
    main()
