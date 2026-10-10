from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.binance_market_data import BinanceMarketDataService
from app.research.multi_timeframe_crypto import MultiTimeframeCryptoAnalyzer
from app.backtest.crypto_strategy import backtest_signals, buy_and_hold
from app.backtest.block_bootstrap import block_bootstrap_mean

THRESHOLDS = (60, 65, 70, 75, 80, 85, 90)
STRATEGIES = {
    "SPOT_MTF": ("MTF_Score_Spot", "Spot"),
    "PERP_MTF": ("MTF_Score_Perp", "Perp"),
    "PERP_SHORT": ("Short_Score_Perp", "Perp"),
    "BOTH_MTF": ("Both_Bullish", "Spot"),
    "COMBINED_70": ("Combined_MTF_Score", "Spot"),
    "COMBINED_80": ("Combined_MTF_Score", "Spot"),
    "PERP_COMBINED_SHORT": ("Short_Combined_Score", "Perp"),
}
FEE = 0.0004
SLIPPAGE = 0.0003
STOP = 0.08
TARGET = 0.16
HORIZON = 72

def merge_symbol(service, analyzer, symbol):
    spot_frames = service.get_multi_timeframe(symbol, "spot")
    perp_frames = service.get_multi_timeframe(symbol, "futures")
    spot = analyzer.calculate(spot_frames)
    perp = analyzer.calculate(perp_frames)
    sh = analyzer.build_1h_confluence_history(spot)
    ph = analyzer.build_1h_confluence_history(perp)
    if sh.empty or ph.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    # Pandas 3.x can preserve different datetime resolutions (ms/us).
    # Normalize all 1H merge/join indexes to UTC nanoseconds so the Spot/Perp
    # confluence merge and OHLC joins use identical datetime dtypes.
    sh = sh.copy()
    ph = ph.copy()
    sh.index = pd.to_datetime(sh.index, utc=True).astype("datetime64[ns, UTC]")
    ph.index = pd.to_datetime(ph.index, utc=True).astype("datetime64[ns, UTC]")

    left = sh.reset_index().rename(columns={"index": "Timestamp"})
    right = ph.reset_index().rename(columns={"index": "Timestamp"})
    left["Timestamp"] = pd.to_datetime(left["Timestamp"], utc=True).astype("datetime64[ns, UTC]")
    right["Timestamp"] = pd.to_datetime(right["Timestamp"], utc=True).astype("datetime64[ns, UTC]")

    m = pd.merge_asof(
        left.sort_values("Timestamp"),
        right.sort_values("Timestamp"),
        on="Timestamp", suffixes=("_Spot", "_Perp"), direction="backward"
    ).set_index("Timestamp")

    raw_s = spot_frames["1H"][["Open","High","Low"]].copy().add_suffix("_Spot")
    raw_p = perp_frames["1H"][["Open","High","Low"]].copy().add_suffix("_Perp")
    raw_s.index = pd.to_datetime(raw_s.index, utc=True).astype("datetime64[ns, UTC]")
    raw_p.index = pd.to_datetime(raw_p.index, utc=True).astype("datetime64[ns, UTC]")
    m = m.join(raw_s, how="left").join(raw_p, how="left")
    m["Both_Bullish"] = (m["MTF_Score_Spot"] >= 70) & (m["MTF_Score_Perp"] >= 70)
    m["Combined_MTF_Score"] = .60*m["MTF_Score_Spot"] + .40*m["MTF_Score_Perp"]
    m["Short_Score_Perp"] = 100.0 - m["MTF_Score_Perp"]
    m["Short_Combined_Score"] = 100.0 - m["Combined_MTF_Score"]
    funding = service.get_funding_history(symbol)
    fs = funding["fundingRate"] if not funding.empty else pd.Series(dtype=float)
    return m.dropna(subset=["Open_Spot","Close_Spot","Open_Perp","Close_Perp"]), fs

def run_one(hist, funding, strategy, threshold, start=0, end=None, costs=True):
    h = hist.iloc[start:end].copy()
    col, market = STRATEGIES[strategy]
    if strategy == "BOTH_MTF":
        mask = h[col].astype(bool)
        h["Signal"] = np.where(mask, 100.0, 0.0)
    else:
        h["Signal"] = h[col]
    if strategy == "COMBINED_70": threshold = 70
    if strategy == "COMBINED_80": threshold = 80
    direction = "short" if "SHORT" in strategy else "long"
    trades, equity = backtest_signals(
        h, "Signal", threshold=threshold, horizon_hours=HORIZON,
        fee=FEE if costs else 0, slippage=SLIPPAGE if costs else 0,
        stop_loss=STOP, take_profit=TARGET,
        direction=direction, funding=funding if market == "Perp" else None,
        price_prefix=market,
    )
    if trades.empty:
        return {
            "Trades": 0,
            "Total_Return_Pct": 0.0,
            "Win_Rate_Pct": np.nan,
            "Average_Trade_Pct": np.nan,
            "Profit_Factor": np.nan,
            "Max_Drawdown_Pct": 0.0,
        }, trades
    return {**{
        "Trades": len(trades), "Win_Rate_Pct": (trades["Net_Return"]>0).mean()*100,
        "Total_Return_Pct": (equity["Equity"].iloc[-1]-1)*100,
        "Average_Trade_Pct": trades["Net_Return"].mean()*100,
        "Profit_Factor": (trades.loc[trades.Net_Return>0,"Net_Return"].sum() /
                          -trades.loc[trades.Net_Return<0,"Net_Return"].sum())
                          if (trades.Net_Return<0).any() else np.nan,
        "Max_Drawdown_Pct": (
            pd.concat([pd.Series([1.0]), equity["Equity"].reset_index(drop=True)], ignore_index=True)
            / pd.concat([pd.Series([1.0]), equity["Equity"].reset_index(drop=True)], ignore_index=True).cummax()
            - 1.0
        ).min()*100,
    }}, trades

def main():
    p=ArgumentParser()
    p.add_argument("--symbols", default="AUTO")
    p.add_argument("--output-dir", default="reports/crypto_mtf_real_backtest")
    args=p.parse_args()
    out=ROOT/args.output_dir; out.mkdir(parents=True,exist_ok=True)
    service=BinanceMarketDataService(); analyzer=MultiTimeframeCryptoAnalyzer()
    if args.symbols.strip().upper() == "AUTO":
        spot = set(service.discover_spot_usdt(10_000_000))
        perp = set(service.discover_usdt_perpetuals(10_000_000))
        symbols = sorted(spot & perp, key=lambda x: x)
    else:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    oos_rows=[]; trade_rows=[]; all_summaries=[]
    status_rows=[]; errors=[]

    for symbol in symbols:
        print("REAL BACKTEST",symbol)
        status = {
            "Symbol": symbol, "Status": "running", "History_Bars": 0,
            "OOS_Folds": 0, "OOS_Trades": 0, "Reason": "", "Error": "",
        }
        try:
            hist,funding=merge_symbol(service,analyzer,symbol)
            status["History_Bars"] = int(len(hist))
            if len(hist)<300:
                status["Status"] = "insufficient_history"
                status["Reason"] = f"Only {len(hist)} aligned 1H bars; at least 300 required"
                status_rows.append(status)
                print("SKIP", symbol, status["Reason"])
                continue
            hist.to_csv(out/f"{symbol}_history.csv")
            n=len(hist)
            for strategy in STRATEGIES:
                # Anchored walk-forward: three chronological OOS windows.
                folds=[(0,int(n*.50),int(n*.50),int(n*.65)),
                       (0,int(n*.65),int(n*.65),int(n*.80)),
                       (0,int(n*.80),int(n*.80),n)]
                for fold,(train0,train1,test0,test1) in enumerate(folds,1):
                    scores=[]
                    train_cutoff_ts = hist.index[train1]
                    # Purged walk-forward selection: allow enough future bars to
                    # resolve the 72h label, but only keep trades whose signal
                    # AND exit are fully known before the train/test boundary.
                    train_end_with_labels = min(n, train1 + HORIZON + 1)
                    for threshold in THRESHOLDS:
                        s, tr = run_one(
                            hist, funding, strategy, threshold,
                            train0, train_end_with_labels
                        )
                        if not tr.empty:
                            tr = tr.loc[
                                (tr["Signal_Date"] < train_cutoff_ts)
                                & (tr["Exit_Date"] < train_cutoff_ts)
                            ]
                        avg = float(tr["Net_Return"].mean() * 100) if not tr.empty else -np.inf
                        s = dict(s)
                        s["Trades"] = len(tr)
                        s["Average_Trade_Pct"] = avg
                        scores.append((avg, threshold, s))
                    valid_scores = [candidate for candidate in scores if np.isfinite(candidate[0])]
                    if valid_scores:
                        best = max(valid_scores, key=lambda x: x[0])
                        selection_status = "TRAINED"
                    else:
                        # Never pick the first threshold arbitrarily when the training
                        # window has no completed trades; use a declared fallback and
                        # mark that this fold had no evidence for threshold selection.
                        best = (np.nan, 80, {})
                        selection_status = "FALLBACK_NO_TRAIN_TRADES"
                    test_stats,trades=run_one(hist,funding,strategy,best[1],test0,test1)
                    if len(trades) < 20:
                        # A bootstrap interval from a handful of trades is degenerate
                        # and should not be presented as a meaningful 95% confidence interval.
                        ci = {"Lower_95": np.nan, "Upper_95": np.nan}
                    else:
                        ci=block_bootstrap_mean(trades["Net_Return"],block_size=8,iterations=2000,random_state=42+fold)
                    row={"Symbol":symbol,"Strategy":strategy,"Fold":fold,
                         "Train_End":hist.index[train1-1],"Test_Start":hist.index[test0],
                         "Test_End":hist.index[test1-1],"Selected_Threshold":best[1],
                         "Threshold_Selection_Status":selection_status,
                         "Train_Avg_Trade_Pct":best[0],**test_stats,
                         "OOS_Mean_95_Lower_Pct":ci["Lower_95"]*100 if pd.notna(ci["Lower_95"]) else np.nan,
                         "OOS_Mean_95_Upper_Pct":ci["Upper_95"]*100 if pd.notna(ci["Upper_95"]) else np.nan}
                    oos_rows.append(row)
                    if not trades.empty:
                        trades=trades.copy(); trades["Symbol"]=symbol; trades["Strategy"]=strategy; trades["Fold"]=fold
                        trade_rows.append(trades)
                for threshold in (70,80):
                    s,tr=run_one(hist,funding,strategy,threshold)
                    all_summaries.append({"Symbol":symbol,"Strategy":strategy,"Threshold":threshold,**s})
            bh=buy_and_hold(hist,fee=FEE,slippage=SLIPPAGE)
            all_summaries.append({"Symbol":symbol,"Strategy":"BUY_AND_HOLD","Threshold":0,**bh})
            status["OOS_Folds"] = sum(1 for row in oos_rows if row["Symbol"] == symbol)
            status["OOS_Trades"] = int(sum(row["Trades"] for row in oos_rows if row["Symbol"] == symbol))
            status["Status"] = "completed" if status["OOS_Trades"] > 0 else "no_oos_trades"
            if status["Status"] == "no_oos_trades":
                status["Reason"] = "The walk-forward windows completed but produced no OOS trades"
            status_rows.append(status)
        except Exception as exc:
            status["Status"] = "error"
            status["Error"] = repr(exc)
            status_rows.append(status)
            errors.append((symbol, repr(exc)))
            print("ERROR",symbol,exc)

    pd.DataFrame(status_rows).to_csv(out/"backtest_status.csv",index=False)
    pd.DataFrame(all_summaries).to_csv(out/"strategy_pnl_comparison.csv",index=False)
    oos=pd.DataFrame(oos_rows); oos.to_csv(out/"walk_forward_oos.csv",index=False)
    if trade_rows: pd.concat(trade_rows,ignore_index=True).to_csv(out/"oos_trades.csv",index=False)
    if not oos.empty:
        fold_agg=(oos.groupby("Strategy").agg(Folds=("Fold","count"),OOS_Trades=("Trades","sum"),
              OOS_Win_Rate_Pct=("Win_Rate_Pct","mean"),Mean_Fold_Return_Pct=("Total_Return_Pct","mean"),
              OOS_Avg_Trade_Pct=("Average_Trade_Pct","mean"),OOS_Max_DD_Pct=("Max_Drawdown_Pct","mean"))
              .reset_index())
        if trade_rows:
            all_oos_trades=pd.concat(trade_rows,ignore_index=True)
            rows=[]
            for strategy,g in all_oos_trades.groupby("Strategy"):
                ci=block_bootstrap_mean(g["Net_Return"],block_size=4,iterations=5000,random_state=123)
                rows.append({"Strategy":strategy,"Aggregate_OOS_Trades":len(g),
                    "Aggregate_OOS_Win_Rate_Pct":float((g["Net_Return"]>0).mean()*100),
                    "Aggregate_OOS_Mean_Trade_Pct":float(g["Net_Return"].mean()*100),
                    "Aggregate_OOS_Median_Trade_Pct":float(g["Net_Return"].median()*100),
                    "Bootstrap_Lower_95_Pct":ci["Lower_95"]*100,
                    "Bootstrap_Upper_95_Pct":ci["Upper_95"]*100})
            agg=fold_agg.merge(pd.DataFrame(rows),on="Strategy",how="left")
        else:
            agg=fold_agg
        agg=agg.sort_values("Aggregate_OOS_Mean_Trade_Pct",ascending=False)
        agg.to_csv(out/"walk_forward_summary.csv",index=False)
        print(agg.to_string(index=False))
    if errors:
        print(f"BACKTEST FAILED: {len(errors)} symbol(s) had exceptions: {errors}")
        raise SystemExit(1)

if __name__=="__main__": main()
