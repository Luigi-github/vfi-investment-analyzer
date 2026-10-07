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
    "PERP_MTF": ("MTF_Score_Perp", "Perp"),\n    "PERP_SHORT": ("Short_Score_Perp", "Perp"),
    "BOTH_MTF": ("Both_Bullish", "Spot"),
    "COMBINED_70": ("Combined_MTF_Score", "Spot"),
    "COMBINED_80": ("Combined_MTF_Score", "Spot"),\n    "PERP_COMBINED_SHORT": ("Short_Combined_Score", "Perp"),
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
    m = pd.merge_asof(
        sh.reset_index().sort_values("Timestamp"),
        ph.reset_index().sort_values("Timestamp"),
        on="Timestamp", suffixes=("_Spot", "_Perp"), direction="backward"
    ).set_index("Timestamp")
    raw_s = spot_frames["1H"][["Open","High","Low","Close"]].add_suffix("_Spot")
    raw_p = perp_frames["1H"][["Open","High","Low","Close"]].add_suffix("_Perp")
    m = m.join(raw_s, how="left").join(raw_p, how="left")
    m["Both_Bullish"] = (m["MTF_Score_Spot"] >= 70) & (m["MTF_Score_Perp"] >= 70)
    m["Combined_MTF_Score"] = .60*m["MTF_Score_Spot"] + .40*m["MTF_Score_Perp"]\n    m["Short_Score_Perp"] = 100.0 - m["MTF_Score_Perp"]\n    m["Short_Combined_Score"] = 100.0 - m["Combined_MTF_Score"]
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
    if strategy == "COMBINED_80": threshold = 80\n    direction = "short" if "SHORT" in strategy else "long"
    trades, equity = backtest_signals(
        h, "Signal", threshold=threshold, horizon_hours=HORIZON,
        fee=FEE if costs else 0, slippage=SLIPPAGE if costs else 0,
        stop_loss=STOP, take_profit=TARGET,
        direction=direction, funding=funding if market == "Perp" else None,
        price_prefix=market,
    )
    if trades.empty:
        return {"Trades":0, "Total_Return_Pct":0.0, "Win_Rate_Pct":np.nan}, trades
    return {**{
        "Trades": len(trades), "Win_Rate_Pct": (trades["Net_Return"]>0).mean()*100,
        "Total_Return_Pct": (equity["Equity"].iloc[-1]-1)*100,
        "Average_Trade_Pct": trades["Net_Return"].mean()*100,
        "Profit_Factor": (trades.loc[trades.Net_Return>0,"Net_Return"].sum() /
                          -trades.loc[trades.Net_Return<0,"Net_Return"].sum())
                          if (trades.Net_Return<0).any() else np.inf,
        "Max_Drawdown_Pct": (equity["Equity"]/equity["Equity"].cummax()-1).min()*100,
    }}, trades

def main():
    p=ArgumentParser()
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT")
    p.add_argument("--output-dir", default="reports/crypto_mtf_real_backtest")
    args=p.parse_args()
    out=ROOT/args.output_dir; out.mkdir(parents=True,exist_ok=True)
    service=BinanceMarketDataService(); analyzer=MultiTimeframeCryptoAnalyzer()
    oos_rows=[]; trade_rows=[]; all_summaries=[]

    for symbol in [x.strip().upper() for x in args.symbols.split(",") if x.strip()]:
        print("REAL BACKTEST",symbol)
        try:
            hist,funding=merge_symbol(service,analyzer,symbol)
            if len(hist)<300: continue
            hist.to_csv(out/f"{symbol}_history.csv")
            n=len(hist)
            for strategy in STRATEGIES:
                # Anchored walk-forward: three chronological OOS windows.
                folds=[(0,int(n*.50),int(n*.50),int(n*.65)),
                       (0,int(n*.65),int(n*.65),int(n*.80)),
                       (0,int(n*.80),int(n*.80),n)]
                for fold,(train0,train1,test0,test1) in enumerate(folds,1):
                    scores=[]
                    for threshold in THRESHOLDS:
                        s,_=run_one(hist,funding,strategy,threshold,train0,train1)
                        scores.append((s.get("Average_Trade_Pct",-np.inf),threshold,s))
                    best=max(scores,key=lambda x:x[0])
                    test_stats,trades=run_one(hist,funding,strategy,best[1],test0,test1)
                    if trades.empty: continue
                    ci=block_bootstrap_mean(trades["Net_Return"],block_size=8,iterations=2000,random_state=42+fold)
                    row={"Symbol":symbol,"Strategy":strategy,"Fold":fold,
                         "Train_End":hist.index[train1-1],"Test_Start":hist.index[test0],
                         "Test_End":hist.index[test1-1],"Selected_Threshold":best[1],
                         "Train_Avg_Trade_Pct":best[0],**test_stats,
                         "OOS_Mean_95_Lower_Pct":ci["Lower_95"]*100,
                         "OOS_Mean_95_Upper_Pct":ci["Upper_95"]*100}
                    oos_rows.append(row)
                    if not trades.empty:
                        trades=trades.copy(); trades["Symbol"]=symbol; trades["Strategy"]=strategy; trades["Fold"]=fold
                        trade_rows.append(trades)
                for threshold in (70,80):
                    s,tr=run_one(hist,funding,strategy,threshold)
                    all_summaries.append({"Symbol":symbol,"Strategy":strategy,"Threshold":threshold,**s})
            bh=buy_and_hold(hist,fee=FEE,slippage=SLIPPAGE)
            all_summaries.append({"Symbol":symbol,"Strategy":"BUY_AND_HOLD","Threshold":0,**bh})
        except Exception as exc:
            print("ERROR",symbol,exc)

    pd.DataFrame(all_summaries).to_csv(out/"strategy_pnl_comparison.csv",index=False)
    oos=pd.DataFrame(oos_rows); oos.to_csv(out/"walk_forward_oos.csv",index=False)
    if trade_rows: pd.concat(trade_rows,ignore_index=True).to_csv(out/"oos_trades.csv",index=False)
    if not oos.empty:
        agg=(oos.groupby("Strategy").agg(Folds=("Fold","count"),OOS_Trades=("Trades","sum"),
              OOS_Win_Rate_Pct=("Win_Rate_Pct","mean"),OOS_Total_Return_Pct=("Total_Return_Pct","mean"),
              OOS_Avg_Trade_Pct=("Average_Trade_Pct","mean"),OOS_Max_DD_Pct=("Max_Drawdown_Pct","mean"),
              CI_Lower_Pct=("OOS_Mean_95_Lower_Pct","mean"),CI_Upper_Pct=("OOS_Mean_95_Upper_Pct","mean"))
              .reset_index().sort_values("OOS_Avg_Trade_Pct",ascending=False))
        agg.to_csv(out/"walk_forward_summary.csv",index=False)
        print(agg.to_string(index=False))

if __name__=="__main__": main()
