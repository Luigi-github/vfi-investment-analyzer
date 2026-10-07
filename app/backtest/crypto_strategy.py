from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_FEE = 0.0004
DEFAULT_SLIPPAGE = 0.0003
DEFAULT_FUNDING_HOURS = 8


def _cost_rate(fee: float, slippage: float) -> float:
    return 2.0 * (float(fee) + float(slippage))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1.0).min())


def _stats(trades: pd.DataFrame, equity: pd.Series) -> dict:
    if trades.empty:
        return {
            "Trades": 0, "Win_Rate_Pct": np.nan, "Total_Return_Pct": 0.0,
            "Max_Drawdown_Pct": 0.0, "Sharpe": np.nan, "Sortino": np.nan,
            "Profit_Factor": np.nan, "Expectancy_Pct": np.nan,
            "Average_Trade_Pct": np.nan, "Time_In_Market_Pct": 0.0,
        }
    r = trades["Net_Return"].astype(float)
    std = r.std(ddof=1)
    downside = r[r < 0].std(ddof=1)
    sharpe = float(r.mean() / std * np.sqrt(len(r))) if len(r) > 1 and std > 0 else np.nan
    sortino = float(r.mean() / downside * np.sqrt(len(r))) if len(r) > 1 and pd.notna(downside) and downside > 0 else np.nan
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return {
        "Trades": int(len(r)),
        "Win_Rate_Pct": float((r > 0).mean() * 100),
        "Total_Return_Pct": float((equity.iloc[-1] - 1.0) * 100),
        "Max_Drawdown_Pct": _max_drawdown(equity) * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Profit_Factor": float(gains / losses) if losses > 0 else np.inf,
        "Expectancy_Pct": float(r.mean() * 100),
        "Average_Trade_Pct": float(r.mean() * 100),
        "Time_In_Market_Pct": float(trades["Holding_Hours"].sum() / max(trades["Calendar_Hours"].sum(), 1.0) * 100),
    }


def backtest_signals(
    history: pd.DataFrame,
    signal_column: str,
    threshold: float = 80.0,
    horizon_hours: int = 72,
    fee: float = DEFAULT_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    direction: str = "long",
    funding: pd.Series | None = None,\n    price_prefix: str = "Spot",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Non-overlapping causal trade backtest on confirmed 1H signals.

    Signal at bar t is executed at the next bar's open. Exit is the first of
    horizon, stop, or target. Funding is charged/credited for perpetual trades.
    """
    if history.empty or signal_column not in history:
        return pd.DataFrame(), pd.DataFrame()
    df = history.sort_index().copy()
    if not {"Open_Spot", "High_Spot", "Low_Spot", "Close_Spot"}.issubset(df.columns):
        raise ValueError("history must contain Spot OHLC columns")
    funding = funding.sort_index() if funding is not None else pd.Series(dtype=float)
    trades = []
    i = 0
    idx = df.index
    while i < len(df) - 2:
        if float(df.iloc[i][signal_column]) < threshold:
            i += 1
            continue
        entry_i = i + 1
        entry_ts = idx[entry_i]
        entry = float(df.iloc[entry_i][f"Open_{price_prefix}"])
        end_i = min(len(df) - 1, entry_i + int(horizon_hours))
        exit_i = end_i
        exit_price = float(df.iloc[end_i][f"Close_{price_prefix}"])
        reason = "HORIZON"
        for j in range(entry_i, end_i + 1):
            hi = float(df.iloc[j][f"High_{price_prefix}"])
            lo = float(df.iloc[j][f"Low_{price_prefix}"])
            if direction == "long":
                if stop_loss is not None and lo <= entry * (1 - stop_loss):
                    exit_i, exit_price, reason = j, entry * (1 - stop_loss), "STOP"
                    break
                if take_profit is not None and hi >= entry * (1 + take_profit):
                    exit_i, exit_price, reason = j, entry * (1 + take_profit), "TARGET"
                    break
            else:
                if stop_loss is not None and hi >= entry * (1 + stop_loss):
                    exit_i, exit_price, reason = j, entry * (1 + stop_loss), "STOP"
                    break
                if take_profit is not None and lo <= entry * (1 - take_profit):
                    exit_i, exit_price, reason = j, entry * (1 - take_profit), "TARGET"
                    break
        gross = exit_price / entry - 1.0 if direction == "long" else entry / exit_price - 1.0
        costs = _cost_rate(fee, slippage)
        funding_pnl = 0.0
        if direction == "long" and len(funding):
            f = funding.loc[(funding.index > entry_ts) & (funding.index <= idx[exit_i])]
            funding_pnl = -float(f.sum())
        net = gross - costs + funding_pnl
        holding_hours = max((idx[exit_i] - entry_ts).total_seconds() / 3600.0, 0.0)
        trades.append({
            "Signal_Date": idx[i], "Entry_Date": entry_ts, "Exit_Date": idx[exit_i],
            "Entry_Price": entry, "Exit_Price": exit_price, "Gross_Return": gross,
            "Costs": costs, "Funding_PnL": funding_pnl, "Net_Return": net,
            "Holding_Hours": holding_hours, "Calendar_Hours": holding_hours,
            "Exit_Reason": reason, "Direction": direction,
        })
        i = exit_i + 1
    t = pd.DataFrame(trades)
    if t.empty:
        return t, pd.DataFrame()
    equity = [1.0]
    for ret in t["Net_Return"]:
        equity.append(equity[-1] * (1.0 + ret))
    eq = pd.DataFrame({"Equity": equity[1:]}, index=t["Exit_Date"])
    return t, eq


def buy_and_hold(history: pd.DataFrame, fee: float = DEFAULT_FEE, slippage: float = DEFAULT_SLIPPAGE) -> dict:
    if history.empty:
        return {"Total_Return_Pct": np.nan}
    first = float(history["Open_Spot"].iloc[0])
    last = float(history["Close_Spot"].iloc[-1])
    return {"Total_Return_Pct": float((last / first - 1.0 - _cost_rate(fee, slippage)) * 100)}
