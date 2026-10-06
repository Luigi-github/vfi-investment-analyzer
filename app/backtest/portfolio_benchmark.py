import numpy as np
import pandas as pd


class PortfolioBenchmark:
    """Passive benchmark curves using the same weekly price universe."""

    @staticmethod
    def _weekly_returns(asset_prices):
        returns = {}
        for symbol, frame in asset_prices.items():
            if "Close" not in frame.columns:
                raise ValueError(f"Close column is required for {symbol}.")
            close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            close.index = pd.DatetimeIndex(close.index)
            returns[symbol] = close.sort_index().pct_change()
        return pd.DataFrame(returns)

    def equal_weight_rebalanced(self, asset_prices, start_date=None):
        returns = self._weekly_returns(asset_prices)
        if start_date is not None:
            returns = returns.loc[pd.Timestamp(start_date):]

        rows = []
        equity = 1.0
        for date, row in returns.iterrows():
            valid = row.dropna()
            if valid.empty:
                continue
            period_return = float(valid.mean())
            equity *= 1.0 + period_return
            rows.append(
                {
                    "Date": date,
                    "Equity": equity,
                    "Net_Return": period_return * 100.0,
                    "Active_Assets": int(valid.size),
                }
            )

        curve = pd.DataFrame(rows)
        if not curve.empty:
            curve["Peak_Equity"] = curve["Equity"].cummax()
            curve["Drawdown"] = curve["Equity"] / curve["Peak_Equity"] - 1.0
        return curve

    def single_asset_buy_hold(self, price_frame, start_date=None):
        if "Close" not in price_frame.columns:
            raise ValueError("Column 'Close' is required.")

        close = pd.to_numeric(price_frame["Close"], errors="coerce").dropna()
        close.index = pd.DatetimeIndex(close.index)
        close = close.sort_index()
        if start_date is not None:
            close = close.loc[pd.Timestamp(start_date):]

        if len(close) < 2:
            return pd.DataFrame()

        equity = close / close.iloc[0]
        returns = close.pct_change().fillna(0.0)
        curve = pd.DataFrame(
            {
                "Date": close.index,
                "Equity": equity.values,
                "Net_Return": returns.values * 100.0,
            }
        )
        curve["Peak_Equity"] = curve["Equity"].cummax()
        curve["Drawdown"] = curve["Equity"] / curve["Peak_Equity"] - 1.0
        return curve

    @staticmethod
    def summarize(curve):
        if curve.empty:
            return {
                "Periods": 0,
                "Total_Return": np.nan,
                "CAGR": np.nan,
                "Annualized_Volatility": np.nan,
                "Sharpe": np.nan,
                "Sortino": np.nan,
                "Max_Drawdown": np.nan,
                "Positive_Period_Rate": np.nan,
            }

        returns = curve["Net_Return"] / 100.0
        equity = curve["Equity"]
        periods = len(returns)
        years = periods / 52.0
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        cagr = (
            float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
            if years > 0
            else np.nan
        )
        volatility = (
            float(returns.std(ddof=1) * np.sqrt(52.0))
            if periods > 1
            else np.nan
        )
        downside = returns[returns < 0]
        downside_dev = (
            float(downside.std(ddof=1) * np.sqrt(52.0))
            if len(downside) > 1
            else np.nan
        )

        return {
            "Periods": periods,
            "Total_Return": total_return * 100.0,
            "CAGR": cagr * 100.0,
            "Annualized_Volatility": volatility * 100.0
            if np.isfinite(volatility)
            else np.nan,
            "Sharpe": (
                float(returns.mean() * 52.0 / volatility)
                if volatility and volatility > 0
                else np.nan
            ),
            "Sortino": (
                float(returns.mean() * 52.0 / downside_dev)
                if downside_dev and downside_dev > 0
                else np.nan
            ),
            "Max_Drawdown": float(curve["Drawdown"].min() * 100.0),
            "Positive_Period_Rate": float((returns > 0).mean() * 100.0),
        }
