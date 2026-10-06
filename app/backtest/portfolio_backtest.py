import numpy as np
import pandas as pd


class PortfolioBacktester:
    """Point-in-time long-only portfolio backtester for causal event signals.

    Signals generated at a bar close become executable on the next available
    market observation. This prevents using the signal bar's closing price as
    an executable entry price.
    """

    def __init__(
        self,
        horizon_weeks=26,
        max_positions=10,
        weighting="equal",
        volatility_lookback=26,
        min_history_weeks=12,
        max_position_weight=None,
        fee_bps=0.0,
        slippage_bps=0.0,
    ):
        if horizon_weeks <= 0 or max_positions <= 0:
            raise ValueError("horizon_weeks and max_positions must be positive.")
        if weighting not in {"equal", "inverse_vol", "risk_parity"}:
            raise ValueError("Unsupported weighting mode.")
        if volatility_lookback <= 1 or min_history_weeks <= 1:
            raise ValueError("History windows must be greater than 1.")
        if max_position_weight is not None and not 0 < max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1].")
        if fee_bps < 0 or slippage_bps < 0:
            raise ValueError("Costs must be non-negative.")

        self.horizon_weeks = int(horizon_weeks)
        self.max_positions = int(max_positions)
        self.weighting = weighting
        self.volatility_lookback = int(volatility_lookback)
        self.min_history_weeks = int(min_history_weeks)
        self.max_position_weight = max_position_weight
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)

    @staticmethod
    def _normalize_prices(asset_prices):
        normalized = {}
        for symbol, frame in asset_prices.items():
            if "Close" not in frame.columns:
                raise ValueError(f"Close column is required for {symbol}.")
            close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            close.index = pd.DatetimeIndex(close.index)
            normalized[symbol] = close.sort_index()
        return normalized

    @staticmethod
    def _risk_parity_weights(covariance):
        from scipy.optimize import minimize
        n = covariance.shape[0]
        if n == 1:
            return np.ones(1)
        cov = np.asarray(covariance, dtype=float)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov = (cov + cov.T) / 2.0
        cov += np.eye(n) * 1e-8

        def objective(weights):
            portfolio_var = float(weights @ cov @ weights)
            if portfolio_var <= 0:
                return 0.0
            marginal = cov @ weights
            contributions = weights * marginal / np.sqrt(portfolio_var)
            target = np.mean(contributions)
            return float(np.sum((contributions - target) ** 2))

        result = minimize(
            objective, np.full(n, 1.0 / n), method="SLSQP",
            bounds=[(1e-8, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            options={"maxiter": 500, "ftol": 1e-10},
        )
        return np.asarray(result.x, dtype=float) if result.success else np.full(n, 1.0 / n)

    @staticmethod
    def _apply_cap(weights, cap):
        if cap is None:
            return np.asarray(weights, dtype=float)
        weights = np.asarray(weights, dtype=float)
        result = np.zeros_like(weights)
        remaining = np.ones(len(weights), dtype=bool)
        target = 1.0
        for _ in range(len(weights) + 1):
            if not remaining.any() or target <= 1e-12:
                break
            base = weights[remaining]
            if base.sum() <= 0:
                break
            indices = np.flatnonzero(remaining)
            proposal = base / base.sum() * target
            over = proposal > cap + 1e-12
            if not over.any():
                result[indices] = proposal
                break
            capped_indices = indices[over]
            result[capped_indices] = cap
            target -= cap * len(capped_indices)
            remaining[capped_indices] = False
        total = result.sum()
        return result / total if total > 1.0 + 1e-12 else result

    def _select_active(self, rows):
        if len(rows) <= self.max_positions:
            return rows
        rows = rows.copy()
        if "Signal_Score" in rows.columns:
            rows["_selection"] = pd.to_numeric(rows["Signal_Score"], errors="coerce").fillna(-np.inf)
            return rows.sort_values(["_selection", "Symbol"], ascending=[False, True]).head(self.max_positions).drop(columns="_selection")
        return rows.sort_values("Symbol").head(self.max_positions)

    def _weights_for_date(self, active, prices, date):
        if active.empty:
            return pd.Series(dtype=float)
        active = self._select_active(active)
        symbols = active["Symbol"].tolist()
        history = pd.DataFrame({symbol: prices[symbol] for symbol in symbols}).loc[:date]
        returns = history.pct_change().tail(self.volatility_lookback)
        valid = returns.count()
        eligible = [s for s in symbols if valid.get(s, 0) >= self.min_history_weeks]
        if not eligible:
            eligible = symbols
        active = active[active["Symbol"].isin(eligible)]
        symbols = active["Symbol"].tolist()
        if not symbols:
            return pd.Series(dtype=float)

        if self.weighting == "equal":
            raw = np.full(len(symbols), 1.0 / len(symbols))
        elif self.weighting == "inverse_vol":
            vol = returns[symbols].std(ddof=1)
            inv = 1.0 / vol.replace(0, np.nan)
            inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            raw = inv.to_numpy(dtype=float)
            raw = raw / raw.sum() if raw.sum() > 0 else np.full(len(symbols), 1.0 / len(symbols))
        else:
            raw = self._risk_parity_weights(returns[symbols].cov().fillna(0.0).to_numpy())

        raw = self._apply_cap(raw, self.max_position_weight)
        return pd.Series(raw, index=symbols, dtype=float)

    def run(self, asset_prices, signals):
        """Run a point-in-time weekly portfolio from causal signals."""
        prices = self._normalize_prices(asset_prices)
        if not isinstance(signals.index, pd.DatetimeIndex):
            raise ValueError("signals must use a DatetimeIndex.")
        if "Symbol" not in signals.columns:
            raise ValueError("Missing signal column: Symbol.")

        signals = signals.copy()
        signals.index = pd.DatetimeIndex(signals.index)
        signals = signals.sort_index()
        signals = signals[signals["Symbol"].isin(prices.keys())]
        if signals.empty:
            return {"equity_curve": pd.DataFrame(), "trades": pd.DataFrame(), "summary": self._summary(pd.Series(dtype=float))}

        calendar = sorted(set().union(*(set(series.index) for series in prices.values())))
        calendar = pd.DatetimeIndex(calendar)
        calendar = calendar[calendar >= signals.index.min()]
        if len(calendar) < 2:
            return {"equity_curve": pd.DataFrame(), "trades": pd.DataFrame(), "summary": self._summary(pd.Series(dtype=float))}

        # A signal observed at t is executable at t+1, never at t.
        signal_positions = {}
        for signal_date, group in signals.groupby(signals.index):
            positions = calendar.searchsorted(signal_date, side="right")
            if positions >= len(calendar):
                continue
            entry_date = calendar[positions]
            signal_positions.setdefault(entry_date, []).append((signal_date, group.copy()))

        active = []
        equity = 1.0
        previous_weights = pd.Series(dtype=float)
        curve_rows = []
        trade_rows = []

        for position in range(len(calendar) - 1):
            date = calendar[position]
            next_date = calendar[position + 1]

            active = [trade for trade in active if date < trade["exit_date"]]

            for signal_date, group in signal_positions.get(date, []):
                for _, signal in group.iterrows():
                    symbol = signal["Symbol"]
                    if date not in prices[symbol].index:
                        continue
                    active.append({
                        "Symbol": symbol,
                        "signal_date": signal_date,
                        "entry_date": date,
                        "exit_date": date + pd.Timedelta(weeks=self.horizon_weeks),
                        "entry_price": float(prices[symbol].loc[date]),
                        "Signal_Score": signal.get("Signal_Score", np.nan),
                    })

            active_frame = pd.DataFrame(active)
            target_weights = self._weights_for_date(active_frame, prices, date) if not active_frame.empty else pd.Series(dtype=float)
            available = [symbol for symbol in target_weights.index if date in prices[symbol].index and next_date in prices[symbol].index]
            target_weights = target_weights.reindex(available).dropna()
            if not target_weights.empty:
                target_weights = target_weights / target_weights.sum()

            turnover = float(previous_weights.sub(target_weights, fill_value=0.0).abs().sum())
            gross_return = 0.0
            realized_positions = []
            for symbol, weight in target_weights.items():
                asset_return = float(prices[symbol].loc[next_date] / prices[symbol].loc[date] - 1.0)
                gross_return += float(weight) * asset_return
                realized_positions.append({
                    "Symbol": symbol,
                    "Weight": float(weight),
                    "Asset_Return": asset_return * 100.0,
                })

            transaction_rate = (self.fee_bps + self.slippage_bps) / 10000.0
            transaction_cost = turnover * transaction_rate
            net_return = gross_return - transaction_cost
            equity *= 1.0 + net_return

            for row in realized_positions:
                trade_rows.append({
                    "Signal_Date": next(
                        (trade["signal_date"] for trade in active if trade["Symbol"] == row["Symbol"] and trade["entry_date"] == date),
                        pd.NaT,
                    ),
                    "Entry_Date": date,
                    "Symbol": row["Symbol"],
                    "Weight": row["Weight"],
                    "Asset_Return": row["Asset_Return"],
                    "Portfolio_Contribution": row["Weight"] * row["Asset_Return"],
                })

            curve_rows.append({
                "Date": next_date,
                "Equity": equity,
                "Gross_Return": gross_return * 100.0,
                "Net_Return": net_return * 100.0,
                "Turnover": turnover,
                "Transaction_Cost": transaction_cost * 100.0,
                "Active_Positions": len(target_weights),
                "Exposure": float(target_weights.sum()),
            })
            previous_weights = target_weights

        curve = pd.DataFrame(curve_rows)
        if not curve.empty:
            curve["Peak_Equity"] = curve["Equity"].cummax()
            curve["Drawdown"] = curve["Equity"] / curve["Peak_Equity"] - 1.0
            curve["Underwater_Weeks"] = self._underwater_duration(curve["Drawdown"])

        trades = pd.DataFrame(trade_rows)
        returns = curve["Net_Return"] / 100.0 if not curve.empty else pd.Series(dtype=float)
        return {"equity_curve": curve, "trades": trades, "summary": self._summary(returns, curve)}

    @staticmethod
    def _underwater_duration(drawdown):
        result, current = [], 0
        for value in drawdown:
            if value < 0:
                current += 1
            else:
                current = 0
            result.append(current)
        return result

    @staticmethod
    def _summary(returns, curve=None):
        if returns.empty:
            return {"Periods": 0, "Total_Return": np.nan, "CAGR": np.nan, "Annualized_Volatility": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "Max_Drawdown": np.nan, "Positive_Period_Rate": np.nan, "Average_Turnover": np.nan, "Profit_Factor": np.nan, "Expectancy": np.nan, "Calmar": np.nan, "Max_Underwater_Weeks": np.nan}

        equity = (1.0 + returns.fillna(0.0)).cumprod()
        periods = len(returns)
        years = periods / 52.0
        total_return = float(equity.iloc[-1] - 1.0)
        cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else np.nan
        vol = float(returns.std(ddof=1) * np.sqrt(52.0)) if periods > 1 else np.nan
        downside = returns[returns < 0]
        downside_dev = float(downside.std(ddof=1) * np.sqrt(52.0)) if len(downside) > 1 else np.nan
        sharpe = float(returns.mean() * 52.0 / vol) if vol and vol > 0 else np.nan
        sortino = float(returns.mean() * 52.0 / downside_dev) if downside_dev and downside_dev > 0 else np.nan
        running = equity.cummax()
        drawdowns = equity / running - 1.0
        max_drawdown = float(drawdowns.min())
        wins, losses = returns[returns > 0], returns[returns < 0]
        profit_factor = float(wins.sum() / abs(losses.sum())) if not losses.empty and losses.sum() != 0 else np.nan
        expectancy = float(returns.mean())
        calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 and np.isfinite(cagr) else np.nan

        summary = {
            "Periods": periods,
            "Total_Return": total_return * 100.0,
            "CAGR": cagr * 100.0 if np.isfinite(cagr) else np.nan,
            "Annualized_Volatility": vol * 100.0 if np.isfinite(vol) else np.nan,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max_Drawdown": max_drawdown * 100.0,
            "Positive_Period_Rate": float((returns > 0).mean() * 100.0),
            "Average_Turnover": float(curve["Turnover"].mean()) if curve is not None and not curve.empty else np.nan,
            "Profit_Factor": profit_factor,
            "Expectancy": expectancy * 100.0,
            "Calmar": calmar,
            "Max_Underwater_Weeks": float(curve["Underwater_Weeks"].max()) if curve is not None and not curve.empty else np.nan,
        }
        return summary
