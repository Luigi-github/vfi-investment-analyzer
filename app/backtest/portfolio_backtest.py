import numpy as np
import pandas as pd


class PortfolioBacktester:
    """Point-in-time long-only portfolio backtester for causal event signals."""

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
        if horizon_weeks <= 0:
            raise ValueError("horizon_weeks must be positive.")
        if max_positions <= 0:
            raise ValueError("max_positions must be positive.")
        if weighting not in {"equal", "inverse_vol", "risk_parity"}:
            raise ValueError("Unsupported weighting mode.")
        if volatility_lookback <= 1:
            raise ValueError("volatility_lookback must be greater than 1.")
        if min_history_weeks <= 1:
            raise ValueError("min_history_weeks must be greater than 1.")
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
        """Equal-risk-contribution weights using a deterministic SLSQP solve."""
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
            objective,
            np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=[(1e-8, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success:
            return np.full(n, 1.0 / n)
        return np.asarray(result.x, dtype=float)

    @staticmethod
    def _apply_cap(weights, cap):
        if cap is None:
            return weights

        weights = np.asarray(weights, dtype=float)
        result = np.zeros_like(weights)
        remaining = np.ones(len(weights), dtype=bool)
        target = 1.0

        for _ in range(len(weights) + 1):
            if not remaining.any() or target <= 1e-12:
                break
            base = weights[remaining]
            base_sum = base.sum()
            if base_sum <= 0:
                break
            proposal = base / base_sum * target
            over = proposal > cap + 1e-12
            indices = np.flatnonzero(remaining)
            if not over.any():
                result[indices] = proposal
                target = 0.0
                break
            capped_indices = indices[over]
            result[capped_indices] = cap
            target -= cap * len(capped_indices)
            remaining[capped_indices] = False

        total = result.sum()
        if total > 1.0 + 1e-12:
            result /= total
        return result

    def _select_active(self, rows):
        if len(rows) <= self.max_positions:
            return rows

        rows = rows.copy()
        if "Signal_Score" in rows.columns:
            rows["_selection"] = pd.to_numeric(
                rows["Signal_Score"], errors="coerce"
            ).fillna(-np.inf)
            return rows.sort_values(
                ["_selection", "Symbol"], ascending=[False, True]
            ).head(self.max_positions).drop(columns="_selection")

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

        if self.weighting == "equal":
            raw = np.full(len(symbols), 1.0 / len(symbols))
        elif self.weighting == "inverse_vol":
            vol = returns[symbols].std(ddof=1)
            inv = 1.0 / vol.replace(0, np.nan)
            inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            raw = inv.to_numpy(dtype=float)
            if raw.sum() <= 0:
                raw = np.ones(len(symbols))
            raw /= raw.sum()
        else:
            covariance = returns[symbols].cov().fillna(0.0).to_numpy()
            raw = self._risk_parity_weights(covariance)

        raw = self._apply_cap(raw, self.max_position_weight)
        return pd.Series(raw, index=symbols, dtype=float)

    def run(self, asset_prices, signals):
        """Run the portfolio and return weekly equity/position diagnostics."""
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
            return {
                "equity_curve": pd.DataFrame(),
                "trades": pd.DataFrame(),
                "summary": self._summary(pd.Series(dtype=float)),
            }

        calendar = sorted(
            set().union(*(set(series.index) for series in prices.values()))
        )
        calendar = pd.DatetimeIndex(calendar)
        calendar = calendar[calendar >= signals.index.min()]
        if len(calendar) < 2:
            return {
                "equity_curve": pd.DataFrame(),
                "trades": pd.DataFrame(),
                "summary": self._summary(pd.Series(dtype=float)),
            }

        entries_by_date = {
            date: group.copy() for date, group in signals.groupby(signals.index)
        }

        active = []
        equity = 1.0
        previous_weights = pd.Series(dtype=float)
        curve_rows = []
        trade_rows = []

        for position in range(len(calendar) - 1):
            date = calendar[position]
            next_date = calendar[position + 1]

            active = [trade for trade in active if date < trade["exit_date"]]

            for _, signal in entries_by_date.get(date, pd.DataFrame()).iterrows():
                symbol = signal["Symbol"]
                if date not in prices[symbol].index:
                    continue
                active.append(
                    {
                        "Symbol": symbol,
                        "entry_date": date,
                        "exit_date": date + pd.Timedelta(weeks=self.horizon_weeks),
                        "entry_price": float(prices[symbol].loc[date]),
                    }
                )

            active_frame = pd.DataFrame(active)
            target_weights = (
                self._weights_for_date(active_frame, prices, date)
                if not active_frame.empty
                else pd.Series(dtype=float)
            )

            turnover = float(
                previous_weights.sub(target_weights, fill_value=0.0).abs().sum()
            )
            gross_return = 0.0
            realized_positions = []

            for symbol, weight in target_weights.items():
                series = prices[symbol]
                if date not in series.index or next_date not in series.index:
                    continue
                asset_return = float(series.loc[next_date] / series.loc[date] - 1.0)
                gross_return += float(weight) * asset_return
                realized_positions.append(
                    {
                        "Symbol": symbol,
                        "Weight": float(weight),
                        "Asset_Return": asset_return * 100.0,
                    }
                )

            transaction_rate = (self.fee_bps + self.slippage_bps) / 10000.0
            transaction_cost = turnover * transaction_rate
            net_return = gross_return - transaction_cost
            equity *= 1.0 + net_return

            for row in realized_positions:
                trade_rows.append(
                    {
                        "Date": date,
                        "Symbol": row["Symbol"],
                        "Weight": row["Weight"],
                        "Asset_Return": row["Asset_Return"],
                        "Portfolio_Contribution": row["Weight"] * row["Asset_Return"],
                    }
                )

            curve_rows.append(
                {
                    "Date": next_date,
                    "Equity": equity,
                    "Gross_Return": gross_return * 100.0,
                    "Net_Return": net_return * 100.0,
                    "Turnover": turnover,
                    "Transaction_Cost": transaction_cost * 100.0,
                    "Active_Positions": len(target_weights),
                    "Exposure": float(target_weights.sum()),
                }
            )
            previous_weights = target_weights

        curve = pd.DataFrame(curve_rows)
        if not curve.empty:
            curve["Peak_Equity"] = curve["Equity"].cummax()
            curve["Drawdown"] = curve["Equity"] / curve["Peak_Equity"] - 1.0

        trades = pd.DataFrame(trade_rows)
        returns = curve["Net_Return"] / 100.0 if not curve.empty else pd.Series(dtype=float)
        return {
            "equity_curve": curve,
            "trades": trades,
            "summary": self._summary(returns, curve),
        }

    @staticmethod
    def _summary(returns, curve=None):
        if returns.empty:
            return {
                "Periods": 0,
                "Total_Return": np.nan,
                "CAGR": np.nan,
                "Annualized_Volatility": np.nan,
                "Sharpe": np.nan,
                "Sortino": np.nan,
                "Max_Drawdown": np.nan,
                "Positive_Period_Rate": np.nan,
                "Average_Turnover": np.nan,
            }

        equity = (1.0 + returns.fillna(0.0)).cumprod()
        periods = len(returns)
        years = periods / 52.0
        total_return = float(equity.iloc[-1] - 1.0)
        cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
        vol = float(returns.std(ddof=1) * np.sqrt(52.0)) if periods > 1 else np.nan
        downside = returns[returns < 0]
        downside_dev = (
            float(downside.std(ddof=1) * np.sqrt(52.0))
            if len(downside) > 1 else np.nan
        )
        sharpe = float(returns.mean() * 52.0 / vol) if vol and vol > 0 else np.nan
        sortino = (
            float(returns.mean() * 52.0 / downside_dev)
            if downside_dev and downside_dev > 0 else np.nan
        )
        running = equity.cummax()
        max_drawdown = float((equity / running - 1.0).min())

        summary = {
            "Periods": periods,
            "Total_Return": total_return * 100.0,
            "CAGR": cagr * 100.0,
            "Annualized_Volatility": vol * 100.0 if np.isfinite(vol) else np.nan,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max_Drawdown": max_drawdown * 100.0,
            "Positive_Period_Rate": float((returns > 0).mean() * 100.0),
            "Average_Turnover": np.nan,
        }
        if curve is not None and not curve.empty:
            summary["Average_Turnover"] = float(curve["Turnover"].mean())
        return summary
