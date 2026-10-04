import numpy as np
import pandas as pd


class StatisticalRobustnessAnalyzer:
    """Assess benchmark-relative evidence and multiple-testing robustness."""

    def __init__(self, bootstrap_samples=5000, confidence=0.95, random_seed=42):
        if bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive.")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1.")
        self.bootstrap_samples = int(bootstrap_samples)
        self.confidence = float(confidence)
        self.random_seed = int(random_seed)

    @staticmethod
    def _finite(values):
        result = pd.Series(values, dtype="float64").dropna().to_numpy()
        if result.size == 0 or not np.isfinite(result).all():
            raise ValueError("Values must contain at least one finite observation.")
        return result

    def bootstrap_excess(self, event_returns, benchmark_returns):
        """Bootstrap paired excess returns when both series are aligned observations."""
        events = pd.Series(event_returns, dtype="float64")
        benchmark = pd.Series(benchmark_returns, dtype="float64")
        aligned = pd.concat([events, benchmark], axis=1).dropna()
        if aligned.empty:
            raise ValueError("Paired event and benchmark returns are required.")

        excess = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).to_numpy()
        rng = np.random.default_rng(self.random_seed)
        samples = rng.choice(
            excess,
            size=(self.bootstrap_samples, excess.size),
            replace=True,
        )
        means = samples.mean(axis=1)
        alpha = 1.0 - self.confidence
        low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
        return {
            "Excess_Mean": float(excess.mean()),
            "CI_Lower": float(low),
            "CI_Upper": float(high),
            "Count": int(excess.size),
        }

    def bootstrap_excess_independent(self, event_returns, benchmark_returns):
        """Bootstrap the difference between two independent return samples."""
        events = self._finite(event_returns)
        benchmark = self._finite(benchmark_returns)
        rng = np.random.default_rng(self.random_seed)
        event_means = rng.choice(
            events, size=(self.bootstrap_samples, events.size), replace=True
        ).mean(axis=1)
        benchmark_means = rng.choice(
            benchmark, size=(self.bootstrap_samples, benchmark.size), replace=True
        ).mean(axis=1)
        excess = event_means - benchmark_means
        alpha = 1.0 - self.confidence
        low, high = np.quantile(excess, [alpha / 2.0, 1.0 - alpha / 2.0])
        return {
            "Excess_Mean": float(events.mean() - benchmark.mean()),
            "CI_Lower": float(low),
            "CI_Upper": float(high),
        }

    @staticmethod
    def benjamini_hochberg(p_values):
        """Return BH-adjusted q-values in original order."""
        values = np.asarray(p_values, dtype="float64")
        if values.size == 0:
            return np.array([], dtype="float64")
        if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("p_values must be finite probabilities in [0, 1].")

        order = np.argsort(values)
        ranked = values[order]
        n = len(ranked)
        adjusted = ranked * n / np.arange(1, n + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)

        result = np.empty(n, dtype="float64")
        result[order] = adjusted
        return result

    @staticmethod
    def one_sample_bootstrap_pvalue(values, null_value=0.0, samples=5000, seed=42):
        """Two-sided bootstrap p-value for a mean against a fixed null."""
        data = StatisticalRobustnessAnalyzer._finite(values)
        centered = data - data.mean() + null_value
        rng = np.random.default_rng(seed)
        sample_means = rng.choice(
            centered,
            size=(samples, data.size),
            replace=True,
        ).mean(axis=1)
        observed = abs(data.mean() - null_value)
        p_value = (np.sum(np.abs(sample_means - null_value) >= observed) + 1) / (samples + 1)
        return float(p_value)

    def test_conditions(self, returns_by_condition):
        """Estimate condition evidence and adjust p-values for multiple testing."""
        rows = []
        raw_p = []

        for name, returns in returns_by_condition.items():
            values = self._finite(returns)
            p = self.one_sample_bootstrap_pvalue(
                values,
                samples=self.bootstrap_samples,
                seed=self.random_seed,
            )
            summary = {
                "Condition": name,
                "Count": len(values),
                "Average_Return": float(values.mean()),
                "Positive_Rate": float((values > 0).mean() * 100.0),
                "Raw_P_Value": p,
            }
            rows.append(summary)
            raw_p.append(p)

        if not rows:
            return pd.DataFrame()

        q = self.benjamini_hochberg(raw_p)
        result = pd.DataFrame(rows)
        result["BH_Q_Value"] = q
        result["BH_Significant_5pct"] = result["BH_Q_Value"] <= 0.05
        return result.sort_values(
            ["BH_Q_Value", "Average_Return"],
            ascending=[True, False],
        ).reset_index(drop=True)

    def stability_score(
        self,
        average_return,
        positive_rate,
        valid_windows,
        worst_return,
        benchmark_excess=0.0,
        significant=False,
    ):
        """Conservative 0-100 research robustness score, not a trading signal."""
        return_score = np.clip((average_return + 20.0) / 60.0, 0, 1) * 30
        positive_score = np.clip(positive_rate / 100.0, 0, 1) * 20
        window_score = np.clip(valid_windows / 8.0, 0, 1) * 15
        drawdown_score = np.clip((worst_return + 40.0) / 80.0, 0, 1) * 15
        excess_score = np.clip((benchmark_excess + 20.0) / 40.0, 0, 1) * 10
        significance_score = 10 if significant else 0
        return float(
            return_score
            + positive_score
            + window_score
            + drawdown_score
            + excess_score
            + significance_score
        )
