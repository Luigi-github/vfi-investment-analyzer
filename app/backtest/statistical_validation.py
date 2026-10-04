import numpy as np
import pandas as pd


class StatisticalValidationAnalyzer:
    """Deterministic statistical validation for event-return research."""

    def __init__(self, bootstrap_samples=5000, confidence=0.95, random_seed=42):
        if bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive.")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1.")
        self.bootstrap_samples = int(bootstrap_samples)
        self.confidence = float(confidence)
        self.random_seed = int(random_seed)

    def validate_returns(self, returns):
        values = pd.Series(returns, dtype="float64").dropna().to_numpy()
        if values.size == 0:
            raise ValueError("At least one finite return is required.")
        if not np.isfinite(values).all():
            raise ValueError("Returns must be finite.")
        return values

    def summarize(self, returns):
        values = self.validate_returns(returns)
        return {
            "Count": int(values.size),
            "Average_Return": float(values.mean()),
            "Median_Return": float(np.median(values)),
            "Std_Return": float(values.std(ddof=1)) if values.size > 1 else np.nan,
            "Positive_Rate": float((values > 0).mean() * 100.0),
            "Negative_Rate": float((values < 0).mean() * 100.0),
            "Best_Return": float(values.max()),
            "Worst_Return": float(values.min()),
        }

    def bootstrap_mean(self, returns):
        values = self.validate_returns(returns)
        rng = np.random.default_rng(self.random_seed)
        samples = rng.choice(
            values,
            size=(self.bootstrap_samples, values.size),
            replace=True,
        )
        means = samples.mean(axis=1)
        alpha = 1.0 - self.confidence
        low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
        return {
            "Mean": float(values.mean()),
            "CI_Lower": float(low),
            "CI_Upper": float(high),
            "Confidence": self.confidence,
            "Samples": self.bootstrap_samples,
        }

    def compare_to_benchmark(self, event_returns, benchmark_returns):
        events = self.validate_returns(event_returns)
        benchmark = self.validate_returns(benchmark_returns)
        excess = float(events.mean() - benchmark.mean())
        return {
            "Event_Count": int(events.size),
            "Benchmark_Count": int(benchmark.size),
            "Event_Average_Return": float(events.mean()),
            "Benchmark_Average_Return": float(benchmark.mean()),
            "Excess_Average_Return": excess,
            "Event_Median_Return": float(np.median(events)),
            "Benchmark_Median_Return": float(np.median(benchmark)),
        }

    def validate_event_table(self, events, return_column, benchmark_column=None):
        if return_column not in events.columns:
            raise ValueError(f"Column '{return_column}' is required.")
        returns = events[return_column].dropna()
        summary = self.summarize(returns)
        bootstrap = self.bootstrap_mean(returns)
        result = {**summary, **{f"Bootstrap_{k}": v for k, v in bootstrap.items()}}
        if benchmark_column is not None:
            if benchmark_column not in events.columns:
                raise ValueError(f"Column '{benchmark_column}' is required.")
            comparison = self.compare_to_benchmark(
                returns,
                events[benchmark_column].dropna(),
            )
            result.update(comparison)
        return result
