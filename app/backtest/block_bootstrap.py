import numpy as np
import pandas as pd


def block_bootstrap_mean(
    values,
    block_size=8,
    iterations=5000,
    random_state=42,
):
    """Moving block bootstrap for dependent event outcomes."""
    series = pd.Series(values).dropna().astype(float)
    if series.empty:
        return {"Mean": np.nan, "Lower_95": np.nan, "Upper_95": np.nan}

    if block_size <= 0 or iterations <= 0:
        raise ValueError("block_size and iterations must be positive.")

    values = series.to_numpy()
    n = len(values)
    block_size = min(int(block_size), n)
    blocks = [values[i:i + block_size] for i in range(n - block_size + 1)]
    rng = np.random.default_rng(random_state)

    estimates = np.empty(iterations)
    for i in range(iterations):
        sample = []
        while len(sample) < n:
            sample.extend(blocks[rng.integers(0, len(blocks))])
        estimates[i] = np.mean(sample[:n])

    return {
        "Mean": float(np.mean(values)),
        "Lower_95": float(np.quantile(estimates, 0.025)),
        "Upper_95": float(np.quantile(estimates, 0.975)),
        "Bootstrap_Iterations": int(iterations),
        "Block_Size": int(block_size),
    }
