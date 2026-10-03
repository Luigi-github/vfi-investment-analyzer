import pandas as pd


class LongOnlyExecutionCostModel:
    """Apply deterministic entry/exit fees and slippage to long returns."""

    def __init__(self, fee_bps=0.0, slippage_bps=0.0):
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)

        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be non-negative.")

    def net_return(self, gross_return):
        """Convert a gross long return into a net return percentage."""
        gross = pd.to_numeric(gross_return, errors="coerce")
        fee = self.fee_bps / 10000.0
        slippage = self.slippage_bps / 10000.0

        buy_multiplier = (1.0 + slippage) * (1.0 + fee)
        sell_multiplier = (1.0 - slippage) * (1.0 - fee)

        net = ((1.0 + gross / 100.0) * sell_multiplier / buy_multiplier - 1.0) * 100.0

        if hasattr(net, "where"):
            return net.where(gross.notna())

        return float(net) if pd.notna(net) else float("nan")
