from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01
    max_exposure: float = 1.0
    min_stop_distance: float = 0.005


def position_fraction(
    equity: float,
    entry_price: float,
    stop_price: float,
    config: RiskConfig = RiskConfig(),
) -> float:
    """Return portfolio fraction sized so the stop risks at most risk_per_trade."""
    if equity <= 0 or entry_price <= 0 or stop_price <= 0:
        return 0.0
    if config.risk_per_trade <= 0 or config.max_exposure <= 0:
        return 0.0
    distance = max(abs(entry_price - stop_price) / entry_price, config.min_stop_distance)
    return min(config.max_exposure, config.risk_per_trade / distance)


def trade_pnl_fraction(net_return: float, exposure_fraction: float) -> float:
    """Convert trade return into contribution to total portfolio equity."""
    return float(net_return) * max(0.0, float(exposure_fraction))
