from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    stop_price: float
    target_price: float
    entry_time: str

    def unrealized_return(self, price: float) -> float:
        if self.direction == "long":
            return price / self.entry_price - 1.0
        return self.entry_price / price - 1.0


@dataclass
class PaperAccount:
    equity: float = 10000.0
    cash: float = 10000.0
    realized_pnl: float = 0.0
    fees: float = 0.0
    positions: dict[str, Position] | None = None

    def __post_init__(self) -> None:
        if self.positions is None:
            self.positions = {}

    def mark_to_market(self, prices: dict[str, float]) -> float:
        value = self.cash
        for symbol, position in self.positions.items():
            price = float(prices[symbol])
            value += position.quantity * (
                price - position.entry_price
                if position.direction == "long"
                else position.entry_price - price
            )
        self.equity = value
        return value

    def snapshot(self) -> dict:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "fees": self.fees,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PaperAccount":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        positions = {k: Position(**v) for k, v in raw.get("positions", {}).items()}
        return cls(
            equity=float(raw["equity"]),
            cash=float(raw["cash"]),
            realized_pnl=float(raw["realized_pnl"]),
            fees=float(raw["fees"]),
            positions=positions,
        )


class PaperTradingEngine:
    """Simulation-only execution engine. It never sends exchange orders."""

    def __init__(self, account: PaperAccount | None = None, fee_rate: float = 0.0004):
        self.account = account or PaperAccount()
        self.fee_rate = float(fee_rate)

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        stop_price: float,
        target_price: float,
        timestamp: datetime | None = None,
    ) -> Position:
        if symbol in self.account.positions:
            raise ValueError(f"position already exists for {symbol}")
        if direction not in {"long", "short"}:
            raise ValueError("direction must be long or short")
        if min(entry_price, quantity, stop_price, target_price) <= 0:
            raise ValueError("prices and quantity must be positive")
        fee = entry_price * quantity * self.fee_rate
        if fee >= self.account.cash:
            raise ValueError("insufficient cash for entry fee")
        self.account.cash -= fee
        self.account.fees += fee
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=float(entry_price),
            quantity=float(quantity),
            stop_price=float(stop_price),
            target_price=float(target_price),
            entry_time=(timestamp or datetime.now(timezone.utc)).isoformat(),
        )
        self.account.positions[symbol] = position
        return position

    def close_position(self, symbol: str, exit_price: float) -> float:
        position = self.account.positions.pop(symbol)
        gross = (
            (exit_price - position.entry_price) * position.quantity
            if position.direction == "long"
            else (position.entry_price - exit_price) * position.quantity
        )
        fee = abs(exit_price * position.quantity) * self.fee_rate
        net = gross - fee
        self.account.cash += net
        self.account.realized_pnl += net
        self.account.fees += fee
        self.account.equity = self.account.cash
        return net

    def process_bar(self, symbol: str, high: float, low: float, close: float) -> str | None:
        position = self.account.positions.get(symbol)
        if position is None:
            return None
        if position.direction == "long":
            if low <= position.stop_price:
                self.close_position(symbol, position.stop_price)
                return "STOP"
            if high >= position.target_price:
                self.close_position(symbol, position.target_price)
                return "TARGET"
        else:
            if high >= position.stop_price:
                self.close_position(symbol, position.stop_price)
                return "STOP"
            if low <= position.target_price:
                self.close_position(symbol, position.target_price)
                return "TARGET"
        self.account.mark_to_market({symbol: close})
        return None
