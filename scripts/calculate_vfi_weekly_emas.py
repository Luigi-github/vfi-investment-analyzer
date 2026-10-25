"""Report the latest weekly VFI and EMAs of VFI for selected market symbols.

The EMA spans are measured in weekly bars and are applied to the VFI series,
not to price. MarketDataService fills missing weeks with the prior weekly
OHLCV row before calculating the indicator. The current/incomplete week is
excluded by MarketDataService.
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator


DEFAULT_SYMBOLS = ["MSFT", "NVDA", "SPY", "AMZN"]
DEFAULT_EMA_PERIODS = [5, 20, 80, 200]


def calculate_latest(symbol: str, period: str, ema_periods: list[int]) -> dict:
    """Calculate the latest completed weekly VFI and its weekly EMAs."""
    market = MarketDataService()
    daily = market.get_daily_data(symbol, period=period)
    weekly = market.convert_to_weekly(daily)

    indicator = VFIIndicator(
        length=130,
        coef=0.2,
        vcoef=2.5,
        signal_length=20,
        smooth_vfi=False,
    )
    result = indicator.calculate(weekly)

    # These are EMAs of the VFI values, not EMAs of Close/price.
    for span in ema_periods:
        result[f"VFI_EMA_{span}"] = result["VFI"].ewm(
            span=span,
            adjust=False,
            min_periods=span,
        ).mean()

    required = ["VFI"] + [f"VFI_EMA_{span}" for span in ema_periods]
    valid = result.dropna(subset=required)
    if valid.empty:
        raise ValueError(
            f"Insufficient completed weekly history for {symbol}; "
            f"EMA periods requested: {ema_periods}"
        )

    latest = valid.iloc[-1]
    row = {
        "Symbol": symbol,
        "Week_Ending": valid.index[-1].strftime("%Y-%m-%d"),
        "Weekly_Bars_For_VFI": int(result["VFI"].notna().sum()),
        "VFI": float(latest["VFI"]),
    }
    for span in ema_periods:
        row[f"VFI_EMA_{span}W"] = float(latest[f"VFI_EMA_{span}"])
    return row


def main() -> int:
    parser = ArgumentParser(
        description=(
            "Calculate weekly VFI and 5/20/80/200-week EMAs of VFI "
            "for MSFT, NVDA, SPY and AMZN."
        )
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated yfinance symbols (default: MSFT,NVDA,SPY,AMZN).",
    )
    parser.add_argument(
        "--period",
        default="max",
        help='Yahoo Finance history period (default: "max").',
    )
    parser.add_argument(
        "--ema-periods",
        default=",".join(map(str, DEFAULT_EMA_PERIODS)),
        help="Comma-separated EMA spans in weekly bars (default: 5,20,80,200).",
    )
    parser.add_argument(
        "--output",
        default="reports/vfi_weekly_ema_comparison.csv",
        help="Output CSV path (default: reports/vfi_weekly_ema_comparison.csv).",
    )
    args = parser.parse_args()

    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    try:
        ema_periods = sorted({int(value.strip()) for value in args.ema_periods.split(",")})
    except ValueError:
        parser.error("--ema-periods must contain comma-separated positive integers")
    if not symbols:
        parser.error("--symbols must contain at least one symbol")
    if not ema_periods or any(span <= 0 for span in ema_periods):
        parser.error("--ema-periods must contain positive integers")

    rows = []
    failures = []
    for symbol in symbols:
        try:
            rows.append(calculate_latest(symbol, args.period, ema_periods))
            print(f"[OK] {symbol}")
        except Exception as exc:  # Report each failed symbol without aborting others.
            failures.append((symbol, str(exc)))
            print(f"[ERROR] {symbol}: {exc}")

    if not rows:
        print("No symbols could be calculated; no report was written.")
        return 1

    table = pd.DataFrame(rows)
    numeric_columns = [
        column for column in table.columns
        if column not in {"Symbol", "Week_Ending"}
    ]
    print("\nWEEKLY VFI AND EMAs OF VFI")
    print(
        table.to_string(
            index=False,
            formatters={column: lambda value: f"{value:.6f}" for column in numeric_columns},
        )
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"\nCSV saved to: {output_path}")

    if failures:
        print("\nSymbols not calculated:")
        for symbol, reason in failures:
            print(f"- {symbol}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
