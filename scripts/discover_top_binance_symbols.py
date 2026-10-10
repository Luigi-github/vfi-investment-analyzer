from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.binance_market_data import BinanceMarketDataService


def main() -> None:
    service = BinanceMarketDataService()

    # These lists are already filtered to active USDT spot / USD-M perpetual
    # markets and sorted by each market's 24-hour quote volume, descending.
    spot = service.discover_spot_usdt(0)
    perp = service.discover_usdt_perpetuals(0)
    spot_rank = {symbol: i + 1 for i, symbol in enumerate(spot)}
    perp_rank = {symbol: i + 1 for i, symbol in enumerate(perp)}

    # Preserve the actual quote-volume values so the final report can explain
    # why each symbol entered the research universe.
    spot_tickers = service._get(service.SPOT_BASE, "/api/v3/ticker/24hr", {})
    perp_tickers = service._get(service.FUTURES_BASE, "/fapi/v1/ticker/24hr", {})
    spot_volume = {
        row["symbol"]: float(row.get("quoteVolume", 0) or 0)
        for row in spot_tickers if row.get("symbol") in spot_rank
    }
    perp_volume = {
        row["symbol"]: float(row.get("quoteVolume", 0) or 0)
        for row in perp_tickers if row.get("symbol") in perp_rank
    }

    common = set(spot_rank) & set(perp_rank)
    records = []
    for symbol in common:
        if symbol not in spot_volume or symbol not in perp_volume:
            continue
        sr, pr = spot_rank[symbol], perp_rank[symbol]
        records.append({
            "Symbol": symbol,
            "Spot_Volume_Rank": sr,
            "Perp_Volume_Rank": pr,
            "Spot_24h_Quote_Volume_USD": spot_volume[symbol],
            "Perp_24h_Quote_Volume_USD": perp_volume[symbol],
            "Combined_Volume_Rank_Sum": sr + pr,
            "Worst_Market_Volume_Rank": max(sr, pr),
        })

    records.sort(key=lambda row: (
        row["Combined_Volume_Rank_Sum"],
        row["Worst_Market_Volume_Rank"],
        row["Symbol"],
    ))
    records = records[:200]
    if len(records) < 200:
        raise SystemExit(f"Expected 200 common liquid markets, discovered only {len(records)}")

    for rank, row in enumerate(records, 1):
        row["Universe_Rank"] = rank

    payload = {
        "retrieved_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "selection_method": (
            "Intersection of active Binance USDT spot and USD-M perpetual markets; "
            "ranked by the sum of each market's 24-hour quote-volume rank, then by "
            "the worse of the two ranks, then symbol. Current liquidity snapshot only."
        ),
        "universe_size": len(records),
        "symbols": [row["Symbol"] for row in records],
        "records": records,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
