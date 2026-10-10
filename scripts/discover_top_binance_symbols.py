from __future__ import annotations
import json
from app.data.binance_market_data import BinanceMarketDataService

service = BinanceMarketDataService()
spot = service.discover_spot_usdt(0)
perp = service.discover_usdt_perpetuals(0)
spot_rank = {symbol: i for i, symbol in enumerate(spot)}
perp_rank = {symbol: i for i, symbol in enumerate(perp)}
common = set(spot_rank) & set(perp_rank)
symbols = sorted(common, key=lambda s: (
    spot_rank[s] + perp_rank[s], max(spot_rank[s], perp_rank[s]), s
))[:200]
if len(symbols) < 200:
    raise SystemExit(f"Expected 200 common liquid markets, discovered only {len(symbols)}")
print(json.dumps({
    "selection_method": "Top 200 intersection of Binance USDT spot and USD-M perpetual markets ranked by combined 24h quote-volume rank",
    "symbols": symbols
}, indent=2))
