"""Research universe definitions grouped by market.

These are research candidates, not investment recommendations. The universe is
kept separate from the VFI formula and can be expanded without changing the
indicator implementation.
"""

MARKET_UNIVERSES = {
    "USA_STOCKS": [
        "SPY", "QQQ", "IWM",
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
        "AMD", "INTC", "AVGO", "TSLA", "JPM", "XOM",
        "CVX", "WMT", "COST", "UNH", "BRK-B",
        "DDD", "EC",
    ],
    "USA_ETFS": [
        "SPY", "QQQ", "IWM", "EEM", "TLT", "GLD", "SLV", "XLE",
    ],
    "CRYPTO": [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
        "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
    ],
    # Colombia candidates. Eligibility is decided by the Colombian
    # recency/liquidity policy before VFI calculation.
    "COLOMBIA": [
        # Common shares
        "ECOPETROL.CL", "CIBEST.CL", "ISA.CL", "MINEROS.CL",
        "GRUPOARGOS.CL", "CEMARGOS.CL", "NUTRESA.CL",
        "BOGOTA.CL", "CELSIA.CL", "CONCONCRET.CL", "CORFICOLCF.CL",
        "EXITO.CL", "GEB.CL", "GRUPOAVAL.CL", "GRUPOSURA.CL",
        "GRUBOLIVAR.CL", "PROMIGAS.CL", "TERPEL.CL", "ENKA.CL", "ETB.CL",
        # Preferred shares
        "PFCIBEST.CL", "PFAVAL.CL", "PFCEMARGOS.CL", "PFCORFICOL.CL",
        "PFDAVVNDA.CL", "PFGRUPOARG.CL", "PFGRUPSURA.CL",
    ],
    "CHILE": [
        "SQM", "SQM-B.SN", "COPEC.SN", "ENELAM.SN",
    ],
    "LATAM": [
        "VALE", "PBR", "ITUB", "NU",
    ],
    "COMMODITIES_ETFS": [
        "GLD", "SLV", "USO", "UNG", "DBA",
    ],
}


def get_universe(market: str):
    """Return a copy of a configured research universe."""
    if market not in MARKET_UNIVERSES:
        raise ValueError(f"Unknown market universe: {market}")
    return list(MARKET_UNIVERSES[market])


def all_symbols():
    """Return unique symbols preserving market-definition order."""
    result = []
    seen = set()
    for symbols in MARKET_UNIVERSES.values():
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                result.append(symbol)
    return result
