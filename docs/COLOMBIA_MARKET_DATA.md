# Colombia market data: free-source policy

## Decision

The first implementation uses Yahoo Finance through yfinance as the free OHLCV source, with a persistent local CSV cache and explicit data-quality audit. This is a practical research feed, not an official BVC feed and not guaranteed to be complete or real-time. It must be validated before using any result for a financial decision.

Marketstack's published free tier is limited to end-of-day data and roughly one year of history, so it cannot alone supply the 130 weekly observations required by the current VFI length of 130. It also consumes requests per symbol, making a 25+ symbol daily universe a poor fit for its free quota. No Marketstack API key is required by this implementation.

The Superintendencia Financiera de Colombia is an official validation source, but its published daily share price is a volume-weighted average price (PPP) calculated from trades that marked price. On days with no trade, it can carry forward the previous day's price. It does not provide the complete OHLCV candle required to calculate the VFI. Therefore its PPP must not silently replace the daily Close used in the indicator. See:
- https://www.superfinanciera.gov.co/publicaciones/14960/metodologa-14960/
- https://www.superfinanciera.gov.co/info/superfinancierav8/media/MercadoAccionario/Precios/Diarios/ppal_promedios.htm

Grupo Aval publishes BVC equity closing information and can be used for manual close-price cross-checks where the relevant species is present:
- https://www.grupoaval.com/wps/portal/grupo-aval/aval/portal-financiero/renta-variable/acciones-bolsa-colombia/datos-historicos

The collector also queries the SFC issuer pages for the last-operation date and PPP, storing sfc_reference.csv and source_comparison.csv. This is an automated date/price-reference check, not a second OHLCV feed. The PPP difference is informational only and is never used to overwrite Yahoo Close. If the SFC page is unavailable or a symbol cannot be mapped, the audit records the failure.

## Cache layout

- daily/<SYMBOL>.csv: daily OHLCV and available corporate-action fields, with no daily imputation.
- latest_quotes.csv: last observation date, close, volume, source, freshness and error state.
- source_audit.csv: quality status for every requested symbol.
- sfc_reference.csv: SFC report date, last-operation date, and weighted-average price when parsed.
- source_comparison.csv: date conflicts and reference differences; PPP is not treated as close.
- metadata.json: retrieval timestamp, counts and source disclaimer.

The collector fetches full history once, then refreshes an overlapping recent interval to capture corrections. The overlap is merged by date, preserving the latest provider record. A failed refresh leaves the last good cache intact and reports the error.

## Quality controls

- OHLCV must be numeric, positive where appropriate, and internally consistent.
- Duplicate and unsorted dates are reported.
- Missing daily records are not filled.
- Each symbol is checked independently for freshness; stale data is never labelled current.
- The collector excludes CIB from the Colombia universe because it is a US-listed ADR rather than a local BVC-listed share. The configured .CL symbols are retained for individual coverage checks.
- The existing weekly conversion remains a research transformation; synthetic weeks should be distinguished from actual trading weeks when interpreting VFI, especially because carried-forward OHLCV can distort volume-flow measures.

## Running locally

    python -m pip install pandas yfinance
    python scripts/refresh_colombia_market_data.py
    python scripts/refresh_colombia_market_data.py --symbols ISA.CL,ECOPETROL.CL,GEB.CL

The scheduled GitHub Actions workflow runs on weekdays after the Colombia market session and publishes the cache/audit as a 30-day artifact. GitHub Actions cache preserves the prior daily files between runs. A scheduled run can fail for a symbol while still producing the audit; inspect source_audit.csv before trusting the ranking.

## Known limitation

Yahoo's Colombian symbol coverage, freshness, and exact OHLCV quality must be tested symbol by symbol. This code records failures rather than inventing values. It does not claim that all tickers have a current, valid record simply because the refresh job completed.
