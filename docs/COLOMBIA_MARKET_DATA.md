# Colombia market data: free-source policy

## Decision

The first implementation tested Yahoo Finance through yfinance as a free OHLCV source, with a persistent local CSV cache and explicit data-quality audit. It is **not currently approved as a usable Colombian OHLCV source**: the live GitHub Actions validation on 2026-10-26 rejected all 27 configured symbols because the returned historical candles contain inconsistent Open/High/Low/Close values. No Colombian ranking should be produced from that feed until a different source passes validation.

This aligns with the reported upstream Nuam-market historical-data problem affecting Colombia, Chile and Peru from 2026-07-17: https://github.com/ranaroussi/yfinance/issues/2966. The issue describes stale/frozen or missing historical candles in Yahoo's chart feed despite live quotes remaining available on the website.

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

The collector fetches full history once, then refreshes an overlapping recent interval to capture corrections. The overlap is merged by date, preserving the latest provider record. A failed refresh leaves the last good cache intact and reports the error. The current Yahoo collector is retained only as a diagnostic candidate; its data-quality failure means it cannot be treated as an accepted source.

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

The GitHub Actions workflow runs on weekdays after the Colombia market session and publishes the cache/audit as a 30-day artifact. GitHub Actions cache preserves prior files between runs. The workflow now fails closed when refresh validation fails; the audit is still uploaded for diagnosis. Do not trust a ranking unless the source audit shows current, internally consistent OHLCV and the analysis step completes.

## Known limitation

A suitable replacement source remains unresolved. The SFC's official PPP and volume are useful for cross-checking, but they are not a full daily OHLCV candle. A short public history table is not enough for VFI length 130 (about 2.5 years of weekly data). The legacy BVC download endpoint documented by the public `descargarbvc` R package was tested on 2026-10-26 and now returns the BVC site's generic Next.js HTML page rather than an XLS file, so it cannot currently be used as a direct data feed. Until a free source with adequate history and valid OHLCV is identified and tested, the project intentionally withholds the Colombian ranking rather than fabricating bars or publishing misleading scores.
