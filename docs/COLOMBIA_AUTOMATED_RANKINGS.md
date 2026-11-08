# Colombia automated three-rankings pipeline

The workflow downloads Colombia OHLCV from Yahoo Finance and calculates three independent outputs:
1. Weekly VFI opportunity ranking.
2. Monthly Volume Divergence by MM ranking.
3. Combined ranking (50% weekly cross-sectional percentile + 50% monthly divergence cross-sectional percentile).

## Run locally

```bash
python -m pip install -r requirements.txt
python scripts/run_colombia_three_rankings.py
```

Optional:
```bash
python scripts/run_colombia_three_rankings.py --symbols ISA.CL,ECOPETROL.CL
```

## Automated schedule

GitHub Actions runs Monday-Friday at 18:30 Colombia time (23:30 UTC), and can also be run manually from Actions -> "Colombia - automatic three rankings" -> Run workflow.

Each run publishes a downloadable artifact with:
- ranking_vfi_semanal.csv
- ranking_volume_divergence_mensual.csv
- ranking_combinado.csv
- cobertura_universo.csv
- errores.csv
- run_metadata.json
- pipeline.log

## Calculation rules

- Yahoo Finance is the sole market-data provider for this pipeline.
- VFI length is 130 weekly bars. EMA values are calculated on VFI, not price.
- Weekly EMA weights: 5/20/80/200 = 5%/15%/30%/50%.
- Monthly Volume Divergence uses the MM WMA cascade 5/8/13/21/34, pivots 5/5 and divergence distance 5-60 monthly bars.
- Monthly divergence EMA weights: 5/20/80 = 10%/30%/60%. No EMA200 is used for divergence.
- The combined score is the mean of each ranking's cross-sectional percentile, so raw scales and volume magnitudes do not bias the result.
- Missing weekly/monthly calendar bars repeat prior OHLC and use zero synthetic volume; synthetic bars are explicitly counted.
- Per-symbol downloads are retried three times. A failed symbol is recorded and does not silently receive fabricated observations.
- A nonzero workflow exit indicates at least one symbol failed. Inspect errores.csv and pipeline.log.

The score is a research ranking, not a return forecast or investment recommendation.
