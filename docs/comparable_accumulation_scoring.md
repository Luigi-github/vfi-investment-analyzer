# Comparable early-accumulation ranking

The scanner is `scripts/study_early_vfi_accumulation.py`. It retains the intended two-stage process:

1. **Weekly setup:** VFI(130) is negative and near zero, its last completed weekly slope is positive, and the short/medium indicator EMAs are negative and above the long EMA.
2. **Daily confirmation:** Volume Divergence by MM itself (not price) satisfies the analogous negative/near-zero/positive-slope rule and EMA structure.

## Why scores are comparable

Raw values from different assets are not compared directly. The script estimates a near-zero boundary from the 25th percentile of each indicator's **prior absolute values**. Slope strength is ranked against the asset's own prior slope distribution, but it earns score points only when the indicator is negative and in that near-zero zone; a positive/high indicator cannot win on slope alone. EMA separation is scaled by the prior median absolute indicator value. Historical baselines exclude the current bar and use trailing windows only.

- Weekly history window: up to 156 bars (3 years); at least 52 baseline observations.
- Daily history window: up to 252 bars (about 1 trading year); at least 100 baseline observations.
- Current/incomplete daily candle is excluded; weekly conversion excludes the current incomplete week.

## Score (0–100)

| Component | Maximum | What it rewards |
| --- | ---: | --- |
| Weekly VFI state | 25 | Negative state, closeness to the asset-specific near-zero zone, and positive slope relative to the asset's past |
| Weekly EMA structure | 20 | Negative short/medium EMAs above long EMA, with partial credit for the structure |
| Daily Volume Divergence state | 25 | Same scale-aware state criteria applied to the divergence indicator |
| Daily EMA structure | 15 | Same EMA structure on daily divergence |
| Persistence | 15 | Consecutive weekly/daily bars where the indicator remains negative but rises |

The grid retains 64 combinations: short EMA (3, 5, 7, 9), medium (15, 20, 25, 30), long (40, 50, 60, 80), with short < medium < long.

### Trend-loss penalty

The score subtracts up to **15 points** for deterioration in the weekly VFI setup:

- **8 points:** the short VFI EMA is currently below the medium EMA after having been at/above it during one of the previous 3 completed weekly bars (a bearish recross).
- **7 points:** the latest VFI slope is non-positive after the average slope of the previous 3 completed weekly bars was positive (loss of the VFI upslope).

The flags and individual penalties are exported in the ranking and EMA-combination CSVs. This penalty is applied to the score, but does not itself decide eligibility; the strict weekly and daily gates remain separate.

### Relative VFI extension and positive long-EMA penalties

To avoid ranking an already mature upward move as an early accumulation opportunity:

- The current positive weekly VFI is ranked by its **empirical percentile against the asset's own prior raw VFI values** (not absolute VFI values and not a shared raw-value threshold).
- No extension penalty applies while VFI is zero/negative or at/below its own historical 75th percentile.
- Above percentile 75, the penalty increases linearly by percentile rank, reaching 15 points at percentile 95 and remaining capped at 15 thereafter.
- **If the weekly long-period VFI EMA is above zero, subtract an additional 25 points.** This is intentionally a very strong warning that the long-term flow is already positive, which is inconsistent with the desired early-accumulation setup. EMA equal to zero does not trigger it.

All historical comparisons use prior bars only and exclude the current observation from the reference distribution. The report exposes the VFI historical percentile, extension penalty, long-EMA-positive flag and its penalty, and total penalty. The final score subtracts trend-loss, relative VFI extension and positive long-EMA penalties from the gross score, clipped to 0–100. These penalties lower ranking points but do not replace the strict weekly and daily eligibility gates.

**Score and eligibility are different.** `Accumulation_Score` orders research candidates; `Eligible=True` is only assigned when both strict weekly and daily filters pass. A high score alone is not a signal or a probability of profit. Because 64 EMA combinations are searched, the selected combination is a research choice and should be checked out-of-sample before use.

## Outputs

Written to `reports/early_vfi_accumulation/`:

- `early_vfi_accumulation_ranking.csv`: one row per asset, score, component scores, percentile rank, eligibility, selected EMA periods, thresholds and dates.
- `qualified_accumulation_candidates.csv`: only strict weekly + daily passes.
- `diagnostic_candidates_not_qualified.csv`: ranked observations that do not pass both filters.
- `ema_combo_scan.csv`: all 64 combinations per scanned asset.
- `early_vfi_accumulation_errors.csv`: symbols skipped and reasons.

Example:

```bash
python scripts/study_early_vfi_accumulation.py
python scripts/study_early_vfi_accumulation.py --symbols "DDD,INTC,NVDA,BTC-USD"
```

Unit tests also run the score calculation against synthetic weekly and daily indicator histories to verify the score stays bounded and its components sum correctly.

Comparative ranks are cross-sectional among the assets in that scan, so the universe should be kept consistent when comparing reports over time.
