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
| Persistence | 15 | Consecutive weekly/daily bars where the indicator remains negative but rises; weekly VFI persistence saturates over 26 weeks |

The grid retains 64 combinations: short EMA (3, 5, 7, 9), medium (15, 20, 25, 30), long (40, 50, 60, 80), with short < medium < long.

### Strong bearish EMA crossover penalties

The score applies cumulative penalties when weekly VFI EMAs have recently crossed bearish. A crossover is considered recent when the latest completed weekly bar has the faster EMA below the slower EMA and at least one of the prior 3 completed weekly bars had it at/above that EMA:

- **20 points:** short EMA crosses below medium EMA.
- **20 points:** short EMA crosses below long EMA.
- **25 points:** medium EMA crosses below long EMA.
- **7 points:** the latest VFI slope is non-positive after the average slope of the previous 3 completed weekly bars was positive (loss of the VFI upslope).

The EMA crossover penalties stack, so all three bearish crosses together subtract **65 points**; including loss of VFI upslope, the weekly trend penalty can reach **72 points**. These are intentionally strong penalties for deterioration in the accumulation structure. Individual flags and penalties are exported in the ranking and EMA-combination CSVs. The penalty lowers the score but does not itself decide eligibility; strict weekly and daily gates remain separate.

### Duration reward for negative rising VFI and its EMAs

The persistence bonus now increases with the **consecutive number of completed weekly bars** where the raw VFI remains below zero and rises. It saturates at 26 consecutive weeks (up to 8 points), so a longer orderly recovery is rewarded more than a short-lived bounce.

The weekly VFI EMA component remains capped at 20 points: 14 points are allocated to the EMA structure and up to 2 additional points to each of the short, medium and long VFI EMA curves according to their own consecutive negative-and-rising streaks. Each EMA streak saturates at 26 weeks. If an EMA reaches or crosses zero, or its slope stops rising, its current streak resets. The CSV exports all three streak lengths and the structure/persistence sub-scores.

The daily Volume Divergence persistence component retains its existing 10-day cap. This duration reward is for the raw weekly VFI and its weekly EMA curves; it does not introduce an R² calculation on divergence.

### Descending slopes and VFI-only smoothness reward

Descending slopes now receive explicit, scale-aware deductions instead of relying only on bearish EMA recrosses:

- **VFI raw line:** separate penalties for 1-, 4- and 8-week descending slopes, each up to 5 points (up to 15 combined).
- **Weekly VFI EMAs:** short, medium and long EMA lines each incur up to 5 points for a descending 4-bar regression slope.
- **Daily Volume Divergence EMAs:** the same separate 0–5 point penalty is applied to each of its short, medium and long EMA lines. This penalizes declining EMA curves but does not apply R² to divergence.
- Each penalty is normalized by that indicator's own trailing median absolute scale, so raw amplitudes across assets are not compared directly. Penalties grow with the size of the descent and are capped per component.

The scanner also calculates a 12-completed-week linear regression **R² on the weekly VFI line only**. A reward of up to 10 points is possible only when the current VFI remains negative and its regression slope is positive. Reward increases with a straighter path, meaningful positive slope and a level that has not moved too far from zero; downward, flat, or already-positive VFI earns no smoothness reward. **No R² is calculated or rewarded for Volume Divergence or price.**

The final score subtracts the new descending-slope penalties and adds the VFI smoothness reward. These values are exported as separate columns for auditability.

### Relative VFI extension and positive long-EMA penalties

To avoid ranking an already mature upward move as an early accumulation opportunity:

- The current positive weekly VFI is ranked by its **empirical percentile against the asset's own prior raw VFI values** (not absolute VFI values and not a shared raw-value threshold).
- No extension penalty applies while VFI is zero/negative or at/below its own historical 75th percentile.
- Above percentile 75, the penalty increases linearly by percentile rank, reaching 15 points at percentile 95 and remaining capped at 15 thereafter.
- **If the weekly long-period VFI EMA is above zero, subtract an additional 25 points.** This is intentionally a very strong warning that the long-term flow is already positive, which is inconsistent with the desired early-accumulation setup. EMA equal to zero does not trigger it.

All historical comparisons use prior bars only and exclude the current observation from the reference distribution. The report exposes the VFI historical percentile, extension penalty, long-EMA-positive flag and its penalty, each of the three EMA bearish crossover flags/penalties, and total penalty. The final score subtracts trend-loss, relative VFI extension and positive long-EMA penalties from the gross score, clipped to 0–100. These penalties lower ranking points but do not replace the strict weekly and daily eligibility gates.

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
