# Free cloud execution for VFI Investment Analyzer

## Recommended route: GitHub Actions on a public repository

The repository is public so the workflow can use standard GitHub-hosted `ubuntu-latest` runners without paying for runner minutes. The workflow does not use larger runners, self-hosted runners, or paid cloud VMs. GitHub documents standard hosted-runner usage for public repositories as free: https://docs.github.com/en/actions/concepts/billing-and-usage.

Before making code public, check the full Git history for API keys, tokens, passwords, private keys, `.env` files, account identifiers, and personal information. Remove any exposed credential and rotate it; deleting it from the latest commit alone does not remove it from history.

## Run the top-200 backtest

1. Open the [repository Actions tab](https://github.com/Luigi-github/vfi-investment-analyzer/actions).
2. Select **Free Binance Top-200 MTF Backtests**.
3. Click **Run workflow**, choose branch `main`, and start it.
4. Open the new run and wait for the discovery, per-symbol backtests, and aggregation jobs to finish.

The workflow discovers the intersection of Binance USDT spot and USD-M perpetual markets and selects 200 symbols using their combined 24-hour quote-volume ranks. It runs up to 10 symbol jobs in parallel and uploads a consolidated result artifact named `crypto-mtf-top200-results-<run number>`.

- Per-symbol artifacts are retained for 7 days.
- The consolidated report is retained for 30 days.
- The workflow records per-symbol status so errors, insufficient history, and assets with no out-of-sample trades are not silently counted as valid results.

## Download and inspect results

Open the completed workflow run and download the consolidated artifact. Important files include:

- `checkpoint.json`: selected universe and completion/warning status.
- `binance_universe_liquidity.csv`: exact 24-hour spot/perpetual quote volumes and ranks explaining why each symbol entered the universe.
- `crypto_mtf_ranking.csv`: global current-signal ranking with per-timeframe VFI stage, value, slope, and timestamp.
- `final_crypto_ranking.csv`: one row per asset, with current MTF score, liquidity evidence, SPOT_MTF walk-forward OOS results, matched buy-and-hold comparison, evidence tier, composite research score, and a human-readable reason.
- `ranking_methodology.md`: ranking formula, selection criteria, costs, and limitations.
- `data_quality_report.json`: number of assets processed, errors, missing statuses, sample-size eligibility, and completeness status.
- `strategy_comparison_oos_global.csv`: pooled per-trade comparison across strategies, including the matched out-of-sample buy-and-hold benchmark.
- `walk_forward_oos.csv`: walk-forward fold results, including folds with no trades and benchmark folds.
- `oos_trades.csv`: out-of-sample trades and benchmark windows.
- `walk_forward_summary_by_symbol.csv`: corrected per-symbol/per-strategy OOS summary.
- `backtest_status_by_symbol.csv`: whether each symbol completed, had no OOS trades, had insufficient history, or failed.

A workflow marked green only means its jobs exited successfully. Check the checkpoint and status CSVs, row counts, missing values, and OOS trades before interpreting the performance.

## Limits and troubleshooting

- The backtest uses up to 3,000 one-hour candles (about 125 days) for trade simulation. The OOS portion covers only part of that period, so this is an initial test, not proof of robustness across multiple market cycles.
- The 200 assets are not guaranteed to have equal history. New listings may have insufficient data or produce no OOS trades.
- Fees, slippage, funding, stop/target execution within hourly candles, and portfolio-level exposure affect real performance. No backtest guarantees future results.
- Standard hosted-runner minutes for public repositories are documented as free, but an account-level billing restriction may still prevent a workflow from starting. Do not raise the spending limit or select paid runners if the requirement is $0; contact [GitHub Support](https://support.github.com/contact) if the billing block persists.
- The workflow needs access to Binance public market-data endpoints. API limits or outages can cause individual symbols to fail; inspect their job logs and the status report before rerunning.
