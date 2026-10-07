# Crypto MTF real strategy backtest

This research layer converts the earlier forward-return study into a causal, non-overlapping trade simulation.

## Execution
- Signal is evaluated on a **closed 1H bar**.
- Entry occurs at the **next 1H bar open**; the signal bar is never traded.
- Exit is the first of: 72H horizon, 8% stop, 16% target.
- Trades are non-overlapping.
- Long-only for the first validated implementation.
- Spot strategies execute on Spot OHLC.
- Perpetual strategy executes on perpetual OHLC and charges/credits historical Binance funding.
- Fee: 0.04% per side.
- Slippage: 0.03% per side.
- Funding is the sum of funding rates observed during the holding interval, with positive funding treated as a cost to a long.

## Walk-forward
Thresholds 60/65/70/75/80/85/90 are selected using only the chronological training segment. The selected threshold is then frozen for the following OOS segment. Three anchored folds are reported.

## Risk
Reports include trades, win rate, total return, average trade, profit factor, max drawdown and bootstrap confidence intervals for mean OOS trade return.

## Limitations
This is still a research backtest, not an execution simulator. It uses bar OHLC and therefore cannot resolve intrabar ordering when stop and target are both touched in the same hour. Position sizing is not yet used to scale returns. No liquidation model is included. The derivative score penalties remain heuristic and are intentionally separated from the causal PnL engine.
