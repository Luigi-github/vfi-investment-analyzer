# Paper Trading

The project now contains a simulation-only execution layer in `app/paper_trading/`.

## Safety boundary

`PaperTradingEngine` never connects to Binance and never sends an order. It only simulates:

- one position per symbol;
- long and short PnL;
- entry and exit fees;
- stop loss;
- take profit;
- mark-to-market equity;
- persistent account state in JSON.

This is intentionally separate from the Binance market-data service.

## Bar execution rule

When a bar is processed, stop is checked before target for both long and short. This is a deterministic conservative convention when OHLC data shows both levels inside the same candle; it does not claim to know the true intrabar order.

## Persistence

An account can be saved and restored with `account.save("data/paper_account.json")` and `PaperAccount.load(...)`. Do not store exchange credentials in the account file.

## Risk sizing

Use `app.backtest.risk_management.position_fraction()` before opening a position. Recommended default: 1% account risk per trade and a 100% maximum exposure cap.

## Smoke test

Run:

```powershell
.\\venv\\Scripts\\python.exe scripts/run_paper_trading_demo.py
```

The demo is offline and uses synthetic prices.

## Not production execution

Before live deployment the project would still need an exchange order adapter, credential management, order reconciliation, partial-fill handling, reduce-only semantics, leverage/margin/liquidation handling, retry/idempotency controls, a kill switch, a persistent event log, monitoring/alerts, and long-duration paper-trading validation.

No live-order adapter is included by design.
