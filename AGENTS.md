# VFI Investment Analyzer — Agent Guidelines

## 1. Project purpose

VFI Investment Analyzer is a Python research project for long-term market analysis centered on the Volume Flow Indicator (VFI).

The project studies historical accumulation, recovery, distribution, and flow regimes across multiple asset classes. It is a research and analysis system, not an automatic trading system.

## 2. Core principles

### No look-ahead bias

Historical signals must be reproducible using only information available at the signal timestamp.

A signal at time T must not depend on:
- prices, volume, indicators, or labels from T+1 onward;
- future events within the same recovery cluster;
- future observations used to select a representative event;
- future information used to normalize, rank, tune, or classify the signal.

Forward returns are outcomes of a signal, not inputs used to define that signal.

### Point-in-time logic

Prefer point-in-time event detection over retrospective selection.

When an event is detected at T:
1. calculate the signal using data available through T;
2. record the event at T;
3. calculate future performance separately using T+N observations;
4. never use future performance to modify the historical signal.

### Reproducibility

Indicators and analytical rules must be deterministic given the same input data and configuration.

Important parameters must be explicit and centralized rather than hidden inside research scripts.

### Research before optimization

Do not tune parameters solely because they improve historical performance.

Evaluate material parameter changes across multiple assets, historical periods, market regimes, and out-of-sample data when available.

### Statistical honesty

Do not treat a high historical return, win rate, or average return as proof that a signal is predictive.

Consider sample size, median versus mean, dispersion, outliers, survivorship bias, selection bias, data snooping, multiple testing, and regime dependence.

### Traceability

Every backtested result should be traceable to source market data, indicator configuration, signal definition, event timestamp, forward-return horizon, and backtest configuration.

## 3. Architecture

Maintain separation between:

- `app/data/` — market-data acquisition and transformation;
- `app/indicators/` — mathematical indicators;
- `app/analysis/` — interpretation of indicator values and market states;
- `app/scoring/` — scoring and classification logic;
- `app/backtest/` — historical evaluation;
- `app/risk/` — risk calculations;
- `app/scanner/` — multi-asset screening;
- `app/ai/` — future machine-learning/AI components;
- `app/models/` — shared domain/data models;
- `config/` — explicit configuration;
- `tests/` — automated tests;
- `dashboard/` — presentation/UI.

Do not move business logic into presentation code or use research scripts as the permanent home for reusable domain logic.

## 4. VFI implementation

The VFI implementation must remain mathematically reproducible and independently testable.

Changes to the VFI formula require a clear explanation, regression tests, comparison against the previous implementation, and validation against an appropriate reference when possible.

Do not change VFI parameters merely to improve historical backtest results.

## 5. Backtesting rules

Backtests must clearly distinguish signal date, entry/observation date, forward-return period, and evaluation date.

Forward returns must be calculated from the full chronological dataset before retrospective filtering that could alter their temporal meaning.

Representative-event methods must be classified as either:
- point-in-time/causal; or
- retrospective/descriptive.

Retrospective methods must never be presented as live-detectable signals.

## 6. Testing requirements

Important changes require automated tests.

Tests should cover indicator calculations, warm-up periods, missing/invalid input data, zero-cross events, cycle transitions, recovery detection, forward-return calculations, cluster/event selection, and explicit no-look-ahead invariants.

Prefer small deterministic fixtures over downloading live market data inside unit tests. Live market-data tests may exist separately as integration tests.

## 7. Data handling

Do not commit secrets, API keys, credentials, local virtual environments, generated caches, or raw/processed datasets that are intentionally ignored.

Use `.env` only for local secrets/configuration and keep `.env.example` free of secrets.

## 8. Git workflow

Do not make experimental changes directly on `main`.

Prefer:
1. create a focused branch;
2. make one logical change;
3. run relevant tests;
4. inspect the diff;
5. open a pull request;
6. review the result;
7. merge only after validation.

Avoid mixing refactoring with quantitative methodology changes unless necessary.

## 9. Quantitative research workflow

Preferred workflow:

```
Market Data
    ↓
Weekly Transformation
    ↓
VFI Indicator
    ↓
State / Cycle Analysis
    ↓
Point-in-Time Signal
    ↓
Forward Outcome
    ↓
Statistical Analysis
    ↓
Out-of-Sample Validation
```

Machine learning should be introduced only after the rule-based signal and backtesting pipeline are validated.

## 10. Change discipline

Before modifying an existing analytical method:
- identify its current inputs and outputs;
- inspect callers;
- preserve existing behavior unless the change is intentional;
- add or update tests;
- compare before/after results.

Do not silently change historical definitions.

When a methodological change invalidates earlier results, label previous results as obsolete rather than mixing them with the new methodology.

## 11. Investment-safety boundary

The project provides research and analytical information.

It should not silently turn an analytical signal into an automated order or trading action.

Any future execution capability must be explicitly separated from research/backtesting logic and independently reviewed.

## 12. Priority order

When requirements conflict, prioritize:

1. correctness;
2. no look-ahead bias;
3. reproducibility;
4. testability;
5. statistical validity;
6. maintainability;
7. performance;
8. convenience.

Historical backtest performance must never override correctness or methodological validity.
