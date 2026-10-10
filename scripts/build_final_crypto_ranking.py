from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.backtest.block_bootstrap import block_bootstrap_mean


def read_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def trade_metrics(group: pd.DataFrame) -> dict:
    if group.empty:
        return {
            "OOS_Trades": 0, "OOS_Total_Return_Pct": np.nan,
            "OOS_Mean_Trade_Pct": np.nan, "OOS_Median_Trade_Pct": np.nan,
            "OOS_Win_Rate_Pct": np.nan, "OOS_Profit_Factor": np.nan,
            "OOS_Max_Drawdown_Pct": np.nan,
            "OOS_Mean_95_Lower_Pct": np.nan, "OOS_Mean_95_Upper_Pct": np.nan,
        }
    trades = group.copy()
    trades["Net_Return"] = pd.to_numeric(trades["Net_Return"], errors="coerce")
    trades = trades.dropna(subset=["Net_Return"])
    if "Exit_Date" in trades:
        trades = trades.sort_values("Exit_Date")
    returns = trades["Net_Return"].astype(float)
    equity = np.cumprod(1.0 + returns.to_numpy())
    path = np.concatenate(([1.0], equity))
    peaks = np.maximum.accumulate(path)
    max_dd = float(np.min(path / peaks - 1.0)) * 100.0
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    if len(returns) >= 20:
        ci = block_bootstrap_mean(
            returns, block_size=4, iterations=5000,
            random_state=123 + len(returns),
        )
        lower = float(ci["Lower_95"]) * 100.0
        upper = float(ci["Upper_95"]) * 100.0
    else:
        lower = upper = np.nan
    return {
        "OOS_Trades": int(len(returns)),
        "OOS_Total_Return_Pct": float((path[-1] - 1.0) * 100.0),
        "OOS_Mean_Trade_Pct": float(returns.mean() * 100.0),
        "OOS_Median_Trade_Pct": float(returns.median() * 100.0),
        "OOS_Win_Rate_Pct": float((returns > 0).mean() * 100.0),
        "OOS_Profit_Factor": float(gains / losses) if losses > 0 else np.nan,
        "OOS_Max_Drawdown_Pct": max_dd,
        "OOS_Mean_95_Lower_Pct": lower,
        "OOS_Mean_95_Upper_Pct": upper,
    }


def build_symbol_strategy_summary(oos: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if oos.empty or not {"Symbol", "Strategy", "Fold"}.issubset(oos.columns):
        return pd.DataFrame()
    trade_groups = {}
    if not trades.empty and {"Symbol", "Strategy", "Net_Return"}.issubset(trades.columns):
        for key, group in trades.groupby(["Symbol", "Strategy"], dropna=False):
            trade_groups[(str(key[0]), str(key[1]))] = group

    rows = []
    for (symbol, strategy), folds in oos.groupby(["Symbol", "Strategy"], dropna=False):
        key = (str(symbol), str(strategy))
        fold_returns = pd.to_numeric(folds.get("Total_Return_Pct", pd.Series(dtype=float)), errors="coerce")
        positives = int((fold_returns > 0).sum())
        group_trades = trade_groups.get(key, pd.DataFrame())
        metrics = trade_metrics(group_trades)
        rows.append({
            "Symbol": symbol,
            "Strategy": strategy,
            "OOS_Folds": int(folds["Fold"].nunique()),
            "OOS_Positive_Folds": positives,
            "OOS_Fold_Consistency_Pct": (
                positives / max(int(folds["Fold"].nunique()), 1) * 100.0
            ),
            "Selected_Thresholds": ",".join(
                str(int(x)) for x in sorted(
                    pd.to_numeric(folds.get("Selected_Threshold", pd.Series(dtype=float)), errors="coerce")
                    .dropna().unique()
                )
            ),
            "Threshold_Selection_Status": ";".join(sorted(
                str(x) for x in folds.get("Threshold_Selection_Status", pd.Series(dtype=str)).dropna().unique()
            )),
            "Mean_Fold_Return_Pct": float(fold_returns.mean()) if fold_returns.notna().any() else np.nan,
            "Mean_Train_Trades_At_Selected_Threshold": float(pd.to_numeric(
                folds.get("Train_Trades_At_Selected_Threshold", pd.Series(dtype=float)), errors="coerce"
            ).mean()),
            "Mean_Fold_Max_Drawdown_Pct": float(pd.to_numeric(
                folds.get("Max_Drawdown_Pct", pd.Series(dtype=float)), errors="coerce"
            ).mean()),
            **metrics,
        })
    return pd.DataFrame(rows)


def build_global_strategy_comparison(trades: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or not {"Strategy", "Net_Return"}.issubset(trades.columns):
        return pd.DataFrame()
    rows = []
    for strategy, group in trades.groupby("Strategy"):
        returns = pd.to_numeric(group["Net_Return"], errors="coerce").dropna()
        per_symbol = group.assign(Net_Return=pd.to_numeric(group["Net_Return"], errors="coerce")).groupby("Symbol")["Net_Return"].mean()
        gains = float(returns[returns > 0].sum())
        losses = float(-returns[returns < 0].sum())
        rows.append({
            "Strategy": strategy,
            "Global_OOS_Trades": int(len(returns)),
            "Global_OOS_Mean_Trade_Pct": float(returns.mean() * 100) if len(returns) else np.nan,
            "Global_OOS_Median_Trade_Pct": float(returns.median() * 100) if len(returns) else np.nan,
            "Global_OOS_Win_Rate_Pct": float((returns > 0).mean() * 100) if len(returns) else np.nan,
            "Global_OOS_Profit_Factor": float(gains / losses) if losses > 0 else np.nan,
            "Symbols_With_OOS_Trades": int(group["Symbol"].nunique()) if "Symbol" in group else np.nan,
            "Symbols_With_Positive_Mean_Trade": int((per_symbol > 0).sum()),
        })
    result = pd.DataFrame(rows)
    if not summary.empty and {"Strategy", "OOS_Folds", "OOS_Positive_Folds"}.issubset(summary.columns):
        fold = summary.groupby("Strategy").agg(
            Symbols_With_Fold_Evidence=("Symbol", "nunique"),
            Mean_Positive_Fold_Rate_Pct=("OOS_Fold_Consistency_Pct", "mean"),
        ).reset_index()
        result = result.merge(fold, on="Strategy", how="left")
    return result.sort_values("Global_OOS_Mean_Trade_Pct", ascending=False, na_position="last")


def build_final_ranking(root: Path) -> None:
    mtf = read_csv(root, "crypto_mtf_ranking.csv")
    derivatives = read_csv(root, "derivatives_snapshot.csv")
    oos = read_csv(root, "walk_forward_oos.csv")
    trades = read_csv(root, "oos_trades.csv")
    statuses = read_csv(root, "backtest_status_by_symbol.csv")

    universe_path = root / "universe_meta" / "reports-universe.json"
    universe = {}
    if universe_path.exists():
        try:
            payload = json.loads(universe_path.read_text(encoding="utf-8"))
            records = payload.get("records", [])
            if records:
                pd.DataFrame(records).to_csv(root / "binance_universe_liquidity.csv", index=False)
                universe = {str(row["Symbol"]): row for row in records}
            (root / "universe_selection.json").write_text(
                json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2),
                encoding="utf-8",
            )
        except (OSError, ValueError, TypeError):
            universe = {}

    summary = build_symbol_strategy_summary(oos, trades)
    if not summary.empty:
        summary.to_csv(root / "walk_forward_summary_by_symbol.csv", index=False)
    global_summary = build_global_strategy_comparison(trades, summary)
    if not global_summary.empty:
        global_summary.to_csv(root / "strategy_comparison_oos_global.csv", index=False)

    if mtf.empty or "Symbol" not in mtf:
        return
    ranking = mtf.copy()
    if not derivatives.empty and "Symbol" in derivatives:
        ranking = ranking.merge(derivatives.drop_duplicates("Symbol", keep="last"), on="Symbol", how="left")
    if not statuses.empty and "Symbol" in statuses:
        status_table = statuses.drop_duplicates("Symbol", keep="last").copy()
        for col in ("OOS_Folds", "OOS_Trades"):
            if col in status_table:
                status_table = status_table.rename(columns={col: f"All_Strategy_{col}"})
        ranking = ranking.merge(status_table, on="Symbol", how="left", suffixes=("", "_Status"))

    # Current-signal rank is kept separate from out-of-sample performance.
    primary = summary[summary["Strategy"] == "SPOT_MTF"].copy() if not summary.empty else pd.DataFrame()
    if not primary.empty:
        ranking = ranking.merge(primary, on="Symbol", how="left", suffixes=("", "_SPOT_MTF"))

    benchmark = summary[summary["Strategy"] == "BUY_AND_HOLD_OOS"].copy() if not summary.empty else pd.DataFrame()
    if not benchmark.empty:
        benchmark = benchmark[[
            "Symbol", "OOS_Total_Return_Pct", "OOS_Mean_Trade_Pct",
            "OOS_Trades", "OOS_Folds", "OOS_Fold_Consistency_Pct",
        ]].rename(columns={
            "OOS_Total_Return_Pct": "BuyHold_OOS_Total_Return_Pct",
            "OOS_Mean_Trade_Pct": "BuyHold_OOS_Mean_Fold_Return_Pct",
            "OOS_Trades": "BuyHold_OOS_Trades",
            "OOS_Folds": "BuyHold_OOS_Folds",
            "OOS_Fold_Consistency_Pct": "BuyHold_Positive_Fold_Rate_Pct",
        })
        ranking = ranking.merge(benchmark, on="Symbol", how="left")
        ranking["OOS_Return_Advantage_vs_BuyHold_PctPoints"] = (
            ranking["OOS_Total_Return_Pct"] - ranking["BuyHold_OOS_Total_Return_Pct"]
        )

    if universe:
        universe_frame = pd.DataFrame(universe.values()).drop_duplicates("Symbol", keep="last")
        ranking = ranking.merge(universe_frame, on="Symbol", how="left")

    numeric_cols = [
        "OOS_Trades", "OOS_Folds", "OOS_Positive_Folds",
        "OOS_Mean_Trade_Pct", "OOS_Total_Return_Pct",
        "OOS_Win_Rate_Pct", "OOS_Profit_Factor",
        "OOS_Max_Drawdown_Pct", "OOS_Mean_95_Lower_Pct",
        "OOS_Mean_95_Upper_Pct", "OOS_Fold_Consistency_Pct",
    ]
    for col in numeric_cols:
        if col not in ranking:
            ranking[col] = np.nan
        ranking[col] = pd.to_numeric(ranking[col], errors="coerce")

    ranking["OOS_Eligible"] = (
        (ranking["OOS_Trades"].fillna(0) >= 20)
        & (ranking["OOS_Folds"].fillna(0) >= 2)
    )
    eligible = ranking["OOS_Eligible"]
    ranking["OOS_Return_Percentile"] = 0.0
    if eligible.any():
        ranking.loc[eligible, "OOS_Return_Percentile"] = (
            ranking.loc[eligible, "OOS_Mean_Trade_Pct"].rank(pct=True) * 100.0
        )
    ranking["Drawdown_Score"] = (
        100.0 - ranking["OOS_Max_Drawdown_Pct"].abs()
    ).clip(0, 100).fillna(0.0)
    ranking["Evidence_Score"] = (
        ranking["OOS_Trades"].fillna(0).clip(upper=50) / 50.0 * 100.0
    )
    ranking["Consistency_Score"] = ranking["OOS_Fold_Consistency_Pct"].fillna(0.0)
    signal = pd.to_numeric(ranking.get("Final_Risk_Adjusted_Score", pd.Series(50.0, index=ranking.index)), errors="coerce")
    ranking["Signal_Score_Component"] = signal.fillna(50.0).clip(0, 100)
    ranking["Final_Research_Score"] = (
        0.40 * ranking["Signal_Score_Component"]
        + 0.25 * ranking["OOS_Return_Percentile"]
        + 0.20 * ranking["Consistency_Score"]
        + 0.10 * ranking["Drawdown_Score"]
        + 0.05 * ranking["Evidence_Score"]
    )
    ranking["Evidence_Tier"] = np.select(
        [
            eligible
            & (ranking["OOS_Mean_95_Lower_Pct"] > 0)
            & (ranking["OOS_Positive_Folds"].fillna(0) >= 2),
            eligible,
        ],
        ["POSITIVE_OOS_EVIDENCE", "MIXED_OOS_EVIDENCE"],
        default="INSUFFICIENT_OOS_SAMPLE",
    )

    def reason(row) -> str:
        parts = []
        if pd.notna(row.get("Spot_Volume_Rank")) and pd.notna(row.get("Perp_Volume_Rank")):
            spot_volume = row.get("Spot_24h_Quote_Volume_USDT")
            perp_volume = row.get("Perp_24h_Quote_Volume_USDT")
            liquidity = (
                f"Universe liquidity: spot rank {int(row['Spot_Volume_Rank'])}, "
                f"perpetual rank {int(row['Perp_Volume_Rank'])}; "
                f"24h quote volume spot ${float(spot_volume):,.0f}, "
                f"perpetual ${float(perp_volume):,.0f}"
            )
            spot_change = row.get("Spot_24h_Price_Change_Pct")
            perp_change = row.get("Perp_24h_Price_Change_Pct")
            if pd.notna(spot_change) and pd.notna(perp_change):
                liquidity += (
                    f"; 24h price change spot {float(spot_change):+.2f}%, "
                    f"perpetual {float(perp_change):+.2f}%"
                )
            parts.append(liquidity)
        score = row.get("Final_Risk_Adjusted_Score")
        if pd.notna(score):
            parts.append(
                f"current risk-adjusted MTF score {float(score):.1f}/100 "
                f"(spot {row.get('Spot_Direction', 'n/a')}, perpetual {row.get('Perp_Direction', 'n/a')})"
            )
        trades_n = row.get("OOS_Trades")
        folds_n = row.get("OOS_Folds")
        mean_trade = row.get("OOS_Mean_Trade_Pct")
        if pd.notna(trades_n) and trades_n > 0:
            parts.append(
                f"SPOT_MTF walk-forward OOS: {int(trades_n)} trades, "
                f"mean {float(mean_trade):+.3f}% per trade, "
                f"{int(row.get('OOS_Positive_Folds', 0) or 0)}/{int(folds_n or 0)} positive folds"
            )
        else:
            parts.append("SPOT_MTF has no OOS trades; current signal is not performance-validated")
        benchmark_return = row.get("BuyHold_OOS_Total_Return_Pct")
        strategy_return = row.get("OOS_Total_Return_Pct")
        if pd.notna(benchmark_return) and pd.notna(strategy_return):
            parts.append(
                f"SPOT_MTF net OOS return {float(strategy_return):+.2f}% vs "
                f"matched spot buy-and-hold {float(benchmark_return):+.2f}% "
                f"(difference {float(row.get('OOS_Return_Advantage_vs_BuyHold_PctPoints', 0)):+.2f} pp)"
            )
        tier = row.get("Evidence_Tier")
        parts.append(str(tier).replace("_", " ").lower())
        return "; ".join(parts)

    ranking["Research_Reason"] = ranking.apply(reason, axis=1)

    # Machine-readable audit summary: distinguish successful processing from
    # statistically persuasive evidence.
    expected_symbols = 200
    status_counts = {}
    if "Status" in ranking:
        status_counts = {
            str(key): int(value)
            for key, value in ranking["Status"].fillna("missing_status").value_counts().items()
        }
    unique_symbols = int(ranking["Symbol"].astype(str).nunique())
    missing_status = int(ranking["Status"].isna().sum()) if "Status" in ranking else expected_symbols
    duplicate_symbols = int(ranking["Symbol"].astype(str).duplicated().sum())
    report = {
        "expected_universe_size": expected_symbols,
        "mtf_ranking_rows": int(len(mtf)),
        "unique_symbols_in_final_ranking": unique_symbols,
        "duplicate_symbol_rows": duplicate_symbols,
        "status_counts": status_counts,
        "oos_eligible_symbols": int(eligible.sum()),
        "symbols_with_positive_oos_mean_ci_lower": int((
            eligible & (ranking["OOS_Mean_95_Lower_Pct"] > 0)
        ).sum()),
        "symbols_with_no_oos_trades": int((ranking["OOS_Trades"].fillna(0) == 0).sum()),
        "symbols_missing_status": missing_status,
        "symbols_with_errors": int((ranking["Status"] == "error").sum()) if "Status" in ranking else expected_symbols,
        "symbols_with_insufficient_history": int((ranking["Status"] == "insufficient_history").sum()) if "Status" in ranking else 0,
        "ranking_complete_for_200": unique_symbols == expected_symbols and duplicate_symbols == 0 and missing_status == 0,
        "important_caveat": "OOS sample spans only the available recent hourly history; rankings are research signals, not return guarantees."
    }
    if not (
        report["ranking_complete_for_200"]
        and report["symbols_with_errors"] == 0
        and report["symbols_with_insufficient_history"] == 0
    ):
        report["overall_status"] = "partial"
    elif report["status_counts"].get("no_oos_trades", 0) > 0:
        report["overall_status"] = "completed_with_warnings"
    else:
        report["overall_status"] = "complete"
    (root / "data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    methodology = """# Top-200 VFI research ranking

## Universe selection
The universe is the intersection of currently trading Binance USDT spot pairs and USD-M USDT perpetual pairs. Assets are ordered by the sum of their spot and perpetual 24-hour quote-volume ranks, then by the worse of the two ranks. The ranking is a current liquidity snapshot, not a survivorship-bias-free historical universe.

## Current signal
The MTF signal score weights weekly 30%, daily 30%, 4-hour 25%, and 1-hour 15%. The spot and perpetual confluence scores are combined 60%/40%. Funding extremes and large 24-hour open-interest changes can reduce the score. Per-timeframe VFI, stage, slope, and timestamp fields are included for audit.

## Out-of-sample test
The primary per-asset performance evidence is the SPOT_MTF strategy in three chronological walk-forward folds. For tunable strategies, thresholds are selected using only the training window and require at least 10 fully completed training trades per candidate threshold; if no candidate meets that minimum, the fold uses a declared threshold-80 fallback. BOTH_MTF, COMBINED_70, and COMBINED_80 use fixed thresholds, which are reported as such. Signals execute at the next hourly bar open. Backtest costs assume 0.04% fee and 0.03% slippage per side, with an 8% stop, 16% target, and 72-hour maximum holding period. These are modeling assumptions and may not match actual fills.

## Final research score
The score is 40% current risk-adjusted MTF signal, 25% percentile of SPOT_MTF mean net OOS trade return, 20% percentage of positive OOS folds, 10% drawdown score, and 5% sample-size score. Assets with fewer than 20 SPOT_MTF OOS trades or fewer than two OOS folds are marked INSUFFICIENT_OOS_SAMPLE and placed below eligible assets. The positive-evidence tier additionally requires a positive 95% block-bootstrap lower bound and at least two positive folds.

## Interpretation limits
A complete workflow means data and jobs were processed, not that the strategy is profitable. The recent sample is short, candidate assets are selected using today's liquidity (survivorship bias), and this score is a research prioritization tool. A matched out-of-sample spot buy-and-hold benchmark is included per fold, but the sample is still short. Validate across multiple market regimes and verify fee, slippage, funding, and intrabar stop/target assumptions before any live use.
"""
    (root / "ranking_methodology.md").write_text(methodology, encoding="utf-8")
    methodology_es = """# Ranking VFI de los 200 criptoactivos

## Selección del universo
Se toman pares USDT activos que existan tanto en spot como en futuros perpetuos USD-M de Binance. Se ordenan por la suma de sus puestos de volumen cotizado de 24 horas en ambos mercados; luego por el peor puesto de los dos. Es una fotografía de liquidez actual, no un universo histórico libre de sesgo de supervivencia.

## Señal actual
El puntaje MTF pondera semanal 30%, diario 30%, 4 horas 25% y 1 hora 15%. La confluencia spot/perpetuo se combina 60%/40%. Funding extremo y cambios grandes de interés abierto en 24 horas pueden restar puntos. El CSV contiene VFI, etapa, pendientes y marcas temporales por marco temporal.

## Validación fuera de muestra
Se usan tres periodos cronológicos walk-forward. Los umbrales ajustables solo se eligen con datos de entrenamiento y cada candidato debe tener al menos 10 operaciones de entrenamiento completamente cerradas. Si ninguno cumple, se registra un umbral de respaldo de 80. BOTH_MTF, COMBINED_70 y COMBINED_80 son estrategias de umbral fijo. Las señales se ejecutan en la apertura de la siguiente vela horaria. Se modelan comisión 0,04% y deslizamiento 0,03% por lado, stop 8%, objetivo 16% y salida máxima a 72 horas.

## Puntaje final de investigación
40% puntaje MTF actual ajustado por riesgo; 25% percentil del rendimiento medio por operación OOS de SPOT_MTF; 20% proporción de periodos OOS positivos; 10% puntaje de drawdown; 5% tamaño de muestra. Se marca muestra insuficiente si hay menos de 20 operaciones SPOT_MTF OOS o menos de dos periodos. La etiqueta de evidencia positiva exige además límite inferior del intervalo bootstrap del 95% mayor que cero y al menos dos periodos positivos.

## Límites
La clasificación sirve para priorizar investigación, no garantiza rentabilidad. El periodo reciente es corto, el universo usa la liquidez de hoy y puede sufrir sesgo de supervivencia. El benchmark comprar-y-mantener se alinea por periodo OOS, pero deben probarse más ciclos de mercado y validar costos, funding y ejecuciones intravela antes de usar dinero real.
"""
    (root / "ranking_methodology_es.md").write_text(methodology_es, encoding="utf-8")
    def reason_es(row) -> str:
        parts = []
        if pd.notna(row.get("Spot_Volume_Rank")) and pd.notna(row.get("Perp_Volume_Rank")):
            parts.append(
                f"Liquidez: puesto {int(row['Spot_Volume_Rank'])} en spot y "
                f"{int(row['Perp_Volume_Rank'])} en perpetuos por volumen cotizado de 24 h"
            )
        score = row.get("Final_Risk_Adjusted_Score")
        if pd.notna(score):
            parts.append(
                f"Señal MTF actual {float(score):.1f}/100; "
                f"spot {row.get('Spot_Direction', 'n/d')}, perpetuo {row.get('Perp_Direction', 'n/d')}"
            )
        trades_n = row.get("OOS_Trades")
        folds_n = row.get("OOS_Folds")
        mean_trade = row.get("OOS_Mean_Trade_Pct")
        if pd.notna(trades_n) and trades_n > 0:
            parts.append(
                f"SPOT_MTF fuera de muestra: {int(trades_n)} operaciones, "
                f"media por operación {float(mean_trade):+.3f}%, "
                f"{int(row.get('OOS_Positive_Folds', 0) or 0)}/{int(folds_n or 0)} periodos positivos"
            )
        else:
            parts.append("SPOT_MTF no generó operaciones fuera de muestra; la señal no está validada por el backtest")
        benchmark_return = row.get("BuyHold_OOS_Total_Return_Pct")
        strategy_return = row.get("OOS_Total_Return_Pct")
        if pd.notna(benchmark_return) and pd.notna(strategy_return):
            parts.append(
                f"Rendimiento OOS SPOT_MTF {float(strategy_return):+.2f}% frente a "
                f"comprar y mantener {float(benchmark_return):+.2f}%"
            )
        tier_es = {
            "POSITIVE_OOS_EVIDENCE": "evidencia OOS positiva, aún provisional",
            "MIXED_OOS_EVIDENCE": "evidencia OOS mixta",
            "INSUFFICIENT_OOS_SAMPLE": "muestra OOS insuficiente",
        }
        parts.append(tier_es.get(str(row.get("Evidence_Tier")), "evidencia no clasificada"))
        return "; ".join(parts)

    ranking["Motivo_Ranking_ES"] = ranking.apply(reason_es, axis=1)

    ranking["Ranking_Method"] = (
        "40% current risk-adjusted MTF signal + 25% SPOT_MTF OOS mean-trade-return percentile "
        "+ 20% positive-fold consistency + 10% OOS drawdown score + 5% sample-size score; "
        "assets with <20 OOS trades or <2 folds are marked insufficient and ranked after eligible assets"
    )
    ranking = ranking.sort_values(
        ["OOS_Eligible", "Final_Research_Score", "Final_Risk_Adjusted_Score", "Symbol"],
        ascending=[False, False, False, True], na_position="last", kind="mergesort"
    ).reset_index(drop=True)
    ranking.insert(0, "Research_Rank", range(1, len(ranking) + 1))
    ranking.to_csv(root / "final_crypto_ranking.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the audited per-asset ranking from MTF signals and walk-forward OOS evidence.")
    parser.add_argument("--output-dir", default="reports/free_cloud_backtest")
    args = parser.parse_args()
    build_final_ranking(Path(args.output_dir))


if __name__ == "__main__":
    main()
