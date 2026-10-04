import pandas as pd

from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.analysis.vfi_analyzer import VFIAnalyzer
from app.scoring.vfi_score import VFIScore
from app.analysis.vfi_cycle import VFICycleAnalyzer
from app.backtest.recovery_signal_analysis import RecoverySignalAnalyzer
from app.backtest.recovery_event_clustering import RecoveryEventClusterer
from app.backtest.multi_asset_recovery_analysis import (
    MultiAssetRecoveryAnalyzer
)
from app.backtest.recovery_cluster_analyzer import (
    RecoveryClusterAnalyzer
)
from app.backtest.benchmark import BenchmarkAnalyzer
from app.backtest.execution_costs import LongOnlyExecutionCostModel
from app.backtest.temporal_validation import TemporalHoldoutAnalyzer


# ============================================================
# CONFIGURACIÓN
# ============================================================

SYMBOLS = [
    "DDD",
    "INTC",
    "BTC-USD",
    "ETH-USD",
    "EEM",
    "NVDA",
    "SPY",
    "AMD",
]

PERIOD = "max"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def print_separator(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def format_dataframe(
    dataframe,
    float_columns=None
):
    if dataframe.empty:
        print("Sin datos.")
        return

    display_df = dataframe.copy()

    if float_columns:
        for column in float_columns:
            if column in display_df.columns:
                display_df[column] = display_df[
                    column
                ].round(4)

    print(display_df.to_string(index=True))


# ============================================================
# ANÁLISIS DE UN ACTIVO
# ============================================================

def analyze_asset(
    symbol,
    market_data_service,
    vfi_indicator,
    vfi_analyzer,
    scorer,
    cycle_analyzer,
    clusterer,
    recovery_analyzer
):

    print_separator(
        f"ANALIZANDO: {symbol}"
    )

    # --------------------------------------------------------
    # 1. DATOS DIARIOS
    # --------------------------------------------------------

    print("1. Descargando datos...")

    daily_data = (
        market_data_service.get_daily_data(
            symbol,
            period=PERIOD
        )
    )

    print(
        f"Datos diarios: {len(daily_data)}"
    )

    # --------------------------------------------------------
    # 2. DATOS SEMANALES
    # --------------------------------------------------------

    print()
    print("2. Convirtiendo a semanal...")

    weekly_data = (
        market_data_service.convert_to_weekly(
            daily_data
        )
    )

    # --------------------------------------------------------
    # 3. VFI
    # --------------------------------------------------------

    print()
    print("3. Calculando VFI...")

    vfi_data = (
        vfi_indicator.calculate(
            weekly_data
        )
    )

    valid_vfi = (
        vfi_data["VFI"]
        .notna()
        .sum()
    )

    print(
        f"Datos semanales: {len(vfi_data)}"
    )

    print(
        f"VFI válidos: {valid_vfi}"
    )

    # --------------------------------------------------------
    # 4. ANÁLISIS VFI
    # --------------------------------------------------------

    print()
    print("4. Analizando estados VFI...")

    analyzed_data = (
        vfi_analyzer.analyze(
            vfi_data
        )
    )

    # --------------------------------------------------------
    # 5. SCORE
    # --------------------------------------------------------

    print()
    print("5. Calculando VFI Score...")

    scored_data = (
        scorer.calculate(
            analyzed_data
        )
    )

    # --------------------------------------------------------
    # 6. CICLO VFI
    # --------------------------------------------------------

    print()
    print("6. Analizando ciclo VFI...")

    cycle_data = (
        cycle_analyzer.analyze(
            scored_data
        )
    )

    # --------------------------------------------------------
    # 7. CLUSTERING EARLY_RECOVERY
    # --------------------------------------------------------

    print()
    print("7. Clustering de EARLY_RECOVERY...")

    cluster_result = (
        clusterer.analyze_causal(
            cycle_data
        )
    )

    events = cluster_result[
        "events"
    ]

    clustered_events = cluster_result[
        "clustered_events"
    ]

    cluster_summary = clusterer.cluster_summary(
        cluster_result["clustered_events"]
    )

    representatives = cluster_result[
        "representatives"
    ]

    print(
        f"Eventos EARLY_RECOVERY: "
        f"{len(events)}"
    )

    print(
        f"Clusters independientes: "
        f"{clustered_events['Recovery_Cluster_ID'].nunique()}"
        if not clustered_events.empty
        else "0"
    )

    print(
        f"Eventos representativos: "
        f"{len(representatives)}"
    )

    # --------------------------------------------------------
    # 8. PROPAGAR ID DEL CLUSTER
    # --------------------------------------------------------

    cycle_data = cycle_data.copy()

    if not clustered_events.empty:

        cycle_data[
            "Recovery_Cluster_ID"
        ] = pd.NA

        for index, row in (
            clustered_events.iterrows()
        ):

            cycle_data.loc[
                index,
                "Recovery_Cluster_ID"
            ] = row[
                "Recovery_Cluster_ID"
            ]

    else:

        cycle_data[
            "Recovery_Cluster_ID"
        ] = pd.NA

    # --------------------------------------------------------
    # 9. ANÁLISIS DE SEÑALES DE RECUPERACIÓN
    # --------------------------------------------------------

    print()
    print(
        "8. Analizando señales de recuperación..."
    )

    recovery_result = (
        recovery_analyzer.analyze(
            cycle_data
        )
    )

    recovery_events = recovery_result[
        "events"
    ]

    # --------------------------------------------------------
    # 10. RESUMEN DE CLUSTERS
    # --------------------------------------------------------

    print()
    print_separator(
        f"RESUMEN DE CLUSTERS: {symbol}"
    )

    if not cluster_summary.empty:

        format_dataframe(
            cluster_summary,
            float_columns=[
                "Minimum_VFI",
                "Maximum_VFI",
                "Average_VFI"
            ]
        )

    else:

        print("No se encontraron clusters.")

    # --------------------------------------------------------
    # 11. ESTADO ACTUAL
    # --------------------------------------------------------

    print()
    print_separator(
        f"ESTADO ACTUAL: {symbol}"
    )

    valid_rows = cycle_data[
        cycle_data["VFI"].notna()
    ]

    if not valid_rows.empty:

        latest = valid_rows.iloc[-1]

        print(
            f"Fecha: {valid_rows.index[-1]}"
        )

        print(
            f"Precio: "
            f"{latest['Close']:.4f}"
        )

        print(
            f"VFI: "
            f"{latest['VFI']:.4f}"
        )

        print(
            f"VFI Signal: "
            f"{latest['VFI_Signal']:.4f}"
        )

        print(
            f"VFI Slope: "
            f"{latest['VFI_Slope']:.4f}"
        )

        print(
            f"Slope 4W: "
            f"{latest['VFI_Slope_4W']:.4f}"
        )

        print(
            f"Slope 8W: "
            f"{latest['VFI_Slope_8W']:.4f}"
        )

        print(
            f"Estado VFI: "
            f"{latest['VFI_State']}"
        )

        print(
            f"Cycle Stage: "
            f"{latest['VFI_Cycle_Stage']}"
        )

    # --------------------------------------------------------
    # 12. RETORNAR DATASET COMPLETO
    # --------------------------------------------------------

    return cycle_data


# ============================================================
# COMPARACIÓN DE REPRESENTANTES
# ============================================================

def analyze_representative_methods(
    asset_data
):

    print_separator(
        "COMPARACIÓN DE MÉTODOS DE REPRESENTANTE"
    )

    analyzer = RecoveryClusterAnalyzer()

    all_results = []

    asset_method_results = {}

    for symbol, data in asset_data.items():

        print()
        print("-" * 80)
        print(
            f"ACTIVO: {symbol}"
        )
        print("-" * 80)

        try:

            result = analyzer.analyze(
                data
            )

            asset_method_results[
                symbol
            ] = result

            summary = result[
                "summary"
            ].copy()

            summary.insert(
                0,
                "Symbol",
                symbol
            )

            all_results.append(
                summary
            )

            print()

            format_dataframe(
                summary,
                float_columns=[
                    "Positive_Rate_26W",
                    "Average_Return_26W",
                    "Median_Return_26W",
                    "Average_Return_52W"
                ]
            )

        except Exception as error:

            print(
                f"Error analizando métodos "
                f"para {symbol}: {error}"
            )

    # --------------------------------------------------------
    # RESUMEN GLOBAL
    # --------------------------------------------------------

    print_separator(
        "COMPARACIÓN GLOBAL DE MÉTODOS"
    )

    if not all_results:
        print(
            "No hay resultados disponibles."
        )

        return {
            "asset_results": {},
            "global_summary": pd.DataFrame(),
            "horizon_summary": pd.DataFrame()
        }

    combined_summary = pd.concat(
        all_results,
        ignore_index=True
    )

    print()

    format_dataframe(
        combined_summary,
        float_columns=[
            "Positive_Rate_26W",
            "Average_Return_26W",
            "Median_Return_26W",
            "Average_Return_52W"
        ]
    )

    # --------------------------------------------------------
    # RESUMEN GLOBAL REAL
    # --------------------------------------------------------
    #
    # Importante:
    # No promediamos porcentajes de los activos.
    #
    # Recalculamos usando todos los representantes
    # disponibles para cada método.
    # --------------------------------------------------------

    global_rows = []

    for method in [
        "MIN_VFI",
        "FIRST_RECOVERY",
        "BEST_RECOVERY"
    ]:

        method_frames = []

        for symbol, result in (
            asset_method_results.items()
        ):

            method_data = result[
                "methods"
            ].get(method)

            if (
                method_data is not None
                and not method_data.empty
            ):

                method_copy = (
                    method_data.copy()
                )

                method_copy[
                    "Symbol"
                ] = symbol

                method_frames.append(
                    method_copy
                )

        if not method_frames:
            continue

        combined_method = pd.concat(
            method_frames,
            ignore_index=False
        )

        valid = combined_method[
            combined_method[
                "Primary_Result"
            ].isin(
                ["POSITIVE", "NEGATIVE"]
            )
        ]

        positive_count = (
            valid["Primary_Result"]
            .eq("POSITIVE")
            .sum()
        )

        negative_count = (
            valid["Primary_Result"]
            .eq("NEGATIVE")
            .sum()
        )

        total = (
            positive_count
            + negative_count
        )

        global_rows.append(
            {
                "Method": method,
                "Assets": valid[
                    "Symbol"
                ].nunique(),
                "Clusters": len(
                    combined_method
                ),
                "Valid_Clusters": total,
                "Positive_26W": positive_count,
                "Negative_26W": negative_count,
                "Positive_Rate_26W": (
                    positive_count
                    / total
                    * 100.0
                    if total > 0
                    else float("nan")
                ),
                "Average_Return_26W": (
                    valid["Return_26W"].mean()
                ),
                "Median_Return_26W": (
                    valid["Return_26W"].median()
                ),
                "Average_Return_52W": (
                    valid["Return_52W"].mean()
                    if "Return_52W"
                    in valid.columns
                    else float("nan")
                )
            }
        )

    global_summary = pd.DataFrame(
        global_rows
    )

    print()
    print(
        "RESUMEN GLOBAL POR MÉTODO"
    )

    format_dataframe(
        global_summary,
        float_columns=[
            "Positive_Rate_26W",
            "Average_Return_26W",
            "Median_Return_26W",
            "Average_Return_52W"
        ]
    )

    # --------------------------------------------------------
    # RESUMEN POR HORIZONTE
    # --------------------------------------------------------

    horizon_frames = []

    for symbol, result in (
        asset_method_results.items()
    ):

        horizon_data = result[
            "horizon_summary"
        ].copy()

        if horizon_data.empty:
            continue

        horizon_data.insert(
            0,
            "Symbol",
            symbol
        )

        horizon_frames.append(
            horizon_data
        )

    if horizon_frames:

        horizon_summary = pd.concat(
            horizon_frames,
            ignore_index=True
        )

    else:

        horizon_summary = (
            pd.DataFrame()
        )

    print()
    print_separator(
        "COMPARACIÓN POR HORIZONTE"
    )

    if not horizon_summary.empty:

        format_dataframe(
            horizon_summary,
            float_columns=[
                "Average_Return",
                "Median_Return",
                "Positive_Rate",
                "Best_Return",
                "Worst_Return"
            ]
        )

    else:

        print(
            "No hay datos suficientes."
        )

    return {
        "asset_results": asset_method_results,
        "global_summary": global_summary,
        "horizon_summary": horizon_summary
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator(
        "VFI MULTI-ASSET ANALYSIS"
    )

    print()
    print("Activos:")

    for symbol in SYMBOLS:
        print(
            f"  - {symbol}"
        )

    print()
    print(
        f"Periodo de datos: {PERIOD}"
    )

    # --------------------------------------------------------
    # SERVICIOS
    # --------------------------------------------------------

    market_data_service = (
        MarketDataService()
    )

    vfi_indicator = (
        VFIIndicator()
    )

    vfi_analyzer = (
        VFIAnalyzer()
    )

    scorer = (
        VFIScore()
    )

    cycle_analyzer = (
        VFICycleAnalyzer()
    )

    clusterer = (
        RecoveryEventClusterer(
            minimum_separation_weeks=8
        )
    )

    recovery_analyzer = (
        RecoverySignalAnalyzer()
    )

    # --------------------------------------------------------
    # ANALIZAR ACTIVOS
    # --------------------------------------------------------

    asset_data = {}

    for symbol in SYMBOLS:

        try:

            data = analyze_asset(
                symbol,
                market_data_service,
                vfi_indicator,
                vfi_analyzer,
                scorer,
                cycle_analyzer,
                clusterer,
                recovery_analyzer
            )

            asset_data[
                symbol
            ] = data

        except Exception as error:

            print()
            print(
                f"ERROR EN {symbol}: "
                f"{error}"
            )

    # --------------------------------------------------------
    # ANÁLISIS MULTI-ASSET ORIGINAL
    # --------------------------------------------------------

    print_separator(
        "MULTI-ASSET CAUSAL RECOVERY ANALYSIS"
    )

    if not asset_data:

        print(
            "No se pudo analizar ningún activo."
        )

        return

    multi_asset_analyzer = (
        MultiAssetRecoveryAnalyzer()
    )

    multi_asset_result = (
        multi_asset_analyzer.analyze_causal(
            asset_data
        )
    )

    # --------------------------------------------------------
    # RESUMEN POR ACTIVO
    # --------------------------------------------------------

    print_separator(
        "RESUMEN POR ACTIVO"
    )

    asset_summary = (
        multi_asset_result[
            "asset_summary"
        ]
    )

    format_dataframe(
        asset_summary,
        float_columns=[
            "Positive_Rate_26W",
            "Average_Return_26W",
            "Median_Return_26W",
            "Average_Return_52W"
        ]
    )

    # --------------------------------------------------------
    # RESUMEN GLOBAL
    # --------------------------------------------------------

    print_separator(
        "RESUMEN GLOBAL"
    )

    global_summary = (
        multi_asset_result[
            "global_summary"
        ]
    )

    format_dataframe(
        global_summary,
        float_columns=[
            "Positive_Rate_26W",
            "Average_Return_26W",
            "Median_Return_26W",
            "Average_Return_52W"
        ]
    )

    # --------------------------------------------------------
    # ANÁLISIS DE CONDICIONES
    # --------------------------------------------------------

    print_separator(
        "ANÁLISIS DE CONDICIONES"
    )

    condition_analysis = (
        multi_asset_result[
            "condition_analysis"
        ]
    )

    format_dataframe(
        condition_analysis,
        float_columns=[
            "Positive_Rate_26W",
            "Average_Return_26W",
            "Median_Return_26W",
            "Average_Return_52W"
        ]
    )

    # --------------------------------------------------------
    # REPRESENTANTES ACTUALES
    # --------------------------------------------------------

    print_separator(
        "REPRESENTANTES DE CLUSTERS"
    )

    combined = (
        multi_asset_result[
            "combined"
        ]
    )

    if not combined.empty:

        representative_columns = [
            "Symbol",
            "Recovery_Cluster_ID",
            "Cluster_Start_Date",
            "Representative_Method",
            "Representative_Causal",
            "Cluster_End_Date",
            "Cluster_Size",
            "VFI",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_Distance_Zero",
            "VFI_Range_Position",
            "Return_4W",
            "Return_8W",
            "Return_12W",
            "Return_26W",
            "Return_52W",
            "Primary_Result"
        ]

        available_columns = [
            column
            for column in representative_columns
            if column in combined.columns
        ]

        representatives = (
            combined[
                available_columns
            ]
        )

        format_dataframe(
            representatives,
            float_columns=[
                "VFI",
                "VFI_Slope",
                "VFI_Slope_4W",
                "VFI_Slope_8W",
                "VFI_Distance_Zero",
                "VFI_Range_Position",
                "Return_4W",
                "Return_8W",
                "Return_12W",
                "Return_26W",
                "Return_52W"
            ]
        )

    else:

        print(
            "No hay representantes."
        )

    # --------------------------------------------------------
    # ANÁLISIS RETROSPECTIVO EXPLORATORIO
    # COMPARACIÓN DE REPRESENTANTES
    # MIN_VFI/BEST_RECOVERY pueden usar información futura dentro del cluster.
    # No deben tratarse como backtest causal.
    # --------------------------------------------------------

    representative_comparison = (
        analyze_representative_methods(
            asset_data
        )
    )

    # --------------------------------------------------------
    # BENCHMARK Y COSTES
    # --------------------------------------------------------

    print_separator(
        "BENCHMARK Y COSTES DE EJECUCION"
    )

    cost_model = LongOnlyExecutionCostModel(
        fee_bps=10,
        slippage_bps=5
    )

    benchmark_analyzer = BenchmarkAnalyzer(
        execution_cost_model=cost_model
    )

    benchmark_frames = []

    for symbol, data in asset_data.items():

        causal_result = multi_asset_analyzer.analyze_asset_causal(
            symbol,
            data
        )

        if causal_result.empty:
            continue

        benchmark = benchmark_analyzer.compare_event_dates(
            data,
            causal_result.index
        )

        if benchmark.empty:
            continue

        benchmark.insert(0, "Symbol", symbol)
        benchmark_frames.append(benchmark)

    if benchmark_frames:
        benchmark_summary = pd.concat(
            benchmark_frames,
            ignore_index=True
        )

        format_dataframe(
            benchmark_summary,
            float_columns=[
                "Event_Average_Return",
                "Benchmark_Average_Return",
                "Excess_Average_Return",
                "Event_Positive_Rate",
                "Benchmark_Positive_Rate"
            ]
        )
    else:
        print("No hay suficientes datos para benchmark.")

    # --------------------------------------------------------
    # VALIDACION TEMPORAL HOLDOUT
    # --------------------------------------------------------

    print_separator(
        "VALIDACION TEMPORAL HOLDOUT"
    )

    temporal_analyzer = TemporalHoldoutAnalyzer(
        primary_horizon=26
    )

    causal_events_by_asset = {}

    for symbol, data in asset_data.items():
        causal_events = multi_asset_analyzer.analyze_asset_causal(
            symbol,
            data
        )

        if not causal_events.empty:
            causal_events_by_asset[symbol] = causal_events

    temporal_cutoff = "2022-01-01"

    if causal_events_by_asset:
        temporal_summary = temporal_analyzer.analyze_assets(
            causal_events_by_asset,
            temporal_cutoff
        )

        format_dataframe(
            temporal_summary,
            float_columns=[
                "Train_Average_Return",
                "Train_Median_Return",
                "Train_Positive_Rate",
                "Test_Average_Return",
                "Test_Median_Return",
                "Test_Positive_Rate"
            ]
        )
    else:
        print("No hay eventos causales para validacion temporal.")

    # --------------------------------------------------------
    # VALIDACION WALK-FORWARD
    # --------------------------------------------------------

    print_separator(
        "VALIDACION WALK-FORWARD"
    )

    walk_forward_summary = multi_asset_analyzer.analyze_assets_walk_forward(
        causal_events_by_asset,
        initial_train_weeks=104,
        test_weeks=52,
        step_weeks=52
    )

    if not walk_forward_summary.empty:
        format_dataframe(
            walk_forward_summary,
            float_columns=[
                "Train_Average_Return",
                "Train_Median_Return",
                "Train_Positive_Rate",
                "Test_Average_Return",
                "Test_Median_Return",
                "Test_Positive_Rate"
            ]
        )

        walk_forward_consistency = (
            multi_asset_analyzer.summarize_walk_forward_assets(
                walk_forward_summary
            )
        )

        print()
        print("CONSISTENCIA OUT-OF-SAMPLE POR ACTIVO")

        format_dataframe(
            walk_forward_consistency,
            float_columns=[
                "Average_Test_Return",
                "Median_Test_Return",
                "Positive_Window_Rate",
                "Worst_Test_Return",
                "Best_Test_Return"
            ]
        )
    else:
        print("No hay suficientes eventos para walk-forward.")

    # --------------------------------------------------------
    # BENCHMARK WALK-FORWARD POR VENTANA
    # --------------------------------------------------------

    print_separator(
        "BENCHMARK WALK-FORWARD"
    )

    benchmark_walk_frames = []

    for symbol, events in causal_events_by_asset.items():
        windows = temporal_analyzer.walk_forward(
            events,
            initial_train_weeks=104,
            test_weeks=52,
            step_weeks=52
        )

        if not windows:
            continue

        benchmark_data = BenchmarkAnalyzer(
            horizons=(26,)
        ).calculate_forward_returns(
            asset_data[symbol]
        )

        benchmark_series = benchmark_data["Return_26W"]

        window_comparison = temporal_analyzer.summarize_benchmark_by_window(
            windows,
            benchmark_series
        )

        if not window_comparison.empty:
            window_comparison.insert(0, "Symbol", symbol)
            benchmark_walk_frames.append(window_comparison)

    if benchmark_walk_frames:
        benchmark_walk_forward = pd.concat(
            benchmark_walk_frames,
            ignore_index=True
        )

        format_dataframe(
            benchmark_walk_forward,
            float_columns=[
                "Test_Average_Return",
                "Benchmark_Average_Return",
                "Excess_Average_Return"
            ]
        )
    else:
        print("No hay suficientes datos para benchmark walk-forward.")

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "ANÁLISIS COMPLETADO"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()