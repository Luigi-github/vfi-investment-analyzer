import streamlit as st
import pandas as pd

from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator
from app.analysis.vfi_analyzer import VFIAnalyzer
from app.scoring.vfi_score import VFIScore
from app.analysis.vfi_cycle import VFICycleAnalyzer
from app.backtest.recovery_event_clustering import RecoveryEventClusterer
from app.backtest.recovery_signal_analysis import RecoverySignalAnalyzer
from app.backtest.multi_asset_recovery_analysis import MultiAssetRecoveryAnalyzer
from app.scoring.research_ranking import ResearchRankingAnalyzer
from app.backtest.portfolio_backtest import PortfolioBacktester
from app.backtest.portfolio_benchmark import PortfolioBenchmark
from app.backtest.liquidity import LiquidityAnalyzer


DEFAULT_SYMBOLS = ["DDD", "INTC", "BTC-USD", "ETH-USD", "EEM", "NVDA", "SPY", "AMD"]


@st.cache_data(ttl=3600)
def build_asset(symbol):
    market = MarketDataService()
    weekly = market.convert_to_weekly(
        market.get_daily_data(symbol, period="max")
    )
    vfi = VFIIndicator().calculate(weekly)
    analyzed = VFIAnalyzer().analyze(vfi)
    scored = VFIScore().calculate(analyzed)
    cycle = VFICycleAnalyzer().analyze(scored)
    clusterer = RecoveryEventClusterer(minimum_separation_weeks=8)
    clustered = clusterer.analyze_causal(cycle)["clustered_events"]

    cycle = cycle.copy()
    cycle["Recovery_Cluster_ID"] = pd.NA
    if not clustered.empty:
        cycle.loc[clustered.index, "Recovery_Cluster_ID"] = clustered["Recovery_Cluster_ID"]

    RecoverySignalAnalyzer().analyze(cycle)
    return cycle


st.set_page_config(
    page_title="VFI Investment Analyzer",
    page_icon="📈",
    layout="wide",
)

st.title("VFI Investment Analyzer")
st.caption("Investigación semanal de ciclos VFI. No es una señal automática de trading.")

symbols = st.multiselect(
    "Activos",
    options=sorted(set(DEFAULT_SYMBOLS)),
    default=DEFAULT_SYMBOLS[:4],
)

if not symbols:
    st.info("Selecciona al menos un activo.")
    st.stop()

asset_data = {}
errors = {}

with st.spinner("Descargando y calculando datos..."):
    for symbol in symbols:
        try:
            asset_data[symbol] = build_asset(symbol)
        except Exception as exc:
            errors[symbol] = str(exc)

if errors:
    st.warning("Algunos activos no pudieron analizarse.")
    st.json(errors)

if not asset_data:
    st.stop()

rows = []
for symbol, data in asset_data.items():
    valid = data[data["VFI"].notna()]
    if valid.empty:
        continue
    latest = valid.iloc[-1]
    rows.append({
        "Symbol": symbol,
        "Date": valid.index[-1],
        "Price": latest["Close"],
        "VFI": latest["VFI"],
        "Slope": latest["VFI_Slope"],
        "Slope_4W": latest["VFI_Slope_4W"],
        "Slope_8W": latest["VFI_Slope_8W"],
        "State": latest.get("VFI_State", "N/A"),
        "Cycle": latest.get("VFI_Cycle_Stage", "N/A"),
    })

latest_frame = pd.DataFrame(rows)

st.subheader("Estado actual")
st.dataframe(latest_frame, use_container_width=True, hide_index=True)

st.subheader("Descubrimiento walk-forward")

analyzer = MultiAssetRecoveryAnalyzer()
causal_events = {}

for symbol, data in asset_data.items():
    events = analyzer.analyze_asset_causal(symbol, data)
    if not events.empty:
        causal_events[symbol] = events

if causal_events:
    discovery = analyzer.analyze_condition_discovery_walk_forward(
        causal_events,
        initial_train_weeks=104,
        test_weeks=52,
        step_weeks=52,
        minimum_events=10,
        top_n=1,
        ranking_metric="Average_Return",
    )
    summary = discovery["summary"]
    if not summary.empty:
        st.dataframe(summary, use_container_width=True, hide_index=True)
        ranking = analyzer.summarize_walk_forward_assets(
            analyzer.analyze_assets_walk_forward(
                causal_events,
                initial_train_weeks=104,
                test_weeks=52,
                step_weeks=52,
            )
        )
        if not ranking.empty:
            st.subheader("Ranking out-of-sample")
            st.dataframe(
                ResearchRankingAnalyzer().rank(ranking),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Aún no hay suficientes eventos para descubrimiento walk-forward.")
else:
    st.info("No hay eventos causales suficientes para el análisis.")

st.subheader("Historial VFI")
selected = st.selectbox("Activo para gráfico", list(asset_data))
chart_data = asset_data[selected][["VFI", "VFI_Signal"]].dropna().tail(156)
st.line_chart(chart_data)


st.subheader("Backtest de cartera causal")

portfolio_col1, portfolio_col2, portfolio_col3 = st.columns(3)
with portfolio_col1:
    portfolio_horizon = st.selectbox("Horizonte", [8, 12, 26, 52], index=2)
with portfolio_col2:
    portfolio_max_positions = st.number_input(
        "Máximo de posiciones",
        min_value=1,
        max_value=max(1, len(asset_data)),
        value=min(10, max(1, len(asset_data))),
        step=1,
    )
with portfolio_col3:
    portfolio_cost_bps = st.number_input(
        "Coste por lado (bps)",
        min_value=0.0,
        max_value=200.0,
        value=25.0,
        step=5.0,
    )

portfolio_capital = st.number_input(
    "Capital de referencia (COP o unidad de cuenta)",
    min_value=1000.0,
    value=100000.0,
    step=10000.0,
)
portfolio_liquidity = st.checkbox(
    "Aplicar filtro de capacidad de liquidez",
    value=False,
)

portfolio_events = {}
portfolio_prices = {}
for symbol, data in asset_data.items():
    events = analyzer.analyze_asset_causal(symbol, data)
    if not events.empty:
        portfolio_events[symbol] = events[["Symbol"]]
    portfolio_prices[symbol] = data[["Close"]]

if portfolio_events:
    portfolio_signals = pd.concat(portfolio_events.values()).sort_index()
    liquidity_analyzer = LiquidityAnalyzer(lookback_weeks=13)
    liquidity_view = liquidity_analyzer.annotate_signals(
        portfolio_signals,
        asset_data,
        capital=portfolio_capital,
        max_participation_pct=1.0,
        target_position_weight=1.0 / portfolio_max_positions,
    )

    if portfolio_liquidity:
        portfolio_signals = liquidity_view[
            liquidity_view["Liquidity_Eligible"]
        ].set_index("Date")
        if portfolio_signals.empty:
            st.warning("El filtro de liquidez eliminó todas las señales.")
            st.stop()

    weighting_results = {}
    summary_rows = []

    for weighting in ("equal", "inverse_vol", "risk_parity"):
        backtester = PortfolioBacktester(
            horizon_weeks=int(portfolio_horizon),
            max_positions=int(portfolio_max_positions),
            weighting=weighting,
            fee_bps=float(portfolio_cost_bps),
            slippage_bps=float(portfolio_cost_bps),
        )
        result = backtester.run(portfolio_prices, portfolio_signals)
        weighting_results[weighting] = result["equity_curve"]

        row = result["summary"].copy()
        row["Weighting"] = weighting
        summary_rows.append(row)

    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True,
    )

    curves = pd.DataFrame()
    for weighting, curve in weighting_results.items():
        if curve.empty:
            continue
        series = curve.set_index("Date")["Equity"].rename(weighting)
        curves = pd.concat([curves, series], axis=1)

    if not curves.empty:
        st.line_chart(curves)

    benchmark = PortfolioBenchmark()
    benchmark_curves = {
        "Universe Equal Weight": benchmark.equal_weight_rebalanced(
            portfolio_prices,
            start_date=portfolio_signals.index.min(),
        )
    }
    for benchmark_symbol in ("SPY", "QQQ", "BTC-USD"):
        if benchmark_symbol not in portfolio_prices:
            try:
                portfolio_prices[benchmark_symbol] = build_asset(benchmark_symbol)[["Close"]]
            except Exception:
                continue
        benchmark_curves[benchmark_symbol] = benchmark.single_asset_buy_hold(
            portfolio_prices[benchmark_symbol],
            start_date=portfolio_signals.index.min(),
        )

    benchmark_rows = []
    for name, curve in benchmark_curves.items():
        row = benchmark.summarize(curve)
        row["Benchmark"] = name
        benchmark_rows.append(row)

    st.subheader("Benchmarks")
    st.dataframe(
        pd.DataFrame(benchmark_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Señales causales: {len(portfolio_signals)}. "
        f"Filtro de liquidez: {'activo' if portfolio_liquidity else 'inactivo'}."
    )
else:
    st.info("No hay señales causales de recuperación para construir la cartera.")
