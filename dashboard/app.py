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
    else:
        st.info("Aún no hay suficientes eventos para descubrimiento walk-forward.")
else:
    st.info("No hay eventos causales suficientes para el análisis.")

st.subheader("Historial VFI")
selected = st.selectbox("Activo para gráfico", list(asset_data))
chart_data = asset_data[selected][["VFI", "VFI_Signal"]].dropna().tail(156)
st.line_chart(chart_data)
