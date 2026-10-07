# VFI Investment Analyzer

Analizador de investigación de inversiones basado en el **Volume Flow Indicator (VFI) de LazyBear**, con enfoque semanal y orientado a estudiar ciclos de flujo, recuperación y acumulación en múltiples activos.

> Este proyecto es una herramienta de investigación cuantitativa. Sus señales no constituyen asesoría financiera ni garantizan rendimientos futuros.

## Objetivo

El proyecto busca identificar y estudiar, de forma reproducible:

- recuperaciones del VFI desde zonas negativas;
- cambios de pendiente de 1, 4 y 8 semanas;
- aproximaciones y cruces del nivel cero;
- posibles fases de acumulación y distribución;
- eventos de recuperación y sus resultados posteriores;
- comportamiento de estas señales en distintos activos y horizontes.

El enfoque principal es **semanal y de largo plazo**, no trading intradía.

## Arquitectura

```
Datos de mercado
      |
      v
MarketDataService
      |
      v
OHLCV semanal
      |
      v
VFIIndicator
      |
      v
VFIAnalyzer
      |
      v
VFIScore
      |
      v
VFICycleAnalyzer
      |
      +----------------------+
      |                      |
      v                      v
RecoveryEventClusterer   RecoverySignalAnalyzer
      |
      v
MultiAssetRecoveryAnalyzer
      |
      v
Backtesting / summaries
```

### Componentes principales

- `app/data/market_data.py`: descarga y normalización de datos mediante yfinance.
- `app/indicators/vfi.py`: cálculo del VFI y métricas derivadas.
- `app/analysis/vfi_analyzer.py`: estados, posición relativa y recuperación.
- `app/scoring/vfi_score.py`: score de recuperación y clasificación de acumulación.
- `app/analysis/vfi_cycle.py`: interpretación de etapas del ciclo.
- `app/backtest/`: clustering, señales, resultados futuros y análisis multi-activo.
- `app/backtest/crypto_strategy.py`: backtest causal de Spot/Perpetual con costes, stops, targets y funding.
- `app/backtest/risk_management.py`: position sizing por riesgo y límite de exposición.
- `tests/`: pruebas automatizadas, incluyendo controles contra look-ahead bias.

## Metodología VFI

El indicador trabaja con datos OHLCV y calcula:

- precio típico;
- cambio logarítmico;
- volatilidad;
- cutoff de movimiento de precio;
- volumen medio y volumen limitado;
- flujo de volumen;
- VFI;
- señal VFI;
- pendiente semanal;
- pendientes de 4 y 8 semanas;
- distancia a cero;
- mínimo y máximo móviles del contexto.

La configuración actual del indicador utiliza:

- `length = 130`
- `coef = 0.2`
- `vcoef = 2.5`
- `signal_length = 5`

Debido a la ventana de 130 periodos y a las ventanas derivadas, se necesita suficiente historial para obtener señales válidas. Por eso la investigación multi-activo utiliza normalmente `period="max"`.

## Punto crítico: look-ahead bias

El backtest distingue entre análisis **causal** y análisis **retrospectivo**.

### Ruta causal

La ruta utilizada para las señales principales:

1. detecta el evento cuando aparece;
2. agrupa eventos únicamente según su orden temporal;
3. selecciona como representante el **primer evento observado** del cluster;
4. calcula los retornos posteriores desde esa fecha.

El representante causal se identifica como:

`Representative_Method = CAUSAL_FIRST_RECOVERY`

y:

`Representative_Causal = True`

### Métodos retrospectivos

Se conservan para investigación exploratoria:

- `MIN_VFI`
- `BEST_RECOVERY`

Estos métodos pueden utilizar información posterior dentro del cluster para escoger el representante. **No deben interpretarse como resultados de un backtest causal.**

Asimismo, `Recovery_Cluster_Size` puede depender del tamaño completo observado del cluster y no debe utilizarse como feature causal de una señal histórica.

## Backtesting

Los horizontes configurables incluyen:

- 4 semanas
- 8 semanas
- 12 semanas
- 26 semanas
- 52 semanas

También se pueden utilizar horizontes pequeños en las pruebas unitarias.

Los retornos forward se calculan desde la fecha de la señal hacia observaciones posteriores. Una observación futura no debe modificar el retorno histórico calculado para una fecha que ya tenía suficiente información.

### Backtest cripto Spot / Perpetual

La estrategia MTF de Binance utiliza señal confirmada en vela 1H cerrada y entrada en la apertura de la siguiente vela.

Incluye:

- Spot y USDⓈ-M Perpetual separados;
- VFI 1W, 1D, 4H y 1H;
- funding para perpetuos;
- long y short;
- comisión y slippage;
- stop loss y take profit;
- operaciones no solapadas;
- walk-forward OOS;
- block bootstrap.

El funding se contabiliza correctamente en ambas direcciones: funding positivo es coste para long y crédito para short.

El backtest sigue siendo un modelo de investigación: no modela liquidaciones, margen, fills parciales, spread dinámico ni impacto de mercado.

### Position sizing por riesgo

`app/backtest/risk_management.py` proporciona un sizing determinista basado en la distancia al stop:

`position_fraction = min(max_exposure, risk_per_trade / stop_distance)`

La configuración por defecto arriesga como máximo el 1% del equity por operación y limita la exposición al 100%.

Ejemplo:

```python
from app.backtest.risk_management import RiskConfig, position_fraction

config = RiskConfig(risk_per_trade=0.01, max_exposure=1.0)
fraction = position_fraction(
    equity=10_000,
    entry_price=100,
    stop_price=90,
    config=config,
)
# 0.10 de la cartera
```

Esta capa calcula tamaño teórico; no envía órdenes y no sustituye un modelo de margen o liquidación.

### Descubrimiento de condiciones sin contaminación del test

`ConditionDiscoveryAnalyzer` permite comparar hipótesis de señal de forma temporalmente separada:

1. las condiciones candidatas se evalúan en el conjunto de entrenamiento;
2. solo las condiciones que cumplen el mínimo de observaciones pueden ser seleccionadas;
3. la selección se ordena usando únicamente la métrica configurada del entrenamiento;
4. las condiciones seleccionadas se llevan al conjunto de prueba sin volver a optimizarlas;
5. el resultado del test se conserva como evaluación out-of-sample.

Esto evita escoger una condición porque funcionó bien en todo el histórico y luego presentar ese mismo histórico como validación independiente.

## Pruebas

Ejecutar:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

La suite cubre:

- look-ahead bias;
- estabilidad de clusters;
- selección causal;
- VFI + análisis + score;
- estrategia cripto;
- entrada en la siguiente vela;
- stop loss;
- funding long/short;
- position sizing y límites de exposición.

GitHub Actions ejecuta automáticamente la suite en Python 3.12.

## Dashboard Streamlit

La interfaz de investigación se encuentra en `dashboard/app.py`.

Ejecutar:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

El dashboard muestra el estado semanal, VFI, señal, pendientes, etapas, resultados walk-forward y gráficos históricos.

## Instalación

Crear el entorno virtual:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Universos de investigación

Los candidatos están centralizados en `app/research/universes.py`:

- `USA_STOCKS`
- `USA_ETFS`
- `CRYPTO`
- `COLOMBIA`
- `CHILE`
- `LATAM`
- `COMMODITIES_ETFS`

El runner admite `ALL` para analizar todos los símbolos únicos.

## Reproducibilidad y evidencia

Cada capa mantiene separadas:

- cálculo del indicador;
- clasificación;
- scoring;
- detección de eventos;
- selección causal;
- retornos posteriores;
- análisis retrospectivo.

Los resultados OOS nunca se reutilizan para seleccionar parámetros del mismo test.

El proyecto también dispone de metadatos de procedencia, hash de datasets y controles de reproducibilidad.

## Limitaciones

1. La calidad depende del proveedor de datos.
2. El VFI no incorpora por sí solo fundamentales, macroeconomía o noticias.
3. El histórico no garantiza resultados futuros.
4. Los costes reales pueden diferir de los supuestos.
5. No hay todavía ejecución real de órdenes.
6. El backtest Perpetual no modela liquidación/margen ni impacto de mercado.
7. La conformidad exacta con TradingView requiere una exportación real de TradingView en `tests/fixtures/vfi_conformance.csv`.

## Estado del proyecto

**Research-ready / paper-trading-ready.**

El núcleo cuantitativo, causalidad, walk-forward, robustez estadística, universos, Binance Spot/Perpetual, MTF, costes, funding, riesgo básico y automatización CI están implementados.

No debe considerarse un bot de trading de producción hasta añadir una capa independiente de ejecución, reconciliación de órdenes, margen/liquidación, controles de seguridad y pruebas prolongadas en paper trading.

## Criterio de finalización

El proyecto se considera operacionalmente completo cuando:

1. la fórmula original del VFI permanece aislada;
2. las señales principales son causales;
3. existe validación forward y walk-forward;
4. existen controles de robustez estadística;
5. los costes se modelan;
6. existe control básico de riesgo y exposición;
7. existen benchmarks;
8. los parámetros pueden validarse OOS;
9. dashboard y runners reutilizan el mismo motor;
10. las capas críticas tienen pruebas automatizadas.

Research-ready no significa rentable ni garantiza resultados futuros.
