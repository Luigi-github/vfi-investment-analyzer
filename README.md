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
- `test_vfi.py`: ejecución de investigación y comparación de activos.
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

La intención es que una observación futura no pueda mover hacia atrás una señal que ya habría sido conocida.

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

### Descubrimiento de condiciones sin contaminación del test

`ConditionDiscoveryAnalyzer` permite comparar hipótesis de señal de forma temporalmente separada:

1. las condiciones candidatas se evalúan en el conjunto de entrenamiento;
2. solo las condiciones que cumplen el mínimo de observaciones pueden ser seleccionadas;
3. la selección se ordena usando únicamente la métrica configurada del entrenamiento;
4. las condiciones seleccionadas se llevan al conjunto de prueba sin volver a optimizarlas;
5. el resultado del test se conserva como evaluación out-of-sample.

Esto evita escoger una condición porque funcionó bien en todo el histórico y luego presentar ese mismo histórico como validación independiente.

El flujo multi-activo expone este proceso mediante
`MultiAssetRecoveryAnalyzer.analyze_condition_discovery_walk_forward()`.
Cada activo obtiene sus propias ventanas de entrenamiento y prueba, y la
condición seleccionada en una ventana no se vuelve a optimizar usando su
respectiva ventana de test.

### Riesgo, costes y benchmark

`ForwardRiskAnalyzer` calcula como resultados posteriores el máximo favorable, máximo adverso y drawdown máximo para cada horizonte. Las ventanas incompletas al final de la muestra se dejan como `NaN` y no se tratan como resultados observados.

`LongOnlyExecutionCostModel` permite aplicar supuestos deterministas de comisión y slippage a retornos largos. El retorno bruto se conserva y el retorno neto se expone por separado.

El descubrimiento walk-forward puede ordenar hipótesis por `Average_Net_Return`, de modo que los costes formen parte de la selección realizada exclusivamente sobre entrenamiento y la condición seleccionada se evalúe después sobre test intacto. El runner genera además `cost_sensitivity.csv` para 0, 10, 25, 50 y 100 bps de coste round-trip supuesto.

`BenchmarkAnalyzer` compara los eventos seleccionados con el retorno forward de todas las fechas elegibles del mismo activo y horizonte. El benchmark es descriptivo y no se utiliza como feature de la señal.


## Pruebas

Ejecutar:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

La suite contiene pruebas para:

- estabilidad de retornos históricos al agregar datos futuros;
- estabilidad de IDs de clusters causales;
- selección del primer evento como representante causal;
- estabilidad del representante frente a eventos futuros;
- selección causal en análisis multi-activo;
- estabilidad end-to-end de las features del pipeline VFI + análisis + score frente a nuevas observaciones futuras.

GitHub Actions ejecuta automáticamente la suite en Python 3.12.

### Ranking out-of-sample

`ResearchRankingAnalyzer` genera un ranking transparente a partir de las métricas
walk-forward ya calculadas. No aprende pesos ni utiliza retornos del conjunto
de entrenamiento para modificar el resultado del test.

La métrica por defecto es `Average_Test_Return`; en caso de empate se usan
`Valid_Windows` y `Median_Test_Return` como criterios secundarios. El ranking
es descriptivo y no constituye una recomendación automática de inversión.

### Dashboard Streamlit

La interfaz de investigación se encuentra en `dashboard/app.py`.

Ejecutar:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

El dashboard muestra:

- estado semanal actual de los activos;
- VFI, señal y pendientes de 1/4/8 semanas;
- etapa del ciclo VFI;
- resultados del descubrimiento walk-forward;
- gráfico histórico de VFI y su señal.

El dashboard reutiliza el mismo motor cuantitativo; no contiene una implementación paralela del VFI.

## Instalación

Crear el entorno virtual:

```powershell
py -3.12 -m venv venv
```

Activarlo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
```

## Ejecución

Para ejecutar la investigación multi-activo:

```powershell
.\venv\Scripts\python.exe test_vfi.py
```

Para una prueba básica de datos de mercado:

```powershell
.\venv\Scripts\python.exe test_market_data.py
```

## Universos de investigación

Los candidatos de investigación están centralizados en `app/research/universes.py`, separados de la fórmula original del VFI:

- `USA_STOCKS`
- `USA_ETFS`
- `CRYPTO`
- `COLOMBIA`
- `CHILE`
- `LATAM`
- `COMMODITIES_ETFS`

El runner admite `ALL` para analizar todos los símbolos únicos.

Ejemplos:

```powershell
.\venv\Scripts\python.exe scripts\run_cycle_analysis.py --universe USA_STOCKS
.\venv\Scripts\python.exe scripts\run_cycle_analysis.py --universe CRYPTO
.\venv\Scripts\python.exe scripts\run_cycle_analysis.py --universe COLOMBIA
.\venv\Scripts\python.exe scripts\run_cycle_analysis.py --universe ALL
.\venv\Scripts\python.exe scripts\run_cycle_analysis.py --symbols DDD,EC,SQM,NVDA
```

Cada ejecución guarda sus resultados en `reports/cycle_analysis/<universe>/` e incluye, además de `asset_event_summary.csv`, los artefactos `asset_walk_forward.csv`, `asset_oos_ranking.csv` y `condition_discovery_walk_forward.csv`. El ranking OOS se construye únicamente con las ventanas de prueba del análisis walk-forward.

Los activos analizados admiten símbolos compatibles con yfinance, incluyendo acciones, ETFs y criptomonedas. La lista de candidatos es una hipótesis de investigación, no una recomendación de inversión.

## Reproducibilidad

El proyecto mantiene una separación explícita entre:

- cálculo de indicadores;
- clasificación;
- scoring;
- detección de eventos;
- clustering;
- selección causal de señales;
- cálculo de retornos posteriores;
- análisis retrospectivo exploratorio.

Los cambios metodológicos importantes deben acompañarse de pruebas que demuestren que no introducen información futura.

## Limitaciones actuales

1. Los datos de mercado dependen de la disponibilidad y calidad del proveedor utilizado.
2. El VFI es un indicador técnico; no incorpora por sí solo fundamentales, valoración, macroeconomía o noticias.
3. Los resultados históricos no garantizan resultados futuros.
4. Los métodos retrospectivos son útiles para explorar patrones, pero no deben confundirse con señales disponibles en tiempo real.
5. Los costes de comisión y slippage son actualmente supuestos deterministas; todavía no modelan impuestos, spreads variables, impacto de mercado ni restricciones de liquidez.
6. La clasificación de acumulación es una hipótesis cuantitativa que debe validarse estadísticamente antes de utilizarse para decisiones reales.

## Estado actual y extensiones pendientes

La base de investigación ya incluye validación causal, walk-forward, análisis por régimen, robustez estadística, dashboard, ranking descriptivo y selección de universos. Lo que queda fuera del núcleo actual es:

- modelado más realista de impuestos, spreads variables, impacto de mercado y liquidez;
- validación adicional de hipótesis descubiertas bajo distintos costes;
- integración futura de modelos de machine learning/IA, manteniendo separación estricta entre información disponible en cada fecha y datos posteriores;
- validación económica/fundamental independiente del VFI.

Estas extensiones no son necesarias para ejecutar el estudio histórico multi-mercado del VFI.


## Metodología de investigación

El proyecto separa explícitamente tres niveles de evidencia:

1. **Señal causal:** usa únicamente información disponible en la fecha del evento. Las entradas de etapa y los primeros eventos de recuperación no usan retornos futuros ni información posterior.
2. **Resultado forward:** los retornos de 4, 8, 12, 26 y 52 semanas se calculan únicamente cuando existe una ventana futura completa.
3. **Análisis retrospectivo:** los ciclos VFI completos se usan para estudiar resultados históricos y calidad de ciclos, pero no se consideran señales point-in-time.

La validación estadística incluye:

- media, mediana, dispersión y tasas de resultados;
- bootstrap reproducible de medias;
- exceso frente a benchmark;
- intervalo bootstrap del exceso;
- corrección Benjamini-Hochberg para múltiples hipótesis;
- validación walk-forward;
- análisis de estabilidad entre ventanas;
- clasificación causal de régimen BULL, BEAR y SIDEWAYS usando el benchmark SPY.

### Interpretación

Un retorno histórico positivo no implica una señal válida. Para considerar una condición candidata se debe comprobar simultáneamente:

- separación temporal entre entrenamiento y prueba;
- ausencia de look-ahead;
- suficiente número de observaciones;
- comportamiento fuera de muestra;
- comparación contra benchmark;
- estabilidad entre ventanas y activos;
- sensibilidad a costos y riesgo.

El **stability score** es un ranking descriptivo de investigación y no una recomendación automática de compra o venta.

## Reporte reproducible

El análisis histórico completo puede ejecutarse manualmente o de forma programada mediante GitHub Actions. El workflow genera los reportes CSV en `reports/cycle_analysis/`, incluyendo ciclos, entradas de etapa, estadísticas, resumen por activo y régimen de mercado.

El análisis histórico multi-mercado se ejecuta sobre los universos definidos en `app/research/universes.py`. El runner separa la selección de activos del cálculo del VFI y no modifica la fórmula original.

El flujo recomendado es:

1. ejecutar `USA_STOCKS`;
2. ejecutar `USA_ETFS`;
3. ejecutar `CRYPTO`;
4. ejecutar `COLOMBIA`;
5. ejecutar `CHILE`;
6. ejecutar `LATAM`;
7. ejecutar `COMMODITIES_ETFS`;
8. consolidar los resultados y aplicar el filtro de robustez antes de interpretar qué activos merecen una investigación adicional.

El resultado final debe distinguir entre rendimiento histórico, evidencia out-of-sample y estado VFI actual. Ninguna de estas métricas constituye por sí sola una predicción de rentabilidad futura.

El workflow automatizado no modifica la fórmula original del VFI.


## Capa de cartera

El proyecto incluye un backtest de cartera point-in-time en `app/backtest/portfolio_backtest.py`.

La estrategia recibe únicamente eventos causales de recuperación. La fecha del evento es la fecha de decisión; el retorno de cartera empieza en la siguiente observación semanal, evitando utilizar el retorno de la misma vela para construir la señal.

Métodos de ponderación disponibles:

- `equal`: peso igual entre posiciones activas;
- `inverse_vol`: ponderación inversa a volatilidad histórica;
- `risk_parity`: búsqueda de pesos de contribución de riesgo aproximadamente iguales mediante la matriz de covarianzas histórica.

Además permite:

- máximo de posiciones;
- peso máximo por activo;
- comisión y slippage;
- turnover;
- exposición;
- curva de equity;
- drawdown;
- CAGR;
- volatilidad anualizada;
- Sharpe;
- Sortino;
- tasa de periodos positivos.

Ejemplo:

```powershell
.\venv\Scripts\python.exe scripts\run_portfolio_backtest.py --symbols DDD,INTC,NVDA,BTC-USD,SQM,EC --horizon-weeks 26 --max-positions 10 --fee-bps 25 --slippage-bps 25
```

Los resultados se guardan en `reports/portfolio_backtest/<universe>/`.

## Benchmarks

`app/backtest/portfolio_benchmark.py` genera referencias pasivas para evitar evaluar una cartera VFI en aislamiento:

- cartera equal-weight reequilibrada del universo;
- Buy & Hold de SPY;
- Buy & Hold de QQQ;
- Buy & Hold de BTC-USD cuando existe el histórico correspondiente.

Los benchmarks no participan en la construcción de la señal.

## Liquidez y capacidad

`app/backtest/liquidity.py` calcula métricas point-in-time usando únicamente el historial disponible hasta la fecha del evento:

- volumen promedio en dinero;
- máximo nocional negociable bajo un porcentaje de participación máximo;
- peso máximo soportable para un capital dado;
- estado de elegibilidad.

El runner permite activar el filtro:

```powershell
.\venv\Scripts\python.exe scripts\run_portfolio_backtest.py --universe USA_STOCKS --capital 100000 --max-positions 10 --max-participation-pct 1 --apply-liquidity-filter
```

La capacidad es una aproximación conservadora. No sustituye un modelo de spread, market impact, latencia o ejecución real.

## Sensibilidad de parámetros VFI

`app/research/parameter_sensitivity.py` permite evaluar configuraciones alternativas de:

- `length`;
- `coef`;
- `vcoef`;
- `signal_length`.

La selección se realiza dentro de cada ventana de entrenamiento y la configuración seleccionada se evalúa después en la ventana de prueba intacta. El cálculo del VFI sigue viviendo exclusivamente en `VFIIndicator`; esta capa no modifica la fórmula original.

Ejemplo:

```powershell
.\venv\Scripts\python.exe scripts\run_parameter_sensitivity.py --universe USA_STOCKS
```

Los reportes se guardan en `reports/parameter_sensitivity/<universe>/`.

## Dashboard de investigación

El dashboard ahora incluye:

- estado actual del VFI;
- ciclos y eventos causales;
- descubrimiento walk-forward;
- ranking OOS;
- backtest de cartera;
- comparación equal-weight / inverse-vol / risk-parity;
- costes;
- filtro de liquidez;
- benchmarks pasivos;
- curva de equity.

Ejecutar:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

## Automatización

Además del workflow histórico multi-mercado, se recomienda ejecutar de forma independiente la investigación de cartera y la sensibilidad de parámetros. Esto evita mezclar un proceso de investigación intensivo con el cálculo normal del VFI y permite revisar los artefactos generados por cada etapa.

## Criterio de finalización

El proyecto se considera operacionalmente completo cuando:

1. la fórmula original del VFI permanece aislada y sin modificaciones por las capas de investigación;
2. las señales principales son causales;
3. los eventos tienen validación forward y walk-forward;
4. existen controles de robustez estadística y múltiples hipótesis;
5. los costes se modelan explícitamente;
6. la cartera tiene restricciones de posiciones y capacidad;
7. existe comparación contra benchmarks;
8. los parámetros pueden validarse fuera de muestra;
9. el dashboard y los runners reproducen el mismo motor cuantitativo;
10. las pruebas automatizadas cubren las capas críticas.

Esto no significa que el sistema pueda garantizar rentabilidad. Significa que existe una infraestructura reproducible para investigar si una hipótesis de VFI tiene evidencia histórica suficientemente robusta.


## ML supervisado OOS

La capa `app/research/ml_event_analyzer.py` es experimental. Utiliza un Random Forest únicamente sobre features disponibles en la fecha del evento y predice si el retorno forward del horizonte elegido será positivo.

La validación es walk-forward:

1. se construye el conjunto de entrenamiento con eventos anteriores al cutoff;
2. se entrena el modelo únicamente con ese conjunto;
3. se predicen los eventos posteriores;
4. se registran accuracy, balanced accuracy, precision, ROC-AUC y retorno medio de las predicciones positivas;
5. nunca se utiliza el retorno futuro de un evento como feature.

Ejemplo:

```powershell
.\venv\Scripts\python.exe scripts\run_ml_research.py --universe USA_STOCKS
```

Esta capa no reemplaza el VFI, no se considera una señal automática y debe superar validaciones OOS adicionales antes de utilizarse en una estrategia.


## Hardening y reproducibilidad

La investigación utiliza dependencias fijadas en `requirements.txt` y cada ejecución puede conservar metadatos de procedencia del dataset:

- proveedor y hora de descarga;
- rango temporal y número de filas;
- hash SHA-256 del dataset;
- parámetros exactos del VFI;
- versión de investigación;
- versión de fórmula VFI;
- definición de ciclo;
- definición de etapas.

Las funciones de `app/research/reproducibility.py` permiten reproducir y auditar estos metadatos.

## Comparabilidad entre activos

El VFI original nunca se modifica para conseguir comparabilidad. Se añaden features auxiliares mediante `app/features/vfi_normalization.py`:

- `VFI_ZScore`;
- `VFI_Percentile`;
- `VFI_Normalized_Range`;
- distancia respecto a extremos históricos.

Esto evita interpretar un valor absoluto de VFI como si tuviera la misma escala económica en todos los activos.

## Señal, ejecución y outcome

La investigación distingue estrictamente:

`Signal_Date -> Entry_Date -> Exit_Date -> Outcome`

Una señal observada al cierre de una semana no se ejecuta en ese mismo cierre. El backtester y `SignalOutcomeBuilder` utilizan la siguiente observación disponible.

El resultado conserva:

- fecha de señal;
- fecha y precio de entrada;
- fecha y precio de salida;
- retorno de precio;
- retorno total cuando existe `Adj Close`;
- VFI y score disponibles en la fecha de señal.

## Retorno total

Cuando el proveedor proporciona `Adj Close`, los análisis de ciclos y outcomes también pueden calcular retorno total ajustado por dividendos y splits. El retorno basado en `Close` se conserva como retorno de precio.

## Robustez estadística

Las estadísticas de etapas incluyen:

- bootstrap reproducible;
- bootstrap agrupado por activo y ciclo causal;
- block bootstrap para observaciones temporalmente dependientes;
- exceso frente a benchmark;
- corrección Benjamini-Hochberg para múltiples hipótesis.

Los eventos de etapa incorporan `Cycle_ID_Causal` para evitar tratar varias etapas del mismo ciclo como observaciones totalmente independientes.

## Validación de la implementación VFI

`tests/test_vfi_conformance.py` proporciona un arnés para comparar el VFI Python contra una exportación congelada de TradingView. La fixture externa se mantiene opcional porque debe provenir de una ejecución/exportación real de TradingView.

Para completar la conformidad externa, colocar un CSV en:

`tests/fixtures/vfi_conformance.csv`

con las columnas:

`Date, Open, High, Low, Close, Volume, VFI_Reference`

La prueba exige una diferencia absoluta máxima de `1e-6`.

## Machine Learning

El módulo ML permanece experimental. El baseline VFI debe evaluarse primero y el modelo solo puede utilizar features disponibles en la fecha del evento. La validación walk-forward impide usar retornos futuros del test como variables de entrenamiento.

El ML no se considera listo para producir señales automáticas hasta demostrar mejora OOS frente al baseline VFI bajo los mismos costes, benchmarks y ventanas temporales.

## Criterio de listo

El proyecto se considera **research-ready** cuando las pruebas automatizadas pasan, el entorno está fijado, la procedencia de datos queda registrada, las señales son ejecutables sin look-ahead, los resultados se separan de las etiquetas futuras y la evidencia OOS se conserva separada de la selección.

Research-ready no significa rentable ni garantiza resultados futuros.
