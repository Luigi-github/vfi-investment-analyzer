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

### Riesgo, costes y benchmark

`ForwardRiskAnalyzer` calcula como resultados posteriores el máximo favorable, máximo adverso y drawdown máximo para cada horizonte. Las ventanas incompletas al final de la muestra se dejan como `NaN` y no se tratan como resultados observados.

`LongOnlyExecutionCostModel` permite aplicar supuestos deterministas de comisión y slippage a retornos largos. El retorno bruto se conserva y el retorno neto se expone por separado.

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

## Activos

Los activos analizados se configuran en `test_vfi.py`. El pipeline admite símbolos compatibles con yfinance, incluyendo acciones, ETFs y criptomonedas.

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

## Próximas extensiones

Una vez estabilizada esta base, las extensiones naturales son:

- impuestos, spreads, impacto de mercado y restricciones de liquidez;
- validación walk-forward;
- análisis por régimen de mercado;
- benchmark y comparación con estrategias simples;
- validación walk-forward;
- análisis por régimen de mercado;
- exportación de resultados;
- dashboard Streamlit;
- ranking multi-activo;
- integración futura de modelos de machine learning/IA, manteniendo separación entre datos disponibles en cada fecha y datos posteriores.

## Estado del proyecto

La rama de auditoría actual prioriza la integridad temporal del backtesting y la reproducibilidad antes de añadir modelos predictivos más complejos.
