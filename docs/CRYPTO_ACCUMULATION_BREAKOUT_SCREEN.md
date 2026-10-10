# Filtro de oportunidades: VFI diario + Volume Divergence by MM semanal

## Objetivo

Buscar activos al comienzo de un ciclo alcista, no los que ya llevan una subida avanzada. El filtro prioriza osciladores que vienen de valores negativos, muestran pendiente positiva sostenida y están cerca de cruzar cero. No es una predicción garantizada ni una orden de compra.

## Marcos temporales (sin intradía)

- **VFI: 1D (velas diarias).** Se usa para localizar recuperación/acumulación y aproximación al cero.
- **Volume Divergence by MM: 1W (velas semanales).** Es el contexto lento que el usuario compara con el gráfico diario del VFI en TradingView.
- **No se usan 1H, 4H ni marcos de minutos.** Las pendientes de 4 y 8 barras significan 4/8 días en VFI y 4/8 semanas en Divergence; se rotulan por separado para evitar confundirlas.

El universo son hasta 200 símbolos USDT que estén activos tanto en Spot como en contratos perpetuos USDⓈ-M de Binance, ordenados por los rangos de volumen cotizado de 24 horas de ambos mercados. La liquidez usada para elegir el universo es una instantánea actual, no una clasificación histórica.

## Reglas de un candidato

Para cada oscilador se calcula:

- Pendiente de 1 barra: valor actual menos valor anterior.
- Pendiente de 4 barras: cambio neto de las últimas 4 barras dividido por 4.
- Pendiente de 8 barras: cambio neto de las últimas 8 barras dividido por 8.
- Proximidad al cero: distancia absoluta al cero frente al rango móvil de 52 barras.
- Fuerza normalizada: cambio neto de 4/8 barras dividido por la desviación estándar móvil de 52 barras.

**PRE_ZERO_BREAK** requiere:
1. Valor actual negativo.
2. Valor cerca de cero: como máximo 15% del rango móvil de 52 barras; en VFI diario además se limita a 2 puntos absolutos.
3. Pendientes de 1, 4 y 8 barras positivas.
4. Aumento normalizado de 4 barras de al menos 0.10 desviaciones estándar y de 8 barras de al menos 0.25.

**ZERO_CROSS_RECENT** es una categoría aparte: el oscilador acaba de pasar de negativo a cero/positivo, sigue dentro del 10% del rango móvil y las tres pendientes son positivas. No se acepta un oscilador que lleva tiempo muy por encima de cero.

La clasificación final da prioridad a:
1. **Tier 1 — VFI diario + Divergence semanal:** ambos cumplen PRE_ZERO_BREAK o ZERO_CROSS_RECENT.
2. **Tier 2 — Solo VFI diario:** el VFI cumple el filtro; Divergence todavía no lo confirma.
3. **Tier 3 — Sin setup completo:** se conserva para auditar falsos positivos y oportunidades que todavía no cumplen todas las condiciones.

Una divergencia alcista regular/oculta confirmada en las últimas 8 velas semanales se reporta como confirmación adicional, pero no sustituye la regla de pendiente/cerocercanía del oscilador.

## Ranking

- 65% del score continuo de VFI diario.
- 35% del score continuo de Divergence semanal.
- Bonificación de 5 puntos por divergencia alcista reciente, con tope de 100.
- El tier tiene prioridad sobre la puntuación: un activo con ambos setups aparece por encima de uno que solo cumple VFI.

Las puntuaciones sirven para ordenar candidatos, no son probabilidades de rentabilidad.

## Archivos

El script `scripts/run_crypto_accumulation_breakout_screen.py` produce:

- `all_assets.csv`: todos los activos evaluados y sus métricas.
- `vfi_daily_candidates.csv`: activos que pasan el filtro del VFI diario.
- `combined_vfi_divergence_candidates.csv`: candidatos que pasan ambos filtros.
- `errors.csv`: símbolos sin datos o con error de cálculo.
- `run_summary.json`: conteos, marcos temporales y reglas aplicadas.

Comando local:

```bash
python scripts/run_crypto_accumulation_breakout_screen.py --top 200 --daily-bars 1000 --weekly-bars 400 --min-volume 0
```

El workflow manual de GitHub Actions se llama **Binance Early Accumulation Breakout Screen** y conserva el artefacto durante 30 días. Si GitHub bloquea Actions por facturación de la cuenta, el código y el workflow quedan preparados, pero no se debe afirmar que el ranking se ejecutó hasta que exista un artefacto generado por una ejecución exitosa.
