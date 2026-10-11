import numpy as np
import pandas as pd


class VFIScore:

    def __init__(self):
        pass

    def calculate(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        df = data.copy()

        # Prefer timeframe-neutral bar names; accept legacy columns from older callers.
        if "VFI_Slope_4B" not in df.columns and "VFI_Slope_4W" in df.columns:
            df["VFI_Slope_4B"] = df["VFI_Slope_4W"]
        if "VFI_Slope_8B" not in df.columns and "VFI_Slope_8W" in df.columns:
            df["VFI_Slope_8B"] = df["VFI_Slope_8W"]

        required_columns = [
            "VFI",
            "VFI_Slope",
            "VFI_Slope_4B",
            "VFI_Slope_8B",
            "VFI_Distance_Zero",
            "VFI_Range_Position",
            "VFI_Recovery_From_Min_Pct",
            "VFI_State"
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # ==========================================================
        # 1. RECOVERY SCORE
        #
        # Mide la fuerza de la recuperación actual.
        #
        # Este score NO significa acumulación confirmada.
        #
        # Máximo: 70 puntos
        # ==========================================================

        df["Score_Direction"] = 0.0
        df["Score_Trend"] = 0.0
        df["Score_Location"] = 0.0
        df["Score_Recovery"] = 0.0
        df["Score_Zero"] = 0.0
        df["Score_State"] = 0.0

        # ==========================================================
        # 2. DIRECCIÓN - 1 SEMANA
        #
        # ¿El VFI está subiendo actualmente?
        #
        # Máximo: 20 puntos
        # ==========================================================

        direction_positive = (
            df["VFI_Slope"] > 0
        )

        df.loc[
            direction_positive,
            "Score_Direction"
        ] = 20

        # ==========================================================
        # 3. TENDENCIA - 4W + 8W
        #
        # Evalúa si la recuperación está adquiriendo duración.
        #
        # Máximo: 30 puntos
        # ==========================================================

        both_positive = (
            (df["VFI_Slope_4B"] > 0)
            &
            (df["VFI_Slope_8B"] > 0)
        )

        four_week_positive = (
            (df["VFI_Slope_4B"] > 0)
            &
            (df["VFI_Slope_8B"] <= 0)
        )

        eight_week_positive = (
            (df["VFI_Slope_8B"] > 0)
            &
            (df["VFI_Slope_4B"] <= 0)
        )

        df.loc[
            both_positive,
            "Score_Trend"
        ] = 30

        df.loc[
            four_week_positive,
            "Score_Trend"
        ] = 15

        df.loc[
            eight_week_positive,
            "Score_Trend"
        ] = 10

        # ==========================================================
        # 4. UBICACIÓN DENTRO DEL RANGO DE 52 SEMANAS
        #
        # Cerca del mínimo = recuperación temprana.
        #
        # Máximo: 20 puntos
        # ==========================================================

        near_minimum = (
            df["VFI_Range_Position"] <= 0.20
        )

        low_range = (
            (df["VFI_Range_Position"] > 0.20)
            &
            (df["VFI_Range_Position"] <= 0.40)
        )

        middle_range = (
            (df["VFI_Range_Position"] > 0.40)
            &
            (df["VFI_Range_Position"] <= 0.60)
        )

        df.loc[
            near_minimum,
            "Score_Location"
        ] = 20

        df.loc[
            low_range,
            "Score_Location"
        ] = 15

        df.loc[
            middle_range,
            "Score_Location"
        ] = 10

        # ==========================================================
        # 5. RECUPERACIÓN DESDE EL MÍNIMO
        #
        # Una recuperación inicial puede representar una etapa
        # temprana del ciclo.
        #
        # Máximo: 15 puntos
        # ==========================================================

        early_recovery = (
            (df["VFI_Recovery_From_Min_Pct"] >= 0)
            &
            (df["VFI_Recovery_From_Min_Pct"] < 10)
        )

        moderate_recovery = (
            (df["VFI_Recovery_From_Min_Pct"] >= 10)
            &
            (df["VFI_Recovery_From_Min_Pct"] < 30)
        )

        established_recovery = (
            (df["VFI_Recovery_From_Min_Pct"] >= 30)
            &
            (df["VFI_Recovery_From_Min_Pct"] < 60)
        )

        advanced_recovery = (
            df["VFI_Recovery_From_Min_Pct"] >= 60
        )

        df.loc[
            early_recovery,
            "Score_Recovery"
        ] = 15

        df.loc[
            moderate_recovery,
            "Score_Recovery"
        ] = 12

        df.loc[
            established_recovery,
            "Score_Recovery"
        ] = 8

        df.loc[
            advanced_recovery,
            "Score_Recovery"
        ] = 3

        # ==========================================================
        # 6. PROXIMIDAD A CERO
        #
        # Es especialmente relevante cuando VFI está negativo.
        #
        # Máximo: 10 puntos
        # ==========================================================

        close_to_zero = (
            df["VFI_Distance_Zero"] <= 0.50
        )

        moderately_close = (
            (df["VFI_Distance_Zero"] > 0.50)
            &
            (df["VFI_Distance_Zero"] <= 1.50)
        )

        df.loc[
            close_to_zero,
            "Score_Zero"
        ] = 10

        df.loc[
            moderately_close,
            "Score_Zero"
        ] = 5

        # ==========================================================
        # 7. SCORE SEGÚN ESTADO VFI
        #
        # Máximo: 5 puntos
        # ==========================================================

        state_scores = {
            "EARLY_ACCUMULATION": 5,
            "SUSTAINED_RECOVERY": 4,
            "RECOVERY": 3,
            "ACCUMULATION": 5,
            "BULLISH_FLOW": 2,
            "NEUTRAL": 0,
            "WEAKENING": 0,
            "SUSTAINED_WEAKENING": 0,
            "DISTRIBUTION": 0,
            "BEARISH_FLOW": 0
        }

        df["Score_State"] = (
            df["VFI_State"]
            .map(state_scores)
            .fillna(0)
        )

        # ==========================================================
        # 8. RECOVERY SCORE
        #
        # 20 + 30 + 20 + 15 + 10 + 5 = 100
        #
        # Representa la calidad general de la recuperación,
        # no una confirmación de acumulación.
        # ==========================================================

        recovery_score_columns = [
            "Score_Direction",
            "Score_Trend",
            "Score_Location",
            "Score_Recovery",
            "Score_Zero",
            "Score_State"
        ]

        df["VFI_Recovery_Score"] = (
            df[recovery_score_columns]
            .sum(axis=1)
        )

        # Mantener VFI_Score temporalmente como alias.
        #
        # Esto evita romper otros archivos mientras evolucionamos
        # el proyecto.
        df["VFI_Score"] = (
            df["VFI_Recovery_Score"]
        )

        # ==========================================================
        # 9. CATEGORÍA DE RECUPERACIÓN
        # ==========================================================

        df["VFI_Recovery_Category"] = "WEAK"

        df.loc[
            df["VFI_Recovery_Score"] >= 70,
            "VFI_Recovery_Category"
        ] = "STRONG"

        df.loc[
            (df["VFI_Recovery_Score"] >= 50)
            &
            (df["VFI_Recovery_Score"] < 70),
            "VFI_Recovery_Category"
        ] = "MODERATE"

        df.loc[
            (df["VFI_Recovery_Score"] >= 30)
            &
            (df["VFI_Recovery_Score"] < 50),
            "VFI_Recovery_Category"
        ] = "EARLY"

        # ==========================================================
        # 10. COMPONENTES DE ACUMULACIÓN
        #
        # Aquí somos mucho más estrictos.
        #
        # La recuperación semanal por sí sola NO es suficiente.
        # ==========================================================

        df["Accumulation_Direction"] = (
            df["VFI_Slope"] > 0
        )

        df["Accumulation_4W"] = (
            df["VFI_Slope_4B"] > 0
        )

        df["Accumulation_8W"] = (
            df["VFI_Slope_8B"] > 0
        )

        df["Accumulation_Negative_VFI"] = (
            df["VFI"] < 0
        )

        df["Accumulation_Near_Zero"] = (
            df["VFI_Distance_Zero"] <= 1.50
        )

        # ==========================================================
        # 11. ACUMULACIÓN SCORE
        #
        # Máximo: 100 puntos
        #
        # 20 = dirección semanal
        # 25 = tendencia 4W
        # 25 = tendencia 8W
        # 15 = VFI cerca de cero
        # 15 = recuperación desde mínimo
        # ==========================================================

        df["Accumulation_Score"] = 0.0

        # Dirección semanal
        df.loc[
            df["Accumulation_Direction"],
            "Accumulation_Score"
        ] += 20

        # Tendencia 4 semanas
        df.loc[
            df["Accumulation_4W"],
            "Accumulation_Score"
        ] += 25

        # Tendencia 8 semanas
        df.loc[
            df["Accumulation_8W"],
            "Accumulation_Score"
        ] += 25

        # Cercanía a cero
        df.loc[
            df["Accumulation_Near_Zero"],
            "Accumulation_Score"
        ] += 15

        # Recuperación desde mínimo
        recovery_for_accumulation = (
            (df["VFI_Recovery_From_Min_Pct"] >= 10)
            &
            (df["VFI_Recovery_From_Min_Pct"] < 60)
        )

        df.loc[
            recovery_for_accumulation,
            "Accumulation_Score"
        ] += 15

        # ==========================================================
        # 12. BONUS POR VFI POSITIVO
        #
        # Un VFI positivo no significa necesariamente que estemos
        # ante el inicio de una acumulación, por lo que NO se suma
        # directamente al score.
        #
        # Lo registramos por separado.
        # ==========================================================

        df["VFI_Positive"] = (
            df["VFI"] >= 0
        )

        # ==========================================================
        # 13. CRUCE DE CERO
        #
        # Detectamos cuando VFI pasa de negativo a positivo.
        # ==========================================================

        previous_vfi = df["VFI"].shift(1)

        df["VFI_Zero_Cross_Up"] = (
            (previous_vfi < 0)
            &
            (df["VFI"] >= 0)
        )

        df["VFI_Zero_Cross_Down"] = (
            (previous_vfi > 0)
            &
            (df["VFI"] <= 0)
        )

        # ==========================================================
        # 14. ACUMULACIÓN CONFIRMADA
        #
        # Requisitos:
        #
        # - VFI negativo o cruce reciente
        # - dirección semanal positiva
        # - tendencia 4W positiva
        # - tendencia 8W positiva
        # - score de acumulación >= 70
        #
        # No utilizamos únicamente el score.
        # ==========================================================

        accumulation_confirmed = (
            (
                df["Accumulation_Negative_VFI"]
                |
                df["VFI_Zero_Cross_Up"]
            )
            &
            df["Accumulation_Direction"]
            &
            df["Accumulation_4W"]
            &
            df["Accumulation_8W"]
            &
            (df["Accumulation_Score"] >= 70)
        )

        df["Accumulation_Confirmed"] = (
            accumulation_confirmed
        )

        # ==========================================================
        # 15. POTENCIAL ACUMULACIÓN
        #
        # Esta categoría es anterior a la confirmación.
        #
        # Requisitos:
        #
        # - Score >= 50
        # - VFI negativo
        # - dirección semanal positiva
        #
        # PERO:
        # no consideramos que sea acumulación confirmada.
        # ==========================================================

        potential_accumulation = (
            (df["Accumulation_Score"] >= 50)
            &
            (df["VFI"] < 0)
            &
            (df["VFI_Slope"] > 0)
        )

        df["Potential_Accumulation"] = (
            potential_accumulation
        )

        # ==========================================================
        # 16. CATEGORÍA DE ACUMULACIÓN
        # ==========================================================

        df["Accumulation_Category"] = (
            "NO_SETUP"
        )

        df.loc[
            df["Potential_Accumulation"],
            "Accumulation_Category"
        ] = "POTENTIAL"

        df.loc[
            df["Accumulation_Confirmed"],
            "Accumulation_Category"
        ] = "CONFIRMED"

        # ==========================================================
        # 17. ETAPA DEL CICLO
        #
        # El ciclo debe representar la evolución del flujo VFI,
        # no solamente detectar acumulación.
        #
        # Etapas alcistas:
        #
        # BEARISH_FLOW
        # EARLY_RECOVERY
        # SUSTAINED_RECOVERY
        # APPROACHING_ZERO
        # ZERO_CROSS_UP
        # ACCUMULATION
        #
        # Etapas posteriores:
        #
        # WEAKENING
        # SUSTAINED_WEAKENING
        # DISTRIBUTION
        # ZERO_CROSS_DOWN
        #
        # NEUTRAL queda reservado para situaciones donde no existe
        # suficiente evidencia para clasificar el ciclo.
        # ==========================================================

        df["VFI_Cycle_Stage"] = "NEUTRAL"

        # ==========================================================
        # 17.1 CRUCE DE CERO HACIA ABAJO
        #
        # Tiene prioridad sobre otras condiciones porque representa
        # un evento explícito del ciclo.
        # ==========================================================

        zero_cross_down = (
            df["VFI_Zero_Cross_Down"]
        )

        df.loc[
            zero_cross_down,
            "VFI_Cycle_Stage"
        ] = "ZERO_CROSS_DOWN"

        # ==========================================================
        # 17.2 FLUJO BAJISTA
        #
        # VFI negativo y deteriorándose.
        # ==========================================================

        bearish_phase = (
            (df["VFI"] < 0)
            &
            (df["VFI_Slope"] < 0)
            &
            (df["VFI_Slope_4B"] < 0)
        )

        df.loc[
            bearish_phase,
            "VFI_Cycle_Stage"
        ] = "BEARISH_FLOW"

        # ==========================================================
        # 17.3 RECUPERACIÓN INICIAL
        #
        # VFI todavía negativo, pero la dirección semanal comienza
        # a mejorar.
        # ==========================================================

        early_recovery_phase = (
            (df["VFI"] < 0)
            &
            (df["VFI_Slope"] > 0)
            &
            (df["VFI_Slope_4B"] <= 0)
        )

        df.loc[
            early_recovery_phase,
            "VFI_Cycle_Stage"
        ] = "EARLY_RECOVERY"

        # ==========================================================
        # 17.4 RECUPERACIÓN SOSTENIDA
        #
        # Las tendencias de 4W y 8W también son positivas.
        # ==========================================================

        sustained_recovery_phase = (
            (df["VFI"] < 0)
            &
            (df["VFI_Slope"] > 0)
            &
            (df["VFI_Slope_4B"] > 0)
            &
            (df["VFI_Slope_8B"] > 0)
        )

        df.loc[
            sustained_recovery_phase,
            "VFI_Cycle_Stage"
        ] = "SUSTAINED_RECOVERY"

        # ==========================================================
        # 17.5 ACERCAMIENTO A CERO
        #
        # VFI todavía negativo, pero ya está relativamente cerca
        # de cero.
        #
        # Se evalúa después de SUSTAINED_RECOVERY para que la
        # clasificación sea más específica.
        # ==========================================================

        near_zero_phase = (
            (df["VFI"] < 0)
            &
            (df["VFI_Distance_Zero"] <= 1.50)
            &
            (df["VFI_Slope"] > 0)
            &
            (df["VFI_Slope_4B"] > 0)
        )

        df.loc[
            near_zero_phase,
            "VFI_Cycle_Stage"
        ] = "APPROACHING_ZERO"

        # ==========================================================
        # 17.6 CRUCE DE CERO HACIA ARRIBA
        # ==========================================================

        zero_cross_up = (
            df["VFI_Zero_Cross_Up"]
        )

        df.loc[
            zero_cross_up,
            "VFI_Cycle_Stage"
        ] = "ZERO_CROSS_UP"

        # ==========================================================
        # 17.7 ACUMULACIÓN
        #
        # Una vez confirmada la acumulación, esta tiene prioridad
        # sobre la clasificación genérica de flujo positivo.
        # ==========================================================

        accumulation_phase = (
            df["Accumulation_Confirmed"]
        )

        df.loc[
            accumulation_phase,
            "VFI_Cycle_Stage"
        ] = "ACCUMULATION"

        # ==========================================================
        # 17.8 DEBILITAMIENTO
        #
        # VFI positivo pero la dirección semanal comienza a caer.
        # ==========================================================

        weakening_phase = (
            (df["VFI"] > 0)
            &
            (df["VFI_Slope"] < 0)
        )

        df.loc[
            weakening_phase,
            "VFI_Cycle_Stage"
        ] = "WEAKENING"

        # ==========================================================
        # 17.9 DEBILITAMIENTO SOSTENIDO
        #
        # Tanto 4W como 8W muestran deterioro.
        # ==========================================================

        sustained_weakening_phase = (
            (df["VFI"] > 0)
            &
            (df["VFI_Slope"] < 0)
            &
            (df["VFI_Slope_4B"] < 0)
            &
            (df["VFI_Slope_8B"] < 0)
        )

        df.loc[
            sustained_weakening_phase,
            "VFI_Cycle_Stage"
        ] = "SUSTAINED_WEAKENING"

        # ==========================================================
        # 17.10 DISTRIBUCIÓN
        #
        # VFI positivo pero acercándose a cero mientras mantiene
        # una tendencia negativa.
        # ==========================================================

        distribution_phase = (
            (df["VFI"] > 0)
            &
            (df["VFI_Distance_Zero"] <= 1.50)
            &
            (df["VFI_Slope"] < 0)
            &
            (df["VFI_Slope_4B"] < 0)
        )

        df.loc[
            distribution_phase,
            "VFI_Cycle_Stage"
        ] = "DISTRIBUTION"

        # ==========================================================
        # 17.11 FLUJO ALCISTA
        #
        # VFI positivo y subiendo, pero todavía no cumple las
        # condiciones estrictas de acumulación confirmada.
        # ==========================================================

        bullish_flow_phase = (
            (df["VFI"] > 0)
            &
            (df["VFI_Slope"] > 0)
        )

        df.loc[
            bullish_flow_phase,
            "VFI_Cycle_Stage"
        ] = "BULLISH_FLOW"

        # ==========================================================
        # 17.12 PRIORIDAD DE ETAPAS
        #
        # Algunas condiciones pueden coincidir.
        # Reaplicamos las etapas más específicas al final para
        # garantizar que no sean sobrescritas por categorías
        # genéricas.
        # ==========================================================

        df.loc[
            sustained_weakening_phase,
            "VFI_Cycle_Stage"
        ] = "SUSTAINED_WEAKENING"

        df.loc[
            distribution_phase,
            "VFI_Cycle_Stage"
        ] = "DISTRIBUTION"

        df.loc[
            sustained_recovery_phase,
            "VFI_Cycle_Stage"
        ] = "SUSTAINED_RECOVERY"

        df.loc[
            near_zero_phase,
            "VFI_Cycle_Stage"
        ] = "APPROACHING_ZERO"

        df.loc[
            zero_cross_down,
            "VFI_Cycle_Stage"
        ] = "ZERO_CROSS_DOWN"

        df.loc[
            zero_cross_up,
            "VFI_Cycle_Stage"
        ] = "ZERO_CROSS_UP"

        df.loc[
            accumulation_phase,
            "VFI_Cycle_Stage"
        ] = "ACCUMULATION"

        return df