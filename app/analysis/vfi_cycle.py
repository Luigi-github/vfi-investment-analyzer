import pandas as pd


class VFICycleAnalyzer:

    def __init__(self):
        pass

    def analyze(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        df = data.copy()

        required_columns = [
            "VFI",
            "VFI_Slope",
            "VFI_Slope_4W",
            "VFI_Slope_8W",
            "VFI_State",
            "VFI_Cycle_Stage",
            "Accumulation_Confirmed"
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
        # 1. ETAPA OBJETIVA
        #
        # Mantemos exatamente a classificação produzida pelo score.
        #
        # NEUTRAL significa que esa semana no cumplió las
        # condiciones específicas de una etapa.
        # ==========================================================

        df["Cycle_Stage"] = (
            df["VFI_Cycle_Stage"]
            .fillna("NEUTRAL")
        )

        # ==========================================================
        # 2. ETAPA ANTERIOR OBJETIVA
        # ==========================================================

        df["Previous_Cycle_Stage"] = (
            df["Cycle_Stage"].shift(1)
        )

        # ==========================================================
        # 3. CAMBIO OBJETIVO DE ETAPA
        #
        # Aquí NO ignoramos NEUTRAL todavía.
        # Esto representa el cambio literal de clasificación.
        # ==========================================================

        df["Cycle_Stage_Changed"] = (
            df["Cycle_Stage"]
            != df["Previous_Cycle_Stage"]
        )

        if not df.empty:
            df.loc[
                df.index[0],
                "Cycle_Stage_Changed"
            ] = False

        # ==========================================================
        # 4. TRANSICIÓN OBJETIVA
        # ==========================================================

        df["Cycle_Transition"] = "NONE"

        changed_mask = (
            df["Cycle_Stage_Changed"]
        )

        previous_stage = (
            df["Previous_Cycle_Stage"]
            .fillna("NONE")
        )

        df.loc[
            changed_mask,
            "Cycle_Transition"
        ] = (
            previous_stage.astype(str)
            + " -> "
            + df.loc[
                changed_mask,
                "Cycle_Stage"
            ].astype(str)
        )

        # ==========================================================
        # 5. FASE DE RECUPERACIÓN
        # ==========================================================

        recovery_stages = [
            "EARLY_RECOVERY",
            "SUSTAINED_RECOVERY",
            "APPROACHING_ZERO",
            "ZERO_CROSS_UP"
        ]

        df["Recovery_Phase"] = (
            df["Cycle_Stage"]
            .isin(recovery_stages)
        )

        # ==========================================================
        # 6. FASE BAJISTA
        # ==========================================================

        bearish_stages = [
            "BEARISH_FLOW",
            "ZERO_CROSS_DOWN"
        ]

        df["Bearish_Phase"] = (
            df["Cycle_Stage"]
            .isin(bearish_stages)
        )

        # ==========================================================
        # 7. FASE DE ACUMULACIÓN
        # ==========================================================

        accumulation_stages = [
            "ZERO_CROSS_UP",
            "ACCUMULATION"
        ]

        df["Accumulation_Phase"] = (
            df["Cycle_Stage"]
            .isin(accumulation_stages)
            |
            df["Accumulation_Confirmed"].fillna(False)
        )

        # ==========================================================
        # 8. IDENTIFICADOR DE ETAPA OBJETIVA
        #
        # Este ID representa cambios reales de clasificación,
        # incluyendo NEUTRAL.
        # ==========================================================

        stage_change = (
            df["Cycle_Stage"]
            != df["Cycle_Stage"].shift(1)
        )

        df["Cycle_ID"] = (
            stage_change
            .fillna(True)
            .cumsum()
        )

        df["Weeks_In_Current_Stage"] = (
            df.groupby("Cycle_ID")
            .cumcount()
            + 1
        )

        # ==========================================================
        # 9. DURACIÓN DE LA ETAPA ANTERIOR
        # ==========================================================

        stage_sizes = (
            df.groupby("Cycle_ID")
            .size()
        )

        previous_stage_duration = (
            stage_sizes.shift(1)
        )

        df["Previous_Stage_Weeks"] = (
            df["Cycle_ID"]
            .map(previous_stage_duration)
        )

        # ==========================================================
        # 10. INICIO Y CONTINUACIÓN DE RECUPERACIÓN
        # ==========================================================

        previous_recovery = (
            df["Recovery_Phase"]
            .shift(1)
            .fillna(False)
        )

        df["Recovery_Started"] = (
            df["Recovery_Phase"]
            &
            ~previous_recovery
        )

        df["Recovery_Continuing"] = (
            df["Recovery_Phase"]
            &
            previous_recovery
        )

        # ==========================================================
        # 11. INICIO DE ACUMULACIÓN
        # ==========================================================

        previous_accumulation = (
            df["Accumulation_Phase"]
            .shift(1)
            .fillna(False)
        )

        df["Accumulation_Started"] = (
            df["Accumulation_Phase"]
            &
            ~previous_accumulation
        )

        # ==========================================================
        # 12. INICIO DE CONFIRMACIÓN
        # ==========================================================

        previous_confirmed = (
            df["Accumulation_Confirmed"]
            .shift(1)
            .fillna(False)
        )

        df["Accumulation_Confirmation_Started"] = (
            df["Accumulation_Confirmed"]
            &
            ~previous_confirmed
        )

        # ==========================================================
        # 13. CRUCES DE CERO
        # ==========================================================

        previous_vfi = (
            df["VFI"].shift(1)
        )

        df["Zero_Cross_Up_Event"] = (
            (previous_vfi < 0)
            &
            (df["VFI"] >= 0)
        )

        df["Zero_Cross_Down_Event"] = (
            (previous_vfi > 0)
            &
            (df["VFI"] <= 0)
        )

        # ==========================================================
        # 14. CALIDAD DE RECUPERACIÓN
        # ==========================================================

        df["Recovery_Quality"] = "NONE"

        strong_recovery = (
            (df["VFI_Slope"] > 0)
            &
            (df["VFI_Slope_4W"] > 0)
            &
            (df["VFI_Slope_8W"] > 0)
        )

        moderate_recovery = (
            (df["VFI_Slope"] > 0)
            &
            (
                (df["VFI_Slope_4W"] > 0)
                |
                (df["VFI_Slope_8W"] > 0)
            )
        )

        early_recovery = (
            (df["VFI_Slope"] > 0)
            &
            (df["VFI_Slope_4W"] <= 0)
            &
            (df["VFI_Slope_8W"] <= 0)
        )

        df.loc[
            early_recovery,
            "Recovery_Quality"
        ] = "EARLY"

        df.loc[
            moderate_recovery,
            "Recovery_Quality"
        ] = "MODERATE"

        df.loc[
            strong_recovery,
            "Recovery_Quality"
        ] = "STRONG"

        # ==========================================================
        # 15. DIRECCIÓN DEL CICLO
        # ==========================================================

        df["Cycle_Direction"] = "NEUTRAL"

        df.loc[
            df["Recovery_Phase"],
            "Cycle_Direction"
        ] = "RECOVERING"

        df.loc[
            df["Bearish_Phase"],
            "Cycle_Direction"
        ] = "DECLINING"

        df.loc[
            df["Accumulation_Phase"],
            "Cycle_Direction"
        ] = "ACCUMULATING"

        # ==========================================================
        # 16. ESTADO DEL CICLO
        # ==========================================================

        df["Cycle_Status"] = "NEUTRAL"

        df.loc[
            df["Bearish_Phase"],
            "Cycle_Status"
        ] = "BEARISH"

        df.loc[
            df["Recovery_Phase"],
            "Cycle_Status"
        ] = "RECOVERY"

        df.loc[
            df["Accumulation_Phase"],
            "Cycle_Status"
        ] = "ACCUMULATION"

        return df