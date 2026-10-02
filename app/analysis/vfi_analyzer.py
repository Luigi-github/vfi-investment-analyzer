import numpy as np
import pandas as pd


class VFIAnalyzer:

    def __init__(
        self,
        zero_threshold: float = 0.50,
        recovery_4w_threshold: float = 0.05,
        recovery_8w_threshold: float = 0.03,
        weakening_4w_threshold: float = -0.05,
        weakening_8w_threshold: float = -0.03,
        near_extreme_percent: float = 0.20
    ):
        self.zero_threshold = zero_threshold

        self.recovery_4w_threshold = (
            recovery_4w_threshold
        )

        self.recovery_8w_threshold = (
            recovery_8w_threshold
        )

        self.weakening_4w_threshold = (
            weakening_4w_threshold
        )

        self.weakening_8w_threshold = (
            weakening_8w_threshold
        )

        self.near_extreme_percent = (
            near_extreme_percent
        )

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
            "VFI_Distance_Zero",
            "VFI_Rolling_Min",
            "VFI_Rolling_Max"
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
        # 1. DEFAULT STATE
        # ==========================================================

        df["VFI_State"] = "NEUTRAL"

        # ==========================================================
        # 2. DISTANCE FROM ZERO
        # ==========================================================

        df["VFI_Near_Zero"] = (
            df["VFI_Distance_Zero"]
            <= self.zero_threshold
        )

        # ==========================================================
        # 3. SUSTAINED RECOVERY
        # ==========================================================

        sustained_recovery = (
            (df["VFI_Slope_4W"]
             > self.recovery_4w_threshold)
            &
            (df["VFI_Slope_8W"]
             > self.recovery_8w_threshold)
        )

        # ==========================================================
        # 4. SUSTAINED WEAKENING
        # ==========================================================

        sustained_weakening = (
            (df["VFI_Slope_4W"]
             < self.weakening_4w_threshold)
            &
            (df["VFI_Slope_8W"]
             < self.weakening_8w_threshold)
        )

        # ==========================================================
        # 5. VFI RANGE POSITION
        # ==========================================================

        vfi_range = (
            df["VFI_Rolling_Max"]
            - df["VFI_Rolling_Min"]
        )

        df["VFI_Range_Position"] = np.where(
            vfi_range > 0,
            (
                df["VFI"]
                - df["VFI_Rolling_Min"]
            ) / vfi_range,
            np.nan
        )

        # ==========================================================
        # 6. RECOVERY FROM 52W MINIMUM (%)
        # ==========================================================

        df["VFI_Recovery_From_Min_Pct"] = (
            df["VFI_Range_Position"] * 100
        )

        # ==========================================================
        # 7. DISTANCE FROM 52W MINIMUM
        # ==========================================================

        df["VFI_Distance_From_Min"] = (
            df["VFI"]
            - df["VFI_Rolling_Min"]
        )

        # ==========================================================
        # 8. DISTANCE FROM 52W MAXIMUM
        # ==========================================================

        df["VFI_Distance_From_Max"] = (
            df["VFI_Rolling_Max"]
            - df["VFI"]
        )

        # ==========================================================
        # 9. NEAR 52W MINIMUM / MAXIMUM
        # ==========================================================

        df["VFI_Near_52W_Min"] = (
            df["VFI_Range_Position"]
            <= self.near_extreme_percent
        )

        df["VFI_Near_52W_Max"] = (
            df["VFI_Range_Position"]
            >= (
                1.0
                - self.near_extreme_percent
            )
        )

        # ==========================================================
        # 10. CLASSIFY STATES
        # ==========================================================

        # ----------------------------------------------------------
        # Recent recovery
        # ----------------------------------------------------------

        recent_recovery = (
            (df["VFI"] < 0)
            &
            (df["VFI_Slope"] > 0)
        )

        df.loc[
            recent_recovery,
            "VFI_State"
        ] = "RECOVERY"

        # ----------------------------------------------------------
        # Sustained recovery
        # ----------------------------------------------------------

        sustained_recovery_negative = (
            (df["VFI"] < 0)
            &
            sustained_recovery
        )

        df.loc[
            sustained_recovery_negative,
            "VFI_State"
        ] = "SUSTAINED_RECOVERY"

        # ----------------------------------------------------------
        # Early accumulation
        # ----------------------------------------------------------

        early_accumulation = (
            (df["VFI"] < 0)
            &
            df["VFI_Near_Zero"]
            &
            sustained_recovery
        )

        df.loc[
            early_accumulation,
            "VFI_State"
        ] = "EARLY_ACCUMULATION"

        # ----------------------------------------------------------
        # Accumulation
        # ----------------------------------------------------------

        accumulation = (
            (df["VFI"] >= 0)
            &
            sustained_recovery
        )

        df.loc[
            accumulation,
            "VFI_State"
        ] = "ACCUMULATION"

        # ----------------------------------------------------------
        # Bullish flow
        # ----------------------------------------------------------

        bullish_flow = (
            (df["VFI"] > self.zero_threshold)
            &
            (df["VFI_Slope"] > 0)
        )

        df.loc[
            bullish_flow,
            "VFI_State"
        ] = "BULLISH_FLOW"

        # ----------------------------------------------------------
        # Recent weakening
        # ----------------------------------------------------------

        recent_weakening = (
            (df["VFI"] > 0)
            &
            (df["VFI_Slope"] < 0)
        )

        df.loc[
            recent_weakening,
            "VFI_State"
        ] = "WEAKENING"

        # ----------------------------------------------------------
        # Sustained weakening
        # ----------------------------------------------------------

        sustained_weakening_positive = (
            (df["VFI"] > 0)
            &
            sustained_weakening
        )

        df.loc[
            sustained_weakening_positive,
            "VFI_State"
        ] = "SUSTAINED_WEAKENING"

        # ----------------------------------------------------------
        # Distribution
        # ----------------------------------------------------------

        distribution = (
            (df["VFI"] > 0)
            &
            df["VFI_Near_Zero"]
            &
            sustained_weakening
        )

        df.loc[
            distribution,
            "VFI_State"
        ] = "DISTRIBUTION"

        # ----------------------------------------------------------
        # Bearish flow
        #
        # IMPORTANT:
        # Do not overwrite a recent recovery.
        # ----------------------------------------------------------

        bearish_flow = (
            (df["VFI"] < -self.zero_threshold)
            &
            sustained_weakening
            &
            ~recent_recovery
        )

        df.loc[
            bearish_flow,
            "VFI_State"
        ] = "BEARISH_FLOW"

        # ==========================================================
        # 11. RECOVERY STRENGTH CATEGORY
        # ==========================================================

        df["VFI_Recovery_Strength"] = "NONE"

        weak_recovery = (
            (df["VFI_Recovery_From_Min_Pct"] > 0)
            &
            (df["VFI_Recovery_From_Min_Pct"] < 10)
            &
            recent_recovery
        )

        df.loc[
            weak_recovery,
            "VFI_Recovery_Strength"
        ] = "EARLY"

        moderate_recovery = (
            (df["VFI_Recovery_From_Min_Pct"] >= 10)
            &
            (df["VFI_Recovery_From_Min_Pct"] < 30)
            &
            recent_recovery
        )

        df.loc[
            moderate_recovery,
            "VFI_Recovery_Strength"
        ] = "MODERATE"

        strong_recovery = (
            (df["VFI_Recovery_From_Min_Pct"] >= 30)
            &
            recent_recovery
        )

        df.loc[
            strong_recovery,
            "VFI_Recovery_Strength"
        ] = "STRONG"

        return df