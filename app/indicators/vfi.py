import numpy as np
import pandas as pd


class VFIIndicator:

    def __init__(
        self,
        length: int = 130,
        coef: float = 0.2,
        vcoef: float = 2.5,
        signal_length: int = 20,
        signal_length_50: int = 50,
        smooth_vfi: bool = False,
        context_window: int = 52
    ):
        self.length = length
        self.coef = coef
        self.vcoef = vcoef
        self.signal_length = signal_length
        self.signal_length_50 = signal_length_50
        self.smooth_vfi = smooth_vfi
        self.context_window = context_window

    def calculate(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        df = data.copy()

        # ==========================================================
        # 1. Typical Price / HLC3
        # ==========================================================

        df["TypicalPrice"] = (
            df["High"]
            + df["Low"]
            + df["Close"]
        ) / 3.0

        # ==========================================================
        # 2. Logarithmic price change
        # ==========================================================

        df["LogChange"] = (
            np.log(df["TypicalPrice"])
            - np.log(df["TypicalPrice"].shift(1))
        )

        # ==========================================================
        # 3. Volatility
        # ==========================================================

        df["Volatility"] = (
            df["LogChange"]
            .rolling(
                window=30,
                min_periods=30
            )
            .std()
        )

        # ==========================================================
        # 4. Cutoff
        # ==========================================================

        df["Cutoff"] = (
            self.coef
            * df["Volatility"]
            * df["Close"]
        )

        # ==========================================================
        # 5. Average Volume
        # ==========================================================

        df["VolumeAverage"] = (
            df["Volume"]
            .rolling(
                window=self.length,
                min_periods=self.length
            )
            .mean()
            .shift(1)
        )

        # ==========================================================
        # 6. Maximum allowed volume
        # ==========================================================

        df["VolumeMax"] = (
            df["VolumeAverage"]
            * self.vcoef
        )

        # ==========================================================
        # 7. Capped Volume
        # ==========================================================

        df["CappedVolume"] = np.where(
            df["Volume"] < df["VolumeMax"],
            df["Volume"],
            df["VolumeMax"]
        )

        # ==========================================================
        # 8. Money Flow
        # ==========================================================

        df["MoneyFlow"] = (
            df["TypicalPrice"]
            - df["TypicalPrice"].shift(1)
        )

        # ==========================================================
        # 9. Price Movement
        # ==========================================================

        df["PriceMovement"] = np.where(
            df["MoneyFlow"] > df["Cutoff"],
            1,
            np.where(
                df["MoneyFlow"] < -df["Cutoff"],
                -1,
                0
            )
        )

        # ==========================================================
        # 10. Volume Flow
        # ==========================================================

        df["VolumeFlow"] = (
            df["PriceMovement"]
            * df["CappedVolume"]
        )

        # ==========================================================
        # 11. Rolling Volume Flow
        # ==========================================================

        df["RollingVolumeFlow"] = (
            df["VolumeFlow"]
            .rolling(
                window=self.length,
                min_periods=self.length
            )
            .sum()
        )

        # ==========================================================
        # 12. VFI
        # ==========================================================

        df["VFI"] = (
            df["RollingVolumeFlow"]
            / df["VolumeAverage"]
        )

        # ==========================================================
        # 13. Optional VFI smoothing
        # ==========================================================

        if self.smooth_vfi:
            df["VFI"] = (
                df["VFI"]
                .rolling(
                    window=3,
                    min_periods=3
                )
                .mean()
            )

        # ==========================================================
        # 14. VFI Signal
        # ==========================================================

        # Keep the explicit 20/50 research lines stable, while also exposing
        # any custom configured spans without mislabeling their column names.
        signal_spans = {20, 50, int(self.signal_length), int(self.signal_length_50)}
        for span in sorted(signal_spans):
            df[f"VFI_Signal_{span}"] = df["VFI"].ewm(span=span, adjust=False).mean()

        # Backward compatibility: the unsuffixed signal follows signal_length.
        df["VFI_Signal"] = df[f"VFI_Signal_{int(self.signal_length)}"]

        # Fixed 20/50 histograms remain comparable across runs.
        df["VFI_Histogram_20"] = df["VFI"] - df["VFI_Signal_20"]
        df["VFI_Histogram_50"] = df["VFI"] - df["VFI_Signal_50"]
        df["VFI_Histogram"] = df["VFI"] - df["VFI_Signal"]

        # ==========================================================
        # 16. Weekly VFI slope
        # ==========================================================

        df["VFI_Slope"] = (
            df["VFI"]
            - df["VFI"].shift(1)
        )

        # ==========================================================
        # 17. Change in VFI slope
        # ==========================================================

        df["VFI_Slope_Change"] = (
            df["VFI_Slope"]
            - df["VFI_Slope"].shift(1)
        )

        # ==========================================================
        # 18. Distance from zero
        # ==========================================================

        df["VFI_Distance_Zero"] = (
            df["VFI"].abs()
        )

        # ==========================================================
        # 19. Bar-based average VFI slopes
        #
        # These are bar counts, not calendar weeks. On a daily input,
        # 4B/8B means 4/8 days; on a weekly input, 4/8 weeks.
        # Keep the legacy 4W/8W aliases for older consumers, but new
        # code should use the timeframe-neutral 4B/8B columns.
        # ==========================================================

        df["VFI_Slope_4B"] = (df["VFI"] - df["VFI"].shift(4)) / 4.0
        df["VFI_Slope_8B"] = (df["VFI"] - df["VFI"].shift(8)) / 8.0

        # Backward-compatible aliases. These names are historical and
        # should not be interpreted as weeks unless the input is weekly.
        df["VFI_Slope_4W"] = df["VFI_Slope_4B"]
        df["VFI_Slope_8W"] = df["VFI_Slope_8B"]

        # ==========================================================
        # 21. Rolling minimum
        #
        # Used to determine whether VFI is recovering from
        # a significant negative level.
        # ==========================================================

        df["VFI_Rolling_Min"] = (
            df["VFI"]
            .rolling(
                window=self.context_window,
                min_periods=self.context_window
            )
            .min()
        )

        # ==========================================================
        # 22. Rolling maximum
        #
        # Used to determine whether VFI is weakening from
        # a significant positive level.
        # ==========================================================

        df["VFI_Rolling_Max"] = (
            df["VFI"]
            .rolling(
                window=self.context_window,
                min_periods=self.context_window
            )
            .max()
        )

        return df