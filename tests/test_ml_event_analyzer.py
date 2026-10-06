import unittest

import numpy as np
import pandas as pd

from app.research.ml_event_analyzer import VFIEventMLAnalyzer


class TestVFIEventMLAnalyzer(unittest.TestCase):
    def make_events(self):
        dates = pd.date_range("2015-01-02", periods=120, freq="W-FRI")
        rng = np.random.default_rng(42)
        signal = rng.normal(size=len(dates))
        target = np.where(signal > 0, 5.0, -3.0)

        return pd.DataFrame(
            {
                "VFI": signal,
                "VFI_Slope": signal,
                "VFI_Slope_4W": signal,
                "VFI_Slope_8W": signal,
                "VFI_Distance_Zero": np.abs(signal),
                "VFI_Range_Position": 0.5 + signal * 0.05,
                "VFI_Recovery_From_Min_Pct": signal * 5.0,
                "VFI_Histogram": signal * 0.2,
                "Return_4W": target,
                "Return_26W": target,
            },
            index=dates,
        )

    def test_prepare_creates_binary_target(self):
        analyzer = VFIEventMLAnalyzer(primary_horizon=26)
        prepared = analyzer.prepare(self.make_events())

        self.assertIn("Target_Positive", prepared.columns)
        self.assertEqual(set(prepared["Target_Positive"].unique()), {0, 1})

    def test_walk_forward_produces_oos_predictions(self):
        analyzer = VFIEventMLAnalyzer(
            primary_horizon=26,
            minimum_train_events=20,
            n_estimators=50,
        )
        summary, predictions = analyzer.walk_forward(
            self.make_events(),
            initial_train_weeks=60,
            test_weeks=20,
            step_weeks=20,
        )

        self.assertFalse(summary.empty)
        self.assertFalse(predictions.empty)
        self.assertIn("ML_Probability", predictions.columns)
        self.assertIn("ML_Prediction", predictions.columns)

        window_end = summary.set_index("Window")["Test_End"]
        for _, row in predictions.iterrows():
            self.assertLess(
                pd.Timestamp(row.name) + pd.Timedelta(weeks=26),
                pd.Timestamp(window_end.loc[row["Window"]]),
            )


if __name__ == "__main__":
    unittest.main()
