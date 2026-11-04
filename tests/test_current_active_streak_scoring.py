import unittest

import pandas as pd

from scripts.run_us_weekly_then_daily_screen import (
    ema_rising_streak_score,
    trailing_true_streak as weekly_trailing_true_streak,
)
from scripts.build_us_three_rankings import trailing_true_streak as monthly_trailing_true_streak


class TestCurrentActiveStreakScoring(unittest.TestCase):
    def test_weekly_streak_does_not_sum_disconnected_rising_periods(self):
        flags = pd.Series([True, True, False, True, False, True, True, True])
        self.assertEqual(weekly_trailing_true_streak(flags), 3)
        self.assertNotEqual(weekly_trailing_true_streak(flags), int(flags.sum()))

    def test_monthly_streak_does_not_sum_disconnected_rising_periods(self):
        flags = pd.Series([True, False, True, True, False, True, True])
        self.assertEqual(monthly_trailing_true_streak(flags), 2)
        self.assertNotEqual(monthly_trailing_true_streak(flags), int(flags.sum()))

    def test_streak_is_zero_when_latest_bar_breaks_the_regime(self):
        flags = pd.Series([True, True, True, False])
        self.assertEqual(weekly_trailing_true_streak(flags), 0)
        self.assertEqual(monthly_trailing_true_streak(flags), 0)

    def test_empty_series_has_no_active_streak(self):
        flags = pd.Series(dtype=bool)
        self.assertEqual(weekly_trailing_true_streak(flags), 0)
        self.assertEqual(monthly_trailing_true_streak(flags), 0)

    def test_ema_only_score_rewards_longer_ema_streaks_more(self):
        ema_series = {
            "ema5": pd.Series([1, 2, 3, 4, 5]),
            "ema20": pd.Series([1, 2, 3, 4, 5]),
            "ema80": pd.Series([1, 2, 3, 4, 5]),
            "ema200": pd.Series([1, 2, 3, 4, 5]),
        }
        score, streaks, contributions = ema_rising_streak_score(ema_series)
        self.assertEqual(set(streaks.values()), {4})
        self.assertAlmostEqual(contributions["ema200"], 100 * 0.50 * 4 / 52)
        self.assertGreater(contributions["ema200"], contributions["ema80"])
        self.assertGreater(contributions["ema80"], contributions["ema20"])
        self.assertGreater(contributions["ema20"], contributions["ema5"])
        self.assertAlmostEqual(score, sum(contributions.values()))

    def test_ema_score_uses_only_latest_uninterrupted_rising_run(self):
        ema_series = {
            "ema5": pd.Series([1, 2, 3, 2, 3, 4]),
            "ema20": pd.Series([1, 2, 3, 2, 3, 4]),
            "ema80": pd.Series([1, 2, 3, 2, 3, 4]),
            "ema200": pd.Series([1, 2, 3, 2, 3, 4]),
        }
        _, streaks, _ = ema_rising_streak_score(ema_series)
        self.assertEqual(set(streaks.values()), {2})

    def test_ema_score_caps_duration_at_52_weeks(self):
        ema_series = {
            name: pd.Series(range(61))
            for name in ("ema5", "ema20", "ema80", "ema200")
        }
        score, streaks, contributions = ema_rising_streak_score(ema_series)
        self.assertEqual(set(streaks.values()), {60})
        self.assertAlmostEqual(score, 100.0)
        self.assertAlmostEqual(contributions["ema200"], 50.0)


if __name__ == "__main__":
    unittest.main()
