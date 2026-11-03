import unittest

import pandas as pd

from scripts.run_us_weekly_then_daily_screen import trailing_true_streak as weekly_trailing_true_streak
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


if __name__ == "__main__":
    unittest.main()
