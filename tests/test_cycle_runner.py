import unittest

import pandas as pd

from scripts.run_cycle_analysis import (
    build_asset_event_summary,
    build_output_dir,
    resolve_symbols,
)


class TestCycleRunner(unittest.TestCase):
    def test_all_universe_is_unique(self):
        symbols = resolve_symbols("ALL")
        self.assertTrue(symbols)
        self.assertEqual(len(symbols), len(set(symbols)))

    def test_universe_selection(self):
        symbols = resolve_symbols("USA_STOCKS")
        self.assertIn("NVDA", symbols)
        self.assertIn("DDD", symbols)

    def test_custom_symbols_override_universe_and_deduplicate(self):
        symbols = resolve_symbols("CRYPTO", "BTC-USD, ETH-USD, BTC-USD")
        self.assertEqual(symbols, ["BTC-USD", "ETH-USD"])

    def test_output_directory_is_grouped_by_universe(self):
        output = build_output_dir("USA_STOCKS")
        self.assertEqual(output.name, "usa_stocks")\n        self.assertEqual(output.parent.name, "cycle_analysis")

    def test_asset_event_summary_is_sorted(self):
        events = pd.DataFrame(
            {
                "Symbol": ["A", "A", "B", "B"],
                "Return_26W": [10.0, 20.0, -5.0, 15.0],
            }
        )
        summary = build_asset_event_summary(events, 26)
        self.assertEqual(summary.iloc[0]["Symbol"], "A")
        self.assertEqual(summary.iloc[0]["Events"], 2)


if __name__ == "__main__":
    unittest.main()
