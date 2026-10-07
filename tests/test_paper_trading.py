import tempfile
import unittest
from pathlib import Path

from app.paper_trading.engine import PaperAccount, PaperTradingEngine


class TestPaperTrading(unittest.TestCase):
    def test_long_target_and_persistence(self):
        account = PaperAccount(equity=10000, cash=10000)
        engine = PaperTradingEngine(account, fee_rate=0)
        engine.open_position("BTCUSDT", "long", 100, 1, 90, 110)
        self.assertEqual(engine.process_bar("BTCUSDT", 111, 99, 108), "TARGET")
        self.assertAlmostEqual(account.realized_pnl, 10)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "account.json"
            account.save(path)
            loaded = PaperAccount.load(path)
            self.assertAlmostEqual(loaded.realized_pnl, 10)
            self.assertEqual(loaded.positions, {})

    def test_short_stop(self):
        account = PaperAccount(equity=10000, cash=10000)
        engine = PaperTradingEngine(account, fee_rate=0)
        engine.open_position("ETHUSDT", "short", 100, 2, 110, 80)
        self.assertEqual(engine.process_bar("ETHUSDT", 111, 95, 105), "STOP")
        self.assertAlmostEqual(account.realized_pnl, -20)

    def test_duplicate_position_is_rejected(self):
        engine = PaperTradingEngine(fee_rate=0)
        engine.open_position("SOLUSDT", "long", 100, 1, 90, 110)
        with self.assertRaises(ValueError):
            engine.open_position("SOLUSDT", "long", 100, 1, 90, 110)


if __name__ == "__main__":
    unittest.main()
