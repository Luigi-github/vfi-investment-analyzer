import unittest

from app.backtest.risk_management import RiskConfig, position_fraction, trade_pnl_fraction


class TestRiskManagement(unittest.TestCase):
    def test_position_fraction_risks_one_percent(self):
        cfg = RiskConfig(risk_per_trade=0.01, max_exposure=1.0)
        self.assertAlmostEqual(position_fraction(10000, 100, 90, cfg), 0.1)

    def test_position_fraction_is_capped(self):
        cfg = RiskConfig(risk_per_trade=0.01, max_exposure=0.25)
        self.assertAlmostEqual(position_fraction(10000, 100, 99.9, cfg), 0.25)

    def test_invalid_inputs_return_zero(self):
        self.assertEqual(position_fraction(0, 100, 90), 0.0)
        self.assertEqual(position_fraction(10000, 0, 90), 0.0)

    def test_pnl_contribution(self):
        self.assertAlmostEqual(trade_pnl_fraction(0.10, 0.20), 0.02)


if __name__ == "__main__":
    unittest.main()
