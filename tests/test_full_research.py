import unittest

from app.research.universes import MARKET_UNIVERSES
from scripts.run_full_research import MARKET_ORDER


class TestFullResearchOrder(unittest.TestCase):
    def test_all_configured_markets_are_orchestrated(self):
        self.assertEqual(set(MARKET_ORDER), set(MARKET_UNIVERSES))
        self.assertEqual(len(MARKET_ORDER), len(MARKET_UNIVERSES))


if __name__ == "__main__":
    unittest.main()
