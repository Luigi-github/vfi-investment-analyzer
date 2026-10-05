import unittest

from app.research.universes import MARKET_UNIVERSES, all_symbols, get_universe


class TestResearchUniverses(unittest.TestCase):
    def test_known_universes_are_non_empty(self):
        for name, symbols in MARKET_UNIVERSES.items():
            self.assertTrue(name)
            self.assertTrue(symbols)

    def test_get_universe_returns_copy(self):
        symbols = get_universe("USA_STOCKS")
        symbols.append("TEST")
        self.assertNotIn("TEST", MARKET_UNIVERSES["USA_STOCKS"])

    def test_all_symbols_are_unique(self):
        symbols = all_symbols()
        self.assertEqual(len(symbols), len(set(symbols)))

    def test_unknown_universe_fails(self):
        with self.assertRaises(ValueError):
            get_universe("UNKNOWN")


if __name__ == "__main__":
    unittest.main()
