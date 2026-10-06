import unittest

from app.research.parameter_sensitivity import VFIParameterSensitivityAnalyzer


class TestVFIParameterSensitivity(unittest.TestCase):
    def test_expand_grid_is_deterministic(self):
        grid = {
            "length": [100, 130],
            "coef": [0.2],
            "vcoef": [2.5, 3.0],
        }

        configurations = VFIParameterSensitivityAnalyzer.expand_grid(grid)

        self.assertEqual(len(configurations), 4)
        self.assertEqual(configurations[0]["length"], 100)
        self.assertEqual(configurations[-1]["vcoef"], 3.0)

    def test_unknown_parameter_is_rejected(self):
        with self.assertRaises(ValueError):
            VFIParameterSensitivityAnalyzer.expand_grid(
                {"length": [130], "future_parameter": [1]}
            )


if __name__ == "__main__":
    unittest.main()
