import unittest

import pandas as pd

from app.scoring.research_ranking import ResearchRankingAnalyzer


class TestResearchRanking(unittest.TestCase):

    def test_ranks_assets_by_out_of_sample_metric(self):
        data = pd.DataFrame(
            {
                "Symbol": ["AAA", "BBB"],
                "Average_Test_Return": [4.0, 7.0],
                "Median_Test_Return": [3.0, 5.0],
                "Positive_Window_Rate": [60.0, 80.0],
                "Worst_Test_Return": [-10.0, -8.0],
                "Best_Test_Return": [12.0, 15.0],
                "Valid_Windows": [5, 5],
            }
        )

        result = ResearchRankingAnalyzer().rank(data)

        self.assertEqual(result.iloc[0]["Symbol"], "BBB")
        self.assertEqual(result.iloc[0]["Research_Rank"], 1)
        self.assertEqual(result.iloc[1]["Research_Rank"], 2)

    def test_empty_input_is_preserved(self):
        data = pd.DataFrame()
        result = ResearchRankingAnalyzer().rank(data)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
