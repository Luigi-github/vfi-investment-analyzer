import pandas as pd


class ResearchRankingAnalyzer:
    """Rank assets from out-of-sample research results without fitting a model."""

    def __init__(self, metric="Average_Test_Return", ascending=False):
        self.metric = metric
        self.ascending = ascending

    def rank(self, walk_forward_summary):
        if walk_forward_summary.empty:
            return walk_forward_summary.copy()

        required = {
            "Symbol",
            "Average_Test_Return",
            "Median_Test_Return",
            "Positive_Window_Rate",
            "Worst_Test_Return",
            "Best_Test_Return",
            "Valid_Windows",
        }
        missing = required - set(walk_forward_summary.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        result = walk_forward_summary.copy()
        result = result.dropna(subset=[self.metric]).copy()

        if result.empty:
            result["Research_Rank"] = pd.Series(dtype="int64")
            return result

        result = result.sort_values(
            [self.metric, "Valid_Windows", "Median_Test_Return"],
            ascending=[self.ascending, False, self.ascending],
            kind="mergesort",
        ).reset_index(drop=True)

        result.insert(
            0,
            "Research_Rank",
            range(1, len(result) + 1),
        )
        return result
