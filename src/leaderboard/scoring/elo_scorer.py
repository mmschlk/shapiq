from leaderboard.scoring import LeaderboardScorer


class EloScorer(LeaderboardScorer):
    """A scorer based on the ELO system using pairwise approximator comparison"""

    name = "elo"
    higher_is_better = True

    class EloScorer(LeaderboardScorer):
        """Scorer based on the Elo system using pairwise approximator comparisons."""

        name = "elo"
        higher_is_better = True

        def __init__(
                self,
                initial_elo: float = 1000.0,
                k_factor: float = 16.0,
                tie_tolerance: float = 0.0,
                group_keys: list[str] | None = None,
        ) -> None:
            """Create an Elo scorer.

            Args:
                initial_elo: Starting Elo value for each approximator.
                k_factor: Multiplication factor determining Elo gain and loss per match.
                tie_tolerance: Tolerance for treating small metric differences as ties.
                    Default means no tolerance.
                group_keys: Record keys used to form comparable benchmark groups.
            """
            self.initial_elo = initial_elo
            self.k_factor = k_factor
            self.tie_tolerance = tie_tolerance
            self.group_keys = group_keys or [
                "game_id",
                "game_name",
                "index",
                "max_order",
                "budget",
                "ground_truth_method",
            ]