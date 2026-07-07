"""Critical Difference scorer for leaderboard approximator comparison.

Implements the Demsar (2006) critical difference analysis: approximators are ranked per comparable benchmark group and metric, Friedman's test checks whether any ranking differences are significant, and Nemenyi's post-hoc CD determines which pairs of approximators differ significantly.

The result exposes per-approximator mean ranks plus a CD value that can be used to render a CD diagram (horizontal axis = mean rank, groups of approximators whose rank ranges overlap within CD are connected by a bar).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

import plotly.graph_objects as go

from leaderboard.metrics.registry import METRIC_ALIASES, METRIC_SPECS
from leaderboard.scoring import LeaderboardScorer, ScoringResult
from leaderboard.scoring.result import LeaderboardRow, ScoringContext
from leaderboard.scoring.scorer_utils import (
    aggregate_seeds_in_group,
    build_context,
    filter_valid_records,
    get_metric_value,
    group_records,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Nemenyi critical value
#
# Demsar (2006), Table 5, tabulates q_alpha for k = 2..10 classifiers only.
# q_alpha is in fact the (1 - alpha) quantile of the studentized range
# distribution for k means with infinite degrees of freedom, divided by sqrt(2):
#
#   q_alpha(k) = Q_{1-alpha}(k, df=inf) / sqrt(2)
#
# We compute this directly from the studentized range distribution, so it
# works for any number of approximators, not just k in [2, 10].
# ---------------------------------------------------------------------------


def _nemenyi_q_alpha(alpha: float, k: int) -> float:
    """Compute the Nemenyi critical value q_alpha for *k* classifiers.

    q_alpha = Q_{1-alpha}(k, df=inf) / sqrt(2), where Q is the quantile function of the studentized range distribution. Ranks entering the Nemenyi test are treated as asymptotically normal, hence df=inf.

    Args:
        alpha: Significance level in (0, 1).
        k: Number of classifiers/approximators being compared.

    Returns:
        The q_alpha critical value. Returns 0.0 if *k* < 2.
    """
    if k < 2:
        return 0.0
    return float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf) / math.sqrt(2.0))


@dataclass(frozen=True)
class RankedGroup:
    """Ranks assigned to approximators in one comparable benchmark group / metric.

    Attributes:
        group_key: Context dictionary describing the benchmark group.
        metric_name: Metric used for ranking.
        ranks: Mapping from approximator name to its rank in this group.
            Tied approximators share the average of the tied positions.
    """

    group_key: dict[str, object]
    metric_name: str
    ranks: dict[str, float]


@dataclass
class CriticalDifferenceResult:
    """Aggregated CD analysis result.

    Attributes:
        mean_ranks: Mean rank per approximator across all groups/metrics.
            Lower rank is better (rank 1 = best in group).
        critical_difference: The CD value at the configured alpha level.
            Approximator pairs whose mean ranks differ by more than this value are considered significantly different.
        friedman_statistic: Friedman chi-squared statistic.
        friedman_p_value: p-value of the Friedman test.
        friedman_significant: Whether the Friedman test rejected H0.
        n_groups: Number of (group, metric) combinations used.
        n_approximators: Number of approximators compared.
        alpha: Significance level used.
        cliques: Groups of approximators not significantly different from each
            other (i.e. whose mean rank difference ≤ CD).
    """

    mean_ranks: dict[str, float]
    critical_difference: float
    friedman_statistic: float
    friedman_p_value: float
    friedman_significant: bool
    n_groups: int
    n_approximators: int
    alpha: float
    cliques: list[list[str]] = field(default_factory=list)


class CriticalDifferenceScorer(LeaderboardScorer):
    """Scorer based on Demsar's critical difference analysis.

    Approximators are compared within each comparable benchmark group and metric. The Critical Difference (CD) analysis consists of two steps:
        1. The Friedman test checks whether there are any significant differences in the rankings of the approximators across the groups.
        2. If the Friedman test is significant, Nemenyi's post-hoc test is used to determine which pairs of approximators are significantly different.

    The ``score`` method returns a :class:`ScoringResult` where each ``LeaderboardRow.score`` is the mean rank. This corresponds to a ``higher_is_better=False``.

    The full CD analysis is accessible via ``result.metadata["cd_result"]``.

    Reference:
        Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.
    """

    name = "critical_difference"
    higher_is_better = False  # mean rank: lower is better

    def __init__(
        self,
        alpha: float = 0.05,
        group_keys: list[str] | None = None,
        metric_names: list[str] | None = None,
        game_names: list[str] | None = None,
        indices: list[str] | None = None,
        budgets: list[int] | None = None,
    ) -> None:
        """Create a Critical Difference scorer.

        Args:
            alpha: Significance level for the Nemenyi post-hoc test and the Friedman omnibus test, e.g. ``0.10`` or ``0.05`` (default).
                Any value strictly between 0 and 1 is accepted; the corresponding q_alpha critical value is derived from the studentized range distribution (see :func:`_nemenyi_q_alpha`).
            group_keys: Record keys used to form comparable benchmark groups.
                Defaults to the same keys used by :class:`EloScorer`.
            metric_names: Metric names to include. ``None`` means all metrics.
            game_names: Game names to include. ``None`` means all games.
            indices: Interaction indices to include. ``None`` means all.
            budgets: Budgets to include. ``None`` means all.

        Raises:
            ValueError: If *alpha* is not strictly between 0 and 1.
        """
        if not (0.0 < alpha < 1.0):
            msg = f"alpha must be in the open interval (0, 1), got {alpha}."
            raise ValueError(msg)

        self.alpha = alpha
        self.group_keys = group_keys or [
            "game_name",
            "index",
            "max_order",
            "budget",
            "ground_truth_method",
        ]
        self.metric_names = self._normalize_metric_names(metric_names)
        self.game_names = game_names
        self.indices = indices
        self.budgets = budgets

    def score(self, records: list[dict[str, object]]) -> ScoringResult:
        """Compute the CD analysis from benchmark records.

        Args:
            records: Raw benchmark records from the database.

        Returns:
            A :class:`ScoringResult` where each row's ``score`` is the
            *negated* mean rank.  The full :class:`CriticalDifferenceResult`
            is stored in ``metadata["cd_result"]``.
        """
        valid_records = filter_valid_records(records)
        selected_records = self._filter_records_by_context(valid_records)

        groups = group_records(selected_records, self.group_keys)
        ranked_groups = self._build_ranked_groups(groups)

        cd_result = self._compute_cd(ranked_groups)
        leaderboard_rows = self._build_leaderboard_rows(cd_result)
        used_metric_names = self._get_used_metric_names(ranked_groups)

        metadata: dict[str, object] = {
            "n_input_records": len(records),
            "n_valid_records": len(valid_records),
            "n_selected_records": len(selected_records),
            "n_groups": cd_result.n_groups,
            "n_approximators": cd_result.n_approximators,
            "alpha": self.alpha,
            "critical_difference": cd_result.critical_difference,
            "friedman_statistic": cd_result.friedman_statistic,
            "friedman_p_value": cd_result.friedman_p_value,
            "friedman_significant": cd_result.friedman_significant,
            "cd_result": cd_result,
        }

        return ScoringResult(
            scorer_name=self.name,
            context=self._build_context(selected_records, used_metric_names),
            rows=leaderboard_rows,
            group_results=[],
            metadata=metadata,
        )

    def _build_ranked_groups(
        self,
        groups: dict[tuple[object, ...], list[dict[str, object]]],
    ) -> list[RankedGroup]:
        """Rank approximators within each (group, metric) combination.

        Only groups where at least two approximators have valid metric values are included.  Ties receive the average of tied positions.

        Args:
            groups: Grouped benchmark records from :func:`group_records`.

        Returns:
            One :class:`RankedGroup` per (group-key, metric) pair.
        """
        ranked_groups: list[RankedGroup] = []

        for group_key_tuple, records_in_group in groups.items():
            group_key = dict(zip(self.group_keys, group_key_tuple, strict=True))
            aggregated = aggregate_seeds_in_group(records_in_group, self.group_keys)

            for metric_name in self.metric_names:
                metric_spec = METRIC_SPECS[metric_name]
                records_with_metric = [
                    r for r in aggregated if get_metric_value(r, metric_name) is not None
                ]
                if len(records_with_metric) < 2:
                    continue

                ranks = self._rank_records(
                    records_with_metric,
                    metric_name=metric_name,
                    higher_is_better=metric_spec.higher_is_better,
                )
                ranked_groups.append(
                    RankedGroup(
                        group_key=group_key,
                        metric_name=metric_name,
                        ranks=ranks,
                    )
                )

        return ranked_groups

    def _rank_records(
        self,
        records: list[dict[str, object]],
        *,
        metric_name: str,
        higher_is_better: bool,
    ) -> dict[str, float]:
        """Assign fractional ranks to approximators for one metric.

        Rank 1 is best.  If ``higher_is_better``, the approximator with the largest metric value receives rank 1.  Ties are resolved by averaging the tied positions.

        Args:
            records: Aggregated records for one comparable group.
            metric_name: Metric column to rank by.
            higher_is_better: Whether larger values are better.

        Returns:
            Mapping from approximator name to fractional rank.
        """
        scored: list[tuple[str, float]] = []
        for record in records:
            approx = record.get("approximator_name")
            value = get_metric_value(record, metric_name)
            if isinstance(approx, str) and value is not None:
                scored.append((approx, float(value)))

        # Sort so that the best performer comes first.
        scored.sort(key=lambda kv: kv[1], reverse=higher_is_better)

        ranks: dict[str, float] = {}
        i = 0
        while i < len(scored):
            j = i
            # Find the end of a run of tied values.
            while j + 1 < len(scored) and scored[j + 1][1] == scored[i][1]:
                j += 1
            avg_rank = mean(range(i + 1, j + 2))
            for k in range(i, j + 1):
                ranks[scored[k][0]] = avg_rank
            i = j + 1

        return ranks

    def _compute_cd(self, ranked_groups: list[RankedGroup]) -> CriticalDifferenceResult:
        """Run the full Demsar CD analysis on the ranked groups.

        Steps:
        1. Collect all approximators and compute mean ranks.
        2. Run the Friedman test.
        3. Compute the Nemenyi CD.
        4. Detect non-significantly-different cliques.

        Args:
            ranked_groups: All ranked (group, metric) combinations.

        Returns:
            A fully populated :class:`CriticalDifferenceResult`.
        """
        if not ranked_groups:
            return CriticalDifferenceResult(
                mean_ranks={},
                critical_difference=0.0,
                friedman_statistic=0.0,
                friedman_p_value=1.0,
                friedman_significant=False,
                n_groups=0,
                n_approximators=0,
                alpha=self.alpha,
                cliques=[],
            )

        # Collect per-approximator rank lists across all groups.
        all_approxs: set[str] = set()
        for rg in ranked_groups:
            all_approxs.update(rg.ranks)

        approx_list = sorted(all_approxs)
        n_approx = len(approx_list)
        n_groups = len(ranked_groups)

        # Build a complete rank matrix (n_groups x n_approx)
        rank_matrix = self._build_rank_matrix(ranked_groups, approx_list)

        mean_ranks: dict[str, float] = {
            approx: float(mean(rank_matrix[:, j])) for j, approx in enumerate(approx_list)
        }

        # Friedman test
        friedman_stat, friedman_p = self._friedman_test(rank_matrix)
        friedman_significant = friedman_p < self.alpha

        # Nemenyi CD
        cd = self._nemenyi_cd(n_approx=n_approx, n_groups=rank_matrix.shape[0])

        # Cliques: groups of approximators not significantly different
        cliques = self._find_cliques(approx_list, mean_ranks, cd)

        return CriticalDifferenceResult(
            mean_ranks=mean_ranks,
            critical_difference=cd,
            friedman_statistic=friedman_stat,
            friedman_p_value=friedman_p,
            friedman_significant=friedman_significant,
            n_groups=n_groups,
            n_approximators=n_approx,
            alpha=self.alpha,
            cliques=cliques,
        )

    def _build_rank_matrix(
        self,
        ranked_groups: list[RankedGroup],
        approx_list: list[str],
    ) -> np.ndarray:
        """Build an (n_complete_groups x n_approx) rank matrix.

        Groups where not all approximators in *approx_list* participated are excluded so the Friedman statistic is computed on a balanced design. Missing approximators within an included group are filled with the worst possible rank (n_approx) as a conservative fallback.

        Args:
            ranked_groups: All ranked groups.
            approx_list: Canonical sorted approximator list.

        Returns:
            Float rank matrix of shape ``(n_rows, len(approx_list))``.
            May have 0 rows if no group has any data.
        """
        n_approx = len(approx_list)
        approx_idx = {a: i for i, a in enumerate(approx_list)}

        rows: list[list[float]] = []
        for rg in ranked_groups:
            row = [float(n_approx)] * n_approx  # worst-rank default
            for approx, rank in rg.ranks.items():
                if approx in approx_idx:
                    row[approx_idx[approx]] = rank
            rows.append(row)

        return np.array(rows, dtype=float) if rows else np.empty((0, n_approx), dtype=float)

    @staticmethod
    def _friedman_test(rank_matrix: np.ndarray) -> tuple[float, float]:
        """Run Friedman's chi-squared test on a rank matrix.

        Uses SciPy's ``friedmanchisquare`` when >= 3 approximators are present.  For exactly 2 approximators falls back to a sign test.

        Args:
            rank_matrix: (n_groups x n_approx) rank matrix.

        Returns:
            Tuple of (statistic, p_value).
        """
        if rank_matrix.shape[0] < 2 or rank_matrix.shape[1] < 2:
            return 0.0, 1.0

        columns = [rank_matrix[:, j] for j in range(rank_matrix.shape[1])]

        if rank_matrix.shape[1] == 2:
            # Two-approximator case: Wilcoxon signed-rank test
            diff = columns[0] - columns[1]
            if np.all(diff == 0):
                return 0.0, 1.0
            stat, p = stats.wilcoxon(diff, alternative="two-sided")
            return float(stat), float(p)

        stat, p = stats.friedmanchisquare(*columns)
        return float(stat), float(p)

    def _nemenyi_cd(self, *, n_approx: int, n_groups: int) -> float:
        """Compute the Nemenyi critical difference.

        CD = q_alpha * sqrt(n_approx * (n_approx + 1) / (6 * n_groups))

        q_alpha is computed from the studentized range distribution for ``n_approx`` classifiers (see :func:`_nemenyi_q_alpha`), so this works for any number of approximators, not just the k in [2, 10] tabulated in Demsar (2006) Table 5.

        Args:
            n_approx: Number of approximators (k).
            n_groups: Number of comparable groups used (N).

        Returns:
            Critical difference value.
        """
        if n_approx < 2 or n_groups < 1:
            return 0.0
        q_alpha = _nemenyi_q_alpha(self.alpha, n_approx)
        return q_alpha * math.sqrt(n_approx * (n_approx + 1) / (6.0 * n_groups))

    @staticmethod
    def _find_cliques(
        approx_list: list[str],
        mean_ranks: dict[str, float],
        cd: float,
    ) -> list[list[str]]:
        """Find maximal cliques of approximators not significantly different.

        A clique is a maximal set of approximators whose pairwise mean rank differences are all ≤ *cd*.  The algorithm sorts approximators by mean rank and greedily extends each clique to include the next approximator as long as the difference from the clique's *first* member does not exceed *cd*.  Only cliques of size >= 2 are returned.

        Args:
            approx_list: All approximator names, in any order.
            mean_ranks: Mean rank per approximator.
            cd: Nemenyi critical difference.

        Returns:
            List of cliques (each clique is a list of approximator names
            sorted by mean rank).
        """
        sorted_approxs = sorted(approx_list, key=lambda a: mean_ranks.get(a, float("inf")))
        cliques: list[list[str]] = []

        for i, anchor in enumerate(sorted_approxs):
            anchor_rank = mean_ranks.get(anchor, float("inf"))

            clique = [anchor] + [
                other
                for other in sorted_approxs[i + 1 :]
                if mean_ranks.get(other, float("inf")) - anchor_rank <= cd
            ]

            if len(clique) >= 2:
                cliques.append(clique)

        # Remove cliques that are strict subsets of another clique.
        maximal: list[list[str]] = []
        for clique in cliques:
            clique_set = set(clique)
            if not any(clique_set < set(other) for other in cliques):
                maximal.append(clique)

        # Deduplicate.
        seen: set[frozenset[str]] = set()
        unique: list[list[str]] = []
        for clique in maximal:
            key = frozenset(clique)
            if key not in seen:
                seen.add(key)
                unique.append(clique)

        return unique

    def _build_leaderboard_rows(
        self,
        cd_result: CriticalDifferenceResult,
    ) -> list[LeaderboardRow]:
        """Convert CD results into ranked leaderboard rows.

        Rows are sorted by ascending mean rank (best first).  The ``score`` field stores the *negated* mean rank so that ``higher_is_better=True`` behaves consistently with the rest of the leaderboard infrastructure.

        Args:
            cd_result: Fully computed CD result.

        Returns:
            Ranked leaderboard rows.
        """
        sorted_approxs = sorted(
            cd_result.mean_ranks.items(),
            key=lambda kv: kv[1],
        )

        rows: list[LeaderboardRow] = []
        for rank, (approx, mean_rank) in enumerate(sorted_approxs, start=1):
            # Find which clique(s) this approximator belongs to.
            clique_ids = [i for i, clique in enumerate(cd_result.cliques) if approx in clique]
            rows.append(
                LeaderboardRow(
                    approximator_name=approx,
                    score=-mean_rank,  # negate so higher = better
                    higher_is_better=False,
                    rank=rank,
                    metadata={
                        "mean_rank": mean_rank,
                        "critical_difference": cd_result.critical_difference,
                        "friedman_p_value": cd_result.friedman_p_value,
                        "friedman_significant": cd_result.friedman_significant,
                        "clique_ids": clique_ids,
                    },
                )
            )

        return rows

    def plot(
        self,
        result: ScoringResult,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Render the CD diagram for a :class:`ScoringResult` from :meth:`score`.

        Convenience wrapper around :meth:`plot_cd_diagram` that pulls the :class:`CriticalDifferenceResult` out of ``result.metadata``.

        Args:
            result: The result returned by :meth:`score`.
            ax: Optional Matplotlib axes to draw on. A new, auto-sized
                figure/axes pair is created if not provided.
            title: Optional title for the plot.
            figsize: Optional minimum figure size (width, height in inches),
                only used when *ax* is not provided. The figure grows beyond
                this if needed to fit long labels.

        Returns:
            The Matplotlib :class:`~matplotlib.figure.Figure` containing the diagram.

        Raises:
            KeyError: If ``result.metadata`` does not contain a ``"cd_result"`` entry (i.e. *result* was not produced by this scorer's :meth:`score`).
        """
        cd_result = result.metadata["cd_result"]
        return self.plot_cd_diagram(cd_result, ax=ax, title=title, figsize=figsize)

    @staticmethod
    def _text_width_inches(strings: list[str], fontsize: float) -> float:
        """Measure the widest rendered text width, in inches, for *strings*.

        We measure the rendered width of each string in a scratch Matplotlib figure and return the maximum. This is used to size the CD diagram figure so that labels are never clipped.

        Args:
            strings: List of strings to measure.
            fontsize: Font size in points.

        Returns:
            Maximum width of the rendered strings, in inches. Returns 0.0 if *strings* is empty.
        """
        import matplotlib.pyplot as plt

        if not strings:
            return 0.0
        scratch_fig = plt.figure()
        try:
            widths_in = []
            for s in strings:
                text = scratch_fig.text(0, 0, s, fontsize=fontsize)
                scratch_fig.canvas.draw()
                renderer = scratch_fig.canvas.get_renderer()
                bbox = text.get_window_extent(renderer=renderer)
                widths_in.append(bbox.width / scratch_fig.dpi)
                text.remove()
            return max(widths_in)
        finally:
            plt.close(scratch_fig)

    def plot_cd_diagram(
        self,
        cd_result: CriticalDifferenceResult,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Draw a critdd/Demsar-style critical difference diagram.

        Follows the layout popularized by Demsar (2006) and used by e.g. the ``critdd`` package (https://mirkobunse.github.io/critdd/): approximators are placed along a horizontal rank axis at their mean rank (rank 1 = best, on the left). Each approximator is connected by an elbow-shaped line running from its position on the axis up to a dedicated row, then out sideways to a label sitting in the left or right margin (the better half of approximators routes left, the worse half routes
        right).

        Rows are assigned so the most extreme ranks (best and worst) sit in the outermost row and ranks near the median sit closest to the axis. A CD-length reference bar is drawn above everything, and approximators whose mean ranks are not significantly different (i.e. members of the same clique, see :meth:`_find_cliques`) are connected by thick horizontal bars.

        When *ax* is not provided, the figure is sized automatically from the rendered width of the longest label on each side (minimal/no label clipping). The *figsize* argument is treated as a lower bound on the figure size; if the labels are too long, the figure will grow beyond *figsize* to accommodate them.  When *ax* is provided, the existing axes size is respected and the data limits are fit to fill it instead.

        Args:
            cd_result: A fully computed :class:`CriticalDifferenceResult`,
                e.g. from ``result.metadata["cd_result"]``.
            ax: Optional Matplotlib axes to draw on. A new, auto-sized
                figure/axes pair is created if not provided.
            title: Optional title for the plot.
            figsize: Optional minimum figure size (width, height in inches),
                only used when *ax* is not provided. The figure grows beyond
                this if needed to fit long labels.

        Returns:
            The Matplotlib :class:`~matplotlib.figure.Figure` containing the
            diagram.

        Raises:
            ValueError: If *cd_result* has fewer than 2 approximators.
        """
        import matplotlib.pyplot as plt

        mean_ranks = cd_result.mean_ranks
        if len(mean_ranks) < 2:
            msg = "Need at least 2 approximators to plot a CD diagram."
            raise ValueError(msg)

        approx_list = sorted(mean_ranks, key=mean_ranks.get)
        n = len(approx_list)
        lo, hi = 1.0, float(n)
        fontsize = 9.0

        # Pair up ranks from the outside in: best-with-worst in the outermost row
        # Better half -> left margin, worse half -> right margin
        half = (n + 1) // 2
        left_names = approx_list[:half]  # best ranks, best-first
        right_names = list(reversed(approx_list[half:]))  # worst ranks, worst-first

        row_spacing = 1.0
        left_rows = {name: row_spacing * (i + 1) for i, name in enumerate(left_names)}
        right_rows = {name: row_spacing * (i + 1) for i, name in enumerate(right_names)}
        max_rows = max(len(left_names), len(right_names))
        n_cliques = len(cd_result.cliques)

        def _label(name: str) -> str:
            return f"{name} ({mean_ranks[name]:.2f})"

        # Absolute inches needed above/below the rows, independent of number of data units
        tick_area_in = 0.22
        cd_gap_in = 0.30
        cd_label_in = 0.22
        bottom_pad_in = 0.12
        vertical_fixed_in = tick_area_in + cd_gap_in + cd_label_in + bottom_pad_in

        core_pad = 0.5  # data units of breathing room around the rank ticks
        gap_in = 0.12  # inches between an elbow's bend and its label
        core_x0, core_x1 = lo - core_pad, hi + core_pad
        d0 = core_x1 - core_x0  # "core" data span the rank axis itself needs

        created_own_figure = ax is None

        if created_own_figure:
            # Absolute inches needed for the longest label on each side
            t_left_in = self._text_width_inches([_label(n_) for n_ in left_names], fontsize)
            t_right_in = self._text_width_inches([_label(n_) for n_ in right_names], fontsize)

            core_in = max(3.0, 0.4 * (n - 1) + 1.5)
            width_frac = 0.96  # fraction of the figure width given to the axes
            fig_width_in = (core_in + t_left_in + t_right_in + 2 * gap_in) / width_frac

            # Fixed inches-per-row (guarantees the CD bar/tick/clique text always gets its full absolute size)
            row_height_in = 0.30
            height_frac = 0.90 if title else 0.97
            axes_height_in = max_rows * row_height_in + vertical_fixed_in
            fig_height_in = axes_height_in / height_frac

            if figsize is not None:
                fig_width_in = max(fig_width_in, figsize[0])
                fig_height_in = max(fig_height_in, figsize[1])

                # Only grow whichever dimension was requested
                if fig_height_in > axes_height_in / height_frac:
                    row_height_in = (fig_height_in * height_frac - vertical_fixed_in) / max(
                        max_rows, 1
                    )

            fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
            left_margin = (1 - width_frac) / 2
            fig.subplots_adjust(
                left=left_margin,
                right=1 - left_margin,
                top=height_frac if title else 0.99,
                bottom=0.02,
            )

            # Data per inch scale gives the padding
            # `core_in + t_left_in + t_right_in + 2*gap_in`
            data_per_inch = d0 / core_in
            pad_left_data = t_left_in * data_per_inch
            pad_right_data = t_right_in * data_per_inch
            gap_data = gap_in * data_per_inch
        else:
            fig = ax.figure

            # Solve for data padding
            renderer = fig.canvas.get_renderer()
            axes_bbox = ax.get_window_extent(renderer=renderer)
            axes_width_px = axes_bbox.width
            dpi = fig.dpi

            def _max_px(names: list[str]) -> float:
                if not names:
                    return 0.0
                scratch = ax.text(0, 0, "", fontsize=fontsize)
                try:
                    widths = []
                    for name_ in names:
                        scratch.set_text(_label(name_))
                        fig.canvas.draw()
                        widths.append(scratch.get_window_extent(renderer=renderer).width)
                    return max(widths)
                finally:
                    scratch.remove()

            gap_px = gap_in * dpi
            left_px = _max_px(left_names) + gap_px
            right_px = _max_px(right_names) + gap_px
            k_left = left_px / axes_width_px
            k_right = right_px / axes_width_px
            k_total = min(k_left + k_right, 0.85)  # guard against runaway growth
            if k_left + k_right > 0:
                scale = k_total / (k_left + k_right)
                k_left, k_right = k_left * scale, k_right * scale

            d_final = d0 / max(1 - k_total, 0.15)
            data_per_inch = d_final / (axes_width_px / dpi)
            pad_left_data = k_left * d_final - (gap_px / dpi) * data_per_inch
            pad_right_data = k_right * d_final - (gap_px / dpi) * data_per_inch
            pad_left_data = max(pad_left_data, 0.0)
            pad_right_data = max(pad_right_data, 0.0)
            gap_data = (gap_px / dpi) * data_per_inch

            # Given axes height is fixed; solve the row height (inches) that exactly fills it
            axes_height_in = axes_bbox.height / dpi
            row_height_in = max((axes_height_in - vertical_fixed_in) / max(max_rows, 1), 0.05)

        # Convert the absolute-inch vertical layout into data units
        v_scale = 1.0 / row_height_in  # data units per inch
        max_row_y = row_spacing * max_rows
        cd_y = max_row_y + cd_gap_in * v_scale
        y_top = cd_y + cd_label_in * v_scale
        tick_gap_data = tick_area_in * v_scale
        y_bottom = -(tick_area_in + bottom_pad_in) * v_scale

        # Crossbars for cliques (approximators not significantly different)
        crossbar_zone = row_spacing * 0.85
        crossbar_spacing = crossbar_zone / (n_cliques + 0.5) if n_cliques else 0.0
        crossbar_base = crossbar_spacing * 0.5

        left_elbow_x = core_x0
        right_elbow_x = core_x1
        left_text_x = left_elbow_x - gap_data
        right_text_x = right_elbow_x + gap_data
        xlim_left = left_elbow_x - gap_data - pad_left_data
        xlim_right = right_elbow_x + gap_data + pad_right_data

        # Rank axis with integer tick marks
        axis_y = 0.0
        ax.hlines(axis_y, lo, hi, color="black", linewidth=1.2, zorder=1)
        for tick in range(1, n + 1):
            ax.vlines(tick, axis_y - 0.02, axis_y + 0.02, color="black", linewidth=1.0)
            ax.text(
                tick,
                -tick_gap_data * 0.35,
                str(tick),
                ha="center",
                va="top",
                fontsize=fontsize,
            )

        def _draw_elbow(name: str, row_y: float, text_x: float, side: int) -> None:
            rank = mean_ranks[name]
            elbow_x = left_elbow_x if side > 0 else right_elbow_x
            ax.plot(rank, axis_y, "o", color="black", markersize=3, zorder=2)
            ax.plot(
                [rank, rank, elbow_x],
                [axis_y, row_y, row_y],
                color="black",
                linewidth=0.8,
                zorder=1,
                clip_on=False,
            )
            ax.text(
                text_x,
                row_y,
                _label(name),
                ha="right" if side > 0 else "left",
                va="center",
                fontsize=fontsize,
                clip_on=False,
            )

        for name in left_names:
            _draw_elbow(name, left_rows[name], left_text_x, side=1)
        for name in right_names:
            _draw_elbow(name, right_rows[name], right_text_x, side=-1)

        # CD reference bar, drawn above the fan-out

        cd = cd_result.critical_difference

        ax.plot([lo, lo + cd], [cd_y, cd_y], color="black", linewidth=1.5)
        ax.plot([lo, lo], [cd_y - 0.05, cd_y + 0.05], color="black", linewidth=1.5)
        ax.plot([lo + cd, lo + cd], [cd_y - 0.05, cd_y + 0.05], color="black", linewidth=1.5)
        ax.text(lo, cd_y + 0.12, f"CD = {cd:.3f}", ha="left", va="bottom", fontsize=fontsize)

        # Cliques: thick crossbars linking approximators not significantly different
        clique_y = crossbar_base
        for clique in sorted(
            cd_result.cliques,
            key=lambda c: max(mean_ranks[a] for a in c) - min(mean_ranks[a] for a in c),
        ):
            ranks = [mean_ranks[a] for a in clique]
            ax.plot(
                [min(ranks), max(ranks)],
                [clique_y, clique_y],
                color="black",
                linewidth=3.0,
                solid_capstyle="butt",
                zorder=3,
            )
            clique_y += crossbar_spacing

        ax.set_xlim(xlim_left, xlim_right)
        ax.set_ylim(y_bottom, y_top)
        ax.axis("off")
        if title:
            ax.set_title(title)

        return fig

    def _normalize_metric_names(self, metric_names: list[str] | None) -> list[str]:
        """Normalize selected metric names and aliases."""
        if metric_names is None:
            return list(METRIC_SPECS)

        normalized: list[str] = []
        for name in metric_names:
            canonical = METRIC_ALIASES.get(name, name)
            if canonical not in METRIC_SPECS:
                msg = f"Unknown metric: {name}"
                raise KeyError(msg)
            normalized.append(canonical)

        return list(dict.fromkeys(normalized))

    def _filter_records_by_context(
        self,
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Filter records to the configured scoring context."""
        return [r for r in records if self._record_matches_context(r)]

    def _record_matches_context(self, record: dict[str, object]) -> bool:
        """Return whether a record belongs to the selected context."""
        if self.game_names is not None and record.get("game_name") not in self.game_names:
            return False
        if self.indices is not None and record.get("index") not in self.indices:
            return False
        return self.budgets is None or record.get("budget") in self.budgets

    def _build_context(
        self,
        records: list[dict[str, object]],
        metric_names: list[str],
    ) -> ScoringContext:
        """Build context describing the selected CD scoring subset."""
        context = build_context(records, self.group_keys)
        return ScoringContext(
            game_names=context.game_names,
            indices=context.indices,
            budgets=context.budgets,
            metric_names=metric_names,
            group_keys=context.group_keys,
        )

    def _get_used_metric_names(self, ranked_groups: list[RankedGroup]) -> list[str]:
        """Return configured metrics that produced at least one ranked group."""
        used = {rg.metric_name for rg in ranked_groups}
        return [m for m in self.metric_names if m in used]

    @staticmethod
    def plot_cd_diagram_plotly(
        cd_result: CriticalDifferenceResult,
        *,
        title: str | None = None,
    ) -> go.Figure:
        """Render the CD diagram as a Plotly figure.

        Approximators are placed along a horizontal rank axis (rank 1 = best,
        left). Better half routes labels to the left margin, worse half to the
        right. Cliques (not significantly different) are shown as thick bars
        below the axis. A CD reference bar is drawn above.

        Args:
            cd_result: A fully computed :class:`CriticalDifferenceResult`.
            title: Optional figure title.

        Returns:
            A ``plotly.graph_objects.Figure``.
        """

        mean_ranks = cd_result.mean_ranks
        if len(mean_ranks) < 2:
            fig = go.Figure()
            fig.update_layout(title=title or "CD Diagram — not enough approximators")
            return fig

        approx_list = sorted(mean_ranks, key=mean_ranks.get)
        n = len(approx_list)
        cd = cd_result.critical_difference

        half = (n + 1) // 2
        left_names = approx_list[:half]
        right_names = list(reversed(approx_list[half:]))

        row_spacing = 1.0
        left_rows: dict[str, float] = {name: row_spacing * (i + 1) for i, name in enumerate(left_names)}
        right_rows: dict[str, float] = {name: row_spacing * (i + 1) for i, name in enumerate(right_names)}
        max_row_y = row_spacing * max(len(left_names), len(right_names))

        axis_y = 0.0
        cd_y = max_row_y + 0.8
        shapes: list[dict] = []
        annotations: list[dict] = []
        traces: list[go.BaseTraceType] = []

        # ── Rank axis ──────────────────────────────────────────────────────────
        shapes.append(dict(type="line", x0=1, x1=n, y0=axis_y, y1=axis_y,
                           line=dict(color="black", width=1.5)))
        for tick in range(1, n + 1):
            shapes.append(dict(type="line", x0=tick, x1=tick,
                               y0=axis_y - 0.06, y1=axis_y + 0.06,
                               line=dict(color="black", width=1)))
            annotations.append(dict(x=tick, y=axis_y - 0.18, text=str(tick),
                                    showarrow=False, font=dict(size=11),
                                    xanchor="center", yanchor="top"))

        # ── CD reference bar ───────────────────────────────────────────────────
        shapes.append(dict(type="line", x0=1, x1=1 + cd, y0=cd_y, y1=cd_y,
                           line=dict(color="black", width=2)))
        for x_end in [1, 1 + cd]:
            shapes.append(dict(type="line", x0=x_end, x1=x_end,
                               y0=cd_y - 0.08, y1=cd_y + 0.08,
                               line=dict(color="black", width=1.5)))
        annotations.append(dict(x=1, y=cd_y + 0.18,
                                 text=f"CD = {cd:.3f}",
                                 showarrow=False, font=dict(size=11),
                                 xanchor="left", yanchor="bottom"))

        # ── Clique crossbars ───────────────────────────────────────────────────
        n_cliques = len(cd_result.cliques)
        clique_spacing = 0.2 if n_cliques else 0
        clique_y = 0.15
        for clique in sorted(cd_result.cliques,
                             key=lambda c: max(mean_ranks[a] for a in c) - min(mean_ranks[a] for a in c)):
            ranks = [mean_ranks[a] for a in clique]
            shapes.append(dict(type="line",
                               x0=min(ranks), x1=max(ranks),
                               y0=clique_y, y1=clique_y,
                               line=dict(color="black", width=4)))
            clique_y += clique_spacing

        # ── Elbow lines + labels ───────────────────────────────────────────────
        left_x = 1 - 0.3
        right_x = n + 0.3

        def _draw_elbow(name: str, row_y: float, label_x: float, anchor: str) -> None:
            rank = mean_ranks[name]
            label = f"{name} ({mean_ranks[name]:.2f})"
            # dot on axis
            traces.append(go.Scatter(x=[rank], y=[axis_y], mode="markers",
                                     marker=dict(color="black", size=5),
                                     showlegend=False, hoverinfo="skip"))
            # elbow: vertical up, then horizontal to margin
            traces.append(go.Scatter(x=[rank, rank, label_x],
                                     y=[axis_y, row_y, row_y],
                                     mode="lines",
                                     line=dict(color="black", width=0.8),
                                     showlegend=False, hoverinfo="skip"))
            annotations.append(dict(x=label_x, y=row_y, text=label,
                                    showarrow=False, font=dict(size=10),
                                    xanchor=anchor, yanchor="middle"))

        for name in left_names:
            _draw_elbow(name, left_rows[name], left_x, "right")
        for name in right_names:
            _draw_elbow(name, right_rows[name], right_x, "left")

        # ── Friedman annotation ────────────────────────────────────────────────
        sig_text = "✓ significant" if cd_result.friedman_significant else "✗ not significant"
        friedman_info = (
            f"Friedman p={cd_result.friedman_p_value:.4g} ({sig_text})  |  "
            f"n_groups={cd_result.n_groups}  |  α={cd_result.alpha}"
        )

        max_label_len = max((len(f"{a} ({mean_ranks[a]:.2f})") for a in approx_list), default=10)
        label_padding = max(1.5, max_label_len * 0.12)

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title or "Critical Difference Diagram",
            shapes=shapes,
            annotations=annotations,
            xaxis=dict(visible=False, range=[left_x - label_padding, right_x + label_padding]),
            yaxis=dict(visible=False, range=[-0.8 - n_cliques * clique_spacing, cd_y + 0.8]),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            margin=dict(l=150, r=150, t=60, b=60),
            height=max(350, 150 + max(len(left_names), len(right_names)) * 40 + n_cliques * 20),
        )
        fig.add_annotation(
            text=friedman_info, xref="paper", yref="paper",
            x=0.5, y=-0.08, showarrow=False,
            font=dict(size=9, color="gray"), xanchor="center",
        )
        return fig
