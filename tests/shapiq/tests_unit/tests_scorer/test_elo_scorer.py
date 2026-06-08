from leaderboard.scoring.elo_scorer import EloScorer
from leaderboard.scoring.scorer_utils import filter_valid_records, group_records
from tests.shapiq.tests_unit.tests_scorer.test_group_rank_scorer import _make_test_records


def test_elo_scorer_filters_matches_by_metric_and_budget():
    scorer = EloScorer(metric_names=["spearman"], budgets=[500])
    records = _make_test_records()

    valid_records = filter_valid_records(records)
    selected_records = scorer._filter_records_by_context(valid_records)
    groups = group_records(selected_records, scorer.group_keys)
    matches = scorer._build_pairwise_matches(groups)

    assert matches
    assert all(match.metric_name == "spearman" for match in matches)
    assert all(match.group_key["budget"] == 500 for match in matches)
