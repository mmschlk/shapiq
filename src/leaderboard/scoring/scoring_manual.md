## Scoring Manual

### Purpose of the scorer

The purpose of a leaderboard scorer is to compute a ranking of approximators based on their performance on a selected benchmark context. The project defines an abstract `LeaderboardScorer` interface and allows different concrete scoring implementations. The primary implementation currently used is the `EloScorer`, which is based on the Elo rating system (https://en.wikipedia.org/wiki/Elo_rating_system). This implementation will be covered in more detail.

The input to a scorer are run records from the database. They contain benchmark results of the approximators across games, indices, budgets, seeds and metrics. After the scorer processes these records, it returns a `ScoringResult`.

The `ScoringResult` describes which scorer was used, which benchmark context was evaluated, and how the approximators were ranked. Its main output is a list of leaderboard rows, where each approximator is assigned a score and ranked accordingly. In addition, the result contains metadata that documents how the score was computed.

### How the EloScorer works

The `EloScorer` is the primary scoring implementation used in the leaderboard. It compares approximators through pairwise matches within benchmark groups. A comparable group consists of run records that share the same benchmark context, such as game, index, maximum interaction order, budget, and ground truth method. Comparable groups are formed during scoring based on the configured group keys. 

Within each comparable group, the scorer first aggregates repeated runs over approximation seeds and then constructs pairwise matches for the selected metrics. All approximators start with the same initial Elo rating (Default: 1000). Each pairwise match updates the ratings of the two involved approximators. The update depends on the match outcome and on the expected win probability, which is derived from the current rating difference between both approximators. A win against a stronger opponent leads to a larger Elo gain, while a win against a weaker opponent leads to a smaller gain. On the other hand, losing against a weaker opponent is penalized more strongly than losing against a stronger opponent.

Since Elo ratings are updated sequentially, the order of pairwise matches can influence the final outcome. This is a known disadvantage of the Elo system. To obtain more stable rankings, the scorer can compute Elo ratings multiple times using different permutations of the match order. The final score is then based on the mean rating across these permutations.

In addition, the scorer supports bootstrapping over comparable groups, inspired by the TabArena(https://github.com/autogluon/tabarena/tree/main) Elo rating system approach. For each bootstrap sample the comparable groups are sampled with replacement and the ratings recomputed. If match-order permutations are enabled, the scorer averages the permutation-based Elo ratings within each sample. The bootstrap score distribution is used to compute confidence intervals. These intervals show how stable the approximator rankings are under resampling of the available comparable groups.

### How to use the EloScorer

Construct the EloScorer by giving the benchmark context over which the approximators should be compared. The context can be restricted by metrics, games, indices, and budgets. If one of these filter arguments is set to `None`, the scorer uses all available values for that argument.
In addition you can decide on the number of order permutations to be calculated and the number of bootstrap samples to be taken. By default permutations and bootstrapping are disabled.
After completing the construction, call the `score` method with the run records as parameter to obtain the `ScoringResult`. It contains the ranked leaderboard rows, the evaluated context, and metadata about the scoring process.

**Usage example:**

```jsx
scorer = EloScorer(
    metric_names=["spearman"],
    game_names=None,
    indices=["SII"],
    budgets=[250, 500, 1000],
    n_permutations=50,
    n_bootstrap_samples=200,
    confidence_level=0.95,
)

result = scorer.score(raw_records)
```

### Addition of a new scorer

New scorers can be defined in the `leaderboard/scoring` package. A new scorer must fulfill the following requirements:

1. It must extend the abstract `LeaderboardScorer` class.
2. It must define a unique scorer name.
3. It must define whether higher final scores are considered better or worse.
4. It must implement the `score` method and return a `ScoringResult`.

The `score` method receives the run records as input and is responsible for computing the final leaderboard. The returned `ScoringResult` should contain the ranked leaderboard rows, the scoring context, and metadata describing the scoring process.

The main output of a scorer is a list of rows in the leaderboard (`LeaderboardRow`). The main data you need to construct these rows is a mapping from approximator names to their computed scores.
If the scorer computes intermediate results for comparable groups, these can be stored as group results (`group_results`) and optionally be provided in the output.
The `metadata` attribute can be used to store additional information about the scoring process. You can also use this information for debugging or visualization in the UI.