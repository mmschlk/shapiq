"""Masked games: batches from the data, input shapes, native data and model dispatch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shapiq.approximator.base import SHAPIQ
from shapiq.coalition.coalition import pack_bool_coalition
from shapiq.game import subset_patterns
from shapiq.index import ShapleyValueIndex
from shapiq.masker import BaselineMasker
from shapiq.players import FinitePlayers
from shapiq.sampling._base import PairedSizeWithoutReplacement
from shapiq.value import VectorValue
from shapiq.xai import MaskedGame

WEIGHTS = jnp.array([1.0, 2.0, 3.0])
X_EXPLAIN = jnp.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [-1.0, -2.0, -3.0]])
BACKGROUND = jnp.array([[0.0, 0.0, 0.0], [1.0, -1.0, 2.0], [0.5, 0.5, 0.5]])
COALITIONS = subset_patterns(3)
PLAYERS = FinitePlayers(3)
TABULAR = (3,)


def model(x):
    """Return a model with an interaction between the first two features."""
    return x @ WEIGHTS + x[:, 0] * x[:, 1]


def vector_model(x):
    """Return two outputs per input: ``model`` and its negated double."""
    return jnp.stack([model(x), -2 * model(x)], axis=-1)


def baseline_values(f, x, baseline):
    """Return ``f`` on ``x`` with every coalition's absent features from ``baseline``."""
    return f(jnp.where(COALITIONS, x, baseline))


@pytest.mark.parametrize("packed", [False, True], ids=["bool", "packed"])
def test_baselines_are_games_and_marginal_averages(packed):
    """Are several baselines separate games, and the marginal game their average?"""
    coalitions = pack_bool_coalition(COALITIONS) if packed else COALITIONS
    baselines = MaskedGame(model, PLAYERS, X_EXPLAIN[0], BACKGROUND, TABULAR, BaselineMasker())
    assert baselines.shape == (3,)
    values = baselines(coalitions[:, None])
    want = jnp.stack([baseline_values(model, X_EXPLAIN[0], b) for b in BACKGROUND], axis=-1)
    np.testing.assert_allclose(values, want, rtol=1e-6)
    marginal = MaskedGame(model, PLAYERS, X_EXPLAIN[0], BACKGROUND, TABULAR)
    assert marginal.shape == ()
    np.testing.assert_allclose(marginal(coalitions), want.mean(axis=-1), rtol=1e-6)


def test_batch_from_the_data():
    """Do the batches of inputs and baselines broadcast into the game batch, also indexed?"""
    game = MaskedGame(model, PLAYERS, X_EXPLAIN, BACKGROUND[:, None, :], TABULAR, BaselineMasker())
    assert game.shape == (3, 3)  # baselines x inputs
    values = game(COALITIONS[:, None, None, :])
    assert values.shape == (8, 3, 3)
    for b in range(3):
        for i in range(3):
            want = baseline_values(model, X_EXPLAIN[i], BACKGROUND[b])
            np.testing.assert_allclose(values[:, b, i], want, rtol=1e-6)
            np.testing.assert_allclose(game[b, i](COALITIONS), want, rtol=1e-6)


def test_jit_for_jax_data_and_models():
    """Does a game over JAX data and a JAX model trace, while other backends raise clearly?"""
    coalitions = COALITIONS[:, None, None, :]
    game = MaskedGame(model, PLAYERS, X_EXPLAIN, BACKGROUND[:, None, :], TABULAR, BaselineMasker())
    traced = jax.jit(lambda c: game(c))(coalitions)  # noqa: PLW0108 - jit hashes its function
    np.testing.assert_allclose(traced, game(coalitions), rtol=1e-6)
    numpy_data = MaskedGame(
        model, PLAYERS, np.asarray(X_EXPLAIN[0]), np.asarray(BACKGROUND), TABULAR
    )
    with pytest.raises(TypeError, match="ndarray data cannot be traced"):
        jax.jit(lambda c: numpy_data(c))(COALITIONS)  # noqa: PLW0108


def test_images_keep_their_shape():
    """Is an image masked in its own shape and passed to the model as a batch of images?"""
    images = np.arange(16.0).reshape(2, 2, 2, 2)  # two images of two channels, 2 x 2 pixels
    seen = []

    def image_model(x):
        seen.append(x.shape)
        return x.sum(axis=(1, 2, 3))

    game = MaskedGame(
        image_model, FinitePlayers(8), images, np.zeros((2, 2, 2)), (2, 2, 2), BaselineMasker()
    )
    assert game.shape == (2,)
    keep = np.arange(8) % 3 == 0
    values = game(jnp.asarray(keep))
    np.testing.assert_allclose(values, (images * keep.reshape(2, 2, 2)).sum(axis=(1, 2, 3)))
    assert seen[-1] == (2, 2, 2, 2)  # the batch merged, the input axes untouched
    with pytest.raises(ValueError, match="does not end with the input shape"):
        MaskedGame(image_model, FinitePlayers(8), images, np.zeros((2, 2)), (2, 2, 2))


def test_batch_size_bounds_model_calls():
    """Is the model called on at most ``batch_size`` inputs, with every input's samples together?"""
    sizes = []

    def counted(x):
        sizes.append(len(x))
        return vector_model(x)

    game = MaskedGame(counted, PLAYERS, X_EXPLAIN, BACKGROUND, TABULAR, batch_size=7)
    values = game(COALITIONS[:, None, :])  # 24 evaluations of 3 background rows each
    assert sizes == [6] * 12  # two evaluations per call
    want = MaskedGame(vector_model, PLAYERS, X_EXPLAIN, BACKGROUND, TABULAR)(COALITIONS[:, None, :])
    np.testing.assert_allclose(values.values, want.values, rtol=1e-6)
    with pytest.raises(ValueError, match="must be positive"):
        MaskedGame(model, PLAYERS, X_EXPLAIN, BACKGROUND, TABULAR, batch_size=0)


def test_vector_outputs():
    """Does a model with several outputs give vector values, one game's values per output?"""
    values = MaskedGame(vector_model, PLAYERS, X_EXPLAIN[0], BACKGROUND, TABULAR)(COALITIONS)
    assert isinstance(values, VectorValue)
    assert values.shape == (8,)
    want = jnp.mean(
        jnp.stack([baseline_values(vector_model, X_EXPLAIN[0], b) for b in BACKGROUND]), axis=0
    )
    np.testing.assert_allclose(values.values, want, rtol=1e-6)


def test_dataframe_keeps_dtypes():
    """Does a pipeline get DataFrames with their categories, against baselines and backgrounds?"""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from sklearn.compose import make_column_transformer  # noqa: PLC0415
    from sklearn.linear_model import LinearRegression  # noqa: PLC0415
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415
    from sklearn.preprocessing import OneHotEncoder  # noqa: PLC0415

    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "size": rng.normal(size=30),
            "color": pd.Categorical(rng.choice(["red", "green", "blue"], size=30)),
            "count": rng.integers(0, 5, size=30),
        }
    )
    y = data["size"] + (data["color"] == "red") + data["count"]
    pipeline = make_pipeline(
        make_column_transformer((OneHotEncoder(), ["color"]), remainder="passthrough"),
        LinearRegression(),
    ).fit(data, y)

    def masked(i, row):
        return pd.DataFrame(
            [
                {
                    name: data.iloc[i][name] if keep else row[name]
                    for name, keep in zip(data, c, strict=True)
                }
                for c in np.asarray(COALITIONS)
            ]
        ).astype(data.dtypes)

    background = data.iloc[2:6]
    marginal = MaskedGame(pipeline, PLAYERS, data.iloc[:2], background, TABULAR)
    values = marginal(COALITIONS[:, None, :])
    for i in range(2):
        want = np.mean([pipeline.predict(masked(i, row)) for _, row in background.iterrows()], 0)
        np.testing.assert_allclose(values[:, i], want, rtol=1e-5)
    baselines = MaskedGame(
        pipeline, PLAYERS, data.iloc[:2], data.iloc[2:4], TABULAR, BaselineMasker()
    )
    values = baselines(COALITIONS[:, None, :])
    for i in range(2):
        want = pipeline.predict(masked(i, data.iloc[2 + i]))
        np.testing.assert_allclose(values[:, i], want, rtol=1e-5)


def test_sklearn_models():
    """Do classifiers give probabilities, also in a pipeline, else decision scores or predictions?"""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    from sklearn.svm import LinearSVC  # noqa: PLC0415

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3)).astype(np.float32)
    y = x @ np.asarray(WEIGHTS)
    baseline = np.asarray(BACKGROUND[1])
    masked = np.where(COALITIONS, x[0], baseline)

    def values(estimator):
        return MaskedGame(estimator, PLAYERS, x[0], baseline, TABULAR, BaselineMasker())(COALITIONS)

    regressor = LinearRegression().fit(x, y)
    np.testing.assert_allclose(values(regressor), regressor.predict(masked), rtol=1e-5)
    # A pipeline over a classifier is no ClassifierMixin, yet has class probabilities.
    for classifier in [LogisticRegression(), make_pipeline(StandardScaler(), LogisticRegression())]:
        classes = values(classifier.fit(x, y > 0))
        assert isinstance(classes, VectorValue)
        np.testing.assert_allclose(classes.values, classifier.predict_proba(masked), rtol=1e-5)
    svc = LinearSVC().fit(x, y > 0)  # no predict_proba
    np.testing.assert_allclose(values(svc), svc.decision_function(masked), rtol=1e-5)
    jax_data = MaskedGame(regressor, PLAYERS, jnp.asarray(x[0]), jnp.asarray(baseline), TABULAR)
    with pytest.raises(TypeError, match="scikit-learn models cannot be traced"):
        jax.jit(lambda c: jax_data(c))(COALITIONS)  # noqa: PLW0108 - jit hashes its function


def test_torch_module():
    """Is a torch module called on tensor data, its output columns kept as a vector value?"""
    torch = pytest.importorskip("torch")
    x, background = torch.tensor(np.asarray(X_EXPLAIN[0])), torch.tensor(np.asarray(BACKGROUND))
    module = torch.nn.Linear(3, 2, dtype=x.dtype)
    values = MaskedGame(module, PLAYERS, x, background, TABULAR)(COALITIONS)
    masked = torch.where(torch.tensor(np.asarray(COALITIONS))[:, None, :], x, background)
    with torch.no_grad():
        want = module(masked).numpy().mean(axis=1)
    assert isinstance(values, VectorValue)
    np.testing.assert_allclose(values.values, want, rtol=1e-5)


def test_shapiq_on_vector_game():
    """Does SHAPIQ explain every class of a vector game as it explains that class's game?"""
    sampler = PairedSizeWithoutReplacement(keys=jax.random.key(0), weights=jnp.ones(4))

    def game(f):
        return MaskedGame(f, PLAYERS, X_EXPLAIN[0], BACKGROUND[1], TABULAR, BaselineMasker())

    estimate, _ = SHAPIQ(sampler=sampler, game=game(vector_model), index=ShapleyValueIndex())(8)
    for k in range(2):
        column = game(lambda x, k=k: vector_model(x)[:, k])
        want, _ = SHAPIQ(sampler=sampler, game=column, index=ShapleyValueIndex())(8)
        np.testing.assert_allclose(
            estimate.contributions().values[..., k], want.contributions(), atol=1e-5
        )


@pytest.mark.parametrize(
    ("outputs", "error", "match"),
    [
        (lambda x: jnp.zeros((x.shape[0], 2, 2)), ValueError, "not one vector"),
        (lambda x: np.zeros(x.shape[0], dtype=int), TypeError, "not float values"),
        (lambda x: (model(x), x), TypeError, "no conversion"),
    ],
    ids=["matrix", "labels", "tuple"],
)
def test_invalid_outputs(outputs, error, match):
    """Are outputs that are no float scalar or vector per input rejected with a clear error?"""
    with pytest.raises(error, match=match):
        MaskedGame(outputs, PLAYERS, X_EXPLAIN[0], BACKGROUND, TABULAR)(COALITIONS)
