"""Behavioural tests for all imputers.

Structured as:
- ``TestImputerProtocol``: shared contract checks, parametrised over every registry entry.
- One ``Test<Imputer>`` class per imputer: only semantics unique to that imputer.
- ``TestCrossImputerAgreement``: relationships between imputers on the same problem.

Protocol tests use ``normalize=False`` for imputers that accept the flag so the raw
``value_function`` output is returned; normalisation behaviour is covered per-imputer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq.imputer import (
    BaselineImputer,
    GaussianCopulaImputer,
    GaussianImputer,
    GenerativeConditionalImputer,
    MarginalImputer,
    TabPFNImputer,
)
from shapiq.imputer.gaussian_imputer_exceptions import CategoricalFeatureError

from .conftest import skip_if_no_tabpfn, skip_if_no_xgboost

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_FEATURES = 5


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def column_picker(i: int) -> Callable[[np.ndarray], np.ndarray]:
    """Probe model that returns feature ``i`` unchanged — isolates a single feature's flow."""
    return lambda X: X[:, i].astype(float)


def sum_model(X: np.ndarray) -> np.ndarray:
    return X.sum(axis=1).astype(float)


def product_model(X: np.ndarray) -> np.ndarray:
    """Multiplicative model — exposes feature dependence for joint vs per-feature sampling."""
    return (X[:, 0] * X[:, 1]).astype(float)


def gaussian_data(n: int = 500, n_features: int = N_FEATURES, seed: int = 42) -> np.ndarray:
    """Default 500-row background — matches the MC ``sample_size`` so nothing gets capped."""
    return _rng(seed).normal(size=(n, n_features))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# ``normalize=False`` is passed where supported so protocol tests can assume raw
# value_function output. Per-imputer classes below exercise the normalised path.

IMPUTER_CONFIGS = [
    pytest.param(
        {
            "cls": BaselineImputer,
            "kwargs": {"normalize": False},
            "is_stochastic": False,
            "preserves_x_exactly": True,
            "mc_tol": 1e-10,
        },
        id="BaselineImputer",
    ),
    pytest.param(
        {
            "cls": MarginalImputer,
            "kwargs": {"sample_size": 500, "normalize": False},
            "is_stochastic": True,
            "preserves_x_exactly": True,
            "mc_tol": 0.2,
        },
        id="MarginalImputer",
    ),
    pytest.param(
        {
            "cls": GaussianImputer,
            "kwargs": {"sample_size": 500},
            "is_stochastic": True,
            "preserves_x_exactly": True,
            "mc_tol": 0.2,
        },
        id="GaussianImputer",
    ),
    pytest.param(
        {
            "cls": GaussianCopulaImputer,
            "kwargs": {"sample_size": 500},
            "is_stochastic": True,
            # Copula transforms x to Gaussian space and back via interpolation on the
            # sorted background; present features get the interpolated round-trip value,
            # not the original x[i].
            "preserves_x_exactly": False,
            "mc_tol": 0.25,
        },
        id="GaussianCopulaImputer",
    ),
    pytest.param(
        {
            "cls": GenerativeConditionalImputer,
            "kwargs": {
                "sample_size": 20,
                "conditional_budget": 32,
                "normalize": False,
            },
            "is_stochastic": True,
            "preserves_x_exactly": True,
            "mc_tol": 0.5,
        },
        marks=[skip_if_no_xgboost, pytest.mark.slow],
        id="GenerativeConditionalImputer",
    ),
]


def _make_imputer(config: dict, *, model=sum_model, data=None, seed=42, **overrides):
    """Instantiate an imputer from a registry config with optional overrides."""
    if data is None:
        data = gaussian_data(seed=seed)
    kwargs = {**config["kwargs"], **overrides}
    return config["cls"](model=model, data=data, random_state=seed, **kwargs)


# ===========================================================================
# TestImputerProtocol — every assertion here must hold for every imputer.
# ===========================================================================


@pytest.mark.parametrize("config", IMPUTER_CONFIGS)
class TestImputerProtocol:
    """Behavioural contract every imputer must satisfy."""

    def test_full_coalition_returns_model_prediction(self, config):
        """All features present → no imputation → output equals ``model(x)``."""
        data = gaussian_data()
        imputer = _make_imputer(config, data=data)
        x = data[0]
        imputer.fit(x)

        coalition_all = np.ones((1, N_FEATURES), dtype=bool)
        result = float(imputer(coalition_all)[0])
        expected = float(sum_model(x.reshape(1, -1))[0])
        assert result == pytest.approx(expected, abs=config["mc_tol"])

    def test_output_shape_matches_coalition_count(self, config):
        """Output is 1-D with length equal to number of coalitions."""
        imputer = _make_imputer(config)
        imputer.fit(gaussian_data()[0])

        for k in (1, 3, 7):
            rng = _rng(k)
            coalitions = rng.integers(0, 2, size=(k, N_FEATURES)).astype(bool)
            out = imputer(coalitions)
            assert out.shape == (k,)
            assert np.all(np.isfinite(out))

    def test_empty_prediction_is_finite_after_construction(self, config):
        """``empty_prediction`` is a finite float; ``normalization_value`` follows ``normalize``."""
        imputer = _make_imputer(config)
        assert np.isfinite(imputer.empty_prediction)
        # With normalize=False (or unsupported), normalisation_value must be 0.
        assert imputer.normalization_value == pytest.approx(0.0)

    def test_present_features_use_x_values(self, config):
        """Probe model ``f(X)=X[:,i]`` + coalition with only feature ``i`` present → result == ``x[i]``.

        For imputers that preserve x exactly (Baseline, Marginal, Gaussian, Generative) the
        result is ``x[i]`` to machine precision. GaussianCopulaImputer round-trips through
        Gaussian space via an interpolating inverse CDF, so only a loose tolerance holds and
        ``x`` must be within the background range (the copula clamps outside it).
        """
        data = gaussian_data()
        # Pick an in-distribution x so the copula round-trip is accurate.
        x = data[0].copy()
        tol = 1e-10 if config["preserves_x_exactly"] else 0.5
        for i in range(N_FEATURES):
            imputer = _make_imputer(config, model=column_picker(i), data=data)
            imputer.fit(x)
            coalition = np.zeros((1, N_FEATURES), dtype=bool)
            coalition[0, i] = True
            result = float(imputer(coalition)[0])
            assert result == pytest.approx(float(x[i]), abs=tol)

    def test_missing_features_do_not_leak_x_values(self, config):
        """Probe model ``f(X)=X[:,i]`` + coalition missing feature ``i`` → output closer to bg mean than to ``x[i]``."""
        data = gaussian_data()
        # Pick an x far from background mean (≈0).
        x = np.full(N_FEATURES, 10.0)
        for i in range(N_FEATURES):
            imputer = _make_imputer(config, model=column_picker(i), data=data)
            imputer.fit(x)
            coalition = np.ones((1, N_FEATURES), dtype=bool)
            coalition[0, i] = False
            result = float(imputer(coalition)[0])
            bg_mean = float(data[:, i].mean())
            assert abs(result - bg_mean) < abs(result - x[i]), (
                f"feature {i}: result={result}, bg_mean={bg_mean}, x={x[i]}"
            )

    def test_random_state_reproducibility(self, config):
        """Two fresh imputer instances with the same ``random_state`` produce identical output."""
        if not config["is_stochastic"]:
            pytest.skip("deterministic imputer")
        data = gaussian_data()
        imputer_a = _make_imputer(config, data=data, seed=123)
        imputer_b = _make_imputer(config, data=data, seed=123)
        x = data[0]
        imputer_a.fit(x)
        imputer_b.fit(x)
        rng = _rng(0)
        coalitions = rng.integers(0, 2, size=(8, N_FEATURES)).astype(bool)
        np.testing.assert_array_equal(imputer_a(coalitions), imputer_b(coalitions))

    def test_fit_changes_explanation_point(self, config):
        """Fitting on a different x changes the output for coalitions that expose x."""
        data = gaussian_data()
        imputer = _make_imputer(config, model=column_picker(0), data=data)
        # x1 and x2 differ in feature 0; coalition has feature 0 present.
        x1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        x2 = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
        coalition = np.zeros((1, N_FEATURES), dtype=bool)
        coalition[0, 0] = True

        imputer.fit(x1)
        out_1 = float(imputer(coalition)[0])
        imputer.fit(x2)
        out_2 = float(imputer(coalition)[0])
        assert out_1 != out_2
        tol = 1e-10 if config["preserves_x_exactly"] else config["mc_tol"]
        assert out_1 == pytest.approx(1.0, abs=tol)
        assert out_2 == pytest.approx(-1.0, abs=tol)

    def test_fit_returns_self(self, config):
        """``fit`` returns the imputer for chaining (base class contract)."""
        imputer = _make_imputer(config)
        assert imputer.fit(gaussian_data()[0]) is imputer


# ===========================================================================
# Per-imputer semantics — only behaviour that cannot be expressed generically.
# ===========================================================================


class TestBaselineImputer:
    """Baseline imputer uses a deterministic constant per feature."""

    def test_baseline_values_equal_mean_of_numerical_background(self):
        data = gaussian_data(n=500)
        imputer = BaselineImputer(model=sum_model, data=data)
        np.testing.assert_allclose(
            imputer.baseline_values[0].astype(float), data.mean(axis=0), atol=1e-12
        )

    def test_baseline_values_use_mode_for_categorical(self):
        data = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [1, 2, 3], [1, 0, 2]], dtype=object)
        imputer = BaselineImputer(
            model=lambda X: np.zeros(X.shape[0]),
            data=data,
            categorical_features=[0, 1, 2],
        )
        np.testing.assert_array_equal(imputer.baseline_values[0].tolist(), [0, 1, 2])

    def test_init_background_with_vector_overrides(self):
        """Passing a vector to ``init_background`` replaces ``baseline_values``.

        Note: ``init_background`` only recomputes ``empty_prediction`` when given a
        matrix (see baseline_imputer.py:136-160). With a vector the caller is expected
        to invoke ``calc_empty_prediction`` themselves.
        """
        data = gaussian_data()
        imputer = BaselineImputer(model=sum_model, data=data)
        baseline_vec = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        imputer.init_background(baseline_vec)
        np.testing.assert_allclose(imputer.baseline_values[0], baseline_vec)
        # After manually recomputing, empty_prediction follows the new baseline.
        imputer.calc_empty_prediction()
        assert imputer.empty_prediction == pytest.approx(float(sum(baseline_vec)))

    def test_value_function_is_exact_where_substitution(self):
        """For every coalition, the imputed row equals ``where(coalition, x, baseline)``."""
        data = gaussian_data()
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        for i in range(N_FEATURES):
            imputer = BaselineImputer(model=column_picker(i), data=data, normalize=False)
            imputer.fit(x)
            baseline = imputer.baseline_values[0]
            # Every 2^5 coalition — deterministic so cheap.
            for mask in range(1 << N_FEATURES):
                coalition = np.array(
                    [(mask >> j) & 1 for j in range(N_FEATURES)], dtype=bool
                ).reshape(1, -1)
                expected = x[i] if coalition[0, i] else baseline[i]
                result = float(imputer(coalition)[0])
                assert result == pytest.approx(float(expected), abs=1e-10)

    def test_repeated_calls_are_identical(self):
        data = gaussian_data()
        imputer = BaselineImputer(model=sum_model, data=data, normalize=False)
        imputer.fit(data[0])
        coalitions = _rng(7).integers(0, 2, size=(10, N_FEATURES)).astype(bool)
        np.testing.assert_array_equal(imputer(coalitions), imputer(coalitions))

    def test_normalize_true_centers_empty_at_zero(self):
        data = gaussian_data()
        imputer = BaselineImputer(model=sum_model, data=data, normalize=True)
        imputer.fit(data[0])
        # With normalize=True, the empty coalition returns 0 via Game.__call__.
        out = float(imputer(np.zeros((1, N_FEATURES), dtype=bool))[0])
        assert out == pytest.approx(0.0, abs=1e-10)


class TestMarginalImputer:
    """Marginal imputer samples replacement rows from the empirical background."""

    def test_empty_prediction_equals_mean_model_over_background(self):
        data = gaussian_data(n=100)
        imputer = MarginalImputer(model=sum_model, data=data, sample_size=100, normalize=False)
        expected = float(sum_model(data).mean())
        assert imputer.empty_prediction == pytest.approx(expected, abs=1e-10)

    def test_converges_to_marginal_expectation(self):
        """Missing feature 1 → output converges to ``data[:,1].mean()`` within ~3 sd/sqrt(N)."""
        data = gaussian_data(n=1000)
        imputer = MarginalImputer(
            model=column_picker(1),
            data=data,
            sample_size=500,
            normalize=False,
            random_state=42,
        )
        x = np.full(N_FEATURES, 100.0)  # far from bg mean so leakage would be obvious
        imputer.fit(x)
        coalition = np.ones((1, N_FEATURES), dtype=bool)
        coalition[0, 1] = False
        result = float(imputer(coalition)[0])
        expected = float(data[:, 1].mean())
        tol = 3.0 * data[:, 1].std() / np.sqrt(500)
        assert result == pytest.approx(expected, abs=tol)

    def test_joint_vs_per_feature_differ_on_dependent_data(self):
        """Perfectly correlated background + multiplicative model → joint ≠ per-feature."""
        rng = _rng(0)
        col = rng.normal(size=400)
        data = np.column_stack([col, col, col, col, col])  # X1 == X2 == ...
        x = np.full(N_FEATURES, 5.0)

        joint = MarginalImputer(
            model=product_model,
            data=data,
            sample_size=400,
            joint_marginal_distribution=True,
            normalize=False,
            random_state=1,
        )
        perfeat = MarginalImputer(
            model=product_model,
            data=data,
            sample_size=400,
            joint_marginal_distribution=False,
            normalize=False,
            random_state=1,
        )
        joint.fit(x)
        perfeat.fit(x)

        # Coalition: both features 0 and 1 missing.
        coalition = np.ones((1, N_FEATURES), dtype=bool)
        coalition[0, :2] = False
        joint_out = float(joint(coalition)[0])
        perfeat_out = float(perfeat(coalition)[0])
        # Joint preserves X1==X2 so E[X1*X2] = E[X^2] = Var(X) + E[X]^2.
        # Per-feature independent ≈ E[X]*E[X].
        joint_expected = float((data[:, 0] ** 2).mean())
        perfeat_expected = float(data[:, 0].mean() ** 2)
        assert abs(joint_out - joint_expected) < 0.2
        assert abs(perfeat_out - perfeat_expected) < 0.2
        assert abs(joint_out - perfeat_out) > 0.5

    def test_sample_size_capped_with_warning(self):
        data = gaussian_data(n=10)
        with pytest.warns(UserWarning, match="sample size is larger"):
            imp = MarginalImputer(model=sum_model, data=data, sample_size=1000, normalize=False)
        assert imp._sample_size == 10

    def test_init_background_swaps_distribution(self):
        data = gaussian_data(n=200, seed=1)
        imputer = MarginalImputer(
            model=sum_model, data=data, sample_size=200, normalize=False, random_state=0
        )
        before = imputer.empty_prediction
        new_data = gaussian_data(n=200, seed=2) + 5.0  # shift so mean differs
        imputer.init_background(new_data)
        after = imputer.empty_prediction
        assert abs(after - before) > 1.0


class TestGaussianImputer:
    """Gaussian imputer samples from the conditional multivariate normal."""

    def test_categorical_features_raise(self):
        rng = _rng(0)
        data = rng.normal(size=(100, 3))
        data_with_binary = np.column_stack([data, rng.integers(0, 2, size=100)])
        with pytest.raises(CategoricalFeatureError):
            GaussianImputer(model=lambda X: X.sum(axis=1), data=data_with_binary, sample_size=10)

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="Background data must not be empty"):
            GaussianImputer(model=lambda X: X.sum(axis=1), data=np.empty((0, 3)), sample_size=10)

    def test_conditional_mean_matches_closed_form(self):
        """For ``N(0, Σ)`` data + linear model, imputer recovers the analytical conditional mean."""
        # Σ with strong correlation between features 0 and 1.
        mu = np.zeros(N_FEATURES)
        cov = np.eye(N_FEATURES)
        cov[0, 1] = cov[1, 0] = 0.8
        rng = _rng(42)
        data = rng.multivariate_normal(mu, cov, size=5000)

        # Known feature 0, missing feature 1; model returns X[:, 1].
        imputer = GaussianImputer(
            model=column_picker(1), data=data, sample_size=500, random_state=0
        )
        x = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
        imputer.fit(x)
        coalition = np.array([[True, False, False, False, False]])
        result = float(imputer(coalition)[0])

        # Closed form for Gaussian conditional mean: μ1 + Σ10/Σ00 · (x0 - μ0).
        cov_emp = np.cov(data.T)
        mu_emp = data.mean(axis=0)
        closed_form = mu_emp[1] + cov_emp[1, 0] / cov_emp[0, 0] * (x[0] - mu_emp[0])
        cond_var = cov_emp[1, 1] - cov_emp[1, 0] ** 2 / cov_emp[0, 0]
        mc_std = np.sqrt(cond_var / 500)
        assert result == pytest.approx(closed_form, abs=4.0 * mc_std)

    def test_cov_mat_regularised_to_psd(self):
        """Rank-deficient background → ``cov_mat`` eigenvalues strictly positive."""
        rng = _rng(0)
        base = rng.normal(size=(50, 3))
        # Duplicate a column so the covariance matrix is rank-deficient.
        data = np.column_stack([base, base[:, 0]])
        imputer = GaussianImputer(model=lambda X: X.sum(axis=1), data=data, sample_size=10)
        eigenvalues = np.linalg.eigvalsh(imputer.cov_mat)
        assert np.all(eigenvalues > 1e-7)

    def test_known_features_threaded_through_samples(self):
        """Samples for present features equal ``x`` on every MC draw."""
        data = gaussian_data(n=200)
        imputer = GaussianImputer(
            model=lambda X: np.zeros(X.shape[0]),
            data=data,
            sample_size=50,
            random_state=0,
        )
        x = np.array([7.0, -3.0, 0.0, 0.0, 0.0])
        coalitions = np.array([[True, True, False, False, False]])
        samples = imputer._sample_monte_carlo(x, coalitions)
        np.testing.assert_allclose(samples[0, :, 0], 7.0)
        np.testing.assert_allclose(samples[0, :, 1], -3.0)


class TestGaussianCopulaImputer:
    """Copula imputer transforms features to Gaussian space via empirical CDF before sampling."""

    def test_transform_round_trip_preserves_ranks(self):
        data = _rng(0).gamma(shape=2, scale=1, size=(200, 3))  # non-Gaussian marginals
        imputer = GaussianCopulaImputer(model=lambda X: X.sum(axis=1), data=data, sample_size=10)
        gaussian_space = imputer._transform_to_gaussian(data)
        back = imputer._transform_from_gaussian(gaussian_space)
        for col in range(data.shape[1]):
            orig_ranks = np.argsort(np.argsort(data[:, col]))
            back_ranks = np.argsort(np.argsort(back[:, col]))
            np.testing.assert_array_equal(orig_ranks, back_ranks)

    def test_matches_gaussian_on_standard_normal_data(self):
        """Copula ≈ Gaussian when data is already standard normal."""
        data = _rng(1).normal(size=(500, N_FEATURES))
        x = data[0]
        coalitions = np.array(
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [False, False, True, True, False],
            ]
        )
        gauss = GaussianImputer(model=sum_model, data=data, sample_size=500, random_state=2)
        copula = GaussianCopulaImputer(model=sum_model, data=data, sample_size=500, random_state=2)
        gauss.fit(x)
        copula.fit(x)
        np.testing.assert_allclose(gauss(coalitions), copula(coalitions), atol=0.3)

    @pytest.mark.slow
    def test_handles_non_gaussian_marginals(self):
        """Heavy-tailed marginals: copula output finite, transformed space ~ standard normal."""
        data = _rng(3).lognormal(size=(500, N_FEATURES))
        imputer = GaussianCopulaImputer(model=sum_model, data=data, sample_size=200, random_state=0)
        imputer.fit(data[0])
        coalitions = _rng(4).integers(0, 2, size=(5, N_FEATURES)).astype(bool)
        out = imputer(coalitions)
        assert np.all(np.isfinite(out))
        # Transformed background should have (per-feature) mean ≈ 0, std ≈ 1.
        transformed = imputer._transform_to_gaussian(data)
        assert np.all(np.abs(transformed.mean(axis=0)) < 0.1)
        assert np.all(np.abs(transformed.std(axis=0) - 1.0) < 0.1)


@skip_if_no_xgboost
@pytest.mark.slow
class TestGenerativeConditionalImputer:
    """Generative imputer uses an XGBoost tree embedder to find conditional neighbourhoods."""

    def test_neighbourhood_sampling_respects_clusters(self):
        """Two-cluster background; fixing the cluster-separating feature → samples stay in cluster."""
        rng = _rng(0)
        n = 300
        cluster_a = rng.normal(loc=-5, scale=0.5, size=(n, N_FEATURES))
        cluster_b = rng.normal(loc=+5, scale=0.5, size=(n, N_FEATURES))
        data = np.vstack([cluster_a, cluster_b])

        # Model reports feature 2 (not the one we fix) so output tracks cluster identity.
        imputer = GenerativeConditionalImputer(
            model=column_picker(2),
            data=data,
            sample_size=30,
            conditional_budget=64,
            conditional_threshold=0.1,
            normalize=False,
            random_state=0,
        )
        # x places us firmly in cluster B via feature 0. Feature 2 is missing.
        x = np.array([5.0, 5.0, 999.0, 5.0, 5.0])
        imputer.fit(x)
        coalition = np.array([[True, True, False, True, True]])
        result = float(imputer(coalition)[0])
        # Cluster B has feature 2 ≈ +5; cluster A has feature 2 ≈ -5.
        assert result > 0, f"expected cluster-B mean (~+5), got {result}"

    def test_init_background_fits_embedder(self):
        data = gaussian_data(n=100)
        imputer = GenerativeConditionalImputer(
            model=sum_model,
            data=data,
            sample_size=10,
            conditional_budget=16,
            normalize=False,
        )
        assert hasattr(imputer, "_tree_embedder")
        assert imputer._data_embedded.shape[0] == data.shape[0]

    def test_empty_prediction_equals_mean_model_over_background(self):
        data = gaussian_data(n=100)
        imputer = GenerativeConditionalImputer(
            model=sum_model,
            data=data,
            sample_size=10,
            conditional_budget=16,
            normalize=False,
        )
        expected = float(sum_model(data).mean())
        assert imputer.empty_prediction == pytest.approx(expected, abs=1e-10)


@skip_if_no_tabpfn
@pytest.mark.slow
class TestTabPFNImputer:
    """TabPFN: remove-and-contextualize paradigm (not imputation)."""

    def _build(self):
        from tabpfn import TabPFNRegressor

        rng = _rng(42)
        x_train = rng.normal(size=(30, N_FEATURES))
        y_train = x_train[:, 0] + 0.5 * x_train[:, 1]
        model = TabPFNRegressor(n_estimators=1)
        try:
            model.fit(x_train, y_train)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"TabPFN model unavailable: {exc}")

        seen_shapes: list[tuple[int, int]] = []

        def predict_fn(mdl, X):
            seen_shapes.append(X.shape)
            return mdl.predict(X)

        imp = TabPFNImputer(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_train,
            predict_function=predict_fn,
        )
        return imp, x_train, seen_shapes

    def test_empty_coalition_uses_precomputed_empty_prediction(self):
        imp, x_train, _ = self._build()
        imp.fit(x_train[0])
        out = float(imp(np.zeros((1, N_FEATURES), dtype=bool))[0])
        # normalize is not a kwarg; Game.__call__ still subtracts normalization_value (0).
        assert out == pytest.approx(imp.empty_prediction, abs=1e-10)

    def test_remove_and_contextualize_uses_subset_features(self):
        imp, x_train, seen_shapes = self._build()
        imp.fit(x_train[0])
        seen_shapes.clear()
        coalitions = np.array(
            [
                [True, False, False, False, False],
                [True, True, True, False, False],
            ]
        )
        imp(coalitions)
        # For each coalition, predict_fn is called with ``k`` feature columns.
        widths = {shape[1] for shape in seen_shapes}
        assert 1 in widths
        assert 3 in widths

    def test_requires_empty_prediction_or_x_test(self):
        from tabpfn import TabPFNRegressor

        rng = _rng(0)
        x_train = rng.normal(size=(10, 3))
        y_train = x_train[:, 0]
        model = TabPFNRegressor(n_estimators=1)
        try:
            model.fit(x_train, y_train)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"TabPFN model unavailable: {exc}")
        with pytest.raises(ValueError, match="empty prediction"):
            TabPFNImputer(
                model=model,
                x_train=x_train,
                y_train=y_train,
                predict_function=lambda m, X: m.predict(X),
            )


# ===========================================================================
# Cross-imputer agreement — relationships on the same problem.
# ===========================================================================


class TestCrossImputerAgreement:
    """Invariants linking multiple imputers on the same data/model."""

    def test_all_imputers_agree_on_full_coalition(self):
        data = gaussian_data()
        x = data[0]
        coalition_all = np.ones((1, N_FEATURES), dtype=bool)
        expected = float(sum_model(x.reshape(1, -1))[0])
        # Use the fast, deterministic subset of the registry.
        cfgs = [
            {"cls": BaselineImputer, "kwargs": {"normalize": False}},
            {"cls": MarginalImputer, "kwargs": {"sample_size": 500, "normalize": False}},
            {"cls": GaussianImputer, "kwargs": {"sample_size": 500}},
            {"cls": GaussianCopulaImputer, "kwargs": {"sample_size": 500}},
        ]
        for cfg in cfgs:
            imp = cfg["cls"](model=sum_model, data=data, random_state=0, **cfg["kwargs"])
            imp.fit(x)
            assert float(imp(coalition_all)[0]) == pytest.approx(expected, abs=1e-10)

    def test_baseline_matches_marginal_on_constant_background(self):
        """Degenerate marginal (identical rows) matches Baseline exactly, coalition-by-coalition."""
        constant_row = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = np.tile(constant_row, (50, 1))
        x = np.array([10.0, -10.0, 10.0, -10.0, 10.0])
        rng = _rng(0)
        coalitions = rng.integers(0, 2, size=(16, N_FEATURES)).astype(bool)

        baseline = BaselineImputer(model=sum_model, data=data, normalize=False)
        marginal = MarginalImputer(
            model=sum_model, data=data, sample_size=50, normalize=False, random_state=0
        )
        baseline.fit(x)
        marginal.fit(x)
        np.testing.assert_allclose(baseline(coalitions), marginal(coalitions), atol=1e-10)

    def test_gaussian_matches_marginal_on_independent_data(self):
        """Independent features → conditional = marginal; Gaussian ≈ Marginal within MC tol."""
        data = _rng(0).normal(size=(1000, N_FEATURES))  # Σ ≈ I
        x = np.full(N_FEATURES, 3.0)
        rng = _rng(1)
        coalitions = rng.integers(0, 2, size=(10, N_FEATURES)).astype(bool)

        gauss = GaussianImputer(model=sum_model, data=data, sample_size=800, random_state=0)
        marginal = MarginalImputer(
            model=sum_model,
            data=data,
            sample_size=800,
            normalize=False,
            random_state=0,
        )
        gauss.fit(x)
        marginal.fit(x)
        np.testing.assert_allclose(gauss(coalitions), marginal(coalitions), atol=0.5)

    def test_copula_matches_gaussian_on_standard_normal(self):
        data = _rng(2).normal(size=(500, N_FEATURES))
        x = data[0]
        rng = _rng(3)
        coalitions = rng.integers(0, 2, size=(10, N_FEATURES)).astype(bool)

        gauss = GaussianImputer(model=sum_model, data=data, sample_size=500, random_state=0)
        copula = GaussianCopulaImputer(model=sum_model, data=data, sample_size=500, random_state=0)
        gauss.fit(x)
        copula.fit(x)
        np.testing.assert_allclose(gauss(coalitions), copula(coalitions), atol=0.3)
