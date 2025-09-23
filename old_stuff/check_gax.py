from shapiq.games.benchmark.local_xai.benchmark_tabular import (
    ForestFires,
    RealEstate,
    BreastCancer,
    NHANESI,
    WineQuality,
    CommunitiesAndCrime,
    IndependentLinear60,
    Corrgroups60,
)

from shapiq.approximator.regression.polyshap import (
    ShapleyGAX,
    ExplanationBasisGenerator,
)
from shapiq.games.benchmark.local_xai import AdultCensus, CaliforniaHousing, BikeSharing


if __name__ == "__main__":
    game = CaliforniaHousing(model_name="neural_network", random_state=40)
    n_players = game.n_players
    explanation_basis = ExplanationBasisGenerator(N=set(range(n_players)))
    kadd2 = explanation_basis.generate_kadd_explanation_basis(max_order=2)
    stoch = explanation_basis.generate_stochastic_explanation_basis(
        n_explanation_terms=20
    )
    kadd1 = explanation_basis.generate_kadd_explanation_basis(max_order=1)

    PAIRING = True
    REPLACEMENT = False

    approximator_1 = ShapleyGAX(
        n=n_players,
        explanation_basis=kadd1,
        random_state=10,
        pairing_trick=PAIRING,
        replacement=REPLACEMENT,
    )
    approximator_2 = ShapleyGAX(
        n=n_players,
        explanation_basis=kadd2,
        random_state=10,
        pairing_trick=PAIRING,
        replacement=REPLACEMENT,
    )
    approximator_stoch = ShapleyGAX(
        n=n_players,
        explanation_basis=stoch,
        random_state=10,
        pairing_trick=PAIRING,
        replacement=REPLACEMENT,
    )

    shap_1 = approximator_1.approximate(game=game, budget=250)
    shap_2 = approximator_2.approximate(game=game, budget=250)
    shap_stoch = approximator_stoch.approximate(game=game, budget=250)

    import numpy as np

    print(
        np.max(
            approximator_1._sampler.coalitions_matrix
            == approximator_2._sampler.coalitions_matrix
        )
    )
    print(shap_1.values)
    print(shap_2.values)
    print(shap_stoch.values)
