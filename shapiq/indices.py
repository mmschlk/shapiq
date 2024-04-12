"""This module contains a summary of all interaction indices and game theoretic concepts available
in the `shapiq` package."""

ALL_AVAILABLE_CONCEPTS = {
    "Moebius": {
        "name": "Moebius Transformation",
        "source": "",  # TODO: Add source
    },
    # Base Interactions
    "SII": {
        "name": "Shapley Interaction Index",
        "source": "https://link.springer.com/article/10.1007/s001820050125",
    },
    "BII": {
        "name": "Banzhaf Interaction Index",
        "source": "https://link.springer.com/article/10.1007/s001820050125",
    },
    "CHII": {
        "name": "Chaining Interaction Index",
        "source": "https://link.springer.com/chapter/10.1007/978-94-017-0647-6_5",
    },
    # Base Generalized Values
    "SGV": {
        "name": "Shapley Generalized Value",
        "source": "https://doi.org/10.1016/S0166-218X(00)00264-X",
    },
    "BGV": {
        "name": "Banzhaf Generalized Value",
        "source": "https://doi.org/10.1016/S0166-218X(00)00264-X",
    },
    "CHGV": {
        "name": "Chaining Generalized Value",
        "source": "https://doi.org/10.1016/j.dam.2006.05.002",
    },
    # Shapley Interactions
    "k-SII": {
        "name": "k-Shapley Interaction Index",
        "source": "https://proceedings.mlr.press/v206/bordt23a.html",
    },
    "STIII": {
        "name": "Shapley-Taylor Interaction Index",
        "source": "https://proceedings.mlr.press/v119/sundararajan20a.html",
    },
    "FSIII": {
        "name": "Faithful Shapley Interaction Index",
        "source": "https://jmlr.org/papers/v24/22-0202.html",
    },
    "kADD-SHAP": {
        "name": "k-additive Shapley Values",
        "source": "https://doi.org/10.1016/j.artint.2023.104014",
    },
    # Probabilistic Values
    "SV": {
        "name": "Shapley Value",
        "source": "https://doi.org/10.1515/9781400881970-018",
    },
    "BV": {
        "name": "Banzhaf Value",
        "source": "",  # TODO: Add source
    },
    # Shapley Generalized Values
    "JointSV": {
        "name": "JointSV",
        "source": "https://openreview.net/forum?id=vcUmUvQCloe",
    },
}

ALL_AVAILABLE_INDICES: set[str] = set(ALL_AVAILABLE_CONCEPTS.keys())
