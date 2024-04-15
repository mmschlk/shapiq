"""This module contains a summary of all interaction indices and game theoretic concepts available
in the `shapiq` package."""

ALL_AVAILABLE_CONCEPTS = {
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
        "source": "Banzhaf III, J. F. (1965). Weighted Voting Doesn’t Work: A Mathematical "
        "Analysis. Rutgers Law Review, 19, 317-343.",  # no doi
    },
    # Shapley Generalized Values
    "JointSV": {
        "name": "Joint Shapley Values",
        "source": "https://openreview.net/forum?id=vcUmUvQCloe",
    },
    # Moebius Transformation
    "Moebius": {
        "name": "Moebius Transformation",
        "source": "https://doi.org/10.2307/2525487",
        # see also 2.10 in "Michel Grabisch (2016) Set Functions, Games and Capacities in Decision
        # Making" Link: http://www.getniif.com/component/rsfiles/previsualizar?path=Set%2BFunctions%2BGames%2Band%2BCapacities%2Bin%2BDecision%2BMaking%2B-%2BMichel%2BGrabisch%2BSpringer.pdf
    },
}

ALL_AVAILABLE_INDICES: set[str] = set(ALL_AVAILABLE_CONCEPTS.keys())
