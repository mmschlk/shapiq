"""Summary of all interaction indices and game theoretic concepts available
in ``shapiq``."""

ALL_AVAILABLE_CONCEPTS: dict[str, dict] = {
    # Base Interactions
    "SII": {
        "name": "Shapley Interaction Index",
        "source": "https://link.springer.com/article/10.1007/s001820050125",
        "generalizes": "SV",
    },
    "BII": {
        "name": "Banzhaf Interaction Index",
        "source": "https://link.springer.com/article/10.1007/s001820050125",
        "generalizes": "BV",
    },
    "CHII": {
        "name": "Chaining Interaction Index",
        "source": "https://link.springer.com/chapter/10.1007/978-94-017-0647-6_5",
        "generalizes": "SV",
    },
    "Co-Moebius": {
        "name": "External Interaction Index",
        "source": "https://www.sciencedirect.com/science/article/abs/pii/S0899825605000278",
        "generalizes": None,
    },
    # Base Generalized Values
    "SGV": {
        "name": "Shapley Generalized Value",
        "source": "https://doi.org/10.1016/S0166-218X(00)00264-X",
        "generalizes": "SV",
    },
    "BGV": {
        "name": "Banzhaf Generalized Value",
        "source": "https://doi.org/10.1016/S0166-218X(00)00264-X",
        "generalizes": "BV",
    },
    "CHGV": {
        "name": "Chaining Generalized Value",
        "source": "https://doi.org/10.1016/j.dam.2006.05.002",
        "generalizes": "SV",
    },
    "IGV": {
        "name": "Internal Generalized Value",
        "source": "https://doi.org/10.1016/j.dam.2006.05.002",
        "generalizes": None,
    },
    "EGV": {
        "name": "External Generalized Value",
        "source": "https://doi.org/10.1016/j.dam.2006.05.002",
        "generalizes": None,
    },
    # Shapley Interactions
    "k-SII": {
        "name": "k-Shapley Interaction Index",
        "source": "https://proceedings.mlr.press/v206/bordt23a.html",
        "generalizes": "SII",
    },
    "STII": {
        "name": "Shapley-Taylor Interaction Index",
        "source": "https://proceedings.mlr.press/v119/sundararajan20a.html",
        "generalizes": "SV",
    },
    "FSII": {
        "name": "Faithful Shapley Interaction Index",
        "source": "https://jmlr.org/papers/v24/22-0202.html",
        "generalizes": "SV",
    },
    "kADD-SHAP": {
        "name": "k-additive Shapley Values",
        "source": "https://doi.org/10.1016/j.artint.2023.104014",
        "generalizes": "SV",
    },
    # Banzhaf Interactions
    "FBII": {
        "name": "Faithful Banzhaf Interaction Index",
        "source": "https://jmlr.org/papers/v24/22-0202.html",
        "generalizes": "BII",
    },
    # Probabilistic Values
    "SV": {
        "name": "Shapley Value",
        "source": "https://doi.org/10.1515/9781400881970-018",
        "generalizes": None,
    },
    "BV": {
        "name": "Banzhaf Value",
        "source": "Banzhaf III, J. F. (1965). Weighted Voting Doesn’t Work: A Mathematical "
        "Analysis. Rutgers Law Review, 19, 317-343.",  # no doi
        "generalizes": None,
    },
    # Shapley Generalized Values
    "JointSV": {
        "name": "Joint Shapley Values",
        "source": "https://openreview.net/forum?id=vcUmUvQCloe",
        "generalizes": "SV",
    },
    # Moebius Transformation
    "Moebius": {
        "name": "Moebius Transformation",
        "source": "https://doi.org/10.2307/2525487",
        # see also 2.10 in "Michel Grabisch (2016) Set Functions, Games and Capacities in Decision
        # Making" Link: http://www.getniif.com/component/rsfiles/previsualizar?path=Set%2BFunctions%2BGames%2Band%2BCapacities%2Bin%2BDecision%2BMaking%2B-%2BMichel%2BGrabisch%2BSpringer.pdf
        "generalizes": None,
    },
}

ALL_AVAILABLE_INDICES: set[str] = set(ALL_AVAILABLE_CONCEPTS.keys())

AVAILABLE_INDICES_REGRESSION = {"k-SII", "SII", "kADD-SHAP", "FSII"}
AVAILABLE_INDICES_MONTE_CARLO = {"k-SII", "SII", "STII", "FSII", "SV", "CHII", "BII"}

AVAILABLE_INDICES_FOR_APPROXIMATION: set[str] = (
    {
        "SII",
        "BII",
        "k-SII",
        "STII",
        "FSII",
        "SV",
        "BV",
        "kADD-SHAP",
        "CHII",
        "FBII",
    }
    .union(AVAILABLE_INDICES_REGRESSION)
    .union(AVAILABLE_INDICES_MONTE_CARLO)
)


def index_generalizes_sv(index: str) -> bool:
    """Checks if the given index generalizes the Shapley Value.

    Args:
        index: The interaction index.

    Returns:
        ``True`` if the index generalizes the Shapley Value, ``False`` otherwise.

    Examples:
        >>> index_generalizes_sv("SII")
        True
        >>> index_generalizes_sv("SV")
        False
        >>> index_generalizes_sv("k-SII")
        True
        >>> index_generalizes_sv("BV")
        False
    """
    return ALL_AVAILABLE_CONCEPTS[index]["generalizes"] == "SV"


def index_generalizes_bv(index: str) -> bool:
    """Checks if the given index generalizes the Banzhaf Value.

    Args:
        index: The interaction index.

    Returns:
        ``True`` if the index generalizes the Banzhaf Value, ``False`` otherwise.

    Examples:
        >>> index_generalizes_bv("BII")
        True
        >>> index_generalizes_bv("SII")
        False
        >>> index_generalizes_bv("BV")
        False
    """
    return ALL_AVAILABLE_CONCEPTS[index]["generalizes"] == "BV"


def get_computation_index(index: str) -> str:
    """Returns the base index of a given interaction index.

    The base index is the index without any aggregation or transformation. The base index is used
    in the approximators to compute the interaction values. After the computation, the interaction
    values are aggregated to original interaction index.

    Args:
        index: The interaction index.

    Returns:
        The base index of the interaction index.

    Examples:
        >>> get_computation_index("k-SII")
        "SII"
        >>> get_computation_index("SII")
        "SII"
        >>> get_computation_index("SV")
        "SII"
        >>> get_computation_index("BV")
        "BII"
    """
    if "k-" in index:
        return index.split("-")[1]  # remove the k- prefix
    if index == "SV":  # for SV we return SII with max order 1
        return "SII"  # TODO this may break with FSII and KernelSHAP
    if index == "BV":  # for SV we return SII with max order 1
        return "BII"
    return index


def get_index_from_computation_index(index: str, max_order: int) -> str:
    """Returns the original interaction index from the base index and the maximum order.

    Args:
        index: The base interaction index.
        max_order: The maximum order of the interaction index.

    Returns:
        The original interaction index.
    """
    if max_order == 1:
        if index == "BII":
            return "BV"
        if index in {"SII", "STII", "FSII"}:
            return "SV"
    return index


def is_index_aggregated(index: str) -> bool:
    """Checks if the given index is an aggregated interaction index as denoted by a ``-``.

    Args:
        index: The interaction index.

    Returns:
        ``True`` if the index is an aggregated interaction index, ``False`` otherwise.

    Examples:
        >>> is_index_aggregated("k-SII")
        True
        >>> is_index_aggregated("SII")
        False
        >>> is_index_aggregated("SV")
        False
        >>> is_index_aggregated("k-FSII")
        True
    """
    return "k-" in index


def is_empty_value_the_baseline(index: str) -> bool:
    """Checks if the empty value stored in the interaction values is the baseline value. This is
    only not the case for the Shapley Interaction Index.

    Args:
        index: The interaction index.

    Returns:
        ``True`` if the empty value is the baseline value, ``False`` otherwise.

    Examples:
        >>> is_empty_value_the_baseline("SII")
        False
        >>> is_empty_value_the_baseline("SV")
        True
        >>> is_empty_value_the_baseline("k-SII")
        True
    """
    return index != "SII"
