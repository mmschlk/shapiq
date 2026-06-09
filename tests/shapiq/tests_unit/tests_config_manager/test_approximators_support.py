from __future__ import annotations

from pydantic import ValidationError

# Import the approximator module to access class objects and SUPPORT lists
import shapiq.approximator as appr
from config_manager import GroundTruthConfig, MVPRunConfig
from config_manager.constants import ALL_SUPPORTED_APPROXIMATORS

# Mapping of indices to their respective registry lists in shapiq
INDEX_LISTS = {
    "SV": getattr(appr, "SV_APPROXIMATORS", []),
    "SII": getattr(appr, "SII_APPROXIMATORS", []),
    "STII": getattr(appr, "STII_APPROXIMATORS", []),
    "FSII": getattr(appr, "FSII_APPROXIMATORS", []),
    "FBII": getattr(appr, "FBII_APPROXIMATORS", []),
}


def supported_indices_for_class(cls) -> list[str]:
    """Returns a list of index names that support the given approximator class."""
    supported = []
    for idx_name, lst in INDEX_LISTS.items():
        if cls in lst:
            supported.append(idx_name)
    return supported


def main() -> None:
    for name in ALL_SUPPORTED_APPROXIMATORS:
        # 1. Assert that the approximator class actually exists in the shapiq module
        assert hasattr(appr, name), (
            f"Approximator '{name}' defined in ALL_SUPPORTED_APPROXIMATORS "
            f"was not found in shapiq.approximator module."
        )

        cls = getattr(appr, name)
        supported_indices = supported_indices_for_class(cls)

        # 2. Assert that the class is registered in at least one index list
        assert supported_indices, (
            f"Approximator '{name}' is present in the module but is not registered "
            f"in any of the INDEX_LISTS (SV, SII, STII, FSII, FBII)."
        )

        # 3. Verify that MVPRunConfig can be successfully instantiated for each supported index
        for idx in supported_indices:
            try:
                # Attempt to initialize the configuration to trigger Pydantic validation
                _ = MVPRunConfig(
                    game="CaliforniaHousing",
                    game_family="local_xai",
                    index=idx,
                    max_order=1 if idx == "SV" else 2,
                    n_players=14,
                    game_seed=0,
                    approximators=[name],
                    budgets=[100],
                    seeds=[0],
                    ground_truth=GroundTruthConfig(strategy="compute", method="ExactComputer"),
                )
            except ValidationError as e:
                # Raise an AssertionError with detailed Pydantic error trace if validation fails
                msg = (
                    f"Configuration validation FAILED for approximator '{name}' "
                    f"with index '{idx}'.\nPydantic Error Details:\n{e}"
                )
                raise AssertionError(msg) from e
            except Exception as e:
                # Catch any other unexpected runtime exceptions
                msg = (
                    f"Unexpected ERROR while validating config for '{name}' "
                    f"with index '{idx}': {type(e).__name__}: {e}"
                )
                raise AssertionError(msg) from e


if __name__ == "__main__":
    main()
